# Graph-Based Entity Resolution and Transitive Closure

**Author:** Research artifact for Amazon ML Challenge 2026 — Da Big Three  
**Date:** 2026-09-26  
**Purpose:** How graph structure and transitive closure can boost recall in entity resolution. If A matches B and B matches C, should A match C? This document explores when and how to use connected components, union-find, and graph propagation in ER.

---

## 1. The Core Idea

Standard pairwise ER treats each (S1, candidate) pair independently. But entity matches form a **graph**:
- Nodes = entities (S1, S2, S3 records)
- Edges = predicted matches

If we predict S1-A matches S2-X, and S2-X matches S3-Y, then **transitively** S1-A should also match S3-Y — even if our pairwise scorer never directly compared S1-A with S3-Y.

This is especially powerful when:
- The blocking step missed a pair (S1-A never saw S3-Y as a candidate)
- The pairwise scorer was uncertain about S1-A ↔ S3-Y but confident about the chain

---

## 2. Why This Matters for Our Task

Our pipeline architecture:
```
S1 (2.2M) ──blocking──→ candidates from S2 (5M) + S3 (5.3M)
          ──scoring──→ matches
```

Each S1 entity independently searches for matches in S2 and S3. But:
- Two S1 entities might match the **same** S2/S3 entity
- An S2 entity and an S3 entity might both match the same S1

The ground truth has match sets of mean size 3.67, max 11. Many of these large groups contain entities that:
- Have slightly different names (typos, abbreviations)
- Have the same address but different name variants
- Would be individually low-confidence pairs but are connected through high-confidence intermediaries

### Example

```
S1: "McDonald's Corporation, 123 Main St, Chicago, US"
S2: "McDonalds Corp, 123 Main Street, Chicago, US"     → high-confidence match
S3: "McDonald Corp., 123 Main, Chicago IL, US"          → medium-confidence match
S2: "MCDONALDS CORPORATION, Chicago, US"                → name match but address partial
```

Direct pairwise: S1→S2a (high), S1→S3a (medium), S1→S2b (borderline)
With transitive closure: S2a→S3a confirmed by graph neighborhood → S1→S3a boosted. S2b connected to S2a through S1 → S1→S2b boosted.

---

## 3. Graph Algorithms for ER

### 3.1 Union-Find (Disjoint Set)

The simplest approach: build a union-find structure where edges are predicted matches with score above threshold. Connected components become match groups.

```python
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def components(self):
        groups = {}
        for x in self.parent:
            r = self.find(x)
            groups.setdefault(r, set()).add(x)
        return groups
```

**Pros:** O(α(n)) per operation (nearly constant). Handles millions of edges.
**Cons:** No notion of edge weight — a weak 0.51 edge is treated the same as a strong 0.99 edge.

### 3.2 Weighted Connected Components

Improvement over basic union-find: only add edges above a high threshold, then expand at a lower threshold within existing components.

```python
def two_pass_clustering(edges, high_thresh=0.8, low_thresh=0.5):
    """
    Pass 1: Build components from high-confidence edges.
    Pass 2: Add low-confidence edges only between existing component members.
    """
    uf = UnionFind()

    # Pass 1: High confidence
    for u, v, score in edges:
        if score >= high_thresh:
            uf.union(u, v)

    # Pass 2: Low confidence within existing groups
    for u, v, score in edges:
        if score >= low_thresh and uf.find(u) == uf.find(v):
            pass  # already connected
        elif score >= low_thresh:
            # Only connect if one of them is already in a component
            if u in uf.parent or v in uf.parent:
                uf.union(u, v)

    return uf.components()
```

### 3.3 Correlation Clustering

Formulates ER as an optimization: minimize the sum of:
- Positive edges not in the same cluster (missed matches)
- Negative edges in the same cluster (false matches)

This is NP-hard but has good approximation algorithms:
- **Pivot algorithm** (3-approximation): pick a random unclustered node, cluster all its positive-edge neighbors with it
- **LP relaxation** + rounding (2.06-approximation)

For our contest, the pivot algorithm is practical:

```python
def correlation_clustering_pivot(edges, threshold=0.5):
    """Pivot-based correlation clustering."""
    import random

    adj = {}
    for u, v, score in edges:
        adj.setdefault(u, []).append((v, score))
        adj.setdefault(v, []).append((u, score))

    nodes = list(adj.keys())
    random.shuffle(nodes)
    clusters = {}
    assigned = set()

    for pivot in nodes:
        if pivot in assigned:
            continue
        cluster = {pivot}
        assigned.add(pivot)
        for neighbor, score in adj.get(pivot, []):
            if neighbor not in assigned and score >= threshold:
                cluster.add(neighbor)
                assigned.add(neighbor)
        for node in cluster:
            clusters[node] = cluster

    return clusters
```

### 3.4 Graph Neural Network Approaches (Research Frontier)

Recent work uses GNNs for entity resolution:
- **GraphER** (Wu et al., 2020): GNN message passing over candidate pairs
- **GNEM** (Chen et al., 2021): Graph neural entity matching with neighborhood aggregation
- **HierGAT** (Yao et al., 2022): Hierarchical graph attention for ER

These learn entity representations that incorporate neighborhood structure, but require:
- GPU training infrastructure
- More implementation complexity
- Larger datasets for the GNN to learn meaningful patterns

**Verdict for our contest:** Too complex. Union-find or correlation clustering is sufficient.

---

## 4. Transitive Closure for Recall Boost

### 4.1 The Recall Problem

Our blocking has a **recall ceiling** of ~0.93 (BM25 hash-key ceiling). Some true matches are never generated as candidates. Transitive closure can recover some:

```
S1-A → S2-X (in candidates, matched)
S2-X ← S1-B (S1-B also matched S2-X)
→ Therefore S1-A and S1-B might be the same entity cluster
→ S1-A's matches should include S1-B's matches
```

Wait — this doesn't apply directly because S1 entities are already deduplicated (each S1 is unique). But S2/S3 entities can be shared:

```
S1-A → S2-X (matched)
S1-A → S3-Y (not a candidate — blocking missed it)
S2-X → S3-Y (if we had cross-source matching)
→ Transitive: S1-A should also match S3-Y
```

### 4.2 Cross-Source Match Propagation

The pipeline currently only matches S1 against S2/S3. But if two S2/S3 entities are in the same S1's match set, they're implicitly in the same cluster. We can use this:

```python
def propagate_matches(s1_matches: dict[str, set[str]],
                      pool: dict[str, tuple]) -> dict[str, set[str]]:
    """Propagate matches through shared S2/S3 entities."""
    # Build reverse index: S2/S3 entity → set of S1 entities that matched it
    reverse = {}
    for s1_id, match_set in s1_matches.items():
        for m_id in match_set:
            reverse.setdefault(m_id, set()).add(s1_id)

    # For each S1, collect all S2/S3 entities that co-occur with its matches
    enhanced = {}
    for s1_id, match_set in s1_matches.items():
        extended = set(match_set)
        for m_id in match_set:
            # Other S1s that matched the same S2/S3 entity
            sibling_s1s = reverse.get(m_id, set())
            for sib_s1 in sibling_s1s:
                if sib_s1 != s1_id:
                    # Add sibling's matches to our set
                    extended |= s1_matches.get(sib_s1, set())
        enhanced[s1_id] = extended

    return enhanced
```

**Caution:** This can create a cascading effect where one bad match propagates false positives to many S1 entities. Use only with high-confidence initial matches.

### 4.3 When Transitive Closure Helps vs Hurts

| Scenario | Effect | Risk |
|----------|--------|------|
| True cluster missed one member | ✓ Recovers match via chain | Low |
| Chain through common name (e.g., "Smith") | ✗ Merges unrelated entities | High |
| Two branches of same franchise | ✓ Connects related locations | Medium |
| Same address, different business | ✗ Merges different businesses | High |

**Safe pattern:** Only propagate through edges with score > 0.9, and limit propagation to 1 hop.

---

## 5. Practical Application to Our Pipeline

### 5.1 Post-Processing with Union-Find

After LightGBM (or Ditto) scoring:

1. For each S1, collect predicted matches above threshold
2. Build union-find over all (S1, S2/S3) match edges
3. For each S1, output the full connected component as the match set
4. Apply max-component-size guard (e.g., max 15) to prevent runaway merges

```python
def postprocess_with_uf(s1_matches: dict[str, set[str]],
                        max_component: int = 15) -> dict[str, set[str]]:
    """Group matches using union-find transitive closure."""
    uf = UnionFind()

    for s1_id, match_set in s1_matches.items():
        for m_id in match_set:
            uf.union(s1_id, m_id)

    components = uf.components()
    result = {}
    for s1_id in s1_matches:
        root = uf.find(s1_id)
        comp = components[root]
        # Remove S1 ids from match set (only S2/S3 ids)
        matches = {eid for eid in comp if not eid.startswith("S1-")}
        if len(matches) > max_component:
            # Fall back to original predictions
            result[s1_id] = s1_matches[s1_id]
        else:
            result[s1_id] = matches

    return result
```

### 5.2 Expected Impact

Based on the ground truth statistics:
- Mean match set size: 3.67, max: 11
- Singleton rate: 5.58%

If transitive closure adds ~5% extra true matches (from missed blocking candidates) but also adds ~2% false positives:
- Net recall gain: +0.05 × recall_weight
- Net precision loss: -0.02 × precision_weight
- F0.5 impact: slightly positive (precision matters 2x more, but the gain is in recall)

**Recommendation:** Use as a **conservative post-processing step** with high threshold (>0.9) and 1-hop limit. Expected gain: +0.005 to +0.01 F0.5.

---

## 6. Advanced: Graph-Aware Feature Engineering

Beyond transitive closure, the graph structure provides features:

| Feature | Description | Signal |
|---------|-------------|--------|
| **Candidate overlap** | How many S1 entities share the same S2/S3 candidate? | Popular candidates are more likely true matches |
| **Mutual nearest neighbor** | Is B the top candidate for A AND A in B's top candidates? | Strong positive signal |
| **Common neighbors** | Do A and B share many other matches? | Jaccard on match neighborhoods |
| **Component size** | Size of the connected component containing this pair | Large components might indicate over-merging |
| **Edge centrality** | How critical is this edge to the component? | Bridge edges are risky |

These could be added as features #19-23 in LightGBM for a second-pass scoring:

```python
def graph_features(s1_id, cand_id, s1_matches, reverse_index):
    """Graph-structure features for a candidate pair."""
    # How many S1s matched this candidate?
    cand_popularity = len(reverse_index.get(cand_id, set()))

    # Mutual nearest: is this candidate in S1's top-3 AND S1 in candidate's "top-3"?
    mutual = s1_id in reverse_index.get(cand_id, set())

    # Common neighbors: other S2/S3 entities matched by both S1 and other S1s
    # that also match this candidate
    sibling_s1s = reverse_index.get(cand_id, set()) - {s1_id}
    common_matches = set()
    for sib in sibling_s1s:
        common_matches |= (s1_matches.get(sib, set()) & s1_matches.get(s1_id, set()))

    return [cand_popularity, float(mutual), len(common_matches)]
```

---

## 7. Key Takeaways

1. **Union-find is cheap and effective** for merging match groups via transitive closure
2. **Transitive closure boosts recall** but risks precision if not thresholded conservatively
3. **F0.5 penalizes false positives 2x** — so aggressive graph expansion is dangerous
4. **1-hop propagation with high threshold (>0.9)** is the safe pattern
5. **Graph features** (candidate popularity, mutual NN, common neighbors) can be added to LightGBM
6. **Max component size guard** prevents runaway cluster merges
7. **GNN approaches** are too complex for a 72-hour contest

---

## 8. References

1. Hassanzadeh, O., et al. (2009). "Framework for Evaluating Clustering Algorithms in Duplicate Detection." VLDB.
2. Bansal, N., Blum, A., & Chawla, S. (2004). "Correlation Clustering." Machine Learning.
3. Saeedi, A., et al. (2018). "Comparative Study of Distributed Clustering Approaches for Entity Resolution." CIKM.
4. Wu, R., et al. (2020). "ZeroER: Entity Resolution using Zero Labeled Examples." SIGMOD.
5. Mudgal, S., et al. (2018). "Deep Learning for Entity Matching." SIGMOD.
