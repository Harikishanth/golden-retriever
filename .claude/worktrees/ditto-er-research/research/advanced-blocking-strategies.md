# Advanced Blocking Strategies for Entity Resolution

**Author:** Research artifact for Amazon ML Challenge 2026 — Da Big Three  
**Date:** 2026-09-26  
**Purpose:** Comprehensive survey of blocking methods beyond hash-key and BM25. Blocking sets the recall ceiling — you cannot match a pair you never generate as a candidate. Our current ceiling is ~0.93 with BM25; reaching 0.97+ requires union of multiple blocking strategies.

---

## 1. The Blocking Problem

We have 1.73M test S1 entities and 9.97M S2+S3 entities. Scoring all pairs:
- 1.73M × 9.97M = **17.2 trillion pairs** — impossible
- At 10μs per pair = **5.5 years**

Blocking reduces this to ~150 candidates per S1 = 260M pairs — manageable.

The tradeoff:
- **Recall ceiling** = fraction of true matches present in candidate sets
- **Reduction ratio** = candidate pairs / all possible pairs
- **Speed** = time to generate all candidate sets

| Method | Typical Recall | Reduction Ratio | Speed |
|--------|---------------|-----------------|-------|
| Hash-key blocking | 0.80-0.93 | 10⁻⁴ to 10⁻⁵ | Very fast |
| BM25 token retrieval | 0.90-0.96 | 10⁻⁴ | Fast |
| Sorted neighborhood | 0.85-0.92 | 10⁻⁴ | Fast |
| LSH / MinHash | 0.90-0.95 | 10⁻⁴ | Fast |
| Dense embedding + FAISS | 0.93-0.98 | 10⁻⁴ | Medium (GPU) |
| Union of above | 0.97-0.99 | 10⁻³ | Sum of parts |

---

## 2. Methods Already Implemented

### 2.1 Hash-Key Blocking (v1-v3)

Build compound keys → entities with the same key are candidates.

Our 7 key types:
1. `(country, squashed_name)` — exact name match within country
2. `(country, squashed_name[:5])` — name prefix
3. `(country, first_name_token, city_token)` — first token + city
4. `(country, house_number, street_token)` — house + street
5. `(country, house_number, city_token)` — house + city
6. `(country, full_addr_key)` — full normalized address
7. `(country, name_token_2gram)` — 2-gram from name tokens

**Ceiling:** 0.9287 (v3, 100k holdout). Hard limit — cannot exceed without adding non-hash methods.

### 2.2 BM25 Token Retrieval (v4, current)

Index individual tokens with IDF weighting:
- Name tokens: IDF × 2.0
- Address tokens: IDF × 1.2
- House numbers: +8.0 fixed bonus

Score = sum of weighted shared tokens. Ranked, top-150.

**Expected ceiling:** 0.93-0.95 (higher than hash keys because it's fuzzy — partial token overlap counts).

---

## 3. Methods Not Yet Implemented

### 3.1 MinHash / Locality-Sensitive Hashing (LSH)

MinHash approximates Jaccard similarity using random hash functions. Two entities with high token Jaccard are likely to share MinHash signatures.

```python
from datasketch import MinHash, MinHashLSH

def build_minhash_index(pool, num_perm=128, threshold=0.3):
    """Build MinHash LSH index for fast approximate Jaccard search."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for eid, (name, addr, country) in pool.items():
        m = MinHash(num_perm=num_perm)
        # Character 4-grams of normalized name+address
        text = fold(name) + " " + fold(addr)
        for i in range(len(text) - 3):
            m.update(text[i:i+4].encode('utf-8'))
        lsh.insert(eid, m)

    return lsh

def query_minhash(lsh, name, addr, num_perm=128):
    """Find approximate Jaccard neighbors."""
    m = MinHash(num_perm=num_perm)
    text = fold(name) + " " + fold(addr)
    for i in range(len(text) - 3):
        m.update(text[i:i+4].encode('utf-8'))
    return lsh.query(m)
```

**Pros:**
- Catches character-level similarity (typos, reordering)
- Sub-linear query time O(1) per query
- Language-agnostic (character n-grams)
- `datasketch` library is MIT licensed

**Cons:**
- Requires tuning: number of permutations, threshold, n-gram size
- Character n-grams may not align well with word boundaries
- Memory: MinHash signatures for 10M entities ≈ 5GB at 128 perms

**Expected recall boost over BM25:** +0.02-0.05 (catches typo variants BM25 misses)

### 3.2 Dense Embedding Retrieval (FAISS)

Encode entities as dense vectors, find nearest neighbors via approximate nearest neighbor search.

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_dense_index(pool, model_name="paraphrase-multilingual-MiniLM-L12-v2",
                      batch_size=1024):
    """Encode pool entities and build FAISS index."""
    model = SentenceTransformer(model_name)

    texts = [f"{name} {addr}" for eid, (name, addr, country) in pool.items()]
    eids = list(pool.keys())

    embeddings = model.encode(texts, batch_size=batch_size,
                              show_progress_bar=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
    index.add(embeddings.astype(np.float32))

    return index, eids, model

def query_dense(index, eids, model, query_name, query_addr, top_k=100):
    """Find top-K nearest neighbors by embedding cosine."""
    query_text = f"{query_name} {query_addr}"
    query_emb = model.encode([query_text], normalize_embeddings=True)
    scores, indices = index.search(query_emb.astype(np.float32), top_k)
    return [(eids[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i >= 0]
```

**Pros:**
- **Semantic matching** — understands "Corp" = "Corporation", "Rue" = "Street"
- **Multilingual** — critical for France
- Handles entity reformulations that share meaning but not tokens
- FAISS on GPU: 10M vectors in < 1 second per query

**Cons:**
- Requires GPU for encoding (10M entities × 384 dim ≈ 40 min on T4)
- Memory: 10M × 384 × 4 bytes = 15GB (fits in 29GB Kaggle RAM)
- Can miss entities that are textually similar but semantically different (same address, different business)

**Expected recall:** 0.93-0.98 (strongest on French, where BM25 IDF weights fail)

### 3.3 Sorted Neighborhood

Sort entities by a key, then slide a window of size W over the sorted list. Entities within the same window are candidates.

```python
def sorted_neighborhood_blocking(s1_records, pool, key_fn, window=10):
    """Sorted neighborhood with configurable sort key."""
    # Combine all entities
    all_entities = []
    for eid, (name, addr, country) in {**s1_records, **pool}.items():
        all_entities.append((key_fn(name, addr, country), eid))

    all_entities.sort(key=lambda x: x[0])

    # Slide window
    candidates = {}
    for i, (key, eid) in enumerate(all_entities):
        if not eid.startswith("S1-"):
            continue
        window_eids = {
            all_entities[j][1] for j in range(max(0, i-window), min(len(all_entities), i+window+1))
            if all_entities[j][1] != eid and not all_entities[j][1].startswith("S1-")
        }
        candidates[eid] = window_eids

    return candidates
```

**Sort keys to try:**
1. `sorted(name_tokens(name))` — alphabetically sorted name tokens
2. `soundex(name_squash(name))` — phonetic sort
3. `name_squash(name)[:8]` — name prefix
4. `f"{country}_{postal_code}"` — geographic sort

**Pros:** Simple, deterministic, good with multiple sort keys (multi-pass)
**Cons:** Sensitive to key choice; doesn't handle long-range similarities

### 3.4 Canopy Clustering

Two-threshold approach:
1. Pick a random entity, find all entities within loose threshold T1 → canopy
2. Remove entities within tight threshold T2 < T1 from the random pool
3. Repeat until pool is empty

Entities in the same canopy are candidates.

```python
def canopy_blocking(entities, embeddings, t1=0.7, t2=0.9):
    """Canopy clustering for blocking."""
    pool = list(range(len(entities)))
    canopies = []

    while pool:
        pivot = pool[0]
        pivot_emb = embeddings[pivot]

        # Find entities within T1 (loose)
        sims = np.dot(embeddings[pool], pivot_emb)
        canopy = [pool[i] for i, s in enumerate(sims) if s >= t1]
        canopies.append(canopy)

        # Remove entities within T2 (tight)
        remove = {pool[i] for i, s in enumerate(sims) if s >= t2}
        pool = [p for p in pool if p not in remove]

    return canopies
```

**Pros:** Produces overlapping canopies (entity can be in multiple). Good recall.
**Cons:** Requires precomputed embeddings. Order-dependent.

### 3.5 Rule-Based Meta-Blocking

After generating candidates from multiple blockers, prune the union using lightweight rules:

```python
def meta_blocking(candidate_sets, max_per_entity=200, min_blockers=1):
    """Prune union of blocking candidates using meta-rules."""
    final = {}
    for s1_id, cand_dict in candidate_sets.items():
        # cand_dict: candidate_id -> set of blockers that generated it
        scored = []
        for cand_id, blockers in cand_dict.items():
            # More blockers = more likely a true match
            score = len(blockers)
            scored.append((cand_id, score))

        scored.sort(key=lambda x: -x[1])
        # Keep candidates that appeared in >= min_blockers methods
        final[s1_id] = [c for c, s in scored[:max_per_entity] if s >= min_blockers]

    return final
```

**This is the key to combining multiple blockers:** BM25 finds some matches, MinHash finds others, dense retrieval finds yet more. The union has high recall but too many candidates. Meta-blocking ranks candidates by how many blockers agree.

---

## 4. Union Blocking Strategy

The optimal approach for our contest combines multiple blockers:

```
                    ┌─── BM25 (token IDF) ──────┐
S1 entity ──────────┼─── MinHash (char 4-grams) ─┼──→ Union → Meta-Block → CAP → Score
                    └─── Dense (embeddings) ─────┘
```

### Expected Recall by Combination

| Blocking | Recall | Candidates/S1 | Notes |
|----------|--------|---------------|-------|
| BM25 only | 0.93-0.95 | 150 | Current |
| MinHash only | 0.90-0.93 | 100-200 | Character-level |
| Dense only | 0.93-0.98 | 100 | Semantic |
| BM25 ∪ MinHash | 0.95-0.97 | 200-300 | +typo variants |
| BM25 ∪ Dense | 0.96-0.98 | 200-250 | +semantic matches |
| BM25 ∪ MinHash ∪ Dense | 0.97-0.99 | 250-400 | Maximum recall |
| Above + Meta-Block (≥2 agree) | 0.96-0.98 | 150-200 | Pruned |

### Implementation Plan

```python
def union_blocking(s1_id, s1_name, s1_addr, s1_country,
                   bm25_index, bm25_idf,
                   minhash_lsh,
                   dense_index, dense_eids, dense_model,
                   cap=200):
    """Union of three blocking methods."""
    cand_votes = {}

    # BM25
    bm25_cands, _ = candidates_for(bm25_index, s1_name, s1_addr, s1_country, bm25_idf)
    for c in bm25_cands:
        cand_votes.setdefault(c, set()).add("bm25")

    # MinHash
    mh_cands = query_minhash(minhash_lsh, s1_name, s1_addr)
    for c in mh_cands:
        cand_votes.setdefault(c, set()).add("minhash")

    # Dense
    dense_cands = query_dense(dense_index, dense_eids, dense_model,
                              s1_name, s1_addr, top_k=100)
    for c, score in dense_cands:
        cand_votes.setdefault(c, set()).add("dense")

    # Rank by number of blockers, then by BM25 score
    ranked = sorted(cand_votes.items(),
                    key=lambda x: (-len(x[1]), x[0]))
    return [c for c, _ in ranked[:cap]]
```

---

## 5. Blocking Evaluation Metrics

### 5.1 Pairs Completeness (PC) = Recall

```
PC = |true pairs in candidates| / |all true pairs|
```

This is what we call "recall ceiling." The most important metric.

### 5.2 Reduction Ratio (RR)

```
RR = 1 - |candidate pairs| / |all possible pairs|
```

Higher is better. Should be > 0.9999 for practical pipelines.

### 5.3 Pairs Quality (PQ)

```
PQ = |true pairs in candidates| / |candidate pairs|
```

The fraction of candidates that are actual matches. Higher PQ = less noise for the matcher.

### 5.4 F-measure for Blocking

```
F_block = 2 * PC * PQ / (PC + PQ)
```

Balanced metric. But for our contest, PC (recall) matters more than PQ (precision) because the matcher handles precision.

---

## 6. Computational Budget

For test set (1.73M S1, 9.97M S2+S3):

| Method | Build Time | Query Time | Total | Memory |
|--------|-----------|------------|-------|--------|
| BM25 index | 10 min | 30 min | 40 min | 8GB |
| MinHash LSH | 20 min | 15 min | 35 min | 5GB |
| Dense encode | 40 min (T4) | 20 min (FAISS) | 60 min | 15GB |
| Union + meta | — | +5 min | +5 min | +2GB |

**Total for union blocking:** ~2 hours on Kaggle T4 with 29GB RAM.
**Memory peak:** ~25GB (fits in Kaggle with careful management).

---

## 7. Key Takeaways

1. **Union of BM25 + Dense is the minimum** for reaching 0.96+ recall ceiling
2. **MinHash adds ~2% recall** by catching character-level variants BM25 misses
3. **Meta-blocking** prunes the union to manageable candidate set sizes
4. **Dense retrieval is critical for France** — the only method that understands cross-lingual semantics
5. **Sorted neighborhood** is a useful secondary blocker when you have good sort keys
6. **The recall ceiling determines the maximum possible F0.5** — investing in blocking has higher ROI than improving the matcher
7. **datasketch (MinHash, MIT) + faiss-cpu (MIT) + sentence-transformers (Apache 2.0)** — all contest-legal

---

## 8. References

1. Papadakis, G., et al. (2020). "Blocking and Filtering Techniques for Entity Resolution: A Survey." ACM Computing Surveys.
2. Christen, P. (2012). "A Survey of Indexing Techniques for Scalable Record Linkage and Deduplication." IEEE TKDE.
3. Steorts, R., et al. (2014). "A Comparison of Blocking Methods for Record Linkage." KDD Workshop.
4. Johnson, J., et al. (2021). "Billion-scale similarity search with GPUs (FAISS)." IEEE TPAMI.
5. Leskovec, J., Rajaraman, A., & Ullman, J. (2014). "Mining of Massive Datasets." Chapter 3: Locality-Sensitive Hashing.
6. Galhotra, S., et al. (2021). "BEER: Blocking for Effective Entity Resolution." VLDB.
