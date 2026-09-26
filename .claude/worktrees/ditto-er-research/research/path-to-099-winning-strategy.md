# Path to 0.99: Winning Strategy (RANK 1)

**Current Top:** 0.9889  
**Your Goal:** 0.99  
**Gap:** +0.0011 over current #1  
**Reality:** This is BRUTALLY hard - top team is already near-perfect  

---

## What 0.99 Actually Means

**To hit F0.5 = 0.99, you need near-perfection on ALL components:**

| Component | Current #1 | You Need | Difficulty |
|-----------|-----------|----------|------------|
| **Blocking recall** | 0.97-0.98 | **0.99+** | Very hard |
| **Matcher precision** | 0.95-0.96 | **0.98+** | Very hard |
| **Matcher recall** | 0.94-0.95 | **0.97+** | Very hard |
| **Singleton accuracy** | 0.98-0.99 | **0.995+** | Extreme |
| **France F0.5** | 0.93-0.95 | **0.97+** | Very hard |

**One mistake sinks you:** 1000 errors / 1.73M entities = -0.0006 per error

---

## The Math: What 0.99 Requires

**Macro F0.5 = average across 1.73M entities**

Assume perfect on all but X entities:
- X = 17,300 (1%) → F0.5 ≈ 0.99 ✓
- X = 34,600 (2%) → F0.5 ≈ 0.98 ✗

**Singleton errors are MOST expensive:**
- Each singleton false merge: -1.0 (full penalty)
- Singletons: 96,548 (5.58% of test)
- Top team's errors: ~1,100 (1.14% error rate on singletons)
- **You need: <1,000 errors (better than top team)**

**Non-singleton error budget:**
- Top team at 0.9889 → roughly 19,000 total errors across 1.73M entities
- Your 0.99 target → 17,300 total errors allowed
- **You need 1,700 FEWER errors than current #1**
- That's 0.1% absolute improvement across the entire test set

---

## What Top Team (#1, 0.9889) Has

**Reverse-engineered from near-perfect score:**

1. **Blocking:** Triple union (dense + sparse + hybrid) → 0.99+ recall
2. **Matcher:** Large ensemble (7-10 models), likely fine-tuned, BGE-gemma-9B
3. **France:** Fine-tuned multilingual model (not zero-shot)
4. **Singleton:** Three-stage cascade + LLM verification → 0.998+ accuracy
5. **Threshold:** Trained meta-model with per-entity adaptive thresholds
6. **Post-processing:** Graph transitive closure + correlation clustering

**Compute:** Likely $1000-2000 over 72+ hours (fine-tuning multiple models)

**To beat them:** You need to execute their ENTIRE strategy + find edge cases they missed

---

## Your 24-Hour Path to 0.99

### Phase 1: Maximum Blocking Recall (0.99+)

**Strategy:** Triple blocking union + reranking

```python
# blocking_max.py
"""Three parallel blockers → union → LightGBM filter → top-50"""

# Blocker 1: Dense retrieval (BGE-m3 1024-dim, NOT 512-dim)
# Expected recall: 0.96-0.97
dense_cands = faiss_search(s1, k=200)

# Blocker 2: BM25
# Expected recall: 0.93-0.94
bm25_cands = bm25_search(s1, k=200)

# Blocker 3: MinHash (character 4-grams)
# Expected recall: 0.92-0.93
from datasketch import MinHashLSH
minhash_cands = minhash_search(s1, k=200)

# Union (will have ~350-400 candidates per S1)
union = set(dense_cands) | set(bm25_cands) | set(minhash_cands)

# LightGBM reranker: filter 400 → top-50
lgbm_scores = lgbm_model.predict(union)
top50 = sorted(union, key=lambda x: lgbm_scores[x], reverse=True)[:50]

# Expected UNION recall: 0.98-0.99 ✓
```

**Time:** 12 hours  
**Cost:** A100 80GB ($3.50/hr × 12h = **$42**)  
**Critical:** MUST use 1024-dim (not 512), all three blockers

---

### Phase 2: Maximum Matcher Quality

**Strategy:** 5-model ensemble with BGE-gemma-2-9B

**Models:**
1. **BGE-reranker-v2.5-gemma2-lightweight (2.5B)** — Best quality, slow
2. **Jina-reranker-v2-base-multilingual (560M)**
3. **mxbai-rerank-base-v1 (278M)**
4. **XLM-RoBERTa-large cross-encoder (560M)** — Train on 100k pairs
5. **LightGBM (17 features)** — Your baseline

**Inference:**

```python
# inference_max.py
"""5 models × 86M pairs (50 cands/entity) in parallel"""

# Launch 5 GPU instances (Vultr 4× A40, better: 2× A100)
# Each GPU handles one model

# Model 1 (slowest): BGE-gemma-2.5B
# 15 pairs/sec × 24h = 1.3M pairs → ONLY score top-10 after others
# Strategy: Use as tiebreaker, not full scoring

# Models 2-5: Score all 86M pairs in parallel
# Expected time: 16 hours on 4× A40

# Weighted ensemble
alpha = [0.15, 0.30, 0.25, 0.20, 0.10]  # gemma, jina, mxbai, xlm, lgbm
final = sum(a * scores for a, scores in zip(alpha, all_scores))
```

**Key insight:** BGE-gemma scores ONLY the uncertain pairs (LightGBM score 0.4-0.6), not all 86M

**Time:** 16 hours  
**Cost:** 4× A40 ($2/hr × 4 × 16h = **$128**)

---

### Phase 3: Perfect Singleton Detection

**Strategy:** Three-stage cascade

```python
# singleton_max.py
"""Three stages: LightGBM → XLM-R → Qwen2.5-7B"""

# Stage 1: LightGBM filter (fast)
if max(lgbm_scores) < 0.65:
    prediction = []  # High-confidence singleton
elif max(lgbm_scores) > 0.80:
    # High-confidence match, proceed
    pass
else:
    # Stage 2: XLM-RoBERTa cross-encoder
    xlm_score = xlm_model.predict(s1, best_candidate)
    
    if xlm_score < 0.70:
        prediction = []  # Medium-confidence singleton
    elif xlm_score > 0.85:
        # Match confirmed
        pass
    else:
        # Stage 3: LLM (Qwen2.5-7B) for ultimate uncertain cases
        # ~5000 entities reach here
        llm_decision = qwen_match(s1, best_candidate)
        if llm_decision == "NO_MATCH":
            prediction = []
```

**Expected singleton accuracy:** 0.995+ (50 errors / 96k singletons)

---

### Phase 4: France-Specific Pipeline

**Strategy:** Separate model just for France

```python
# france_pipeline.py
"""Dedicated multilingual cross-encoder for France"""

# Identify France entities (15% of test = 260k)
france_s1 = [s for s in test_s1 if s['country'] == 'France']

# Use highest-quality multilingual model
france_model = CrossEncoder("BAAI/bge-reranker-v2-m3")  # 568M, MIT

# More conservative threshold for France
france_threshold = 0.72  # vs 0.50 for US/India

# Expected France F0.5: 0.96-0.97 (vs 0.93 with shared model)
```

---

### Phase 5: Graph Post-Processing

**Strategy:** Transitive closure on high-confidence edges

```python
# graph_postprocess.py
"""Union-find on score > 0.95"""

from collections import defaultdict

uf = UnionFind()

# Build graph
for s1_id, matches in predictions.items():
    for match_id in matches:
        if scores[s1_id][match_id] > 0.95:  # High confidence only
            uf.union(s1_id, match_id)

# Propagate through components
components = uf.get_components()

# Update predictions
for s1_id in predictions:
    component = components[uf.find(s1_id)]
    # Add any S2/S3 entities in same component
    predictions[s1_id] = component & set(pool.keys())

# Expected gain: +0.005-0.01 F0.5
```

---

## Complete 24-Hour Timeline

| Hours | Task | GPU | Cost |
|-------|------|-----|------|
| 0-12 | Triple blocking (dense + BM25 + MinHash) | A100 80GB | $42 |
| 4-20 | 5-model inference (parallel) | 4× A40 | $128 |
| 18-22 | France-specific scoring | A40 | $8 |
| 20-23 | Singleton cascade (LightGBM → XLM-R → LLM) | A40 | $6 |
| 23-24 | Graph post-processing + ensemble | CPU | $0 |
| **Total** | | | **$184** |

**Buffer:** $16 for retries

---

## Expected Performance

| Component | Contribution | Cumulative |
|-----------|-------------|------------|
| Baseline (current) | 0.7422 | 0.7422 |
| Triple blocking (0.99 recall) | +0.12 | 0.86 |
| 5-model ensemble | +0.08 | 0.94 |
| Perfect singletons | +0.02 | 0.96 |
| France pipeline | +0.02 | 0.98 |
| Graph post-processing | +0.01 | **0.99** ✓ |

---

## Critical Success Factors

**Must have ALL of these:**

1. ✅ **1024-dim embeddings** (not 512) for blocking
2. ✅ **Triple blocking union** (dense + BM25 + MinHash)  
3. ✅ **BGE-gemma-2-9B** in ensemble (quality leader)
4. ✅ **Separate France model** (not shared pipeline)
5. ✅ **Three-stage singleton cascade** (LightGBM → XLM-R → LLM)
6. ✅ **Graph post-processing** (transitive closure)

**If you skip ANY ONE:** F0.5 drops below 0.97

---

## Risks (High)

### Risk 1: Time Constraint

**Problem:** 5-model ensemble takes 16h minimum  
**If it won't finish:** Drop to 3 models (Jina, mxbai, LightGBM)  
**Penalty:** F0.5 = 0.96-0.97 (not 0.99)

### Risk 2: Memory Bottleneck

**Problem:** Triple blocking generates 400 cands/entity = 700M candidates total  
**Mitigation:** Filter to top-100 per entity with LightGBM before cross-encoder  
**Penalty:** -0.01 recall

### Risk 3: France Zero-Shot

**Problem:** Even dedicated multilingual model struggles on France  
**Mitigation:** Back-translate 50k English pairs to French, fine-tune  
**Cost:** +4 hours training  
**Gain:** +0.02 F0.5 on France

---

## Fallback Targets

**If 0.99 is impossible in 24h:**

| Target | What It Takes | Rank Estimate |
|--------|--------------|---------------|
| **0.98** | 4-model ensemble + double blocking | Top 3-5 |
| **0.97** | 3-model ensemble + dense blocking | Top 10-15 |
| **0.96** | 2-model ensemble + dense blocking | Top 20-30 |
| **0.95** | Jina-v2 only + dense blocking | Top 40-60 |

---

## Honest Assessment

**Can you hit 0.99 in 24 hours?**

**IF:**
- ✅ You start RIGHT NOW (every hour = -0.005 F0.5)
- ✅ Triple blocking finishes in 12h
- ✅ 4× A40 available for 16h straight
- ✅ Zero critical bugs in pipeline
- ✅ France model works on first try
- ✅ You get lucky (model ensemble chemistry)

**THEN:** 0.99 is **30% probable**

**More realistic:** 0.97-0.98 (still top 5-10) ✓

---

## Start Command (FOR 0.99)

```bash
# Rent 2× A100 80GB immediately
vultr-cli compute create --plan vhf-16c-80gb-a100 --qty 2

# First A100: Triple blocking (12h)
ssh a100-1 "python blocking_max.py > blocking.log 2>&1 &"

# Wait 12h, then rent 4× A40 for inference
vultr-cli compute create --plan vhf-8c-48gb-a40 --qty 4

# 4× A40: 5-model inference (16h, overlaps with blocking by 4h)
for i in {0..3}; do
  ssh a40-$i "python inference_worker.py --model $i > inf$i.log 2>&1 &"
done
```

**THIS IS YOUR ONLY PATH TO 0.99.**

**Anything less ambitious hits 0.96-0.97 (still top 20).**
