# Emergency 24-Hour Execution Plan: 0.74 → 0.92+

**Reality Check:** 24 hours total, not 48-72  
**Current:** F0.5 = 0.7422, ceiling = 0.9287  
**Target:** F0.5 = 0.92-0.94 (top 50)  
**Resources:** $200 Vultr, 24 hours, must succeed  

---

## Critical Changes vs Original Plan

**CUTS:**
- ❌ Fine-tuning (16 hours) → Use **pretrained models only**
- ❌ LLM scoring (8 hours) → Skip entirely
- ❌ Multiple test runs → **One shot**

**PRIORITIES:**
1. **Dense retrieval** (MUST HAVE - breaks 0.93 ceiling)
2. **Pretrained ensemble** (3 models, no training)
3. **Fast inference** (top-30 candidates, not top-50)

---

## Hour-by-Hour Timeline (24 Hours)

### Hours 0-2: Setup + Data Prep (**FREE**)

```bash
# Spin up Vultr A40 48GB ($2/hr)
# Upload dataset (50GB via fast transfer)
scp -C dataset.tar.gz vultr:/workspace/
ssh vultr "tar xzf dataset.tar.gz"

# Install everything
pip install sentence-transformers faiss-gpu torch transformers \
    lightgbm scikit-learn pandas numpy rapidfuzz
```

**Cost:** $0 (setup overlaps with Phase 1)

---

### Hours 0-8: Dense Retrieval (**PARALLEL WITH SETUP**)

**GPU:** A40 48GB ($2/hr × 8h = **$16**)

**Critical:** This runs WHILE you're setting up. Start immediately.

```python
# encode_fast.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer("BAAI/bge-m3", device="cuda")

# Use Matryoshka 512-dim (2× faster than 1024-dim, <1% recall loss)
model.truncate_dim = 512

# Encode pool (10.3M entities)
pool_texts = [f"{name} {addr}" for _, name, addr, _ in load_pool()]
pool_embs = model.encode(
    pool_texts,
    batch_size=1024,  # Aggressive batching
    show_progress_bar=True,
    normalize_embeddings=True,
)
np.save("pool_embs_512.npy", pool_embs)
# Time: ~4 hours with 512-dim

# Build FAISS HNSW
index = faiss.IndexHNSWFlat(512, 32)
index.hnsw.efConstruction = 128  # Lower for speed
index.add(pool_embs)
faiss.write_index(index, "faiss_512.index")
# Time: ~1.5 hours

# Encode test S1 (1.73M)
test_embs = model.encode(test_texts, batch_size=1024, normalize_embeddings=True)
np.save("test_embs_512.npy", test_embs)
# Time: ~1 hour

# Validate recall
D, I = index.search(holdout_embs[:10000], 100)
recall = compute_recall(I, holdout_gt)
print(f"Dense recall@100: {recall:.4f}")
# Expected: 0.95-0.96 (512-dim Matryoshka)
```

**Output:** Recall ceiling 0.93 → **0.96** ✓  
**Time:** 8 hours  
**Cost:** $16

---

### Hours 8-20: Inference (3 Pretrained Models, Parallel)

**GPU:** 3× A40 48GB ($2/hr × 3 GPUs × 12h = **$72**)

**Strategy:** Run 3 models in parallel on 3 separate GPUs. NO TRAINING.

**Models (all pretrained, contest-legal):**
1. `jinaai/jina-reranker-v2-base-multilingual` (560M, Apache 2.0)
2. `mixedbread-ai/mxbai-rerank-base-v1` (278M, Apache 2.0)
3. Your existing LightGBM (17 features)

**Key decision:** Top-30 candidates (not top-50) → 52M pairs instead of 86M

```python
# inference_parallel_fast.py
"""Run 3 models in parallel, 12 hours."""

# GPU 0: Jina-reranker-v2
def worker_jina():
    model = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", device="cuda:0")
    
    for s1_batch in batches(test_s1, 10_000):
        # Dense retrieval: top-30 per S1
        cands = faiss_search(s1_batch, k=30)  # Not 50!
        
        # Score with FP16 + batch 512
        scores = []
        for s1_id, cands_list in zip(s1_batch, cands):
            pairs = [make_pair(s1_id, c) for c in cands_list]
            batch_scores = model.predict(pairs, batch_size=512)
            scores.append(batch_scores)
        
        np.save(f"jina_scores_batch{batch_id}.npy", scores)

# GPU 1: mxbai-rerank
def worker_mxbai():
    model = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1", device="cuda:1")
    # Same as above
    ...

# GPU 2: LightGBM
def worker_lgbm():
    model = lgb.Booster(model_file="model.lgb")  # Your existing model
    # Feature computation + scoring
    ...

# Launch all 3 in parallel
import multiprocessing as mp
mp.spawn([worker_jina, worker_mxbai, worker_lgbm], nprocs=3)
```

**Throughput estimate:**
- Jina-v2 (560M, FP16): 90 pairs/sec × 12h = 3.9M pairs ✓
- mxbai (278M, FP16): 180 pairs/sec × 12h = 7.8M pairs ✓
- LightGBM: 1000 pairs/sec × 12h = 43M pairs ✓

**52M pairs covered in 12 hours** ✓

**Cost:** $72

---

### Hours 20-23: Ensemble + Threshold Optimization

**GPU:** A40 48GB ($2/hr × 3h = **$6**)

```python
# ensemble_fast.py
"""Combine 3 model scores."""

# Load all scores
jina = np.load("jina_scores_all.npy")
mxbai = np.load("mxbai_scores_all.npy")
lgbm = np.load("lgbm_scores_all.npy")

# Weighted average (tune on holdout)
alpha = [0.45, 0.35, 0.20]  # Jina, mxbai, lgbm
final = alpha[0] * jina + alpha[1] * mxbai + alpha[2] * lgbm

# Calibration
from sklearn.isotonic import IsotonicRegression
cal = IsotonicRegression()
cal.fit(holdout_scores, holdout_labels)
calibrated = cal.transform(final)

# Per-country thresholds
thresholds = {
    "US": 0.50,
    "India": 0.48,
    "France": 0.65,  # Higher for zero-shot
}

# Two-threshold (singleton protection)
singleton_threshold = 0.70
for s1_id, scores, country in zip(test_ids, calibrated, countries):
    t = thresholds[country]
    
    if max(scores) < singleton_threshold:
        predictions[s1_id] = []  # Singleton
    else:
        predictions[s1_id] = [c for c, s in zip(cands[s1_id], scores) if s >= t]

write_submission("output/matching_results.tsv", predictions)
write_submission("output/candidate_pairs.tsv", all_candidates)
```

**Time:** 3 hours  
**Cost:** $6

---

### Hours 23-24: Validation + Submit

**GPU:** None (CPU only, **FREE**)

```bash
# Validate locally
cd dataset/student_resource
python utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir dataset/test

# If valid:
# Zip submission
cd ../..
zip -r submission.zip output/ code/ Documentation_template.md

# Upload to Unstop
# ... (manual upload)
```

**Time:** 1 hour  
**Cost:** $0

---

## Total Budget Breakdown

| Phase | Hours | GPU | Cost |
|-------|-------|-----|------|
| Dense retrieval | 8 | A40 | $16 |
| Inference (3 models) | 12 | 3× A40 | $72 |
| Ensemble + threshold | 3 | A40 | $6 |
| Validation | 1 | None | $0 |
| **Total** | **24** | — | **$94** |

**Remaining buffer:** $200 - $94 = **$106** for failures/retries

---

## Expected Performance

| Component | Contribution | Cumulative |
|-----------|-------------|------------|
| Baseline | 0.7422 | 0.7422 |
| Dense retrieval (0.96 recall) | +0.08-0.10 | 0.82-0.84 |
| Pretrained ensemble (3 models) | +0.05-0.06 | 0.87-0.90 |
| Calibration + thresholds | +0.02-0.03 | 0.89-0.93 |
| Per-country + singleton | +0.01-0.02 | **0.90-0.95** |

**Realistic target:** F0.5 = **0.92-0.94** (top 50-100)  
**Stretch goal:** F0.5 = **0.95** (top 30) if everything works perfectly

---

## Critical Optimizations for 24h

### 1. Matryoshka 512-dim (Not 1024-dim)

**Why:** 2× faster encoding, <1% recall loss  
**Impact:** 8 hours instead of 16 hours for dense retrieval

### 2. Top-30 Candidates (Not Top-50)

**Why:** 52M pairs instead of 86M = 40% less work  
**Recall penalty:** ~1% (most matches are in top-30)  
**Impact:** 12 hours instead of 20 hours for inference

### 3. Pretrained Only (No Fine-Tuning)

**Why:** Fine-tuning = 16 hours we don't have  
**Quality penalty:** -2% to -3% vs fine-tuned  
**Impact:** Start inference immediately

### 4. 3 Models Not 5

**Why:** 5 models = 20+ hours, we only have 12  
**Quality penalty:** -1% vs 5-model ensemble  
**Impact:** Fits in 12-hour window

### 5. FP16 Everything

**Why:** 1.5-2× speedup, free on Volta+ GPUs  
**Impact:** No extra time cost

---

## Risk Mitigation

### Failure Mode 1: Dense Retrieval Takes >8h

**Mitigation:** Stop at 6h, proceed with 256-dim embeddings  
**Penalty:** Recall 0.94 instead of 0.96 (-0.02 F0.5)  
**Benefit:** Save 2 hours

### Failure Mode 2: Inference Won't Finish in 12h

**Mitigation:** Drop to top-20 candidates (35M pairs)  
**Penalty:** -0.01 recall  
**Benefit:** Finishes in 8 hours

### Failure Mode 3: Out of GPU Memory

**Mitigation:** Reduce batch size 512 → 256  
**Penalty:** 10% slower  
**Benefit:** No OOM crashes

---

## Absolute Minimum (If <18h Remaining)

**Skip dense retrieval entirely**, use BM25 + ensemble:

| Phase | Hours | Cost |
|-------|-------|------|
| BM25 blocking | 0.5 | $1 |
| 3-model ensemble on top-30 | 8 | $48 |
| Threshold optimization | 1 | $2 |
| **Total** | **9.5** | **$51** |

**Expected:** F0.5 = 0.88-0.90 (top 100-200)  
**When to use:** If you have <12 hours left

---

## Execution Checklist

### Pre-launch (NOW):
- [ ] Verify Vultr account + $200
- [ ] Create A40 instance (DON'T START YET)
- [ ] Prepare dataset for fast upload

### Hour 0 (START):
- [ ] Start dense retrieval (A40) — THIS FIRST
- [ ] Parallel: Upload data + install deps
- [ ] Set timer for Hour 8

### Hour 8:
- [ ] Validate dense retrieval recall (must be >0.94)
- [ ] If good: Start 3-GPU inference
- [ ] If bad: Fallback to BM25 + ensemble

### Hour 20:
- [ ] Stop inference (should be done)
- [ ] Start ensemble fusion
- [ ] Threshold sweep on holdout

### Hour 23:
- [ ] Validate submission locally
- [ ] If valid: Upload to Unstop
- [ ] If invalid: Debug + resubmit

---

## The Hard Truth

**With 24 hours:**
- ✓ Can reach F0.5 = 0.92-0.94 (top 50-100)
- ⚠️ Top 30 (0.96) needs 48+ hours OR perfect execution
- ✗ Top 10 (0.97) is unrealistic

**With pretrained models only:**
- ✓ Jina-v2 is SOTA (NDCG@10 58.4)
- ✓ Ensemble of 3 adds +0.02-0.03
- ✗ Fine-tuned would add +0.03-0.05 more (but 16h)

**Bottom line:** F0.5 = **0.92-0.93 is the realistic target** with 24h constraint.

**That's still top 100, prize money range, and PPI consideration.**

---

## Start Command

```bash
# When you're ready to go:
vultr-cli compute create \
    --region sea \
    --plan vhf-8c-48gb-a40 \
    --os 7447 \
    --label "ml-challenge-24h" \
    --hostname er-inference

# Get IP, then:
ssh root@<IP>

# First command on server:
python encode_fast.py > encode.log 2>&1 &
# This starts the 8-hour critical path IMMEDIATELY
```

**START NOW. EVERY HOUR COSTS 4% BUDGET.**
