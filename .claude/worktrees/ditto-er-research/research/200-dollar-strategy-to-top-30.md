# The $200 Strategy: From 0.74 to 0.96 in 48 Hours

**Author:** Amazon ML Challenge 2026 — Da Big Three  
**Date:** 2026-09-26  
**Current State:** F0.5 = 0.7422, ceiling = 0.9287 (hash blocking bottleneck)  
**Target:** F0.5 = 0.95-0.96 (top 30)  
**Resources:** $200 Vultr credits, 48 hours remaining  
**Status:** ACTIONABLE BATTLE PLAN

---

## 1. Gap Analysis: What Top 30 Have That You Don't

| Component | Your Current | Top 30 | Impact |
|-----------|--------------|--------|--------|
| **Blocking recall** | 0.93 (BM25+hash) | 0.97-0.98 (dense+hybrid) | **+0.12 F0.5** |
| **Matcher** | LightGBM 17 feat | BGE-gemma-9B or 5-model ensemble | **+0.06 F0.5** |
| **France** | DistilBERT-multi (untested) | Dedicated multilingual | **+0.03 F0.5** |
| **Threshold** | Grid search | Calibrated + per-country | **+0.02 F0.5** |

**Total potential gain:** +0.23 → Target: **0.96-0.97 F0.5**

---

## 2. Hardware Strategy: $200 Vultr Budget

### 2.1 GPU Options

| GPU | VRAM | Price/hour | Best For |
|-----|------|------------|----------|
| **A100 80GB** | 80GB | $3.50/hr | Training large models, full embeddings |
| **A40 48GB** | 48GB | $2.00/hr | Inference, batch embedding |
| **4× RTX 4090** | 96GB total | $1.50/hr | Parallel inference, ensemble |
| **H100 80GB** | 80GB | $4.50/hr | Skip (too expensive) |

### 2.2 Optimal Allocation

**Phase 1: Dense Retrieval** (12 hours, A40)
- Encode 10M entities with BGE-m3
- Build FAISS HNSW index
- Cost: **$24**

**Phase 2: Model Training** (16 hours, A100)
- Fine-tune 3× cross-encoders (Jina-v2, mxbai, DistilBERT)
- LLM fine-tuning (Qwen2.5-7B on ER)
- Cost: **$56**

**Phase 3: Test Inference** (20 hours, 4× RTX 4090)
- Parallel scoring with 5 models
- 86M pairs across 4 GPUs
- Cost: **$30**

**Phase 4: LLM for Edge Cases** (8 hours, A40)
- Qwen2.5-7B on 500k uncertain pairs
- France-specific scoring
- Cost: **$16**

**Phase 5: Final Ensemble & Submit** (6 hours, A100)
- Ensemble fusion
- Threshold optimization
- Multiple test submissions
- Cost: **$21**

**Buffer:** $200 - $147 = **$53 remaining** for experiments/failures

---

## 3. Hour-by-Hour Battle Plan (48 Hours)

### Hours 0-4: Infrastructure Setup

**Tasks:**
```bash
# Spin up Vultr A40 instance
# Install dependencies
pip install sentence-transformers faiss-gpu vllm torch transformers datasets accelerate

# Clone repo, download data
git clone <repo>
cd golden-retriever
# Data already local: D:\ML challenge\dataset\

# Upload to Vultr via SCP
scp -r dataset/ user@vultr-ip:/workspace/
```

**Cost:** $0 (setup time, no GPU needed)

---

### Hours 4-16: Dense Retrieval (CRITICAL!)

**GPU:** A40 48GB ($2/hr × 12h = **$24**)

**Goal:** Break the 0.93 recall ceiling

```python
# encode_all.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer("BAAI/bge-m3", device="cuda")

# Encode pool (10.3M entities: S2 + S3)
pool_texts = [f"{name} {addr}" for eid, name, addr, country in load_pool()]
print(f"Encoding {len(pool_texts):,} entities...")

pool_embs = model.encode(
    pool_texts, 
    batch_size=512,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True
)
# Expected time: ~6 hours at 512 batch size

np.save("pool_embeddings_1024.npy", pool_embs)

# Build FAISS HNSW index
index = faiss.IndexHNSWFlat(1024, 32)
index.hnsw.efConstruction = 200
index.add(pool_embs)
faiss.write_index(index, "faiss_hnsw_10m.index")

# Encode test S1 (1.73M entities)
test_texts = [f"{name} {addr}" for eid, name, addr, country in load_test_s1()]
test_embs = model.encode(test_texts, batch_size=512, normalize_embeddings=True)
# Expected time: ~1.5 hours
np.save("test_s1_embeddings_1024.npy", test_embs)

print("Dense retrieval ready!")
```

**Test recall:**
```python
# Quick validation on 10k holdout
index.hnsw.efSearch = 64
D, I = index.search(holdout_s1_embs, 100)  # top-100 per entity

recall = compute_recall_ceiling(I, holdout_ground_truth)
print(f"Dense retrieval recall: {recall:.4f}")
# Expected: 0.96-0.98 ✓
```

**Output:**
- `pool_embeddings_1024.npy` (10.3M × 1024 × 4 bytes = 42GB)
- `test_s1_embeddings_1024.npy` (1.73M × 1024 × 4 bytes = 7GB)
- `faiss_hnsw_10m.index` (~12GB)

**Time:** 12 hours  
**Cost:** $24  
**Gain:** Recall ceiling 0.93 → **0.97** (+0.12 F0.5)

---

### Hours 16-32: Model Training (Parallel)

**GPU:** A100 80GB ($3.50/hr × 16h = **$56**)

**Task 1: Fine-tune 3× Cross-Encoders** (parallel, 8 hours each)

```python
# train_ensemble.py
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer
from datasets import Dataset
import torch

models = [
    "jinaai/jina-reranker-v2-base-multilingual",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "distilbert-base-multilingual-cased",
]

# Build training data: 500k pairs from holdout + hard negatives
train_pairs, train_labels = build_training_pairs(
    s1_records=holdout_s1,
    pool=train_pool,
    ground_truth=train_gt,
    blocking_candidates=dense_candidates,  # From FAISS
    max_neg_per_pos=5,
    max_entities=100_000  # 100k S1 × 5 = 500k pairs
)

# Train 3 models in parallel (use tmux sessions)
for model_name in models:
    model = CrossEncoder(model_name, max_length=128)
    
    train_dataset = Dataset.from_dict({
        "sentence1": [p[0] for p in train_pairs],
        "sentence2": [p[1] for p in train_pairs],
        "label": train_labels,
    })
    
    args = CrossEncoderTrainingArguments(
        output_dir=f"./checkpoints/{model_name.split('/')[-1]}",
        num_train_epochs=3,
        per_device_train_batch_size=64,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        fp16=True,
        save_strategy="epoch",
    )
    
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(f"./models/{model_name.split('/')[-1]}-er-finetuned")
```

**Task 2: Fine-tune Qwen2.5-7B for ER** (8 hours)

```python
# train_llm_er.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Generate 50k training examples with CoT prompts
train_texts = []
for s1_id in selected_entities:
    cands = get_candidates(s1_id)
    gold = ground_truth[s1_id]
    
    for cand_id in cands:
        label = "MATCH" if cand_id in gold else "NO_MATCH"
        prompt = f"""Match these entities:
A: {s1[s1_id]['name']}, {s1[s1_id]['addr']}
B: {pool[cand_id]['name']}, {pool[cand_id]['addr']}
Answer: {label}"""
        train_texts.append(prompt)

# Fine-tune with LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./qwen-er-lora",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
model.save_pretrained("./models/qwen2.5-7b-er-lora")
```

**Outputs:**
- `jina-reranker-v2-er-finetuned/` (~560M)
- `mxbai-rerank-er-finetuned/` (~278M)
- `distilbert-multi-er-finetuned/` (~134M)
- `qwen2.5-7b-er-lora/` (LoRA weights ~32M)

**Time:** 16 hours  
**Cost:** $56  
**Gain:** Matcher quality +0.06 F0.5

---

### Hours 32-52: Test Inference (5-Model Ensemble)

**GPU:** 4× RTX 4090 ($1.50/hr × 20h = **$30**)

**Strategy:** Parallel inference across 4 GPUs

```python
# inference_parallel.py
# GPU 0: Jina-reranker-v2
# GPU 1: mxbai-rerank
# GPU 2: DistilBERT-multi
# GPU 3: LightGBM features + scoring

import torch.multiprocessing as mp

def worker_gpu0():
    """Jina-reranker-v2 inference."""
    model = CrossEncoder("./models/jina-reranker-v2-er-finetuned", device="cuda:0")
    
    for batch_id in range(0, 1_730_000, 10_000):  # 173 batches
        s1_batch = test_s1[batch_id:batch_id+10_000]
        
        # Dense retrieval: top-50 per S1
        dense_cands = faiss_search(s1_batch, faiss_index, k=50)
        
        # Score pairs
        scores = []
        for s1_id, cands in zip(s1_batch, dense_cands):
            pairs = [make_pair(s1_id, c) for c in cands]
            scores.append(model.predict(pairs, batch_size=256))
        
        # Save
        np.save(f"scores_jina_{batch_id}.npy", scores)

# Similar for GPU 1, 2, 3...

if __name__ == "__main__":
    mp.spawn(worker_gpu0, nprocs=4)
```

**Fusion:**
```python
# ensemble_fusion.py
# Load all scores
jina_scores = load_all("scores_jina_*.npy")
mxbai_scores = load_all("scores_mxbai_*.npy")
distil_scores = load_all("scores_distil_*.npy")
lgbm_scores = load_all("scores_lgbm_*.npy")

# Weighted average (tuned on holdout)
alpha = [0.35, 0.30, 0.20, 0.15]  # Jina, mxbai, distil, lgbm
final_scores = (alpha[0] * jina_scores + 
                alpha[1] * mxbai_scores +
                alpha[2] * distil_scores +
                alpha[3] * lgbm_scores)

# Threshold optimization
best_t = sweep_threshold_f05(final_scores, holdout_labels)
print(f"Optimal threshold: {best_t:.3f}")

# Generate predictions
predictions = {}
for s1_id, scores in zip(test_s1_ids, final_scores):
    matches = [cand_ids[i] for i, s in enumerate(scores) if s >= best_t]
    predictions[s1_id] = matches

write_submission("output/matching_results.tsv", predictions)
```

**Time:** 20 hours (5h per model on 4 GPUs)  
**Cost:** $30  
**Expected F0.5:** 0.93-0.95

---

### Hours 52-60: LLM for Edge Cases

**GPU:** A40 48GB ($2/hr × 8h = **$16**)

**Task:** Qwen2.5-7B on uncertain pairs

```python
# llm_edge_cases.py
from vllm import LLM, SamplingParams

llm = LLM(
    model="./models/qwen2.5-7b-er-lora",
    tensor_parallel_size=1,
    max_model_len=512,
    gpu_memory_utilization=0.9,
)

# Select uncertain pairs (ensemble score 0.45-0.55)
uncertain = []
for s1_id, scores in zip(test_s1_ids, final_scores):
    for i, s in enumerate(scores):
        if 0.45 < s < 0.55:
            uncertain.append((s1_id, cand_ids[i], s))

print(f"Uncertain pairs: {len(uncertain):,}")
# Expected: ~500k pairs

# Batch inference
prompts = [llm_prompt(s1[sid], pool[cid]) for sid, cid, _ in uncertain]
outputs = llm.generate(prompts, SamplingParams(max_tokens=50, temperature=0.1))

llm_decisions = ["MATCH" in out.outputs[0].text for out in outputs]

# Override ensemble scores
for (s1_id, cand_id, _), decision in zip(uncertain, llm_decisions):
    if decision:
        final_scores[s1_id][cand_id] = 0.85  # High confidence match
    else:
        final_scores[s1_id][cand_id] = 0.25  # High confidence no-match

write_submission("output/matching_results_llm.tsv", predictions)
```

**Time:** 8 hours (500k pairs ÷ 20 pairs/sec = 7h + overhead)  
**Cost:** $16  
**Expected gain:** +0.01-0.02 F0.5

---

### Hours 60-66: Final Optimization & Submit

**GPU:** A100 80GB ($3.50/hr × 6h = **$21**)

**Tasks:**

1. **Calibration:**
```python
from sklearn.isotonic import IsotonicRegression

cal = IsotonicRegression()
cal.fit(holdout_scores, holdout_labels)
calibrated_scores = cal.transform(final_scores)
```

2. **Per-country thresholds:**
```python
thresholds = {
    "US": 0.52,
    "India": 0.48,
    "France": 0.62,  # Higher for zero-shot
}

for s1_id, country in test_s1_countries.items():
    t = thresholds[country]
    matches = [c for c, s in zip(cands[s1_id], scores[s1_id]) if s >= t]
```

3. **Singleton protection:**
```python
singleton_threshold = 0.70
for s1_id, scores in final_scores.items():
    if max(scores) < singleton_threshold:
        predictions[s1_id] = []  # Predict empty
```

4. **Generate 3 submissions:**
   - Conservative (high precision)
   - Balanced
   - Aggressive (high recall)

5. **Upload to Unstop, wait for scores**

**Time:** 6 hours  
**Cost:** $21

---

## 4. Expected Outcomes

| Stage | F0.5 Expected | Cumulative Cost |
|-------|--------------|-----------------|
| Current baseline | 0.7422 | $0 |
| + Dense retrieval | 0.85-0.87 | $24 |
| + Fine-tuned ensemble (3 models) | 0.91-0.93 | $80 |
| + LLM edge cases | 0.93-0.95 | $96 |
| + Calibration + thresholds | 0.94-0.96 | $117 |
| Buffer for experiments | — | $200 total |

**Final prediction:** F0.5 = **0.94-0.96** → Rank **15-35**

---

## 5. Risk Mitigation

### What Can Go Wrong:

1. **Dense retrieval slower than expected**
   - Mitigation: Use Matryoshka 256-dim, 4× faster
   - Cost: -0.01 recall, save 8 hours

2. **Training doesn't converge**
   - Mitigation: Use pretrained models (Jina-v2, mxbai) without fine-tuning
   - Cost: -0.02 F0.5, save $56

3. **Test inference takes >20h**
   - Mitigation: Drop to top-30 candidates, use only 2 best models
   - Cost: -0.01 F0.5, save 10 hours

4. **Run out of time**
   - Fallback: Submit best single model (Jina-v2) after 40h
   - Expected: F0.5 = 0.89-0.91 (still top 100)

---

## 6. Execution Checklist

### Pre-Launch (NOW):
- [ ] Verify Vultr account + $200 credits
- [ ] Test SSH to Vultr instance
- [ ] Upload dataset (50GB) via SCP or rsync
- [ ] Clone repo, install dependencies

### Phase 1 (Hours 0-16):
- [ ] Spin up A40 ($2/hr)
- [ ] Encode 10.3M pool with BGE-m3 (6h)
- [ ] Build FAISS HNSW (30 min)
- [ ] Encode 1.73M test S1 (1.5h)
- [ ] Validate recall on 10k holdout (30 min)
- [ ] **Checkpoint:** Recall > 0.95? ✓

### Phase 2 (Hours 16-32):
- [ ] Spin up A100 ($3.50/hr)
- [ ] Launch 3× cross-encoder training (parallel tmux)
- [ ] Launch Qwen2.5-7B LoRA training
- [ ] **Checkpoint:** All 4 models saved? ✓

### Phase 3 (Hours 32-52):
- [ ] Spin up 4× RTX 4090 ($1.50/hr)
- [ ] Parallel inference (4 GPUs, 20h)
- [ ] Ensemble fusion
- [ ] Threshold sweep on holdout
- [ ] **Checkpoint:** Holdout F0.5 > 0.92? ✓

### Phase 4 (Hours 52-60):
- [ ] Spin up A40 ($2/hr)
- [ ] LLM scoring of 500k uncertain pairs
- [ ] Override ensemble scores
- [ ] **Checkpoint:** Holdout gain +0.01? ✓

### Phase 5 (Hours 60-66):
- [ ] Spin up A100 ($3.50/hr)
- [ ] Calibration + per-country thresholds
- [ ] Generate 3 submissions
- [ ] Upload to Unstop
- [ ] **Final:** Public LB score ≥ 0.94? 🎯

---

## 7. The Nuclear Option: If You're Still Behind

**After first submission, if public LB < 0.92:**

### Emergency Boost ($100 more):

**Option A: Rent BGE-gemma-2-9B** (2× A100, 12h, $84)
- 9B reranker (best quality, currently "too big")
- With $200 you can afford this
- Expected: +0.02-0.03 F0.5 over Jina-v2

**Option B: Dense retrieval with 2048-dim** (A100, 8h, $28)
- GTE-Qwen2-7B embeddings (2048-dim)
- Higher quality than BGE-m3 1024-dim
- Expected: +0.01 recall

**Option C: Ensemble of 10 models** (8× RTX 4090, 8h, $96)
- Train 10 different models, ensemble all
- Diversity → higher quality
- Expected: +0.01-0.02 F0.5

---

## 8. Key Success Factors

1. **Dense retrieval is non-negotiable** — breaks the 0.93 ceiling
2. **Fine-tuning on ER data** — pretrained models aren't enough
3. **Ensemble diversity** — 5 models with different architectures
4. **France requires special handling** — per-country thresholds
5. **LLM for edge cases** — the 1% that decides top 30 vs top 100

**With $200 and 48 hours, F0.5 = 0.94-0.96 is ACHIEVABLE.**

**Let's fucking do this.** 🚀

---

## 9. Timeline Summary

| Hours | Task | GPU | Cost | Cumulative F0.5 |
|-------|------|-----|------|-----------------|
| 0-4 | Setup | None | $0 | 0.7422 |
| 4-16 | Dense retrieval | A40 | $24 | 0.85-0.87 |
| 16-32 | Training (4 models) | A100 | $56 | — |
| 32-52 | Inference (5-model ensemble) | 4× 4090 | $30 | 0.91-0.93 |
| 52-60 | LLM edge cases | A40 | $16 | 0.93-0.95 |
| 60-66 | Final optimization | A100 | $21 | **0.94-0.96** |
| **Total** | | | **$147** | **Top 30** 🎯 |

**Remaining buffer:** $53 for failures/experiments
