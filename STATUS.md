# Amazon ML Challenge 2026 — Session Status

**Team:** Da Big Three  
**Contest deadline:** 27 Sep 2026, 23:59 IST (~22 hours from now)  
**Current submissions:** 0  
**Baseline score (BM25 + LightGBM, previous run):** 0.7422  
**Leaderboard #1:** 0.9889  

---

## What We've Done

### 1. Research & Model Validation (real-time, not training data)

Validated every model claim against current HuggingFace/PyPI sources:

| Model | Status | Reason |
|-------|--------|--------|
| Qwen3-Embedding-0.6B | ✅ SELECTED | Apache-2.0, 614M params, TRUE Matryoshka (512-dim works), MTEB multilingual 64.64 |
| Qwen3-Reranker-0.6B | ✅ SELECTED | Apache-2.0, 614M params, drop-in CrossEncoder, needs `activation_fn=torch.nn.Sigmoid()` |
| BGE-m3 | ❌ REPLACED | NOT true Matryoshka — 512-dim truncation degrades quality |
| Jina-reranker-v2 | ❌ DISQUALIFIED | CC-BY-NC-4.0 license (contest requires MIT/Apache 2.0) |
| Jina-reranker-v3 | ❌ DISQUALIFIED | CC-BY-NC-4.0 license |
| Jina-embeddings-v3 | ❌ DISQUALIFIED | CC-BY-NC-4.0 license |
| BGE-reranker-v2.5-gemma | ❌ DISQUALIFIED | Gemma license |
| INT8 GPU quantization | ❌ DOES NOT WORK | ONNX dynamic INT8 is CPU-only |
| FP16 + ONNX GPU | ✅ VALID | 3-4× speedup over naive PyTorch |

### 2. Code Written / Updated

#### `src/pipeline.py` — Main compound pipeline orchestrator
- **BM25 + Dense + MinHash** union blocking (CAP=200 each)
- **20-feature LightGBM** classifier
- **Qwen3-Reranker-0.6B** cross-encoder cascade (top-50 pairs only)
- **Two-threshold sweep** (match_t, singleton_t) with France +0.10 bump
- **Isotonic calibration** fitted AFTER LightGBM retrain on full 100K (not before)
- **Graph transitive closure** (union-find, score > 0.9) — OFF by default (risky for F0.5)
- **ONNX export** with 100-pair validation sanity check
- **Multi-GPU auto-detection** for pool encoding (both holdout and test paths)
- **`build_final_scores()`** — ensures iso-transformed scores are consistent between sweep and inference
- **7 bugs fixed** by code-reviewer agent (threshold mismatch, in-sample stacking, iso timing, graph safety, OOM)

Key config (`Cfg` dataclass):
```python
dense_model: str = "Qwen/Qwen3-Embedding-0.6B"
dense_dim: int = 512          # True Matryoshka
ce_model: str = "Qwen/Qwen3-Reranker-0.6B"
bm25_cap: int = 200
dense_top_k: int = 200
graph_enabled: bool = False   # OFF — too risky for F0.5 precision metric
```

#### `src/dense.py` — Dense retrieval with FAISS GPU
- Default model: `Qwen/Qwen3-Embedding-0.6B`
- `truncate_dim=512` (true Matryoshka, re-normalized after truncation)
- `prompt_name="query"` for S1 entities, `prompt_name="document"` for pool
- `encode_and_index_multigpu()` — splits pool across N GPUs via multiprocessing (~N× speedup on g5.12xlarge)
- `encode_shard()` static method (per-GPU worker, sets `CUDA_VISIBLE_DEVICES`)
- `self.model_name` attribute saved on init (fixes crash in multi-GPU path)
- `save_pool()` / `load_pool()` checkpoints (.faiss + .eids.npy)
- FAISS `index_cpu_to_all_gpus()` for multi-GPU search

#### `src/benchmark_recall.py` — Recall gate (MUST RUN FIRST on GPU instance)
- Measures Recall@25/50/100/200/500 for each blocking strategy on 100K holdout
- Benchmarks: BM25 | Dense | BM25∪Dense | BM25∪MinHash (char 3-gram) | All three
- **Decision rule:** R@200 ≥ 0.93 → proceed | R@200 < 0.88 → architecture change
- MinHash uses character 3-gram shingles (catches typos/formatting variants BM25 misses)
- datasketch MinHashLSH, threshold=0.4, num_perm=64

#### `src/setup_gpu.sh` — GPU instance bootstrap
- Installs: torch (cu121), faiss-gpu, sentence-transformers≥6.0.0, lightgbm, rapidfuzz, optimum[onnxruntime-gpu]
- `sentence-transformers>=6.0.0` pin is CRITICAL — v6.0.0 (Aug 2025) fixed silent fp16 scoring bug in cross-encoders
- `faiss-gpu` (NOT `faiss-gpu-cu12` — that package doesn't exist on PyPI)
- Runs nvidia-smi + Python verification at the end

### 3. Architecture Decisions Made

- **Macro F0.5 is precision-heavy:** Empty prediction on singleton = 1.0. Any wrong match on singleton = 0.0. This drives the two-threshold strategy and why graph closure is OFF.
- **BM25 recall ceiling ~0.835** — dense + MinHash expected to push to 0.93+
- **France entities** are the hardest subset — get a +0.10 score bump in the threshold sweep
- **Stacking ensemble removed** — was training in-sample (data leakage). Replaced with direct CE score override.
- **VRAM budget on A10G 24GB:** 10M × 512-dim × fp16 = 10.2 GB index + 1.2 GB model = 12 GB total — fits comfortably.

---

## What's Pending / Next Steps

### Immediate (right now)

- [ ] **Push code to GitHub** so the GPU instance can `git clone` instead of SCP
  ```bash
  git add code/business_entity_resolution/src/pipeline.py \
          code/business_entity_resolution/src/dense.py \
          code/business_entity_resolution/src/benchmark_recall.py \
          code/business_entity_resolution/src/setup_gpu.sh
  git commit -m "Switch to Qwen3-Embedding-0.6B, add multi-GPU encoding, fix model_name attr"
  git push
  ```

- [ ] **Sign up on Lambda Labs** (lambdalabs.com) — backup GPU since AWS quota was denied
  - `gpu_1x_a10` = 1× A10G 24GB @ ~$0.60/hr → $70 buys 116 hours
  - No quota system, instant spin-up

- [ ] **AWS appeal** (already drafted) — re-appealing for g5.xlarge (4 vCPUs only)
  - If approved: g5.xlarge ~$1.006/hr on-demand
  - Smaller ask = higher approval chance

### GPU Instance Launch Sequence (once compute available)

```bash
# 1. SSH into instance
# 2. Clone code
git clone https://github.com/<repo>.git ml-challenge
cd ml-challenge/code/business_entity_resolution

# 3. Download dataset (use the URL you have)
wget <DATASET_URL> -O dataset.zip && unzip dataset.zip

# 4. Bootstrap
bash src/setup_gpu.sh

# 5. Run recall benchmark FIRST (evidence gate)
python -X utf8 src/benchmark_recall.py 2>&1 | tee recall_benchmark.log

# 6. If R@200 >= 0.93, run full pipeline
tmux new -s pipeline
python -X utf8 src/pipeline.py all 2>&1 | tee pipeline.log

# 7. Submit matching_results.tsv
```

### Submission #1 (CPU EC2 — already running)

- `run_test_parallel.py` with 7 workers under nohup
- Was at ~750K/1.73M when last checked
- ETA: ~7 AM IST Sunday (Sep 27)
- This gives submission #1 with the old BM25+LightGBM pipeline (~0.74 expected)
- 5 submissions/day limit — use them wisely

### After First GPU Run

1. Check recall benchmark output → validate R@200 ≥ 0.93
2. Check holdout F0.5 from pipeline.py output
3. If F0.5 ≥ 0.90 → submit immediately
4. Tune two-threshold (match_t, singleton_t) if needed
5. Consider enabling graph closure ONLY if precision is already high (>0.95)

---

## Known Issues / Watch Out For

| Issue | Status |
|-------|--------|
| `fast_features.py` jaro_winkler bug | `_fuzz.jaro_winkler_similarity` doesn't exist in rapidfuzz 3.x — crashes if the old run.py path is used. Our new pipeline.py uses different feature code. |
| sentence-transformers < 6.0.0 | Silent fp16 scoring bug. Must pin ≥6.0.0 in setup_gpu.sh (already done). |
| ONNX INT8 on GPU | Does not work. ONNX dynamic INT8 is CPU-only. Not used. |
| Graph closure | OFF by default. Each false merge on a singleton = F0.5 of 0.0 for that entity. |
| AWS vCPU quota | Denied for G instances. Appeal filed for g5.xlarge (4 vCPU). Using Lambda Labs as backup. |

---

## Contest Rules Constraints

- **No external APIs or databases** for entity lookup — disqualification
- **No internet data augmentation** — disqualification  
- **Models:** MIT or Apache 2.0 only, ≤8B parameters
- **5 submissions per day**
- **One machine per participant** (simultaneous login can terminate session)
- Use `tmux` on EC2 so SSH disconnects don't kill the run

---

## Compute Budget

| Option | Instance | GPU | $/hr | Hours on $70 |
|--------|----------|-----|------|-------------|
| Lambda Labs (backup, ready now) | gpu_1x_a10 | 1× A10G 24GB | ~$0.60 | ~116h |
| AWS g5.xlarge (appeal pending) | g5.xlarge | 1× A10G 24GB | $1.006 | ~69h |
| AWS g5.12xlarge (denied) | g5.12xlarge | 4× A10G | $4.024 | ~17h |

**Recommendation:** Lambda Labs `gpu_1x_a10` — available now, no approval needed, same GPU hardware as g5.xlarge.
