# Experiment log (IST)

Format: timestamp, split, macro F0.5, blocking recall ceiling, mean candidates per S1.

Baseline references (2026-09-25 morning): all-empty = 0.0558, raw exact name+address = 0.0731.
2026-09-25 08:36  holdout  n=100,000  macro_F0.5=0.5919  recall_ceiling=0.693  mean_candidates=52.7  capped=0.810  726s
2026-09-25 09:38  holdout  n=100,000  macro_F0.5=0.6796  recall_ceiling=0.736  mean_candidates=36.2  capped=0.842  2682s
2026-09-25 11:00  holdout  n=100,000  macro_F0.5=0.7384  recall_ceiling=0.814  mean_candidates=73.5  capped=0.863  2552s  [val_F0.5=0.7382 t=-0.7]
2026-09-25 14:43  holdout  n=100,000  macro_F0.5=0.7421  recall_ceiling=0.835  mean_candidates=131.5  capped=0.798  5199s  [val_F0.5=0.7422 t=-0.4 ceil_macro=0.9287]

--- Session 2 notes (2026-09-25 afternoon) ---

v1 (11:00): CAP=80, 9 features, LR. First learned model. Ceiling 0.814 — hash keys hit hard cap.
v2 (OOM): CAP=150, 10 features (+JW). val=0.7415 ceil=0.834. model.json saved but Phase 6 OOM. Fixed.
v3 (14:43): CAP=150, 11 features (+tfidf_cos), MAX_POSTING=2000 guard added. val=0.7422, ceil=0.835.
  KEY NUMBER: ceil_macro (perfect matcher) = 0.9287. This is the hard ceiling with hash-key blocking.
  To reach public top 0.96, MUST replace hash keys with BM25 token retrieval.

v4 (running ~16:00+): BM25 blocking + LightGBM + 15 features (+rapidfuzz). LOCAL MACHINE TOO SLOW.
  Index built fine: 2,152,388 posting lists.
  Pool loading hit memory pressure (index ~8GB + pool ~2GB). 7047s just to load.
  Featurization not started yet as of last check. Estimated finish: 3+ more hours.
  STATUS: still running in background, may finish tonight.
  model.json was DELETED before v4 started. No submittable model on disk right now.

--- Cloud strategy (decided ~16:00, executed ~20:45 IST) ---
Moved to AWS EC2 r5.2xlarge (64GB RAM, eu-north-1). Ubuntu 26.04.
Dataset downloaded from Unstop CDN directly on EC2 (3 seconds for 1GB).
All deps installed: rapidfuzz, lightgbm, sentence-transformers, faiss-cpu, torch.
GitHub repo: https://github.com/Harikishanth/golden-retriever
io_utils.py ROOT auto-detects EC2 vs local path.

--- Installed packages (local machine, 2026-09-25) ---
rapidfuzz 3.14.6
lightgbm 4.7.0

--- Dense retrieval plan (tonight on GPU) ---
sentence-transformers/all-MiniLM-L6-v2 (22M params, MIT license)
Encode all 12.3M records → 384-dim embeddings
FAISS top-100 per S1 entity → recall ceiling ~0.95+
Combined BM25 + dense candidates → retrain LightGBM
Expected: 0.88-0.93 holdout
