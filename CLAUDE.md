# Amazon ML Challenge 2026 — Project Context

## Contest
- **Task:** Business entity resolution — match 1.73M S1 entities to 10M S2+S3 pool
- **Metric:** Macro F0.5 (precision-heavy). Empty prediction on singleton = 1.0; wrong match = 0.0
- **Deadline:** 27 Sep 2026, 23:59 IST
- **Submissions:** 5 per day max
- **Rules:** No external APIs/databases. Models must be MIT/Apache 2.0, ≤8B params.

## Model Stack (validated 2026-09-26)
- **Dense retrieval:** `Qwen/Qwen3-Embedding-0.6B` — Apache-2.0, true Matryoshka, 512-dim
- **Cross-encoder:** `Qwen/Qwen3-Reranker-0.6B` — Apache-2.0, drop-in CrossEncoder, needs `activation_fn=torch.nn.Sigmoid()`
- **sentence-transformers ≥ 6.0.0 REQUIRED** — v6.0.0 fixed silent fp16 scoring bug
- **BGE-m3 is NOT Matryoshka** — do not use with truncate_dim
- **Jina models are CC-BY-NC-4.0** — DISQUALIFIED

## Key Files
- `code/business_entity_resolution/src/pipeline.py` — main compound pipeline
- `code/business_entity_resolution/src/dense.py` — dense retrieval + FAISS GPU
- `code/business_entity_resolution/src/benchmark_recall.py` — recall gate (run first)
- `code/business_entity_resolution/src/setup_gpu.sh` — GPU instance bootstrap
- `STATUS.md` — detailed progress log (what's done, what's next)

## Compute
- **Target:** Lambda Labs `gpu_1x_a10` (A10G 24GB, ~$0.60/hr) — no quota needed
- **AWS:** G-instance quota denied; appeal filed for g5.xlarge (4 vCPU)
- **CPU EC2:** Running `run_test_parallel.py` (7 workers, nohup) → submission #1 ~7 AM IST Sep 27
- **HF Pro + $200 credits:** Not useful for batch pipeline (no SSH access)
- Do NOT commit `ml-challenge.pem` (EC2 private key)

## Architecture Notes
- Run `benchmark_recall.py` BEFORE full pipeline — confirms R@200 ≥ 0.93
- Graph closure is OFF by default (`graph_enabled=False`) — too risky for F0.5
- France entities get +0.10 score bump in threshold sweep
- Isotonic calibration must be fitted AFTER LightGBM retrain on full 100K
- Use `tmux` on remote instances so SSH disconnects don't kill runs
