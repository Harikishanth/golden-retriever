# Claude / agent handoff — Amazon ML Challenge 2026

**Stamp:** `2026-09-25 ~21:00 IST`
**Team:** Da Big Three (Harikishanth R +2)
**Machine (primary):** AWS EC2 r5.2xlarge (64GB RAM, eu-north-1), Ubuntu 26.04, Python 3.14.4
**Machine (local):** `D:\ML challenge` (Windows 11, Python 3.14.2) — code editing only, too slow for pipeline
**Contest clock:** 25 Sep 00:00 → 27 Sep 23:59 IST. **5 submits/day.** Public + private; final = private.
**EC2 installed:** rapidfuzz 3.14.6, lightgbm 4.7.0, sentence-transformers 6.1.0, faiss-cpu 1.15.1, torch 2.14.0, numpy, pandas, scikit-learn
**EC2 data:** `/home/ubuntu/student_resource/dataset/` (downloaded from Unstop CDN in 3s)
**EC2 cost:** ~$0.50/hr, ~$100 AWS credits available. STOP WHEN IDLE.
**GitHub:** https://github.com/Harikishanth/golden-retriever (code only, no dataset)
**Status:** Zero Unstop submissions. About to run first BM25+LightGBM holdout on EC2.

---

## 1. What the problem is (FACT)

Business **entity resolution**. Three sources, no shared ID. Fields: `entity_id`, `business_name`, `business_address`, `country`.

- **S1** = clean reference. For each S1 row, list matching S2/S3 ids (zero/one/many).
- **Metric:** macro **F0.5** per S1, then average. Empty-on-singleton = **1.0**. Precision ~2× recall.
- **LB file:** `output/matching_results.tsv` (`source1_entity_id`, `matched_entity_ids` comma-joined).
- **Also in zip:** `output/candidate_pairs.tsv` = exact candidate set fed to matcher.
- **France in test only** (15% of test S1). Country is open-set string.
- **Banned:** geocoding APIs, business registries, commercial ER APIs, external data.
- **Model cap:** ≤8B params, MIT/Apache 2.0. Libraries (rapidfuzz, lightgbm, sentence-transformers) are fine.
- **Validator:** `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` (run from `dataset/student_resource/`)

---

## 2. Full conversation arc

### Session 1 (before this chat, morning 25 Sep)
1. Pre-contest research (15–24 Sep): scraped 2025 Reddit/PJ/Jatin into `research/`. Built `CONTEXT.md`.
2. Drop ~05:11 IST: Unstop video + PDF + LB screenshot. Task = ER confirmed. Team = Da Big Three.
3. Data loaded ~07:35: streamed TSVs. Exact dummy on full train = 0.073.
4. Rule pipeline written: 5-key hash blocking, 3 precision rules. Best holdout **0.680** at ranked CAP=40.

### Session 2 (this chat, ~11:00–16:30 IST 25 Sep)
5. **Architecture replaced:** rules → logistic regression on pair features.
6. **v1** (CAP=80, 9 feat, LR): val F0.5 **0.738**, ceil **0.814**. Confirmed sklearn+numpy available.
7. **v2** (CAP=150, +Jaro-Winkler): val **0.7415**, ceil **0.834**. OOM on Phase 6 → fixed (cast coef to float32).
8. **v3** (CAP=150, +tfidf_cos, MAX_POSTING=2000 guard): val **0.7422**, ceil **0.835**. KEY: `ceil_macro=0.9287`. Hash-key blocking cannot produce score above ~0.93. BM25 rewrite decided.
9. **Diagnosed hash-key OOM pattern:** NP (name-prefix) key had posting lists of millions → OOM. Fixed permanently with `MAX_POSTING=2000` guard in `candidates_for`.
10. **rapidfuzz + lightgbm installed.**
11. **BM25 blocking written** (blocking.py rewritten, individual tokens not compound keys).
12. **run.py rewritten** for LightGBM (predict_proba, batched test inference 20k entities/batch).
13. **match.py updated** to 15 features (+rapidfuzz token_sort, token_set, partial_ratio, addr_sort).
14. **v4 launched locally** — RAM bottleneck: pool loaded at 7047s (117 min). Featurization not started. Estimated finish: 3+ more hours. Running in background.
15. **model.json and model.lgb deleted** before v4 started. No submittable model on disk.
16. **Cloud strategy decided:** Kaggle (29GB RAM, T4 GPU, free) or AWS ml.r5.2xlarge (64GB, ~$0.50/hr) or Vultr GPU. User has all three. Move there to fix RAM bottleneck.
17. **Dense retrieval planned** for tonight: sentence-transformers/all-MiniLM-L6-v2 + FAISS → recall ceiling >0.95 → expected 0.88-0.93 holdout.

---

## 3. Data on this machine (FACT)

Root: `D:\ML challenge\dataset\student_resource\dataset\`
Always `pd.read_csv(..., sep="\t")` or manual split on `\t`. Always `python -X utf8` on Windows.

| File | Rows |
|---|---:|
| train S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 |
| train GT | 2,206,821 |
| test S1 / S2 / S3 | 1,732,544 / 4,887,273 / 5,082,316 |

Test S1: US 663,106 · India 809,986 · **France 259,452 (15.0%)**.
Train GT singletons: **5.58%**. Mean match count 3.67, max 11.

---

## 4. Experiment log summary

| Run | Blocking | Classifier | Features | Val F0.5 | Ceil (recall) | ceil_macro |
|---|---|---|---|---:|---:|---:|
| rule-cap60 | hash 5-key | rules | — | 0.592 | 0.693 | — |
| rule-cap40-ranked | hash 5-key | rules | — | 0.680 | 0.736 | — |
| v1 CAP=80 | hash 7-key | LR | 9 | 0.738 | 0.814 | — |
| v2 CAP=150 | hash 7-key | LR | 10 | 0.7415 | 0.834 | — |
| **v3 CAP=150** | hash 7-key | LR | 11 | **0.7422** | **0.835** | **0.9287** |
| v4 (running) | **BM25** | **LightGBM** | **15** | TBD | TBD | TBD |

`ceil_macro` = macro F0.5 with a perfect classifier on the current candidates. 0.9287 means hash-key blocking cannot score above ~0.93 no matter how good the model. **BM25 / dense retrieval is required to compete for top 3.**

Public LB top: ~0.96. Gap to close: from 0.9287 ceiling to 0.96 = needs BM25 (ceiling ~0.93-0.95) + dense retrieval (ceiling ~0.97).

---

## 5. Code on disk (current state)

```
code/business_entity_resolution/src/
  io_utils.py        # unchanged; ROOT hardcoded to dataset path
  normalize.py       # fold, suffixes, French included; no abbrev expansion yet
  blocking.py        # BM25 token retrieval; build_index_and_idf + candidates_for
  match.py           # 15 features; LR fallback; rapidfuzz with graceful import
  evaluate.py        # macro F0.5, recall_ceiling, f05_one
  run.py             # LightGBM + LR fallback; batched test inference; holdout/test modes
  model.json         # DOES NOT EXIST (deleted for v4)
  model.lgb          # DOES NOT EXIST (v4 not finished)
notes.md             # experiment log
output/              # DOES NOT EXIST (no test run yet)
```

**Run commands (from `code/business_entity_resolution/`):**
```bash
python -X utf8 src/run.py holdout   # train LightGBM, save model.lgb + model.json
python -X utf8 src/run.py test       # batch inference → output/*.tsv
```

---

## 6. Immediate next steps (ordered by priority)

### 6.1 Get on Kaggle/AWS NOW (highest priority)

**Kaggle (free, fastest):**
1. kaggle.com → New Notebook → GPU T4 accelerator
2. In notebook: `!pip install rapidfuzz lightgbm -q`
3. Upload TSV files as a Kaggle Dataset (or re-download from Unstop)
4. Clone/upload the code, update `ROOT` in `io_utils.py` to `/kaggle/input/...`
5. `!python -X utf8 src/run.py holdout` → ~35 min on 29GB RAM
6. `!python -X utf8 src/run.py test` → ~20 min
7. Download `output/matching_results.tsv`, upload to Unstop

**AWS SageMaker (if Kaggle is slow):**
1. Open `ml-challenge-notebook` → change instance to `ml.r5.2xlarge` (64GB, stop first)
2. `pip install rapidfuzz lightgbm` in a terminal cell
3. Same run commands. Faster than Kaggle for large jobs.

### 6.2 Dense retrieval tonight (GPU required)

```python
# On Kaggle T4 or Vultr GPU:
pip install sentence-transformers faiss-gpu

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Encode name + " " + address for all S2/S3 records
# FAISS IndexFlatIP (inner product = cosine after normalize) for top-100 per S1
# Combine with BM25 candidates (union)
# Add cosine_sim as a feature in match.py
# Retrain LightGBM
```

Expected outcome: recall ceiling >0.95, holdout F0.5 ~0.88-0.93, public LB top 10-20 range.

### 6.3 Teammate task

Add abbreviation expansion to `normalize.py`:
```python
ABBREV_MAP = {
    "corp": "corporation", "corporation": "corp",
    "inc": "incorporated", "incorporated": "inc",
    "ltd": "limited", "limited": "ltd",
    "pvt": "private", "private": "pvt",
}
def expand_abbrevs(toks):
    out = []
    for t in toks:
        out.append(t)
        if t in ABBREV_MAP:
            out.append(ABBREV_MAP[t])
    return out
# Call inside name_tokens() before returning
```

This adds both forms so "McDonald Corp" and "McDonald Corporation" generate the same blocking tokens.

---

## 7. Architecture detail

### Blocking (BM25, current blocking.py)

```
Index: (country, 'N', token) → [eid, eid, ...]   # name tokens, country-scoped
       (country, 'A', token) → [eid, eid, ...]   # address tokens
       (country, 'H', house) → [eid, eid, ...]   # house numbers

Score per S1 query:
  Σ IDF(t) * 2.0  for each shared name token  (MAX_POSTING=2000 guard)
  Σ IDF(t) * 1.2  for each shared addr token
  + 8.0           for each shared house number

CAP=150, ranked by score. ~2.15M posting lists over 10M records.
```

### Features (15, match.py)

| # | Name | Description |
|---|---|---|
| 1 | idf_name_jac | IDF-weighted Jaccard on name tokens |
| 2 | name_contain | fraction of S1 name tokens in S2/S3 |
| 3 | char_sim | trigram Jaccard on squashed name (+9.45 weight) |
| 4 | house_agree | shared house number binary |
| 5 | addr_jac | Jaccard on address tokens (+6.69 weight) |
| 6 | addr_exact | full normalized addr exact match |
| 7 | city_agree | last addr token matches |
| 8 | name_rare | name appears <4 times in S1 |
| 9 | name_len_ratio | min/max squashed name length |
| 10 | jaro_winkler | JW on folded full name (+4.37 weight) |
| 11 | tfidf_cos | TF-IDF cosine on name tokens |
| 12 | rf_token_sort | rapidfuzz token_sort_ratio |
| 13 | rf_token_set | rapidfuzz token_set_ratio |
| 14 | rf_partial | rapidfuzz partial_ratio |
| 15 | rf_addr_sort | rapidfuzz token_sort on addr key |

### Classifier (run.py)

LightGBM: n_estimators=500, num_leaves=63, lr=0.05, scale_pos_weight=auto.
Falls back to sklearn LR if lightgbm not installed.
Threshold sweep: 0.05 to 0.93 in steps of 0.02, optimize val 30k F0.5.
Model saved: `model.lgb` (LightGBM native) + `model.json` (threshold + type).

---

## 8. Pitfalls already paid for

- `pd.read_csv` without `sep="\t"` smashes rows.
- Always `python -X utf8` on Windows (Tamil/French break cp1252).
- `candidates_for` without `idf` arg falls back to uniform weights — pass it.
- `MAX_POSTING=2000` guard is essential; any broad key (name prefix, common token) without it OOMs `scored` dict.
- BM25 index + pool = ~10GB local RAM → swapping on laptop. **Use cloud.**
- Batch test inference (BATCH_ENTITIES=20k in run.py) — 226M pairs total for test cannot fit in memory at once.
- `model.json` was deleted before v4. Do not try to run test locally without retraining.
- Hard first-N without ranking discards true matches (learned from cap-60 failure).
- Name-only accept kills singletons (63% share normalized name with some non-match).
- Do not submit without `candidate_pairs.tsv` in the zip.

---

## 9. UNKNOWN as of this stamp

- v4 BM25 holdout F0.5 / ceil_macro (running locally, slow)
- Dense retrieval recall ceiling (planned tonight)
- Any public LB score for this team — **zero Unstop submits so far**
- Whether France performs differently on BM25 vs hash keys
- Who clicks submit; whether failed validate burns a daily shot
- How many of Friday's 5 shots already used
- Test singleton rate (no test GT)
