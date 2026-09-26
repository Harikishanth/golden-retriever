# Research-Backed Improvement Plan

**Source:** 3 parallel research sweeps (ER SOTA, multilingual/France, quick wins)
**Generated:** 2026-09-25 ~22:00 IST

---

## Priority-Ranked Improvements

### Tier 1 — Do Tonight (free/fast, biggest gains)

| # | What | Why | Effort | Expected Gain |
|---|---|---|---|---|
| 1 | **Switch to `paraphrase-multilingual-MiniLM-L12-v2`** in dense_retrieval.py | English-only model misses France (15% of test). Multilingual model is same speed, same size (118M/384d), Apache 2.0 | Change 1 line | +5-15% France recall |
| 2 | **Make `name_rare` continuous** instead of binary | `log(1/name_freq)` gives LightGBM way more signal than a binary <4 cutoff. "Sharma" matching is weak, "Quattrocchi" matching is near-certain | Change 3 lines in match.py | +0.01-0.02 |
| 3 | **Singleton two-stage threshold** | If best candidate score is below a strict threshold, predict empty. Prevents false merges on singletons (5.58% of data, each worth 1.0 if correct, 0.0 if wrong) | Add 10 lines in run.py | +0.01-0.03 |

### Tier 2 — Saturday Morning (1-2 hours each)

| # | What | Why | Effort | Expected Gain |
|---|---|---|---|---|
| 4 | **Embedding cosine as feature #16** | Single strongest feature we're missing. Top ER repos confirm: embedding features dominate string similarity | Wire dense_retrieval.py output into match.py | +0.03-0.05 |
| 5 | **Union blocking: BM25 + dense + MinHash** | Each method catches ~5% the others miss. Union → recall ceiling ~0.97-0.99. `datasketch` library (MIT) for MinHash on char 4-grams | 1 hour | +0.02-0.03 |
| 6 | **CAP 150 → 300 + train on 500k** | More candidates = higher recall. More training data = better LightGBM | Change 2 constants | +0.01-0.02 |
| 7 | **Phonetic features** (Soundex/Metaphone) | "McDonald"="MacDonald", "Smith"="Smyth". `jellyfish` library (BSD) | 30 min | +0.005-0.01 |

### Tier 3 — Saturday Afternoon (the big gun)

| # | What | Why | Effort | Expected Gain |
|---|---|---|---|---|
| 8 | **Ditto-style cross-encoder** | Serialize pairs as `"COL name VAL X COL addr VAL Y"`, fine-tune DistilBERT (66M, Apache 2.0) as match/no-match. Got 96.5% F1 on similar company matching task. Replaces LightGBM as scorer on top-50 candidates | 3-4 hours | +0.03-0.05 |
| 9 | **Upgrade to `BAAI/bge-m3`** (568M, MIT) if multilingual-MiniLM isn't enough | Best multilingual retrieval model. 768-dim. Handles French natively + scores 10+ points above MiniLM on retrieval benchmarks | Swap 1 line, re-encode (~40 min) | +0.01-0.03 over MiniLM |

### Tier 4 — Sunday Polish

| # | What | Why | Effort | Expected Gain |
|---|---|---|---|---|
| 10 | **Ensemble LightGBM + cross-encoder scores** | Different models catch different errors | 30 min | +0.005-0.01 |
| 11 | **Precision floor guard** | Sweep threshold with constraint precision >= 0.90 | 15 min | +0.005 |
| 12 | **France-specific threshold** | If France underperforms, tighten threshold for country="France" | 15 min | +0.005-0.01 |

---

## Model Choices (all legal: <8B params, MIT or Apache 2.0)

| Model | Params | Dim | License | Use |
|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 118M | 384 | Apache 2.0 | Dense blocking (default) |
| `BAAI/bge-m3` | 568M | 768 | MIT | Dense blocking (upgrade if MiniLM recall is low) |
| `distilbert-base-uncased` | 66M | 768 | Apache 2.0 | Ditto-style cross-encoder fine-tuning |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | — | Apache 2.0 | Pretrained cross-encoder (no fine-tune needed) |

---

## Feature Set Roadmap

**Current (15 features):**
idf_name_jac, name_contain, char_sim, house_agree, addr_jac, addr_exact, city_agree, name_rare, name_len_ratio, jaro_winkler, tfidf_cos, rf_token_sort, rf_token_set, rf_partial, rf_addr_sort

**Add Tonight (+1):**
- `name_rarity_continuous` — replace binary name_rare with log(1/freq)

**Add Saturday (+5):**
- `embedding_cosine` — cosine similarity from sentence-transformer embeddings
- `phonetic_name_match` — soundex(name1) == soundex(name2)
- `edit_distance_ratio_name` — normalized Levenshtein on name
- `edit_distance_ratio_addr` — normalized Levenshtein on address
- `digit_overlap` — Jaccard on digit sequences (catches zip/pin matches)

**Total: 21 features → LightGBM handles this easily**

---

## Key Insight from Research

The gap between 0.85 and 0.987 is NOT about better string matching. It's about:
1. **Recall ceiling** — dense retrieval gets you candidates that BM25 never finds
2. **Semantic scoring** — embedding cosine and cross-encoders understand that "Corp" = "Corporation" and "Boulangerie" = "Bakery" without explicit rules
3. **Singleton protection** — two-stage thresholding prevents the most expensive errors (each false merge on a singleton costs 1.0 in the macro average)
