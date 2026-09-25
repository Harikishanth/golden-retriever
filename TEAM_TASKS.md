# Team Task Board — Amazon ML Challenge 2026

**Last updated:** 2026-09-25 21:30 IST
**Team:** Da Big Three
**Deadline:** 27 Sep 23:59 IST (5 submits/day)
**Current score on Unstop:** NONE (zero submissions)
**Leaderboard:** Top 0.987, Rank 433 = 0.908, Rank 445 = 0.903

---

## EC2 Instance (shared)

```
Host: ec2-16-171-199-121.eu-north-1.compute.amazonaws.com
User: ubuntu
Key: ml-challenge.pem (shared via team chat)
Cost: ~$0.50/hr — STOP WHEN NOT IN USE
```

**Connect (Mac/Linux):**
```bash
chmod 400 ml-challenge.pem
ssh -i ml-challenge.pem ubuntu@ec2-16-171-199-121.eu-north-1.compute.amazonaws.com
```

**Connect (Windows PowerShell):**
```powershell
icacls "ml-challenge.pem" /inheritance:r
icacls "ml-challenge.pem" /grant:r "$($env:USERNAME):R"
ssh -i "ml-challenge.pem" ubuntu@ec2-16-171-199-121.eu-north-1.compute.amazonaws.com
```

**IMPORTANT — always use tmux so your process survives disconnects:**
```bash
# First time: create a named session
tmux new -s myname

# Run your commands inside tmux...

# Disconnected? SSH back in, then reattach:
tmux attach -t myname

# Detach on purpose (keep it running): press Ctrl+B, then D
# List sessions: tmux ls
# Kill a session: tmux kill-session -t myname
```

Each person should use their own session name (`hari`, `reshma`, `jehrome`).
If you see "sessions should be nested" — you're already inside tmux. Just run your commands.
If you see no output after starting a script — **WAIT**. Loading millions of rows takes 1-2 min of silence.

**EC2 layout:**
```
~/student_resource/dataset/       <- all TSV files (train + test)
~/golden-retriever/               <- git clone of our code
~/golden-retriever/code/business_entity_resolution/src/  <- pipeline code
~/golden-retriever/output/        <- will contain submission TSVs after test run
```

**Installed on EC2:** Python 3.14, rapidfuzz, lightgbm, sentence-transformers, faiss-cpu, torch, numpy, pandas, scikit-learn

---

## Who does what

### Harikishanth (Windows) — Pipeline Lead

**RIGHT NOW:**
- [ ] Monitor `python3 src/run.py holdout` on EC2 (~30-40 min)
- [ ] When holdout finishes, run `python3 src/run.py test` (~20 min)
- [ ] Validate output: `cd ~/student_resource && python3 utils/validate_submission.py --matching ~/golden-retriever/output/matching_results.tsv --candidate ~/golden-retriever/output/candidate_pairs.tsv --test-dir dataset/test`
- [ ] Download `matching_results.tsv` and submit to Unstop
- [ ] Record score in this file

**AFTER FIRST SUBMIT:**
- [ ] Run Reshma's dense retrieval script
- [ ] Integrate dense features into LightGBM pipeline
- [ ] Retrain and submit round 2

### Reshma (Mac) — Dense Retrieval

**RIGHT NOW:**
- [ ] SSH into EC2 (see commands above)
- [ ] Create `dense_retrieval.py` (code below in Appendix A)
- [ ] DO NOT run it yet — wait until holdout finishes (they share RAM)
- [ ] Read through `normalize.py` and `match.py` to understand the feature set

**AFTER HOLDOUT FINISHES:**
- [ ] Run `python3 src/dense_retrieval.py` on EC2
- [ ] Report the dense recall@200 number in team chat
- [ ] Help integrate cosine_sim as feature #16 in match.py

### Jehrome (Windows) — Docs + Submission Packaging

**RIGHT NOW:**
- [ ] Clone repo: `git clone https://github.com/Harikishanth/golden-retriever.git`
- [ ] Download `Documentation_template.md` from Unstop (or grab from EC2: `~/student_resource/Documentation_template.md`)
- [ ] Fill in the template (details in Appendix B)
- [ ] Create the zip packaging script (Appendix C)

**AFTER FIRST SUBMIT:**
- [ ] Update docs with actual scores
- [ ] Package final zip for Unstop
- [ ] Review code comments and README

---

## Pipeline Architecture (what we built)

```
Input: S1 entity (name, address, country)
  |
  v
[BM25 Blocking] — country-scoped token index, IDF-weighted scoring
  |  returns top-150 candidates from S2+S3
  v
[Feature Extraction] — 15 similarity features per (S1, candidate) pair
  |  name: char_sim, jaro_winkler, token_sort, token_set, idf_jaccard
  |  addr: jaccard, exact match, city agree, house number agree
  |  meta: name_rare, name_len_ratio, tfidf_cosine
  v
[LightGBM Classifier] — binary: match or not
  |  threshold tuned on F0.5 (precision ~2x recall)
  v
Output: matching_results.tsv (S1 id -> comma-separated matched S2/S3 ids)
```

**Best holdout so far:** 0.7422 (v3, hash-key blocking)
**Expected with BM25:** 0.80-0.87
**Expected with BM25 + dense retrieval:** 0.88-0.93

---

## Key Rules (read this or get DQ'd)

1. **Model cap:** 8B params max, MIT or Apache 2.0 license only
2. **No external data:** no geocoding APIs, no business registries, no internet lookups
3. **Submit file:** `matching_results.tsv` (tab-separated: `source1_entity_id\tmatched_entity_ids`)
4. **Candidate file:** `candidate_pairs.tsv` must be a superset of matches (audited)
5. **France is in test only** (15% of test S1) — not in training data. Do not filter it out.
6. **Singletons:** predicting empty = score 1.0 on that entity. Predicting ANY match on a true singleton = score 0.0. When in doubt, predict empty.
7. **5 submissions per day.** Do not waste on broken format — validate first.
8. **One machine per person.** Don't dual-login to Unstop.
9. **Stop EC2 when idle.** It costs $0.50/hr. We have ~$100 credits.

---

## Submission Checklist

Before every Unstop upload:

```bash
# On EC2, from ~/student_resource/
python3 utils/validate_submission.py \
  --matching ~/golden-retriever/output/matching_results.tsv \
  --candidate ~/golden-retriever/output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Must say VALID. If it warns, fix before uploading.

---

## Score Log

| # | Time (IST) | What | Holdout F0.5 | Public LB | Notes |
|---|---|---|---|---|---|
| - | 25 Sep ~21:30 | BM25+LightGBM v4 | TBD | TBD | running on EC2 |
| | | | | | |

---

## Appendix A — Dense Retrieval Script (Reshma)

Save as `~/golden-retriever/code/business_entity_resolution/src/dense_retrieval.py`:

```python
"""Dense retrieval blocking with sentence-transformers + FAISS.
Usage: python3 src/dense_retrieval.py
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from io_utils import ROOT, iter_source

BATCH_SIZE = 4096
TOP_K = 200

def encode_records(model, paths, label=""):
    eids, texts = [], []
    for path in paths:
        for eid, name, addr, country in iter_source(path):
            eids.append(eid)
            texts.append(f"{name} {addr} {country}")
    print(f"{label}: {len(eids):,} records, encoding...", flush=True)
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE,
        show_progress_bar=True, normalize_embeddings=True
    )
    return eids, np.array(embeddings, dtype=np.float32)

def main():
    t0 = time.time()
    import faiss
    from sentence_transformers import SentenceTransformer

    print("Loading model...", flush=True)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    pool_paths = [ROOT / "train/train_source2.tsv", ROOT / "train/train_source3.tsv"]
    pool_eids, pool_emb = encode_records(model, pool_paths, "Pool")
    print(f"Pool encoded: {pool_emb.shape} ({time.time()-t0:.0f}s)", flush=True)

    dim = pool_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(pool_emb)
    print(f"FAISS index built ({time.time()-t0:.0f}s)", flush=True)

    s1_paths = [ROOT / "train/train_source1.tsv"]
    s1_eids, s1_emb = encode_records(model, s1_paths, "S1")
    print(f"S1 encoded: {s1_emb.shape} ({time.time()-t0:.0f}s)", flush=True)

    print(f"Searching top-{TOP_K}...", flush=True)
    scores, indices = index.search(s1_emb, TOP_K)
    print(f"Search done ({time.time()-t0:.0f}s)", flush=True)

    out_dir = Path(__file__).resolve().parents[1] / "dense_cache"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "s1_eids.npy", np.array(s1_eids))
    np.save(out_dir / "pool_eids.npy", np.array(pool_eids))
    np.save(out_dir / "dense_scores.npy", scores)
    np.save(out_dir / "dense_indices.npy", indices)
    print(f"Saved to {out_dir} ({time.time()-t0:.0f}s)", flush=True)

    from io_utils import iter_ground_truth
    gt = dict(iter_ground_truth(ROOT / "train/train_ground_truth.tsv"))
    s1_eid_list = list(s1_eids)
    pool_eid_list = list(pool_eids)

    total_gold, found = 0, 0
    for i, sid in enumerate(s1_eid_list[:100_000]):
        gold = gt.get(sid, set())
        if not gold:
            continue
        retrieved = {pool_eid_list[j] for j in indices[i] if j >= 0}
        total_gold += len(gold)
        found += len(gold & retrieved)

    recall = found / total_gold if total_gold else 0
    print(f"Dense recall@{TOP_K} (100k sample): {recall:.4f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
```

**Expected runtime:** ~20-30 min on EC2 (CPU encoding, no GPU).
**Expected recall@200:** >0.95 (vs 0.83 with hash keys, ~0.93 with BM25).

---

## Appendix B — Documentation Template (Jehrome)

Fill these sections in `Documentation_template.md`:

**Approach:**
> Two-stage entity resolution pipeline: blocking (candidate retrieval) followed by learned matching (binary classification).

**Blocking Strategy:**
> BM25-style token retrieval. Country-scoped inverted index on normalized name and address tokens. IDF-weighted scoring with separate weights for name tokens (2.0), address tokens (1.2), and house numbers (8.0). Top-150 candidates per S1 entity. MAX_POSTING=2000 guard prevents memory issues on common tokens.

**Feature Engineering (15 features):**
> Name similarity: IDF-weighted Jaccard, character trigram Jaccard, Jaro-Winkler, rapidfuzz token_sort_ratio/token_set_ratio/partial_ratio, name containment, name length ratio, TF-IDF cosine, rare name flag.
> Address similarity: token Jaccard, exact match, city agreement, house number agreement, rapidfuzz token_sort on address.

**Model:**
> LightGBM classifier (500 estimators, 63 leaves, lr=0.05). Threshold swept on holdout F0.5. Trained on 70k entities, validated on 30k. Falls back to logistic regression if LightGBM unavailable.

**France handling:**
> Country is treated as open-set string. Blocking is country-scoped so France entities only match France pool records. No country-specific rules hardcoded.

**Scores:** (fill after submission)
> Holdout F0.5: ___
> Public LB: ___

---

## Appendix C — Zip Packaging Script (Jehrome)

Save as `~/golden-retriever/package_submission.sh`:

```bash
#!/bin/bash
set -e

TEAM="Da_Big_Three"
OUT_DIR=~/golden-retriever/output
CODE_DIR=~/golden-retriever/code/business_entity_resolution
ZIP_NAME="${TEAM}_submission.zip"

# Check files exist
for f in "$OUT_DIR/matching_results.tsv" "$OUT_DIR/candidate_pairs.tsv"; do
    [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done

# Build zip structure
STAGING=$(mktemp -d)
mkdir -p "$STAGING/output"
mkdir -p "$STAGING/code/business_entity_resolution/src"

cp "$OUT_DIR/matching_results.tsv" "$STAGING/output/"
cp "$OUT_DIR/candidate_pairs.tsv" "$STAGING/output/"
cp "$CODE_DIR/src/"*.py "$STAGING/code/business_entity_resolution/src/"
cp "$CODE_DIR/README.md" "$STAGING/code/business_entity_resolution/" 2>/dev/null || true
cp "$CODE_DIR/requirements.txt" "$STAGING/code/business_entity_resolution/" 2>/dev/null || true
cp ~/student_resource/Documentation_template.md "$STAGING/" 2>/dev/null || true

cd "$STAGING"
zip -r ~/"$ZIP_NAME" .
echo "Created ~/$ZIP_NAME"
rm -rf "$STAGING"
```

Run with: `bash ~/golden-retriever/package_submission.sh`
