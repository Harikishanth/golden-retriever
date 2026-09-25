"""BM25 blocking + LightGBM (or LR fallback) matcher.

    python -X utf8 src/run.py holdout   # train model, sweep threshold, save
    python -X utf8 src/run.py test       # batch inference → output/*.tsv

Run from code/business_entity_resolution/.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from blocking import build_index_and_idf, candidates_for  # noqa: E402
from evaluate import f05_one, fmt, summarize  # noqa: E402
from io_utils import ROOT, iter_ground_truth, iter_source, write_submission  # noqa: E402
from match import FEATURE_NAMES, features, name_freq, precompute  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
HOLDOUT_N = 100_000
SEED = 42
MODEL_LGB = HERE / "src/model.lgb"
MODEL_JSON = HERE / "src/model.json"


# ── data loading ──────────────────────────────────────────────────────

def load_s1(paths):
    """eid -> (name, addr, country, precomputed)."""
    rec = {}
    for path in paths:
        for eid, name, addr, country in iter_source(path):
            rec[eid] = (name, addr, country, precompute(name, addr))
    return rec


def load_pool(paths):
    """eid -> (name, addr, country)."""
    rec = {}
    for path in paths:
        for eid, name, addr, country in iter_source(path):
            rec[eid] = (name, addr, country)
    return rec


# ── model helpers ─────────────────────────────────────────────────────

def _clf_scores(clf, X: np.ndarray) -> np.ndarray:
    """Return 1-D probability array for positive class."""
    try:
        return clf.predict_proba(X)[:, 1].astype(np.float32)
    except AttributeError:
        # LR stored as linear weights in model.json — handled via score()
        w = np.array(clf["coef"], dtype=np.float32)
        b = np.float32(clf["intercept"])
        raw = X @ w + b
        return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)


def save_model_lgbm(clf, threshold: float):
    try:
        clf.booster_.save_model(str(MODEL_LGB))
    except AttributeError:
        pass
    with open(MODEL_JSON, "w") as f:
        json.dump({"threshold": round(float(threshold), 4), "type": "lgbm"}, f, indent=2)
    print(f"Model saved → {MODEL_LGB.name} + {MODEL_JSON.name}", flush=True)


def save_model_lr(coef, intercept, threshold: float):
    data = {
        "type": "lr",
        "coef": [round(float(c), 6) for c in coef],
        "intercept": round(float(intercept), 6),
        "threshold": round(float(threshold), 4),
    }
    with open(MODEL_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Model saved → {MODEL_JSON.name}", flush=True)


def load_model():
    """Returns (clf_or_meta, threshold). clf is None if LR stored in meta."""
    meta = json.loads(MODEL_JSON.read_text()) if MODEL_JSON.exists() else {}
    threshold = float(meta.get("threshold", 0.5))
    if MODEL_LGB.exists():
        import lightgbm as lgb
        return lgb.Booster(model_file=str(MODEL_LGB)), threshold
    if meta.get("type") == "lr":
        return meta, threshold
    return None, threshold


# ── holdout ───────────────────────────────────────────────────────────

def run_holdout():
    t0 = time.time()

    s1 = load_s1([ROOT / "train/train_source1.tsv"])
    print(f"S1: {len(s1):,} ({time.time()-t0:.0f}s)", flush=True)
    freq = name_freq(s1)
    gt = dict(iter_ground_truth(ROOT / "train/train_ground_truth.tsv"))

    rng = random.Random(SEED)
    holdout = rng.sample(list(s1), HOLDOUT_N)
    n_train = int(HOLDOUT_N * 0.7)

    pool_paths = [ROOT / "train/train_source2.tsv", ROOT / "train/train_source3.tsv"]
    index, idf = build_index_and_idf(pool_paths)
    pool = load_pool(pool_paths)
    print(f"Pool: {len(pool):,} ({time.time()-t0:.0f}s)", flush=True)

    # ── Phase 1: featurize ────────────────────────────────────────────
    entity_data = []
    total_pairs = 0
    for i, sid in enumerate(holdout):
        _name, _addr, country, s1f = s1[sid]
        cands, was_capped = candidates_for(index, _name, _addr, country, idf)
        gold = gt.get(sid, set())
        scored = []
        for c in cands:
            cname, caddr, _ = pool[c]
            candf = precompute(cname, caddr)
            scored.append((c, features(s1f, country, candf, idf, freq)))
        entity_data.append((sid, scored, gold, set(cands), was_capped))
        total_pairs += len(scored)
        if (i + 1) % 20000 == 0:
            print(f"  {i+1:,} featurized  {total_pairs:,} pairs  ({time.time()-t0:.0f}s)", flush=True)

    print(f"Features done: {total_pairs:,} pairs ({time.time()-t0:.0f}s)", flush=True)

    # ── Phase 2: numpy ────────────────────────────────────────────────
    X_list, y_list, boundaries = [], [], []
    for sid, scored, gold, cand_set, _ in entity_data:
        start = len(X_list)
        for c, feat in scored:
            X_list.append(feat)
            y_list.append(1 if c in gold else 0)
        boundaries.append((start, len(X_list)))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    del X_list, y_list

    n_pos = int(y.sum())
    print(f"Pairs: {X.shape[0]:,}  pos: {n_pos:,} ({100*n_pos/max(1,len(y)):.1f}%)", flush=True)

    # ── Phase 3: train ────────────────────────────────────────────────
    train_end = boundaries[n_train - 1][1]
    X_train, y_train = X[:train_end], y[:train_end]

    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            min_child_samples=20, n_jobs=-1, verbose=-1,
            scale_pos_weight=float((y_train == 0).sum()) / max(1, y_train.sum()),
        )
        clf.fit(X_train, y_train)
        use_lgbm = True
        print(f"Classifier: LightGBM ({time.time()-t0:.0f}s)", flush=True)
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)
        use_lgbm = False
        print(f"Classifier: LogisticRegression ({time.time()-t0:.0f}s)", flush=True)

    # ── Phase 4: threshold sweep on val 30k ───────────────────────────
    scores_all = _clf_scores(clf, X)

    best_f05, best_t = 0.0, 0.5
    print("\nThreshold sweep (val 30k):", flush=True)
    thresholds = [i / 100.0 for i in range(5, 95, 2)]
    for t_val in thresholds:
        rows = []
        for ei in range(n_train, HOLDOUT_N):
            sid, scored, gold, cand_set, _ = entity_data[ei]
            start, end = boundaries[ei]
            pred = {scored[j][0] for j in range(end - start) if scores_all[start + j] >= t_val}
            rows.append((pred, gold, cand_set))
        stats = summarize(rows)
        hit = stats["macro_f05"] > best_f05
        if hit:
            best_f05 = stats["macro_f05"]
            best_t = t_val
        if int(t_val * 100) % 10 == 0 or hit:
            mark = " ***" if hit else ""
            print(f"  t={t_val:.2f}  F0.5={stats['macro_f05']:.4f}  ceil={stats['recall_ceiling']:.3f}{mark}", flush=True)

    print(f"\nBest val: t={best_t:.2f}  F0.5={best_f05:.4f}", flush=True)

    # ── Phase 5: retrain on 100k, save ───────────────────────────────
    try:
        clf.fit(X, y)
    except Exception:
        pass
    if use_lgbm:
        save_model_lgbm(clf, best_t)
    else:
        save_model_lr(clf.coef_[0], clf.intercept_[0], best_t)
    print(f"Saved ({time.time()-t0:.0f}s)", flush=True)

    # ── Phase 6: full holdout report + ceiling ────────────────────────
    scores_all = _clf_scores(clf, X)
    rows, capped_n = [], 0
    for ei in range(HOLDOUT_N):
        sid, scored, gold, cand_set, was_capped = entity_data[ei]
        start, end = boundaries[ei]
        pred = {scored[j][0] for j in range(end - start) if scores_all[start + j] >= best_t}
        rows.append((pred, gold, cand_set))
        capped_n += was_capped

    stats = summarize(rows)
    ceil_macro = sum(f05_one(gold & cand_set, gold) for _, gold, cand_set in rows) / HOLDOUT_N
    print(f"Macro F0.5 ceiling (perfect matcher): {ceil_macro:.4f}", flush=True)

    line = (f"{datetime.now():%Y-%m-%d %H:%M}  holdout  {fmt(stats)}  "
            f"capped={capped_n/HOLDOUT_N:.3f}  {time.time()-t0:.0f}s  "
            f"[val={best_f05:.4f} t={best_t:.2f} ceil_macro={ceil_macro:.4f}]")
    print(f"\n{line}", flush=True)
    with open(HERE.parents[1] / "notes.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── test ──────────────────────────────────────────────────────────────

BATCH_ENTITIES = 20_000  # process this many S1 per scoring batch


def run_test():
    t0 = time.time()

    s1 = load_s1([ROOT / "test/test_source1.tsv"])
    order = list(s1)
    print(f"Test S1: {len(s1):,} ({time.time()-t0:.0f}s)", flush=True)
    freq = name_freq(s1)

    pool_paths = [ROOT / "test/test_source2.tsv", ROOT / "test/test_source3.tsv"]
    index, idf = build_index_and_idf(pool_paths)
    pool = load_pool(pool_paths)
    print(f"Pool: {len(pool):,} ({time.time()-t0:.0f}s)", flush=True)

    clf, threshold = load_model()
    print(f"Threshold: {threshold}", flush=True)

    matches: dict[str, list[str]] = {}
    cands_out: dict[str, list[str]] = {}
    capped_total = 0

    # Batch to avoid OOM on full test set
    for batch_start in range(0, len(order), BATCH_ENTITIES):
        batch = order[batch_start: batch_start + BATCH_ENTITIES]
        feat_list, spans = [], []
        for sid in batch:
            _name, _addr, country, s1f = s1[sid]
            cands, was_capped = candidates_for(index, _name, _addr, country, idf)
            capped_total += was_capped
            cands_out[sid] = cands
            start = len(feat_list)
            for c in cands:
                cname, caddr, _ = pool[c]
                candf = precompute(cname, caddr)
                feat_list.append(features(s1f, country, candf, idf, freq))
            spans.append((sid, start, len(feat_list), cands))

        if feat_list:
            X_batch = np.array(feat_list, dtype=np.float32)
            scores = _clf_scores(clf, X_batch)
            for sid, start, end, cands in spans:
                matches[sid] = [cands[j - start] for j in range(start, end) if scores[j] >= threshold]
        else:
            for sid, _, _, _ in spans:
                matches[sid] = []

        if (batch_start + BATCH_ENTITIES) % 100_000 < BATCH_ENTITIES:
            done = min(batch_start + BATCH_ENTITIES, len(order))
            print(f"  {done:,}/{len(order):,}  elapsed={time.time()-t0:.0f}s", flush=True)

    out = HERE.parents[1] / "output"
    write_submission(out / "matching_results.tsv", "matched_entity_ids", matches, order)
    write_submission(out / "candidate_pairs.tsv", "candidate_entity_ids", cands_out, order)
    n_matched = sum(1 for v in matches.values() if v)
    print(f"Wrote {out}  matched={n_matched:,}  capped={capped_total/len(order):.3f}  "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "holdout"
    {"holdout": run_holdout, "test": run_test}[mode]()
