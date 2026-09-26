"""Holdout-free train diagnostic: exact (name, address) match vs GT. Stdlib only."""
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\ML challenge\dataset\student_resource\dataset\train")


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def f05(pred: set, gold: set) -> float:
    if not gold and not pred:
        return 1.0
    if not gold and pred:
        return 0.0
    if gold and not pred:
        return 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    den = 0.25 * p + r
    if den == 0:
        return 0.0
    return (1.25 * p * r) / den


def load_index(path: Path) -> dict:
    idx = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            while len(parts) < 4:
                parts.append("")
            eid, name, addr = parts[0], parts[1], parts[2]
            if not addr.strip():
                continue
            idx[(norm(name), norm(addr))].append(eid)
    return idx


def load_gt(path: Path) -> dict:
    gt = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            mids = parts[1] if len(parts) > 1 else ""
            gt[sid] = set(x for x in mids.split(",") if x.strip()) if mids.strip() else set()
    return gt


def main():
    print("indexing S2...", flush=True)
    idx = load_index(ROOT / "train_source2.tsv")
    print("indexing S3...", flush=True)
    idx3 = load_index(ROOT / "train_source3.tsv")
    for k, v in idx3.items():
        idx[k].extend(v)
    print(f"index keys={len(idx)}", flush=True)
    gt = load_gt(ROOT / "train_ground_truth.tsv")
    print("scoring S1...", flush=True)
    scores = []
    n_pred = n_empty_pred = 0
    with open(ROOT / "train_source1.tsv", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            while len(parts) < 4:
                parts.append("")
            sid, name, addr = parts[0], parts[1], parts[2]
            pred = set(idx.get((norm(name), norm(addr)), []))
            if pred:
                n_pred += 1
            else:
                n_empty_pred += 1
            scores.append(f05(pred, gt.get(sid, set())))
    print(f"n={len(scores)} nonempty_pred={n_pred} empty_pred={n_empty_pred}")
    print(f"macro_F0.5={sum(scores)/len(scores):.6f}")


if __name__ == "__main__":
    main()
