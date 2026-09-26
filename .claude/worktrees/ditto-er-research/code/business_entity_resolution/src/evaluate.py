"""Metrics.

macro F0.5 is computed per Source-1 entity and then averaged, exactly as the
scorer does: empty prediction on a singleton scores 1.0, any prediction on a
singleton scores 0.0.
"""
from __future__ import annotations


def f05_one(pred: set[str], gold: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gold)
    return (1.25 * p * r) / (0.25 * p + r)


def summarize(rows: list[tuple[set[str], set[str], set[str]]]) -> dict:
    """rows: (predicted, gold, candidates) per S1."""
    n = len(rows)
    score = sum(f05_one(p, g) for p, g, _ in rows) / n
    gold_total = sum(len(g) for _, g, _ in rows)
    covered = sum(len(g & c) for _, g, c in rows)
    cand_total = sum(len(c) for *_, c in rows)
    return {
        "n": n,
        "macro_f05": score,
        "recall_ceiling": covered / gold_total if gold_total else 1.0,
        "mean_candidates": cand_total / n if n else 0.0,
        "gold_links": gold_total,
    }


def fmt(stats: dict) -> str:
    return (f"n={stats['n']:,}  macro_F0.5={stats['macro_f05']:.4f}  "
            f"recall_ceiling={stats['recall_ceiling']:.3f}  "
            f"mean_candidates={stats['mean_candidates']:.1f}")
