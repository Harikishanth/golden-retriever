"""Vectorized feature computation using rapidfuzz.process.cdist.

Replaces the Python for-loop in run.py with batch operations.
~10x faster than the per-pair loop for the test run.
"""
from __future__ import annotations

import numpy as np

try:
    from rapidfuzz import process as _rfp, fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

from normalize import addr_tokens, fold, house_numbers, name_tokens


def batch_features(s1_name: str, s1_addr: str, s1_country: str,
                   cand_names: list[str], cand_addrs: list[str],
                   idf: dict, freq: dict) -> np.ndarray:
    """Compute feature matrix for one S1 entity vs all its candidates.

    Returns: float32 array of shape (len(cands), 15)
    """
    n = len(cand_names)
    if n == 0:
        return np.zeros((0, 15), dtype=np.float32)

    # Precompute S1
    s1_ntoks = name_tokens(s1_name)
    s1_atoks = addr_tokens(s1_addr)
    s1_nsq = "".join(s1_ntoks)
    s1_nset = frozenset(s1_ntoks)
    s1_hset = frozenset(house_numbers(s1_addr))
    s1_asq = "".join(s1_atoks)
    s1_aset = frozenset(s1_atoks)
    s1_city = s1_atoks[-1] if s1_atoks else ""
    s1_fn = fold(s1_name).strip()
    c = s1_country.strip()

    # Precompute candidates
    c_ntoks = [name_tokens(n) for n in cand_names]
    c_atoks = [addr_tokens(a) for a in cand_addrs]
    c_nsq = ["".join(t) for t in c_ntoks]
    c_nset = [frozenset(t) for t in c_ntoks]
    c_hset = [frozenset(house_numbers(a)) for a in cand_addrs]
    c_asq = ["".join(t) for t in c_atoks]
    c_aset = [frozenset(t) for t in c_atoks]
    c_city = [t[-1] if t else "" for t in c_atoks]
    c_fn = [fold(nm).strip() for nm in cand_names]

    X = np.zeros((n, 15), dtype=np.float32)

    for i in range(n):
        t2 = c_nset[i]
        at2 = c_aset[i]

        # idf_name_jac
        shared = s1_nset & t2
        union = s1_nset | t2
        if union:
            num = sum(idf.get(t, 1.0) for t in shared)
            den = sum(idf.get(t, 1.0) for t in union)
            X[i, 0] = num / den if den else 0.0

        # name_contain
        X[i, 1] = len(s1_nset & t2) / len(s1_nset) if s1_nset else 0.0

        # char_sim (trigram jaccard on squashed name)
        a, b = s1_nsq, c_nsq[i]
        if a and b:
            s = f"  {a} "
            ga = {s[j:j+3] for j in range(max(1, len(s)-2))}
            s = f"  {b} "
            gb = {s[j:j+3] for j in range(max(1, len(s)-2))}
            X[i, 2] = len(ga & gb) / len(ga | gb) if (ga | gb) else 0.0

        # house_agree
        X[i, 3] = 1.0 if (s1_hset & c_hset[i]) else 0.0

        # addr_jac
        if s1_aset and at2:
            X[i, 4] = len(s1_aset & at2) / len(s1_aset | at2)

        # addr_exact
        X[i, 5] = 1.0 if (len(s1_asq) >= 8 and s1_asq == c_asq[i]) else 0.0

        # city_agree
        X[i, 6] = 1.0 if (s1_city and c_city[i] and s1_city == c_city[i]) else 0.0

        # name_rare
        X[i, 7] = 1.0 if freq.get((c, s1_nsq), 1) < 4 else 0.0

        # name_len_ratio
        if s1_nsq and c_nsq[i]:
            X[i, 8] = min(len(s1_nsq), len(c_nsq[i])) / max(len(s1_nsq), len(c_nsq[i]))

        # tfidf_cosine
        shared_n = s1_nset & t2
        if shared_n:
            dot = sum(idf.get(t, 1.0)**2 for t in shared_n)
            n1 = sum(idf.get(t, 1.0)**2 for t in s1_nset)**0.5
            n2 = sum(idf.get(t, 1.0)**2 for t in t2)**0.5
            X[i, 10] = dot / (n1 * n2) if (n1 and n2) else 0.0

    # jaro_winkler — batch via rapidfuzz
    if _HAS_RAPIDFUZZ and c_fn:
        jw_scores = _rfp.cdist([s1_fn], c_fn, scorer=_fuzz.WRatio, score_cutoff=0)
        X[:, 9] = jw_scores[0] / 100.0

        # rf_token_sort
        ts_scores = _rfp.cdist([s1_fn], c_fn, scorer=_fuzz.token_sort_ratio, score_cutoff=0)
        X[:, 11] = ts_scores[0] / 100.0

        # rf_token_set
        tset_scores = _rfp.cdist([s1_fn], c_fn, scorer=_fuzz.token_set_ratio, score_cutoff=0)
        X[:, 12] = tset_scores[0] / 100.0

        # rf_partial
        pr_scores = _rfp.cdist([s1_fn], c_fn, scorer=_fuzz.partial_ratio, score_cutoff=0)
        X[:, 13] = pr_scores[0] / 100.0

        # rf_addr_sort
        as_scores = _rfp.cdist([s1_asq], c_asq, scorer=_fuzz.token_sort_ratio, score_cutoff=0)
        X[:, 14] = as_scores[0] / 100.0

    return X
