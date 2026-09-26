"""Scoring-based entity matcher with learned linear model.

Each candidate pair is scored on 9 features via logistic regression
(coefficients in model.json, trained by run.py holdout). Inference is a
dot product — no sklearn needed at predict time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from normalize import addr_tokens, fold, house_numbers, name_tokens

HERE = Path(__file__).resolve().parent

try:
    from rapidfuzz import fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

try:
    import jellyfish as _jf
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False

FEATURE_NAMES = [
    "idf_name_jac", "name_contain", "char_sim", "house_agree",
    "addr_jac", "addr_exact", "city_agree", "name_rare", "name_len_ratio",
    "jaro_winkler", "tfidf_cos",
    "rf_token_sort", "rf_token_set", "rf_partial",
    "rf_addr_sort",
    "phonetic_match", "name_rarity_log",
]

_FALLBACK_COEF = [2.0, 1.0, 2.5, 2.5, 1.0, 2.5, 0.5, 1.0, 0.5, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
_FALLBACK_BIAS = -4.5
_FALLBACK_THRESHOLD = 0.0


def _grams(s: str, n: int = 3) -> set[str]:
    s = f"  {s} "
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def char_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ga, gb = _grams(a), _grams(b)
    return len(ga & gb) / len(ga | gb)


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    window = max(l1, l2) // 2 - 1
    if window < 0:
        window = 0
    s1m = [False] * l1
    s2m = [False] * l2
    matches = transpositions = 0
    for i in range(l1):
        lo = max(0, i - window)
        hi = min(i + window + 1, l2)
        for j in range(lo, hi):
            if s2m[j] or s1[i] != s2[j]:
                continue
            s1m[i] = s2m[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(l1):
        if not s1m[i]:
            continue
        while not s2m[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    jaro = (matches / l1 + matches / l2 + (matches - transpositions / 2) / matches) / 3
    prefix = 0
    for i in range(min(4, l1, l2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


def idf_sim(toks1: frozenset, toks2: frozenset, idf: dict) -> float:
    if not toks1 or not toks2:
        return 0.0
    shared = toks1 & toks2
    if not shared:
        return 0.0
    union = toks1 | toks2
    num = sum(idf.get(t, 1.0) for t in shared)
    den = sum(idf.get(t, 1.0) for t in union)
    return num / den if den > 0 else 0.0


def tfidf_cosine(toks1: frozenset, toks2: frozenset, idf: dict) -> float:
    if not toks1 or not toks2:
        return 0.0
    shared = toks1 & toks2
    if not shared:
        return 0.0
    dot = sum(idf.get(t, 1.0) ** 2 for t in shared)
    n1 = sum(idf.get(t, 1.0) ** 2 for t in toks1) ** 0.5
    n2 = sum(idf.get(t, 1.0) ** 2 for t in toks2) ** 0.5
    return dot / (n1 * n2) if (n1 and n2) else 0.0


def name_freq(s1_records: dict) -> dict[tuple, int]:
    """(country, squashed_name) -> count. s1_records values: (name, addr, country, pc)."""
    freq: dict[tuple, int] = {}
    for _eid, (name, addr, country, pc) in s1_records.items():
        k = (country.strip(), pc[0])
        freq[k] = freq.get(k, 0) + 1
    return freq


def precompute(name: str, addr: str) -> tuple:
    """(squash, name_tokens_fset, house_numbers_fset, addr_key, addr_tokens_fset, city, folded_name)."""
    nt = name_tokens(name)
    at = addr_tokens(addr)
    return (
        "".join(nt),
        frozenset(nt),
        frozenset(house_numbers(addr)),
        "".join(at),
        frozenset(at),
        at[-1] if at else "",
        fold(name).strip(),
    )


def features(s1f: tuple, s1_country: str, candf: tuple,
             idf: dict, freq: dict) -> list[float]:
    n1, t1, h1, a1, at1, c1, fn1 = s1f
    n2, t2, h2, a2, at2, c2, fn2 = candf
    c = s1_country.strip()
    # raw address strings for rapidfuzz (stored in precompute slot 3 = addr_key)
    # fn1/fn2 are folded names; a1/a2 are squashed addr keys
    base = [
        idf_sim(t1, t2, idf),
        len(t1 & t2) / len(t1) if t1 else 0.0,
        char_sim(n1, n2),
        1.0 if (h1 & h2) else 0.0,
        len(at1 & at2) / len(at1 | at2) if (at1 and at2) else 0.0,
        1.0 if (len(a1) >= 8 and a1 == a2) else 0.0,
        1.0 if (c1 and c2 and c1 == c2) else 0.0,
        1.0 if freq.get((c, n1), 1) < 4 else 0.0,
        min(len(n1), len(n2)) / max(len(n1), len(n2)) if (n1 and n2) else 0.0,
        jaro_winkler(fn1, fn2),
        tfidf_cosine(t1, t2, idf),
    ]
    if _HAS_RAPIDFUZZ:
        base += [
            _fuzz.token_sort_ratio(fn1, fn2) / 100.0,
            _fuzz.token_set_ratio(fn1, fn2) / 100.0,
            _fuzz.partial_ratio(fn1, fn2) / 100.0,
            _fuzz.token_sort_ratio(a1, a2) / 100.0,
        ]
    else:
        base += [0.0, 0.0, 0.0, 0.0]

    # phonetic_match
    if _HAS_JELLYFISH and n1 and n2:
        base.append(1.0 if _jf.soundex(n1) == _jf.soundex(n2) else 0.0)
    else:
        base.append(0.0)

    # name_rarity_log — continuous version of name_rare
    import math
    f = freq.get((c, n1), 1)
    base.append(math.log(1.0 / max(f, 1) + 1.0))

    return base


def _load_model():
    path = HERE / "model.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


_MODEL = _load_model()


def score(feats: list[float]) -> float:
    m = _MODEL
    if m:
        w, b = m["coef"], m["intercept"]
    else:
        w, b = _FALLBACK_COEF, _FALLBACK_BIAS
    return sum(wi * fi for wi, fi in zip(w, feats)) + b


def get_threshold() -> float:
    return _MODEL["threshold"] if _MODEL else _FALLBACK_THRESHOLD


def decide_pre(s1f: tuple, s1_country: str, candf: tuple,
               idf: dict, freq: dict) -> bool:
    return score(features(s1f, s1_country, candf, idf, freq)) >= get_threshold()


def save_model(coef, intercept, thresh, path=None):
    if path is None:
        path = HERE / "model.json"
    data = {"coef": [round(float(c), 6) for c in coef],
            "intercept": round(float(intercept), 6),
            "threshold": round(float(thresh), 4)}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    global _MODEL
    _MODEL = data
