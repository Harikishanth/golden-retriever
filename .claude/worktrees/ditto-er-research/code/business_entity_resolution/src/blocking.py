"""BM25-style token retrieval blocking.

Replaces exact hash-key matching. Every individual name/address token becomes
a posting-list entry; candidates are scored by IDF-weighted token overlap.

Hash keys gave recall ceiling ~0.834 because they required EXACT match on
a compound normalized field. Token retrieval finds partial matches:
  "McDonald's Restaurant Corp" ↔ "McDonalds Inc"  (shared: 'mcdonald')
  "MG Road Bengaluru" ↔ "M.G. Road Bangalore"    (shared: 'mg', 'road')

Scoring (higher = more evidence of match):
  shared name token:    IDF(token) * NAME_W   (name more discriminative)
  shared address token: IDF(token) * ADDR_W
  shared house number:  HOUSE_SCORE           (fixed, very discriminative)

MAX_POSTING guard: skip any posting list > 2000 entries. A token appearing
in >2000 records contributes near-zero IDF anyway and would OOM the scored dict.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from io_utils import iter_source
from normalize import addr_tokens, house_numbers, name_tokens

CAP = 150
MAX_POSTING = 2000

NAME_W = 2.0
ADDR_W = 1.2
HOUSE_SCORE = 8.0


def build_index_and_idf(paths: list[Path]) -> tuple[dict, dict]:
    """Single-pass build of BM25 blocking index and feature IDF.

    Blocking index keys: (country, 'N'|'A'|'H', token)
      N = name token, A = address token, H = house number

    Feature IDF (not country-scoped, for use in match.py features):
      {token: log(N / (1 + df))}

    Returns (index, name_idf).
    """
    index: dict[tuple, list[str]] = defaultdict(list)
    name_df: dict[str, int] = {}
    n = 0

    for path in paths:
        for eid, name, addr, country in iter_source(path):
            c = country.strip()
            seen: set[tuple] = set()

            for t in name_tokens(name):
                k = (c, "N", t)
                if k not in seen:
                    index[k].append(eid)
                    seen.add(k)
                name_df[t] = name_df.get(t, 0) + 1

            for t in addr_tokens(addr):
                k = (c, "A", t)
                if k not in seen:
                    index[k].append(eid)
                    seen.add(k)

            for h in house_numbers(addr):
                k = (c, "H", h)
                index[k].append(eid)

            n += 1

    feature_idf = {t: math.log(n / (1 + df)) for t, df in name_df.items()} if n else {}
    print(
        f"BM25 index: {n:,} records, {len(index):,} posting lists, "
        f"{len(feature_idf):,} IDF tokens",
        flush=True,
    )
    return index, feature_idf


def candidates_for(index: dict, name: str, addr: str, country: str,
                   idf: dict | None = None) -> tuple[list[str], bool]:
    """Retrieve top-CAP candidates by BM25-style IDF-weighted token overlap.

    idf: feature IDF dict (from build_index_and_idf). If None, falls back to
         uniform weight 1.0 per token (still works, just less discriminative).
    """
    if idf is None:
        idf = {}
    c = country.strip()
    scored: dict[str, float] = {}

    for t in set(name_tokens(name)):
        k = (c, "N", t)
        bucket = index.get(k, ())
        if len(bucket) > MAX_POSTING:
            continue
        w = idf.get(t, 1.0) * NAME_W
        for eid in bucket:
            scored[eid] = scored.get(eid, 0.0) + w

    for t in set(addr_tokens(addr)):
        k = (c, "A", t)
        bucket = index.get(k, ())
        if len(bucket) > MAX_POSTING:
            continue
        w = idf.get(t, 1.0) * ADDR_W
        for eid in bucket:
            scored[eid] = scored.get(eid, 0.0) + w

    for h in house_numbers(addr):
        k = (c, "H", h)
        for eid in index.get(k, ()):
            scored[eid] = scored.get(eid, 0.0) + HOUSE_SCORE

    if not scored:
        return [], False

    capped = len(scored) > CAP
    if not capped:
        return list(scored), False
    ranked = sorted(scored.items(), key=lambda kv: -kv[1])
    return [e for e, _ in ranked[:CAP]], True
