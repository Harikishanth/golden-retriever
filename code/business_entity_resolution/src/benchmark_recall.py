"""Candidate recall benchmark — run BEFORE committing to full pipeline.

Measures Recall@K for each blocking strategy on the 100K holdout.
Run this after pool encoding to validate the architecture decision.

Usage:
    python -X utf8 src/benchmark_recall.py

Output:
                    R@25   R@50  R@100  R@200  R@500  mean_cands  runtime
    BM25            0.???  0.???  0.???  0.???   n/a     ???       ???s
    Dense (Qwen3)   0.???  0.???  0.???  0.???  0.???   ???       ???s
    BM25 ∪ Dense    0.???  0.???  0.???  0.???  0.???   ???       ???s
    BM25 ∪ MinHash  0.???  0.???  0.???  0.???   n/a    ???       ???s
    All three       0.???  0.???  0.???  0.???  0.???   ???       ???s

Decision rule:
    R@200 >= 0.93 → proceed with full pipeline (CE + LightGBM worthwhile)
    R@200 < 0.93  → architecture change needed before full 10M run
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from blocking import build_index_and_idf, candidates_for
from io_utils import ROOT, iter_ground_truth, iter_source
from match import name_freq, precompute
from normalize import name_tokens, fold

import blocking as _blk

HOLDOUT_N = 100_000
SEED = 42
KS = [25, 50, 100, 200, 500]
CKPT = Path(__file__).resolve().parents[1] / "checkpoints"


def load_s1_and_gt():
    s1 = {}
    for eid, name, addr, country in iter_source(ROOT / "train/train_source1.tsv"):
        s1[eid] = (name, addr, country, precompute(name, addr))
    gt = dict(iter_ground_truth(ROOT / "train/train_ground_truth.tsv"))
    rng = random.Random(SEED)
    holdout = rng.sample(list(s1), HOLDOUT_N)
    return s1, gt, holdout


def recall_at_k(candidates_per_entity, gold_per_entity, ks):
    """Compute Recall@K. gold_per_entity[i] = set of true matches for entity i."""
    results = {}
    for k in ks:
        hits = 0
        total_gold = 0
        for cands, gold in zip(candidates_per_entity, gold_per_entity):
            if not gold:
                continue
            topk = set(cands[:k])
            hits += len(topk & gold)
            total_gold += len(gold)
        results[k] = hits / total_gold if total_gold > 0 else 0.0
    return results


def print_row(name, recall_dict, mean_cands, elapsed):
    ks = sorted(recall_dict.keys())
    scores = "  ".join(f"R@{k}={recall_dict[k]:.4f}" for k in ks)
    print(f"  {name:<25} {scores}  mean={mean_cands:.0f}  {elapsed:.0f}s", flush=True)


def benchmark_bm25(s1, gt, holdout, pool, idf, cap=200):
    _blk.CAP = cap
    t0 = time.time()
    cands_list, gold_list = [], []
    for sid in holdout:
        _name, _addr, country, _ = s1[sid]
        cands, _ = candidates_for(idf_dict_to_index(None), _name, _addr, country, idf)
        cands_list.append(cands)
        gold_list.append(gt.get(sid, set()) & set(pool.keys()))
    elapsed = time.time() - t0
    rec = recall_at_k(cands_list, gold_list, [k for k in KS if k <= cap])
    mean_c = sum(len(c) for c in cands_list) / len(cands_list)
    return rec, mean_c, elapsed, cands_list


def benchmark_bm25_proper(s1, gt, holdout, pool, bm25_index, idf, cap=200):
    _blk.CAP = cap
    t0 = time.time()
    cands_list, gold_list = [], []
    for sid in holdout:
        _name, _addr, country, _ = s1[sid]
        cands, _ = candidates_for(bm25_index, _name, _addr, country, idf)
        cands_list.append(cands)
        gold_list.append(gt.get(sid, set()) & set(pool.keys()))
    elapsed = time.time() - t0
    rec = recall_at_k(cands_list, gold_list, [k for k in KS if k <= cap])
    mean_c = sum(len(c) for c in cands_list) / len(cands_list)
    return rec, mean_c, elapsed, cands_list


def benchmark_dense(s1, gt, holdout, pool, faiss_index, pool_eids, top_k=500):
    """Dense retrieval recall using the pre-built FAISS index."""
    from dense import DenseRetrieval
    import gc

    t0 = time.time()
    retriever = DenseRetrieval()
    retriever._index = faiss_index
    retriever._pool_eids = pool_eids

    holdout_records = {sid: s1[sid] for sid in holdout}
    dense_cands, _ = retriever.search_chunked(holdout_records, top_k=top_k)

    cands_list = [dense_cands.get(sid, []) for sid in holdout]
    gold_list = [gt.get(sid, set()) & set(pool.keys()) for sid in holdout]
    elapsed = time.time() - t0

    rec = recall_at_k(cands_list, gold_list, KS)
    mean_c = sum(len(c) for c in cands_list) / len(cands_list)
    return rec, mean_c, elapsed, cands_list


def union_cands(cands_a, cands_b, max_k=500):
    """Union two candidate lists, preserving order, deduplicating."""
    seen = set()
    out = []
    for c in cands_a + cands_b:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= max_k:
            break
    return out


def _char_shingles(text: str, n: int = 3) -> set:
    """Character n-gram shingles on folded text.

    Catches typos/formatting variants that word-token MinHash misses:
    'McDonalds' ↔ 'Mc Donald's' share many 3-grams even with space/punct changes.
    Word-token MinHash would only match if full tokens overlap.
    """
    s = fold(text).replace(" ", "")  # fold + remove spaces before shingling
    if len(s) < n:
        return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def benchmark_minhash(s1, gt, holdout, pool, bm25_cands_per_entity, top_k=200):
    """MinHash LSH blocking using character 3-gram shingles.

    Character shingling (not word tokens) catches typos and formatting variants:
    - 'McDonalds' ↔ 'Mc Donald's' — same 3-grams after space removal
    - 'Patisserie' ↔ 'Patisserie' with accent normalization
    Word-level MinHash would NOT catch these.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  datasketch not installed — skipping MinHash", flush=True)
        return None, 0, 0, [[] for _ in holdout]

    t0 = time.time()
    lsh = MinHashLSH(threshold=0.4, num_perm=64)

    # Index pool with character 3-gram shingles
    print("  Building MinHash pool index (char 3-gram shingles)...", flush=True)
    pool_mh = {}
    for i, (peid, (pname, _, _)) in enumerate(pool.items()):
        shingles = _char_shingles(pname)
        if not shingles:
            continue
        m = MinHash(num_perm=64)
        for s in shingles:
            m.update(s.encode())
        pool_mh[peid] = m
        try:
            lsh.insert(peid, m)
        except Exception:
            pass
        if (i + 1) % 500_000 == 0:
            print(f"  MinHash indexed {i+1:,} pool records ({time.time()-t0:.0f}s)",
                  flush=True)

    print(f"  MinHash pool index built ({time.time()-t0:.0f}s)", flush=True)

    cands_list, gold_list = [], []
    for i, sid in enumerate(holdout):
        _name, _, _, _ = s1[sid]
        shingles = _char_shingles(_name)
        mh_cands = []
        if shingles:
            m = MinHash(num_perm=64)
            for s in shingles:
                m.update(s.encode())
            try:
                mh_cands = lsh.query(m)[:top_k]
            except Exception:
                pass

        # Union with BM25
        combined = union_cands(bm25_cands_per_entity[i], mh_cands, top_k)
        cands_list.append(combined)
        gold_list.append(gt.get(sid, set()) & set(pool.keys()))

    elapsed = time.time() - t0
    rec = recall_at_k(cands_list, gold_list, [k for k in KS if k <= top_k])
    mean_c = sum(len(c) for c in cands_list) / len(cands_list)
    return rec, mean_c, elapsed, cands_list


def run_benchmark():
    t0 = time.time()
    print(f"\n{'='*70}", flush=True)
    print("CANDIDATE RECALL BENCHMARK", flush=True)
    print(f"{'='*70}\n", flush=True)

    print("Loading data...", flush=True)
    s1, gt, holdout = load_s1_and_gt()
    pool_paths = [ROOT / "train/train_source2.tsv", ROOT / "train/train_source3.tsv"]
    bm25_index, idf = build_index_and_idf(pool_paths)

    from run import load_pool
    pool = load_pool(pool_paths)

    n_singletons = sum(1 for sid in holdout if not gt.get(sid, set()))
    print(f"Holdout: {HOLDOUT_N:,}  singletons: {n_singletons:,} "
          f"({100*n_singletons/HOLDOUT_N:.1f}%)\n", flush=True)

    # ── 1. BM25 ──────────────────────────────────────────────────────
    print("1. BM25 blocking...", flush=True)
    rec_bm25, mean_bm25, t_bm25, bm25_cands = benchmark_bm25_proper(
        s1, gt, holdout, pool, bm25_index, idf, cap=500)
    print_row("BM25 (CAP=500)", rec_bm25, mean_bm25, t_bm25)

    # ── 2. Dense (Qwen3-Embedding-0.6B) ──────────────────────────────
    # Load from checkpoint if available
    dense_cands_per_ent = None
    faiss_path = CKPT / "pool_train"
    if faiss_path.with_suffix(".faiss").exists():
        print("\n2. Dense retrieval (loading cached index)...", flush=True)
        from dense import DenseRetrieval
        retriever = DenseRetrieval()
        retriever.load_pool(faiss_path)

        holdout_records = {sid: s1[sid] for sid in holdout}
        t_dense_start = time.time()
        dense_cands_map, _ = retriever.search_chunked(
            holdout_records, top_k=500)
        t_dense = time.time() - t_dense_start

        dense_cands_per_ent = [dense_cands_map.get(sid, []) for sid in holdout]
        gold_list = [gt.get(sid, set()) & set(pool.keys()) for sid in holdout]
        rec_dense = recall_at_k(dense_cands_per_ent, gold_list, KS)
        mean_dense = sum(len(c) for c in dense_cands_per_ent) / len(dense_cands_per_ent)
        print_row("Dense Qwen3-0.6B (K=500)", rec_dense, mean_dense, t_dense)

        # ── 3. BM25 ∪ Dense ──────────────────────────────────────────
        print("\n3. BM25 ∪ Dense union...", flush=True)
        t_union = time.time()
        union_cands_list = [
            union_cands(bm25_cands[i], dense_cands_per_ent[i], 500)
            for i in range(HOLDOUT_N)
        ]
        t_union = time.time() - t_union
        rec_union = recall_at_k(union_cands_list, gold_list, KS)
        mean_union = sum(len(c) for c in union_cands_list) / len(union_cands_list)
        print_row("BM25 ∪ Dense", rec_union, mean_union, t_union)
    else:
        print("\n2. No FAISS index found at checkpoints/pool_train.faiss", flush=True)
        print("   Run pipeline.py holdout first to build the pool index.", flush=True)
        dense_cands_per_ent = None
        union_cands_list = bm25_cands

    # ── 4. BM25 ∪ MinHash ────────────────────────────────────────────
    print("\n4. MinHash blocking (adds typo/variant recovery)...", flush=True)
    rec_mh, mean_mh, t_mh, mh_cands = benchmark_minhash(
        s1, gt, holdout, pool, bm25_cands, top_k=200)
    if rec_mh:
        print_row("BM25 ∪ MinHash (K=200)", rec_mh, mean_mh, t_mh)

    # ── 5. All three ─────────────────────────────────────────────────
    if dense_cands_per_ent is not None and rec_mh is not None:
        print("\n5. BM25 ∪ Dense ∪ MinHash...", flush=True)
        gold_list = [gt.get(sid, set()) & set(pool.keys()) for sid in holdout]
        triple_cands = [
            union_cands(union_cands_list[i], [c for c in mh_cands[i]
                        if c not in set(union_cands_list[i])], 500)
            for i in range(HOLDOUT_N)
        ]
        rec_triple = recall_at_k(triple_cands, gold_list, KS)
        mean_triple = sum(len(c) for c in triple_cands) / len(triple_cands)
        print_row("BM25 ∪ Dense ∪ MinHash", rec_triple, mean_triple, 0)

    # ── Decision ─────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("DECISION RULE:", flush=True)

    best_r200 = rec_bm25.get(200, 0)
    if dense_cands_per_ent is not None:
        best_r200 = rec_union.get(200, best_r200)
    if best_r200 >= 0.93:
        print(f"  R@200 = {best_r200:.4f} ≥ 0.93 → PROCEED with full pipeline", flush=True)
        print("  CE + LightGBM reranking will be worthwhile.", flush=True)
    elif best_r200 >= 0.88:
        print(f"  R@200 = {best_r200:.4f} (0.88–0.93) → MARGINAL", flush=True)
        print("  Proceed but expect F0.5 ceiling ~0.88–0.92.", flush=True)
    else:
        print(f"  R@200 = {best_r200:.4f} < 0.88 → ARCHITECTURE CHANGE NEEDED", flush=True)
        print("  Dense model not lifting recall enough. Try K=500 or different model.", flush=True)

    print(f"\nTotal benchmark time: {time.time()-t0:.0f}s", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    run_benchmark()
