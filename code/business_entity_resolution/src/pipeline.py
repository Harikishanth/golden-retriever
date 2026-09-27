#!/usr/bin/env python3
"""Compound GPU pipeline for Amazon ML Challenge entity resolution.

Synthesizes all research artifacts into one executable pipeline:
  1. Qwen3-Embedding-0.6B dense retrieval (512-dim true Matryoshka, FAISS GPU fp16)
  2. BM25 token retrieval (existing blocking.py)
  3. Hybrid blocking union → ~250-350 candidates per entity
  4. 20-feature LightGBM (17 existing + cosine + 2× IDF containment)
  5. Retrain LightGBM on full 100K → isotonic calibration
  6. mDeBERTa-v3-base cross-encoder (ONNX + fp16, cascade on uncertain band)
  7. Two-threshold grid sweep on consistent iso-transformed scores
  8. Graph transitive closure (opt-in, disabled by default — risky for F0.5)

Usage:
    python -X utf8 src/pipeline.py holdout   # train models, sweep thresholds
    python -X utf8 src/pipeline.py test      # inference → submission
    python -X utf8 src/pipeline.py all       # holdout then test

Acceleration: FP16 + ONNX Runtime + batch=512 ≈ 3-4x over naive PyTorch.
"""
from __future__ import annotations

import gc
import json
import math
import os
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from blocking import build_index_and_idf, candidates_for
from evaluate import f05_one, fmt, summarize
from fast_features import batch_features, NUM_FEATURES as BASE_FEATURES
from io_utils import ROOT, iter_ground_truth, iter_source, write_submission
from match import name_freq, precompute
from normalize import fold, name_tokens

import blocking as _blk

HERE = Path(__file__).resolve().parents[1]
CKPT = HERE / "checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)

NUM_FEATURES = BASE_FEATURES + 3  # +cosine, +idf_contain_fwd, +idf_contain_rev


# ═══════════════════════════════════════════════════════════════════════
# Configuration — every hyperparameter in one place, no hardcoding
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Cfg:
    seed: int = 42
    holdout_n: int = 100_000
    train_frac: float = 0.7

    # BM25
    bm25_cap: int = 200

    # Dense retrieval
    dense_model: str = "Qwen/Qwen3-Embedding-0.6B"
    dense_dim: int = 512          # True Matryoshka — 512-dim works correctly
    dense_batch: int = 512
    dense_top_k: int = 200
    dense_fp16_index: bool = True

    # LightGBM
    lgbm_trees: int = 1000
    lgbm_leaves: int = 63
    lgbm_lr: float = 0.05

    # Cross-encoder — Qwen3-Reranker-0.6B (Apache-2.0, drop-in CrossEncoder,
    # validated 2026-09-26: works with sentence_transformers.CrossEncoder directly,
    # needs activation_fn=Sigmoid() for 0-1 probs. Purpose-built reranker,
    # better than fine-tuning mDeBERTa from scratch.)
    ce_model: str = "Qwen/Qwen3-Reranker-0.6B"
    ce_epochs: int = 3
    ce_batch: int = 32
    ce_lr: float = 2e-5
    ce_maxlen: int = 128
    ce_grad_accum: int = 2
    ce_top_k: int = 20
    ce_uncertain_lo: float = 0.2
    ce_uncertain_hi: float = 0.8
    ce_max_train_ent: int = 50_000
    ce_neg_ratio: int = 5
    ce_pos_weight: float = 5.0

    # Data augmentation
    augment_prob: float = 0.3

    # Thresholds
    match_t_range: tuple = (0.20, 0.81, 0.05)
    singleton_t_range: tuple = (0.40, 0.96, 0.05)
    france_t_bump: float = 0.10
    margin_gap: float = 0.05

    # Graph post-processing — OFF by default (risky for F0.5 precision)
    graph_enabled: bool = False
    graph_edge_threshold: float = 0.9
    graph_max_component: int = 15

    # Inference
    test_batch: int = 50_000
    dense_search_chunk: int = 50_000  # S1 entities per dense-search chunk

    def to_json(self) -> str:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = list(v) if isinstance(v, tuple) else v
        return json.dumps(d, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _timer():
    return time.time()


def load_s1(paths):
    rec = {}
    for p in paths:
        for eid, name, addr, country in iter_source(p):
            rec[eid] = (name, addr, country, precompute(name, addr))
    return rec


def load_pool(paths):
    rec = {}
    for p in paths:
        for eid, name, addr, country in iter_source(p):
            rec[eid] = (name, addr, country)
    return rec


def idf_containment(toks_a: frozenset, toks_b: frozenset, idf: dict) -> float:
    if not toks_a:
        return 0.0
    shared = toks_a & toks_b
    num = sum(idf.get(t, 1.0) for t in shared)
    den = sum(idf.get(t, 1.0) for t in toks_a)
    return num / den if den > 0 else 0.0


def augment_name(name: str, rng: random.Random) -> str:
    toks = name.split()
    if not toks:
        return name
    r = rng.random()
    if r < 0.25 and len(toks) > 1:
        toks.pop(rng.randint(0, len(toks) - 1))
    elif r < 0.50:
        i = rng.randint(0, len(toks) - 1)
        if len(toks[i]) > 2:
            j = rng.randint(0, len(toks[i]) - 2)
            chars = list(toks[i])
            chars[j], chars[j+1] = chars[j+1], chars[j]
            toks[i] = "".join(chars)
    elif r < 0.75:
        suffixes = ["Inc", "LLC", "Ltd", "Corp", "Co", "SARL", "SAS", "Pvt"]
        if toks[-1].lower().rstrip(".") in {s.lower() for s in suffixes}:
            toks[-1] = rng.choice(suffixes)
    return " ".join(toks)


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def component_size(self, x: str) -> int:
        root = self.find(x)
        return sum(1 for k in self.parent if self.find(k) == root)


def serialize_entity(name: str, addr: str, country: str) -> str:
    parts = []
    if name and name.strip():
        parts.append(f"COL name VAL {name.strip()}")
    if addr and addr.strip():
        parts.append(f"COL address VAL {addr.strip()}")
    if country and country.strip():
        parts.append(f"COL country VAL {country.strip()}")
    return " ".join(parts)


def _clf_scores(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1].astype(np.float32)
    if hasattr(clf, "predict") and callable(clf.predict):
        return np.asarray(clf.predict(X), dtype=np.float32)
    w = np.array(clf["coef"], dtype=np.float32)
    b = np.float32(clf["intercept"])
    raw = X @ w + b
    return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Extended features: base 17 + cosine + 2× IDF containment = 20
# ═══════════════════════════════════════════════════════════════════════

def extended_features(s1_name, s1_addr, s1_country,
                      cand_names, cand_addrs,
                      idf, freq, cos_dict, cand_eids):
    n = len(cand_names)
    if n == 0:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32)

    X_base = batch_features(s1_name, s1_addr, s1_country,
                            cand_names, cand_addrs, idf, freq)
    X_ext = np.zeros((n, 3), dtype=np.float32)

    for i, ceid in enumerate(cand_eids):
        X_ext[i, 0] = cos_dict.get(ceid, 0.0)

    s1_ntoks = frozenset(name_tokens(s1_name))
    for i in range(n):
        c_ntoks = frozenset(name_tokens(cand_names[i]))
        X_ext[i, 1] = idf_containment(s1_ntoks, c_ntoks, idf)
        X_ext[i, 2] = idf_containment(c_ntoks, s1_ntoks, idf)

    return np.column_stack([X_base, X_ext])


# ═══════════════════════════════════════════════════════════════════════
# Dense retrieval
# ═══════════════════════════════════════════════════════════════════════

def init_dense(cfg: Cfg):
    from dense import DenseRetrieval
    return DenseRetrieval(
        model_name=cfg.dense_model,
        device=_device(),
        batch_size=cfg.dense_batch,
        truncate_dim=cfg.dense_dim,
    )


# ═══════════════════════════════════════════════════════════════════════
# Cross-encoder
# ═══════════════════════════════════════════════════════════════════════

def build_ce_pairs(s1, pool, gt, blocking_cands, cfg: Cfg, rng: random.Random):
    pairs, labels = [], []
    entities = list(blocking_cands.keys())[:cfg.ce_max_train_ent]
    for sid in entities:
        s1_name, s1_addr, s1_country = s1[sid][0], s1[sid][1], s1[sid][2]
        gold = gt.get(sid, set())
        cands = blocking_cands.get(sid, [])
        pos_ids = [c for c in cands if c in gold and c in pool]
        neg_ids = [c for c in cands if c not in gold and c in pool]
        for pid in pos_ids:
            pn, pa, pc = pool[pid][0], pool[pid][1], pool[pid][2]
            pairs.append((serialize_entity(s1_name, s1_addr, s1_country),
                          serialize_entity(pn, pa, pc)))
            labels.append(1.0)
            if rng.random() < cfg.augment_prob:
                aug = augment_name(s1_name, rng)
                pairs.append((serialize_entity(aug, s1_addr, s1_country),
                               serialize_entity(pn, pa, pc)))
                labels.append(1.0)
        max_neg = cfg.ce_neg_ratio * max(1, len(pos_ids))
        for nid in neg_ids[:max_neg]:
            nn, na, nc = pool[nid][0], pool[nid][1], pool[nid][2]
            pairs.append((serialize_entity(s1_name, s1_addr, s1_country),
                          serialize_entity(nn, na, nc)))
            labels.append(0.0)
    print(f"CE training pairs: {len(pairs):,} "
          f"(pos={sum(1 for l in labels if l > 0.5):,})", flush=True)
    return pairs, labels


def train_cross_encoder(pairs, labels, val_pairs, val_labels, cfg: Cfg):
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder
    try:
        from sentence_transformers.cross_encoder import (
            CrossEncoderTrainer, CrossEncoderTrainingArguments)
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
        from datasets import Dataset

        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder(cfg.ce_model, max_length=cfg.ce_maxlen, device=device_str)

        train_ds = Dataset.from_dict({
            "sentence1": [p[0] for p in pairs],
            "sentence2": [p[1] for p in pairs],
            "label": labels,
        })
        eval_ds = Dataset.from_dict({
            "sentence1": [p[0] for p in val_pairs],
            "sentence2": [p[1] for p in val_pairs],
            "label": val_labels,
        }) if val_pairs else None

        _dev = next(model.model.parameters()).device
        loss = BinaryCrossEntropyLoss(model=model,
                                      pos_weight=torch.tensor(cfg.ce_pos_weight, device=_dev))
        outdir = str(CKPT / "ce-mdeberta")
        args = CrossEncoderTrainingArguments(
            output_dir=outdir,
            num_train_epochs=cfg.ce_epochs,
            per_device_train_batch_size=cfg.ce_batch,
            learning_rate=cfg.ce_lr,
            warmup_steps=0.1,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            logging_steps=200,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_ds else "no",
            load_best_model_at_end=bool(eval_ds),
            gradient_accumulation_steps=cfg.ce_grad_accum,
        )
        trainer = CrossEncoderTrainer(model=model, args=args,
                                      train_dataset=train_ds,
                                      eval_dataset=eval_ds, loss=loss)
        # Prevent DataParallel wrapping attribute error in sentence-transformers
        if hasattr(trainer.model, "device") is False and hasattr(trainer.model, "module"):
            trainer.model.device = next(trainer.model.module.parameters()).device
        trainer.train()
        model.save_pretrained(outdir)
        print(f"Cross-encoder saved → {outdir}", flush=True)
        return model
    except (ImportError, Exception) as e:
        print(f"CrossEncoderTrainer failed ({e}), falling back to manual", flush=True)
        return _train_ce_manual(pairs, labels, cfg)


def _train_ce_manual(pairs, labels, cfg: Cfg):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW

    class OnTheFlyDataset(Dataset):
        def __init__(self, pairs, labels, tokenizer, max_len):
            self.pairs = pairs
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            p1, p2 = self.pairs[idx]
            enc = self.tokenizer(p1, p2, truncation=True, max_length=self.max_len,
                                 padding="max_length", return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)
            }

    tokenizer = AutoTokenizer.from_pretrained(cfg.ce_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.ce_model, num_labels=2)
    ds = OnTheFlyDataset(pairs, labels, tokenizer, cfg.ce_maxlen)
    loader = DataLoader(ds, batch_size=cfg.ce_batch * cfg.ce_grad_accum,
                        shuffle=True, num_workers=4)
    device = torch.device(_device())
    model.to(device)
    if device.type == "cuda":
        model = model.half()
    optimizer = AdamW(model.parameters(), lr=cfg.ce_lr, weight_decay=0.01)
    model.train()
    for epoch in range(cfg.ce_epochs):
        total_loss, n_batch = 0, 0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=lbl)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()
            n_batch += 1
            if n_batch % 500 == 0:
                print(f"  CE epoch {epoch+1}/{cfg.ce_epochs} batch {n_batch}/{len(loader)} "
                      f"loss={total_loss/n_batch:.4f}", flush=True)
        print(f"  CE epoch {epoch+1}/{cfg.ce_epochs}  "
              f"loss={total_loss/max(n_batch,1):.4f}", flush=True)
    outdir = str(CKPT / "ce-mdeberta")
    os.makedirs(outdir, exist_ok=True)
    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    return model


def predict_ce(model, pairs, batch_size=512):
    if not pairs:
        return np.array([], dtype=np.float32)
    scores = model.predict(pairs, batch_size=batch_size,
                           show_progress_bar=len(pairs) > 50000,
                           convert_to_numpy=True)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 2:
        from scipy.special import softmax
        scores = softmax(scores, axis=1)[:, 1]
    elif scores.ndim == 1 and (scores.max() > 1.0 or scores.min() < 0.0):
        scores = 1.0 / (1.0 + np.exp(-scores))
    return scores


def try_onnx_export(model_path: str, ce_model, n_validate: int = 50):
    """Export to ONNX and validate scores match PyTorch within tolerance.

    Returns ONNX path on success, None on any failure or score mismatch.
    """
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        import torch

        onnx_path = model_path + "-onnx"
        ort = ORTModelForSequenceClassification.from_pretrained(
            model_path, export=True, provider="CUDAExecutionProvider")
        ort.save_pretrained(onnx_path)

        # Validation: score n_validate dummy pairs, compare to PyTorch
        probe_pairs = [
            ("COL name VAL Test Corp COL address VAL 123 Main St",
             "COL name VAL Test Corp COL address VAL 123 Main Street"),
        ] * n_validate

        # PyTorch scores
        pt_scores = predict_ce(ce_model, probe_pairs[:10], batch_size=10)

        # ONNX scores — use sentence-transformers CrossEncoder wrapper
        from sentence_transformers.cross_encoder import CrossEncoder
        onnx_ce = CrossEncoder(onnx_path, max_length=128)
        onnx_scores = predict_ce(onnx_ce, probe_pairs[:10], batch_size=10)

        max_diff = float(np.abs(pt_scores - onnx_scores).max())
        if max_diff > 0.02:
            print(f"ONNX validation FAILED: max_diff={max_diff:.4f} > 0.02, "
                  f"falling back to PyTorch", flush=True)
            return None

        print(f"ONNX exported + validated (max_diff={max_diff:.4f}) → {onnx_path}",
              flush=True)
        return onnx_path

    except Exception as e:
        print(f"ONNX export skipped ({e}), using PyTorch", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════════
# Calibration and threshold optimization
# ═══════════════════════════════════════════════════════════════════════

def calibrate_isotonic(scores: np.ndarray, labels: np.ndarray):
    """Fit isotonic regression calibration.

    NOTE: must be fitted on scores from the SAME model that will run at test time
    (i.e. after retraining on full 100K, not on the 70K training-split model).
    """
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, labels)
    print(f"Isotonic calibration fitted on {len(scores):,} pairs", flush=True)
    return ir


def build_final_scores(lgbm_scores: np.ndarray, iso,
                       entity_data: list, boundaries: list,
                       ce_scores_map: dict, n_train: int) -> np.ndarray:
    """Compute consistent final scores for threshold sweep and test inference.

    All pairs get iso(lgbm_score). CE-scored pairs (uncertain band) override
    with the CE score directly — CE is fine-tuned for this task and already
    produces well-calibrated probabilities.

    Consistency guarantee: this EXACT same logic is used at test inference time,
    so the threshold found during sweep is valid on the test set.
    """
    final = iso.transform(lgbm_scores).astype(np.float32)

    ce_overrides = 0
    for ei in range(n_train, len(entity_data)):
        sid, scored, gold, cand_set, _, country = entity_data[ei]
        start, end = boundaries[ei]
        for j in range(end - start):
            ceid = scored[j][0]
            if (sid, ceid) in ce_scores_map:
                final[start + j] = ce_scores_map[(sid, ceid)]
                ce_overrides += 1

    print(f"Final scores: {len(final):,} pairs, "
          f"CE overrides: {ce_overrides:,}", flush=True)
    return final


def sweep_two_threshold(scores_all: np.ndarray, entity_data: list,
                        boundaries: list, cfg: Cfg,
                        start_ei: int = 0, end_ei: int | None = None):
    """Grid search (match_t, singleton_t) to maximize macro F0.5.

    Applies France +0.10 bump and margin-based singleton guard.
    start_ei / end_ei allow running on a subset (e.g. val only).
    """
    n_total = len(entity_data)
    if end_ei is None:
        end_ei = n_total

    mt_lo, mt_hi, mt_step = cfg.match_t_range
    st_lo, st_hi, st_step = cfg.singleton_t_range

    best_f05, best_mt, best_st = 0.0, 0.5, 0.7

    for mt in np.arange(mt_lo, mt_hi, mt_step):
        for st in np.arange(max(mt, st_lo), st_hi, st_step):
            f05_sum = 0.0
            for ei in range(start_ei, end_ei):
                sid, scored, gold, cand_set, _, country = entity_data[ei]
                start, end = boundaries[ei]
                if end == start:
                    f05_sum += (1.0 if not gold else 0.0)
                    continue

                pair_scores = scores_all[start:end]
                max_score = pair_scores.max()
                eff_st = st + (cfg.france_t_bump
                               if country.strip().lower() == "france" else 0.0)
                if len(pair_scores) >= 2:
                    sorted_s = np.sort(pair_scores)[::-1]
                    if sorted_s[0] - sorted_s[1] < cfg.margin_gap:
                        eff_st = max(eff_st, st + 0.10)

                if max_score < eff_st:
                    pred = set()
                else:
                    pred = {scored[j][0] for j in range(end - start)
                            if scores_all[start + j] >= mt}

                f05_sum += f05_one(pred, gold)

            macro = f05_sum / (end_ei - start_ei)
            if macro > best_f05:
                best_f05 = macro
                best_mt, best_st = float(mt), float(st)

    print(f"Two-threshold: match_t={best_mt:.2f}  singleton_t={best_st:.2f}  "
          f"F0.5={best_f05:.4f}", flush=True)
    return best_mt, best_st, best_f05


# ═══════════════════════════════════════════════════════════════════════
# Graph post-processing (conservative, opt-in)
# ═══════════════════════════════════════════════════════════════════════

def graph_closure(matches: dict[str, list[str]], scores_dict: dict,
                  cfg: Cfg) -> dict[str, list[str]]:
    """Conservative transitive closure: only propagate direct high-confidence
    match chains within a single S1 entity's match set.

    Does NOT cross-propagate matches between different S1 entities (that
    would introduce false merges on singletons — a 26× more expensive error
    than a missed match under F0.5).

    Only enabled when cfg.graph_enabled = True.
    """
    if not cfg.graph_enabled:
        return matches

    uf = UnionFind()

    # Only union pool→pool via shared high-confidence S1 connections
    # Conservative: only extend if score ≥ graph_edge_threshold
    pool_s1_scores: dict[str, list[tuple[str, float]]] = {}
    for sid, mids in matches.items():
        for mid in mids:
            score = scores_dict.get((sid, mid), 0.0)
            if score >= cfg.graph_edge_threshold:
                pool_s1_scores.setdefault(mid, []).append((sid, score))

    # Extend S1 matches only to other pool entities matched by the same S1
    # (transitivity within one entity's candidate set, not across S1 entities)
    improved = {sid: list(mids) for sid, mids in matches.items()}
    propagated = 0
    for sid, mids in matches.items():
        high_conf = [m for m in mids
                     if scores_dict.get((sid, m), 0.0) >= cfg.graph_edge_threshold]
        if len(high_conf) < 2:
            continue
        # Within one S1: if A and B are both high-conf matches for the same S1,
        # and they co-occur in the same component, add them to each other
        for m in mids:
            if scores_dict.get((sid, m), 0.0) >= 0.7:
                for hm in high_conf:
                    if m != hm and m not in improved[sid]:
                        improved[sid].append(m)
                        propagated += 1

    print(f"Graph closure: {propagated:,} matches propagated", flush=True)
    return improved


# ═══════════════════════════════════════════════════════════════════════
# HOLDOUT
# ═══════════════════════════════════════════════════════════════════════

def run_holdout(cfg: Cfg):
    t0 = _timer()
    print(f"{'='*60}\nHOLDOUT — {cfg.holdout_n:,} entities\n{'='*60}", flush=True)

    # ── Load data ─────────────────────────────────────────────────
    _blk.CAP = cfg.bm25_cap
    s1 = load_s1([ROOT / "train/train_source1.tsv"])
    print(f"S1: {len(s1):,} ({_timer()-t0:.0f}s)", flush=True)
    freq = name_freq(s1)
    gt = dict(iter_ground_truth(ROOT / "train/train_ground_truth.tsv"))

    pool_paths = [ROOT / "train/train_source2.tsv", ROOT / "train/train_source3.tsv"]
    index, idf = build_index_and_idf(pool_paths)
    pool = load_pool(pool_paths)
    print(f"Pool: {len(pool):,} ({_timer()-t0:.0f}s)", flush=True)

    rng = random.Random(cfg.seed)
    holdout = rng.sample(list(s1), cfg.holdout_n)
    n_train = int(cfg.holdout_n * cfg.train_frac)

    # ── Dense retrieval (pool encoding + index) ───────────────────
    print(f"\n--- Dense retrieval ({cfg.dense_model}, {cfg.dense_dim}-dim) ---",
          flush=True)
    dense = init_dense(cfg)

    # Re-use cached pool index if available
    pool_cache = CKPT / "pool_train"
    if pool_cache.with_suffix(".faiss").exists():
        print("Loading cached pool index...", flush=True)
        dense.load_pool(pool_cache)
    else:
        import faiss as _faiss
        n_gpus = _faiss.get_num_gpus()
        if n_gpus > 1:
            print(f"Multi-GPU encoding ({n_gpus} GPUs detected)", flush=True)
            dense.encode_and_index_multigpu(pool, CKPT / "tmp_shards")
        else:
            dense.encode_and_index(pool, fp16=cfg.dense_fp16_index)
        dense.save_pool(pool_cache)

    # Search S1 in chunks to bound memory
    holdout_records = {sid: s1[sid] for sid in holdout}
    dense_cands, cosine_scores = dense.search_chunked(
        holdout_records, top_k=cfg.dense_top_k, chunk=cfg.dense_search_chunk)

    del dense
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"Dense retrieval done ({_timer()-t0:.0f}s)", flush=True)

    # ── Hybrid blocking ───────────────────────────────────────────
    print(f"\n--- Hybrid blocking (BM25 ∪ dense) ---", flush=True)
    _N_WORKERS = min(32, (os.cpu_count() or 8))
    print(f"Parallel featurization using {_N_WORKERS} threads...", flush=True)

    def _featurize_holdout_ent(item):
        i, sid = item
        _name, _addr, country, s1f = s1[sid]
        bm25_c, _ = candidates_for(index, _name, _addr, country, idf)
        dense_c = dense_cands.get(sid, [])
        seen: set[str] = set()
        hybrid: list[str] = []
        for c in bm25_c + dense_c:
            if c not in seen and c in pool:
                seen.add(c)
                hybrid.append(c)

        gold = gt.get(sid, set())
        cos_dict = cosine_scores.get(sid, {})
        if hybrid:
            cand_names = [pool[c][0] for c in hybrid]
            cand_addrs = [pool[c][1] for c in hybrid]
            X_ent = extended_features(
                _name, _addr, country, cand_names, cand_addrs,
                idf, freq, cos_dict, hybrid)
            scored = [(hybrid[j], X_ent[j].tolist()) for j in range(len(hybrid))]
        else:
            scored = []
        return i, (sid, scored, gold, set(hybrid), False, country), len(scored), sid

    entity_data = [None] * len(holdout)
    total_pairs = 0
    items = list(enumerate(holdout))

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as executor:
        futures = {executor.submit(_featurize_holdout_ent, item): item[0] for item in items}
        for count, future in enumerate(as_completed(futures), 1):
            i, res_tuple, n_pairs, sid = future.result()
            entity_data[i] = res_tuple
            total_pairs += n_pairs
            cosine_scores.pop(sid, None)
            if count % 20000 == 0 or count == len(holdout):
                print(f"  {count:,}/{len(holdout):,} entities featurized  {total_pairs:,} pairs  "
                      f"({_timer()-t0:.0f}s)", flush=True)

    del cosine_scores, dense_cands
    gc.collect()

    # ── Build numpy arrays ────────────────────────────────────────
    X_list, y_list, boundaries = [], [], []
    for sid, scored, gold, cand_set, _, _ in entity_data:
        start = len(X_list)
        for c, feat in scored:
            X_list.append(feat)
            y_list.append(1 if c in gold else 0)
        boundaries.append((start, len(X_list)))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    del X_list, y_list

    n_pos = int(y.sum())
    print(f"Pairs: {X.shape[0]:,}  pos: {n_pos:,} ({100*n_pos/max(1,len(y)):.1f}%)",
          flush=True)

    # Recall ceiling
    ceil_macro = sum(
        f05_one(gold & cand_set, gold)
        for _, _, gold, cand_set, _, _ in entity_data
    ) / cfg.holdout_n
    print(f"Recall ceiling (hybrid blocking): {ceil_macro:.4f}", flush=True)

    # ── Train LightGBM on 70K training split ─────────────────────
    print(f"\n--- LightGBM ({cfg.lgbm_trees} trees) ---", flush=True)
    train_end = boundaries[n_train - 1][1]
    X_train, y_train = X[:train_end], y[:train_end]

    import lightgbm as lgb
    clf = lgb.LGBMClassifier(
        n_estimators=cfg.lgbm_trees, num_leaves=cfg.lgbm_leaves,
        learning_rate=cfg.lgbm_lr, min_child_samples=20, n_jobs=-1, verbose=-1,
        scale_pos_weight=float((y_train == 0).sum()) / max(1, y_train.sum()),
    )
    clf.fit(X_train, y_train)
    print(f"LightGBM trained on 70K ({_timer()-t0:.0f}s)", flush=True)

    # ── Train cross-encoder on 70K train split ────────────────────
    print(f"\n--- Cross-encoder training ({cfg.ce_model}) ---", flush=True)
    train_cands = {entity_data[ei][0]: [c for c, _ in entity_data[ei][1]]
                   for ei in range(n_train)}
    ce_pairs, ce_labels = build_ce_pairs(s1, pool, gt, train_cands, cfg, rng)

    # Val pairs for CE validation during training (10K)
    val_cands = {entity_data[ei][0]: [c for c, _ in entity_data[ei][1]]
                 for ei in range(n_train, min(n_train + 10000, cfg.holdout_n))}
    val_ce_pairs, val_ce_labels = build_ce_pairs(
        s1, pool, gt, val_cands, cfg, rng)

    ce_model = train_cross_encoder(ce_pairs, ce_labels,
                                   val_ce_pairs, val_ce_labels, cfg)
    del ce_pairs, ce_labels, val_ce_pairs, val_ce_labels
    gc.collect()

    # ── CE scoring on validation uncertain band ───────────────────
    print(f"\n--- CE cascade scoring (uncertain band [{cfg.ce_uncertain_lo},"
          f"{cfg.ce_uncertain_hi}]) ---", flush=True)

    # Use 70K model to identify uncertain entities for CE
    lgbm_scores_70k = _clf_scores(clf, X)
    ce_scores_map: dict[tuple, float] = {}
    n_ce_scored = 0

    for ei in range(n_train, cfg.holdout_n):
        sid, scored, gold, cand_set, _, country = entity_data[ei]
        start, end = boundaries[ei]
        if end == start:
            continue
        pair_lgbm = lgbm_scores_70k[start:end]
        max_lgbm = pair_lgbm.max()
        if not (cfg.ce_uncertain_lo <= max_lgbm <= cfg.ce_uncertain_hi):
            continue

        top_idx = np.argsort(pair_lgbm)[::-1][:cfg.ce_top_k]
        ce_pairs_batch = []
        valid_idx = []
        for j in top_idx:
            ceid = scored[j][0]
            if ceid in pool:
                cn, ca, cc = pool[ceid]
                ce_pairs_batch.append((
                    serialize_entity(s1[sid][0], s1[sid][1], s1[sid][2]),
                    serialize_entity(cn, ca, cc),
                ))
                valid_idx.append(j)
        if ce_pairs_batch:
            ce_s = predict_ce(ce_model, ce_pairs_batch)
            for k, j in enumerate(valid_idx[:len(ce_s)]):
                ce_scores_map[(sid, scored[j][0])] = float(ce_s[k])
            n_ce_scored += len(ce_s)

    print(f"CE scored {n_ce_scored:,} pairs ({_timer()-t0:.0f}s)", flush=True)
    del lgbm_scores_70k

    # Try ONNX export for test inference speed
    onnx_path = try_onnx_export(str(CKPT / "ce-mdeberta"), ce_model)
    del ce_model
    gc.collect()

    # ── Retrain LightGBM on full 100K ─────────────────────────────
    # MUST retrain before calibration so iso is fitted on the same model
    # that will run at test time.
    print(f"\n--- Retraining LightGBM on full 100K ---", flush=True)
    clf.fit(X, y)
    lgbm_path = CKPT / "lgbm_full.txt"
    clf.booster_.save_model(str(lgbm_path))
    print(f"LightGBM retrained ({_timer()-t0:.0f}s)", flush=True)

    # ── Isotonic calibration on retrained model ───────────────────
    # Fit on full 100K scores from the retrained model.
    # In-sample for isotonic regression is acceptable (monotone, no complex overfitting).
    # Critical: iso must be from the SAME model used at test time.
    lgbm_scores_100k = _clf_scores(clf, X)
    iso = calibrate_isotonic(lgbm_scores_100k, y.astype(np.float32))

    # ── Build final scores (consistent with test inference) ───────
    # All pairs: iso(lgbm). CE-scored pairs: CE score directly.
    # Same logic used verbatim at test time → thresholds are valid.
    final_scores = build_final_scores(
        lgbm_scores_100k, iso, entity_data, boundaries,
        ce_scores_map, n_train)

    # ── Two-threshold sweep on validation portion only ────────────
    best_mt, best_st, best_f05 = sweep_two_threshold(
        final_scores, entity_data, boundaries, cfg,
        start_ei=n_train, end_ei=cfg.holdout_n)

    # ── Full holdout report ───────────────────────────────────────
    rows_all, scores_dict = [], {}
    for ei in range(cfg.holdout_n):
        sid, scored, gold, cand_set, _, country = entity_data[ei]
        start, end = boundaries[ei]
        pair_scores = final_scores[start:end]

        if end == start:
            pred = set()
        else:
            max_score = pair_scores.max()
            eff_st = best_st + (cfg.france_t_bump
                                if country.strip().lower() == "france" else 0.0)
            if len(pair_scores) >= 2:
                sorted_s = np.sort(pair_scores)[::-1]
                if sorted_s[0] - sorted_s[1] < cfg.margin_gap:
                    eff_st = max(eff_st, best_st + 0.10)
            if max_score < eff_st:
                pred = set()
            else:
                pred = {scored[j][0] for j in range(end - start)
                        if pair_scores[j] >= best_mt}
                for ceid in pred:
                    scores_dict[(sid, ceid)] = float(
                        pair_scores[[scored[j][0] for j in range(end-start)].index(ceid)])

        rows_all.append((pred, gold, cand_set))

    stats = summarize(rows_all)
    print(f"\nFull holdout: {fmt(stats)}", flush=True)

    # ── Save ──────────────────────────────────────────────────────
    meta = {
        "match_threshold": round(best_mt, 4),
        "singleton_threshold": round(best_st, 4),
        "holdout_f05": round(best_f05, 4),
        "full_holdout_f05": round(stats["macro_f05"], 4),
        "recall_ceiling": round(ceil_macro, 4),
        "num_features": NUM_FEATURES,
        "france_t_bump": cfg.france_t_bump,
        "margin_gap": cfg.margin_gap,
        "ce_uncertain_lo": cfg.ce_uncertain_lo,
        "ce_uncertain_hi": cfg.ce_uncertain_hi,
        "ce_top_k": cfg.ce_top_k,
        "onnx_path": onnx_path or "",
    }
    with open(CKPT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(CKPT / "iso.pkl", "wb") as f:
        pickle.dump(iso, f)

    print(f"\n{'='*60}\nHoldout: match_t={best_mt:.2f}  singleton_t={best_st:.2f}  "
          f"val_F0.5={best_f05:.4f}  full_F0.5={stats['macro_f05']:.4f}  "
          f"ceil={ceil_macro:.4f}  {_timer()-t0:.0f}s\n{'='*60}", flush=True)

    with open(HERE.parents[1] / "notes.md", "a", encoding="utf-8") as f:
        import datetime
        f.write(
            f"{datetime.datetime.now():%Y-%m-%d %H:%M}  "
            f"pipeline holdout  {fmt(stats)}  "
            f"t={best_mt:.2f}/{best_st:.2f}  "
            f"ceil={ceil_macro:.4f}\n"
        )


# ═══════════════════════════════════════════════════════════════════════
# TEST INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def run_test(cfg: Cfg):
    t0 = _timer()
    print(f"{'='*60}\nTEST INFERENCE\n{'='*60}", flush=True)

    # ── Load models and meta ──────────────────────────────────────
    meta = json.loads((CKPT / "meta.json").read_text())
    match_t = meta["match_threshold"]
    singleton_t = meta["singleton_threshold"]
    france_t_bump = meta.get("france_t_bump", cfg.france_t_bump)
    margin_gap = meta.get("margin_gap", cfg.margin_gap)
    ce_uncertain_lo = meta.get("ce_uncertain_lo", cfg.ce_uncertain_lo)
    ce_uncertain_hi = meta.get("ce_uncertain_hi", cfg.ce_uncertain_hi)
    ce_top_k = meta.get("ce_top_k", cfg.ce_top_k)

    import lightgbm as lgb
    lgbm_clf = lgb.Booster(model_file=str(CKPT / "lgbm_full.txt"))
    with open(CKPT / "iso.pkl", "rb") as f:
        iso = pickle.load(f)

    onnx_path = meta.get("onnx_path", "")
    ce_path = onnx_path if onnx_path and Path(onnx_path).exists() else str(CKPT / "ce-mdeberta")
    print(f"Loading cross-encoder from {ce_path}", flush=True)
    from sentence_transformers.cross_encoder import CrossEncoder
    ce_model = CrossEncoder(ce_path, max_length=cfg.ce_maxlen, device=_device())

    # ── Load data ─────────────────────────────────────────────────
    _blk.CAP = cfg.bm25_cap
    s1 = load_s1([ROOT / "test/test_source1.tsv"])
    order = list(s1)
    print(f"Test S1: {len(s1):,} ({_timer()-t0:.0f}s)", flush=True)
    freq = name_freq(s1)

    pool_paths = [ROOT / "test/test_source2.tsv", ROOT / "test/test_source3.tsv"]
    index, idf = build_index_and_idf(pool_paths)
    pool = load_pool(pool_paths)
    print(f"Pool: {len(pool):,} ({_timer()-t0:.0f}s)", flush=True)

    # ── Dense retrieval in chunks ─────────────────────────────────
    # Process in chunks matching test_batch size to avoid holding all
    # cosine_scores in memory simultaneously.
    print(f"\n--- Dense retrieval (chunked to bound RAM) ---", flush=True)
    dense = init_dense(cfg)

    pool_cache = CKPT / "pool_test"
    if pool_cache.with_suffix(".faiss").exists():
        print("Loading cached pool index...", flush=True)
        dense.load_pool(pool_cache)
    else:
        import faiss as _faiss
        n_gpus = _faiss.get_num_gpus()
        if n_gpus > 1:
            print(f"Multi-GPU encoding ({n_gpus} GPUs detected)", flush=True)
            dense.encode_and_index_multigpu(pool, CKPT / "tmp_shards_test")
        else:
            dense.encode_and_index(pool, fp16=cfg.dense_fp16_index)
        dense.save_pool(pool_cache)

    # ── Batched inference (dense search inline with scoring) ───────
    matches: dict[str, list[str]] = {}
    cands_out: dict[str, list[str]] = {}
    scores_dict_graph: dict[tuple, float] = {}
    n_ce_total = 0
    _N_FEAT_WORKERS = min(32, (os.cpu_count() or 8))
    print(f"Test inference using {_N_FEAT_WORKERS} parallel featurization threads", flush=True)

    def _featurize_entity(sid):
        """Pure CPU: BM25 retrieval + feature extraction for one entity.
        Thread-safe: reads only shared immutable dicts (pool, index, idf, freq).
        Returns (sid, hybrid, X_ent) or (sid, [], None) for empty."""
        s1_name, s1_addr, country, _ = s1[sid]
        bm25_c, _ = candidates_for(index, s1_name, s1_addr, country, idf)
        dense_c = _batch_dense_cands_ref[0].get(sid, [])
        cos_dict = _batch_cosines_ref[0].get(sid, {})
        seen: set[str] = set()
        hybrid: list[str] = []
        for c in bm25_c + dense_c:
            if c not in seen and c in pool:
                seen.add(c)
                hybrid.append(c)
        if not hybrid:
            return sid, [], None
        cand_names = [pool[c][0] for c in hybrid]
        cand_addrs = [pool[c][1] for c in hybrid]
        X_ent = extended_features(
            s1_name, s1_addr, country, cand_names, cand_addrs,
            idf, freq, cos_dict, hybrid)
        return sid, hybrid, X_ent

    # Mutable reference boxes so the closure can be updated each batch
    _batch_dense_cands_ref = [{}]
    _batch_cosines_ref = [{}]

    for batch_start in range(0, len(order), cfg.test_batch):
        batch = order[batch_start:batch_start + cfg.test_batch]

        # Dense search for this batch only — cosine_scores freed after use
        batch_records = {sid: s1[sid] for sid in batch}
        batch_dense_cands, batch_cosines = dense.search(
            batch_records, top_k=cfg.dense_top_k)

        # Update the closure-visible references
        _batch_dense_cands_ref[0] = batch_dense_cands
        _batch_cosines_ref[0] = batch_cosines

        # Parallel featurization across all entities in the batch
        feat_results = {}
        with ThreadPoolExecutor(max_workers=_N_FEAT_WORKERS) as ex:
            futs = {ex.submit(_featurize_entity, sid): sid for sid in batch}
            for fut in as_completed(futs):
                sid, hybrid, X_ent = fut.result()
                feat_results[sid] = (hybrid, X_ent)

        # Sequential scoring (LightGBM + iso + CE) — not thread-safe to parallelize
        for sid in batch:
            hybrid, X_ent = feat_results[sid]
            s1_name, s1_addr, country, _ = s1[sid]
            cands_out[sid] = hybrid

            if X_ent is None:
                matches[sid] = []
                continue

            # LightGBM → calibrated with iso
            lgbm_s = lgbm_clf.predict(X_ent).astype(np.float32)
            max_lgbm = lgbm_s.max()

            # Exact same logic as build_final_scores in holdout:
            # All pairs: iso(lgbm). CE-scored uncertain band: CE direct.
            final_s = iso.transform(lgbm_s).astype(np.float32)

            if ce_uncertain_lo <= max_lgbm <= ce_uncertain_hi:
                top_idx = np.argsort(lgbm_s)[::-1][:ce_top_k]
                ce_pairs_batch = []
                valid_idx = []
                for j in top_idx:
                    cn, ca, cc = pool[hybrid[j]]
                    ce_pairs_batch.append((
                        serialize_entity(s1_name, s1_addr, country),
                        serialize_entity(cn, ca, cc),
                    ))
                    valid_idx.append(j)
                if ce_pairs_batch:
                    ce_s = predict_ce(ce_model, ce_pairs_batch)
                    for k, j in enumerate(valid_idx[:len(ce_s)]):
                        final_s[j] = ce_s[k]
                    n_ce_total += len(ce_s)

            max_final = final_s.max()
            eff_st = singleton_t + (france_t_bump
                                    if country.strip().lower() == "france" else 0.0)
            if len(final_s) >= 2:
                sorted_fs = np.sort(final_s)[::-1]
                if sorted_fs[0] - sorted_fs[1] < margin_gap:
                    eff_st = max(eff_st, singleton_t + 0.10)

            if max_final < eff_st:
                matches[sid] = []
            else:
                m = [hybrid[j] for j in range(len(hybrid)) if final_s[j] >= match_t]
                matches[sid] = m
                for ceid in m:
                    scores_dict_graph[(sid, ceid)] = float(
                        final_s[[hybrid[j] for j in range(len(hybrid))].index(ceid)])

        # Free batch cosines immediately
        del batch_cosines, batch_dense_cands
        gc.collect()

        done = min(batch_start + cfg.test_batch, len(order))
        if done % 50_000 < cfg.test_batch or done == len(order):
            print(f"  {done:,}/{len(order):,}  ce_pairs={n_ce_total:,}  "
                  f"elapsed={_timer()-t0:.0f}s", flush=True)

    del dense
    gc.collect()

    # ── Graph post-processing (opt-in) ─────────────────────────────
    if cfg.graph_enabled:
        print(f"\n--- Graph closure (enabled) ---", flush=True)
        matches = graph_closure(matches, scores_dict_graph, cfg)

    # ── Write output ──────────────────────────────────────────────
    out = HERE.parents[1] / "output"
    write_submission(out / "matching_results.tsv", "matched_entity_ids",
                     matches, order)
    write_submission(out / "candidate_pairs.tsv", "candidate_entity_ids",
                     cands_out, order)

    n_matched = sum(1 for v in matches.values() if v)
    n_france = sum(1 for sid in order if s1[sid][2].strip().lower() == "france")
    fr_matched = sum(1 for sid in order
                     if s1[sid][2].strip().lower() == "france"
                     and matches.get(sid))

    print(f"\n{'='*60}\nTest: {out}\n"
          f"  matched={n_matched:,}/{len(order):,}\n"
          f"  france={fr_matched:,}/{n_france:,} matched\n"
          f"  CE pairs={n_ce_total:,}\n"
          f"  elapsed={_timer()-t0:.0f}s\n{'='*60}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = Cfg()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    print(f"Pipeline mode: {mode}")
    print(f"Device: {_device()}")
    print(f"Dataset root: {ROOT}")
    print(f"Config:\n{cfg.to_json()}\n")

    if mode == "holdout":
        run_holdout(cfg)
    elif mode == "test":
        run_test(cfg)
    elif mode == "all":
        run_holdout(cfg)
        run_test(cfg)
    else:
        print(f"Unknown mode: {mode}. Use holdout/test/all")
