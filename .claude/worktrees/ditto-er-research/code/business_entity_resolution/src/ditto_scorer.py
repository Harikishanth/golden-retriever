"""Ditto-style cross-encoder scorer for entity resolution.

Serializes entity pairs as structured text with COL/VAL markers, then
scores them with a fine-tuned transformer cross-encoder.  Designed to
slot into the existing BM25 + LightGBM pipeline as either:
  - Feature #18 (ditto_score) fed to LightGBM
  - Standalone re-ranker on top-K LightGBM candidates
  - Ensemble (alpha * lgbm_prob + (1-alpha) * ditto_prob)

Usage:
    # Training
    scorer = DittoScorer("distilbert-base-multilingual-cased")
    scorer.train(pairs, labels, val_pairs, val_labels, epochs=3)
    scorer.save("ditto-ckpt")

    # Inference
    scorer = DittoScorer.load("ditto-ckpt")
    scores = scorer.predict(pairs, batch_size=256)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

_HAS_CROSS_ENCODER = False
_HAS_TORCH = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    _HAS_CROSS_ENCODER = True
except ImportError:
    try:
        from sentence_transformers import CrossEncoder
        _HAS_CROSS_ENCODER = True
    except ImportError:
        pass


def serialize_entity(name: str, addr: str, country: str) -> str:
    """Ditto-style serialization for one entity record."""
    parts = []
    if name and name.strip():
        parts.append(f"COL name VAL {name.strip()}")
    if addr and addr.strip():
        parts.append(f"COL address VAL {addr.strip()}")
    if country and country.strip():
        parts.append(f"COL country VAL {country.strip()}")
    return " ".join(parts)


def make_pair(s1_name: str, s1_addr: str, s1_country: str,
              cand_name: str, cand_addr: str, cand_country: str) -> tuple[str, str]:
    """Returns (text_a, text_b) for cross-encoder input."""
    return (
        serialize_entity(s1_name, s1_addr, s1_country),
        serialize_entity(cand_name, cand_addr, cand_country),
    )


def build_training_pairs(s1_records: dict, pool: dict, ground_truth: dict,
                         blocking_candidates: dict,
                         max_neg_per_pos: int = 5,
                         max_entities: int | None = None) -> tuple[list, list]:
    """Build training pairs from blocking candidates + ground truth.

    Returns (pairs, labels) where pairs is list of (text_a, text_b) tuples.
    """
    pairs = []
    labels = []
    entities = list(blocking_candidates.keys())
    if max_entities:
        entities = entities[:max_entities]

    for s1_id in entities:
        s1_name, s1_addr, s1_country = s1_records[s1_id][:3]
        gold = ground_truth.get(s1_id, set())
        cands = blocking_candidates.get(s1_id, [])

        pos_ids = [c for c in cands if c in gold and c in pool]
        neg_ids = [c for c in cands if c not in gold and c in pool]

        for pid in pos_ids:
            p_name, p_addr, p_country = pool[pid][:3]
            pair = make_pair(s1_name, s1_addr, s1_country,
                             p_name, p_addr, p_country)
            pairs.append(pair)
            labels.append(1.0)

        for nid in neg_ids[:max_neg_per_pos * max(1, len(pos_ids))]:
            n_name, n_addr, n_country = pool[nid][:3]
            pair = make_pair(s1_name, s1_addr, s1_country,
                             n_name, n_addr, n_country)
            pairs.append(pair)
            labels.append(0.0)

    return pairs, labels


class DittoScorer:
    """Cross-encoder wrapper for Ditto-style entity matching."""

    def __init__(self, model_name: str = "distilbert-base-multilingual-cased",
                 max_length: int = 128, device: str | None = None):
        if not _HAS_CROSS_ENCODER:
            raise ImportError(
                "sentence-transformers is required for DittoScorer. "
                "Install with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.max_length = max_length
        self._device = device
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )
        self.threshold = 0.5

    def train(self, train_pairs: list[tuple[str, str]],
              train_labels: list[float],
              val_pairs: list[tuple[str, str]] | None = None,
              val_labels: list[float] | None = None,
              epochs: int = 3, batch_size: int = 64,
              learning_rate: float = 3e-5,
              output_dir: str = "./ditto-ckpt",
              pos_weight: float = 5.0):
        """Fine-tune the cross-encoder on entity pairs."""
        try:
            from sentence_transformers.cross_encoder import (
                CrossEncoderTrainer,
                CrossEncoderTrainingArguments,
            )
            from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
            from datasets import Dataset

            train_dict = {
                "sentence1": [p[0] for p in train_pairs],
                "sentence2": [p[1] for p in train_pairs],
                "label": train_labels,
            }
            train_dataset = Dataset.from_dict(train_dict)

            eval_dataset = None
            if val_pairs and val_labels:
                val_dict = {
                    "sentence1": [p[0] for p in val_pairs],
                    "sentence2": [p[1] for p in val_pairs],
                    "label": val_labels,
                }
                eval_dataset = Dataset.from_dict(val_dict)

            loss = BinaryCrossEntropyLoss(
                model=self.model,
                pos_weight=torch.tensor(pos_weight) if _HAS_TORCH else None,
            )

            args = CrossEncoderTrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_ratio=0.1,
                weight_decay=0.01,
                fp16=_HAS_TORCH and torch.cuda.is_available(),
                logging_steps=100,
                save_strategy="epoch",
                eval_strategy="epoch" if eval_dataset else "no",
                load_best_model_at_end=bool(eval_dataset),
            )

            trainer = CrossEncoderTrainer(
                model=self.model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss=loss,
            )
            trainer.train()
            print(f"Training complete. Model saved to {output_dir}")

        except ImportError as e:
            print(f"Cannot use CrossEncoderTrainer ({e}). Falling back to manual training.")
            self._train_manual(train_pairs, train_labels, epochs, batch_size,
                               learning_rate, output_dir)

    def _train_manual(self, pairs, labels, epochs, batch_size, lr, output_dir):
        """Fallback manual training loop using raw transformers."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )

        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]
        encodings = tokenizer(texts_a, texts_b, truncation=True,
                              padding=True, max_length=self.max_length,
                              return_tensors="pt")

        dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                ids, mask, lbl = [b.to(device) for b in batch]
                out = model(input_ids=ids, attention_mask=mask, labels=lbl)
                loss = out.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Manual training complete. Saved to {output_dir}")

    def predict(self, pairs: list[tuple[str, str]],
                batch_size: int = 256) -> np.ndarray:
        """Score entity pairs. Returns 1-D array of match probabilities."""
        if not pairs:
            return np.array([], dtype=np.float32)

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=len(pairs) > 10000,
            convert_to_numpy=True,
        )
        scores = np.asarray(scores, dtype=np.float32)

        if scores.ndim == 2:
            from scipy.special import softmax
            scores = softmax(scores, axis=1)[:, 1]
        elif scores.ndim == 1:
            if scores.max() > 1.0 or scores.min() < 0.0:
                scores = 1.0 / (1.0 + np.exp(-scores))

        return scores

    def predict_pairs(self, s1_name: str, s1_addr: str, s1_country: str,
                      cand_names: list[str], cand_addrs: list[str],
                      cand_countries: list[str],
                      batch_size: int = 256) -> np.ndarray:
        """Score one S1 entity against multiple candidates."""
        pairs = [
            make_pair(s1_name, s1_addr, s1_country, cn, ca, cc)
            for cn, ca, cc in zip(cand_names, cand_addrs, cand_countries)
        ]
        return self.predict(pairs, batch_size=batch_size)

    def sweep_threshold(self, pairs: list[tuple[str, str]],
                        labels: list[float],
                        entity_boundaries: list[tuple[int, int]],
                        gold_sets: list[set],
                        cand_lists: list[list]) -> float:
        """Sweep threshold on holdout data to maximize macro F0.5."""
        from evaluate import f05_one

        scores = self.predict(pairs)
        best_f05, best_t = 0.0, 0.5

        for t_val in [i / 100.0 for i in range(5, 95, 2)]:
            f05_sum = 0.0
            for ei, (start, end) in enumerate(entity_boundaries):
                gold = gold_sets[ei]
                cands = cand_lists[ei]
                pred = {cands[j] for j in range(end - start)
                        if scores[start + j] >= t_val}
                f05_sum += f05_one(pred, gold)

            macro = f05_sum / len(entity_boundaries)
            if macro > best_f05:
                best_f05 = macro
                best_t = t_val

        self.threshold = best_t
        print(f"Ditto threshold sweep: best t={best_t:.2f}  F0.5={best_f05:.4f}")
        return best_t

    def save(self, path: str):
        """Save model + threshold."""
        self.model.save_pretrained(path)
        meta = {"threshold": self.threshold, "model_name": self.model_name,
                "max_length": self.max_length}
        with open(os.path.join(path, "ditto_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DittoScorer":
        """Load a saved DittoScorer."""
        meta_path = os.path.join(path, "ditto_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        scorer = cls.__new__(cls)
        scorer.model_name = meta.get("model_name", path)
        scorer.max_length = meta.get("max_length", 128)
        scorer.threshold = meta.get("threshold", 0.5)
        scorer._device = None
        scorer.model = CrossEncoder(path, max_length=scorer.max_length)
        return scorer
