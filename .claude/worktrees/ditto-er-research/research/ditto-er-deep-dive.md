# Ditto: A Deep Dive into Deep Entity Resolution

**Author:** Research artifact for Amazon ML Challenge 2026 — Da Big Three  
**Date:** 2026-09-26  
**Purpose:** Technical deep dive into the Ditto entity resolution framework, its serialization scheme, training methodology, and how to adapt it for business entity resolution with name+address+country fields.

---

## 1. What Is Ditto?

Ditto (Li et al., 2020 — "Deep Entity Matching with Pre-Trained Language Models") is a landmark entity resolution (ER) system that reframes record matching as a **sequence-pair classification** task for pre-trained language models. Instead of hand-engineering features like Jaccard similarity, TF-IDF cosine, or edit distance, Ditto feeds serialized entity pairs directly to a transformer (BERT/DistilBERT/RoBERTa) and fine-tunes it to predict match/no-match.

### Key Insight

Traditional ER pipelines break records into tokens and compute similarity features manually. Ditto's insight is that a pre-trained language model already understands:
- That "Corp" and "Corporation" are the same thing
- That "123 Main St" and "123 Main Street" refer to the same location
- That word order variations ("McDonald's Corp" vs "Corp McDonald's") are semantically equivalent
- That abbreviations, typos, and transliterations can be resolved from context

By serializing both records into a single text input with special column markers, the transformer's attention mechanism can learn cross-attribute alignment (e.g., matching name tokens in record A against name tokens in record B) without explicit feature engineering.

---

## 2. Serialization Format

Ditto introduces a structured serialization that preserves column semantics:

```
[CLS] COL name VAL McDonald's Corporation COL address VAL 123 Main Street Chicago IL 60601 COL country VAL US [SEP] COL name VAL McDonalds Corp COL address VAL 123 Main St Chicago Illinois 60601 COL country VAL US [SEP]
```

### Why This Works

1. **Column markers (`COL`, `VAL`)** tell the model which attribute is which — it learns that `name` tokens in entity A should attend to `name` tokens in entity B, not `address` tokens.
2. **The `[SEP]` token** separates the two entities, giving the model a clear boundary.
3. **Pre-training** gives the model knowledge of abbreviations, synonyms, and world knowledge (e.g., "IL" = "Illinois") without any ER-specific training.

### Serialization Variants

| Variant | Format | When to Use |
|---------|--------|-------------|
| **Standard** | `COL name VAL X COL addr VAL Y` | Default, best balance |
| **No markers** | `X Y [SEP] X' Y'` | Ablation only; worse performance |
| **Attribute-aligned** | `COL name VAL X [SEP] COL name VAL X'` per column, concatenated | Good for many columns; worse for few |
| **With domain knowledge** | Inject `[ABBR Corporation=Corp]` tokens | Marginal gain, adds complexity |

For our task (3 columns: name, address, country), the standard format is optimal.

---

## 3. Architecture

```
Input: serialized pair string
  ↓
Tokenizer (WordPiece/BPE, max_length=128-256)
  ↓
Transformer Encoder (DistilBERT 66M / BERT-base 110M / RoBERTa-base 125M)
  ↓
[CLS] token embedding (768-dim)
  ↓
Dropout (0.1)
  ↓
Linear layer (768 → 2)  [or 768 → 1 for regression]
  ↓
Softmax / Sigmoid → match probability
```

### Model Choices for Our Contest (≤8B params, MIT/Apache 2.0)

| Model | Params | License | Speed (pairs/sec on T4) | Quality |
|-------|--------|---------|------------------------|---------|
| `distilbert-base-uncased` | 66M | Apache 2.0 | ~1200 | Good baseline |
| `distilbert-base-multilingual-cased` | 134M | Apache 2.0 | ~900 | Handles France |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | Apache 2.0 | ~2500 | Pre-trained reranker |
| `BAAI/bge-reranker-v2-m3` | 568M | MIT | ~200 | State-of-the-art multilingual |
| `microsoft/deberta-v3-small` | 44M | MIT | ~1500 | Strong for NLI tasks |

**Recommendation:** Start with `distilbert-base-multilingual-cased` for French support, or `cross-encoder/ms-marco-MiniLM-L-6-v2` for speed if French is handled by other features.

---

## 4. Training Strategy

### 4.1 Data Construction

For each training S1 entity with ground truth:
1. **Positive pairs:** S1 × each matched S2/S3 entity (from ground truth)
2. **Hard negatives:** S1 × top-K BM25/embedding candidates that are NOT in ground truth
3. **Ratio:** 1:5 to 1:10 positive:negative is typical for ER

```python
# Pseudo-code for training data construction
training_pairs = []
for s1_id in train_s1_ids:
    gold_matches = ground_truth[s1_id]
    candidates = blocking_candidates[s1_id]  # from BM25
    
    # Positives
    for match_id in gold_matches:
        if match_id in pool:
            pair_text = serialize(s1[s1_id], pool[match_id])
            training_pairs.append((pair_text, 1))
    
    # Hard negatives (candidates that are NOT matches)
    negatives = [c for c in candidates if c not in gold_matches]
    for neg_id in negatives[:5]:  # 5 hard negatives per positive
        pair_text = serialize(s1[s1_id], pool[neg_id])
        training_pairs.append((pair_text, 0))
```

### 4.2 Why Hard Negatives Matter

Random negatives (e.g., "McDonald's in Chicago" vs "Tokyo Ramen in Mumbai") are trivially distinguishable — the model learns nothing. Hard negatives from the BM25 blocker are entities that look similar but aren't matches — these force the model to learn subtle distinctions:
- Same name, different city
- Same address, different business
- Similar name (typo), completely different business
- Same brand chain, different location

### 4.3 Training Hyperparameters (Ditto paper + practical tuning)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 2e-5 to 5e-5 | Standard for fine-tuning transformers |
| Epochs | 3-5 | More can overfit on small ER datasets |
| Batch size | 32-64 | Larger is better for stability |
| Max sequence length | 128 | Name+addr+country fits easily |
| Warmup ratio | 0.1 | 10% of steps |
| Weight decay | 0.01 | Standard AdamW |
| pos_weight | ~5.0 | Ratio of negatives to positives |
| Dropout | 0.1 | On [CLS] before classifier |

### 4.4 Training with sentence-transformers CrossEncoder

```python
from sentence_transformers.cross_encoder import (
    CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from datasets import Dataset
import torch

model = CrossEncoder(
    "distilbert-base-multilingual-cased",
    num_labels=1,  # regression mode → sigmoid → match probability
)

train_dataset = Dataset.from_dict({
    "sentence1": [serialize(s1_rec) for s1_rec, _, _ in pairs],
    "sentence2": [serialize(cand_rec) for _, cand_rec, _ in pairs],
    "label": [float(label) for _, _, label in pairs],
})

loss = BinaryCrossEntropyLoss(
    model=model,
    pos_weight=torch.tensor(5.0),  # 5 negatives per positive
)

args = CrossEncoderTrainingArguments(
    output_dir="./ditto-ckpt",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    save_strategy="epoch",
)

trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()
```

---

## 5. Ditto's Three Optimizations

The original Ditto paper introduces three techniques beyond basic fine-tuning:

### 5.1 Domain Knowledge Injection (DK)

Inject ER-specific tokens into the serialized text:
- `[ABBR]` for abbreviation expansion: "Corp" → "Corp [ABBR] Corporation"
- `[SPELL]` for spell correction: "Chcago" → "Chcago [SPELL] Chicago"

**For our task:** Not worth the complexity. The transformer already handles "Corp"/"Corporation" and similar patterns. Our normalize.py handles the rest.

### 5.2 Summarization (SU)

For long records, summarize by keeping only the most "important" tokens (highest TF-IDF). Truncation to 128 tokens rarely triggers for name+address+country, so this is irrelevant to our task.

### 5.3 Data Augmentation (DA)

Augment training data by:
- **Span deletion:** Remove random spans from one entity (simulates missing data)
- **Attribute swap:** Swap attribute values between entities
- **Entry swap:** Swap entries between entity pairs

**For our task:** Span deletion is useful — simulates the ~3% empty addresses in S2/S3. Attribute swap less so (our entities have only 3 columns).

---

## 6. Performance: Ditto vs Traditional ER

From the Ditto paper (Table 3) on standard ER benchmarks:

| Dataset | Task | Traditional Best | Ditto F1 | Gain |
|---------|------|-----------------|----------|------|
| Abt-Buy | Product matching | 62.8 (Magellan) | 89.3 | +26.5 |
| Amazon-Google | Product matching | 69.3 (DeepMatcher) | 75.6 | +6.3 |
| DBLP-ACM | Publication matching | 98.4 (DeepMatcher) | 99.0 | +0.6 |
| DBLP-Scholar | Publication matching | 94.7 (DeepMatcher) | 95.6 | +0.9 |
| Walmart-Amazon | Product matching | 71.9 (DeepMatcher) | 86.8 | +14.9 |
| Company (WDC) | Company matching | 92.7 (manual) | 96.5 | +3.8 |

**The Company matching result (96.5% F1) is directly relevant to our business entity resolution task.** The gains are largest on "dirty" datasets with abbreviations, typos, and missing values — exactly our scenario.

---

## 7. Adaptation for Our Pipeline

### 7.1 Two-Stage Architecture

```
Stage 1: BM25 Blocking (existing) → CAP=150 candidates per S1
Stage 2: LightGBM filter → top-50 candidates per S1 (score > low_threshold)
Stage 3: Ditto cross-encoder → final match/no-match per candidate pair
```

Why keep LightGBM as a filter:
- Cross-encoder is slow (~1200 pairs/sec on T4)
- 1.73M test S1 × 150 candidates = 260M pairs → 60 GPU-hours
- 1.73M test S1 × 50 candidates = 86M pairs → 20 GPU-hours
- With LightGBM pre-filter to top-50: manageable in ~20h on T4

### 7.2 Serialization Function

```python
def serialize_entity(name: str, addr: str, country: str) -> str:
    """Ditto-style serialization for one entity."""
    parts = []
    if name:
        parts.append(f"COL name VAL {name.strip()}")
    if addr:
        parts.append(f"COL address VAL {addr.strip()}")
    if country:
        parts.append(f"COL country VAL {country.strip()}")
    return " ".join(parts)

def serialize_pair(s1_name, s1_addr, s1_country,
                   cand_name, cand_addr, cand_country) -> tuple[str, str]:
    """Returns (text_a, text_b) for cross-encoder input."""
    return (
        serialize_entity(s1_name, s1_addr, s1_country),
        serialize_entity(cand_name, cand_addr, cand_country),
    )
```

### 7.3 Integration with Existing Pipeline

The cross-encoder score becomes an additional feature or a replacement scorer:

**Option A: Feature fusion (recommended)**
- Add `ditto_score` as feature #18 in LightGBM
- LightGBM learns optimal weighting between string features and semantic score
- Expected: +0.03-0.05 F0.5

**Option B: Cascade replacement**
- LightGBM filters to top-50 → Ditto rescores → threshold
- Simpler, but loses LightGBM's feature interactions
- Expected: +0.02-0.04 F0.5

**Option C: Score ensemble**
- `final_score = alpha * lgbm_prob + (1-alpha) * ditto_prob`
- Sweep alpha on holdout
- Expected: +0.03-0.05 F0.5 (best if both models are strong)

---

## 8. Practical Considerations for Our Contest

### 8.1 Compute Budget

| Scenario | Pairs | Time on T4 (Kaggle) | Time on A10 (cloud) |
|----------|-------|---------------------|---------------------|
| Train (500k pairs, 3 epochs) | 1.5M forward passes | ~25 min | ~12 min |
| Holdout inference (100k S1 × 50 cands) | 5M pairs | ~70 min | ~35 min |
| Test inference (1.73M S1 × 50 cands) | 86M pairs | ~20 hours | ~10 hours |
| Test inference (1.73M S1 × 20 cands) | 35M pairs | ~8 hours | ~4 hours |

**Critical:** Test inference with 50 candidates per S1 takes ~20h on a single T4. Must either:
- Reduce to top-20 candidates (8h)
- Use multiple GPUs / Kaggle sessions
- Use cross-encoder only on "uncertain" pairs (LightGBM score 0.2-0.8)

### 8.2 France (15% of Test, 0% of Train)

`distilbert-base-multilingual-cased` handles French natively — it was pre-trained on 104 languages including French. This is one of Ditto's strongest advantages over string-similarity features: the model understands that "Boulangerie" is a bakery, "Rue" is a street, and French address conventions.

However, we have **no French training pairs**. Mitigation:
- Cross-lingual transfer from English/Hindi ER patterns
- The model learns "same name + same address = match" in English; this transfers to French
- Add a few synthetic French pairs if possible (but external data is banned)

### 8.3 Sequence Length

Typical entity pair in our data:
```
COL name VAL McDonald's Corporation COL address VAL 123 Main Street Chicago IL 60601 COL country VAL US
```
≈ 25-40 tokens per entity, ≈ 50-80 tokens per pair. Max length 128 is sufficient.

### 8.4 Label Noise

Ground truth has some noise (entities that are truly the same but not linked, or spurious links). Ditto is robust to moderate label noise due to pre-training regularization, but:
- Use hard negatives from BM25 (not random) to avoid trivially-correct labels
- Train on more data rather than more epochs to dilute noise
- Consider label smoothing (0.05) if training loss plateaus but validation loss increases

---

## 9. Comparison with Other Deep ER Approaches

| Approach | Architecture | Key Difference from Ditto | Pros | Cons |
|----------|-------------|--------------------------|------|------|
| **Ditto** | BERT [CLS] → classifier | Serialization + 3 optimizations | Simple, effective, well-studied | Slow inference |
| **EMBER** (Kannan et al.) | Bi-encoder + interaction layer | Separate encoders, then merge | Faster (can cache embeddings) | Lower quality than cross-encoder |
| **JointBERT** | BERT with attribute-level attention | Multi-head attention per column | Better attribute alignment | More complex training |
| **PromptEM** (Wang et al., 2022) | Prompt-based LLM matching | Uses LLM prompting, no fine-tuning | Zero-shot capable | Slow, expensive, hard to threshold |
| **CollabER** | Collaborative ER with foundation models | Multiple foundation models vote | Ensemble-robust | Compute-heavy |
| **Sudowoodo** (Wang et al., 2023) | Contrastive pre-training for ER | Self-supervised pre-training | Better with less labeled data | Needs pre-training step |

**For our contest:** Ditto is the sweet spot — well-understood, fast to implement, and proven on company matching. Bi-encoder (EMBER-style) could be used for blocking but not for the final scorer.

---

## 10. Implementation Roadmap

1. **Write `ditto_scorer.py`** — serialization + cross-encoder wrapper
2. **Generate training pairs** from existing holdout with BM25 candidates + ground truth labels
3. **Fine-tune** `distilbert-base-multilingual-cased` with `CrossEncoderTrainer`
4. **Evaluate on holdout** — compare Ditto-only, LightGBM-only, and ensemble
5. **Integrate** — add `ditto_score` as feature or cascade replacement
6. **Batch inference** on test set with progress tracking

---

## 11. References

1. Li, Y., et al. (2020). "Deep Entity Matching with Pre-Trained Language Models." VLDB.
2. Brunner, U., & Stockinger, K. (2020). "Entity Matching with Transformer Architectures — A Step Forward in Data Integration." EDBT.
3. Peeters, R., & Bizer, C. (2021). "Dual-Objective Fine-Tuning of BERT for Entity Matching." VLDB.
4. Wang, R., et al. (2022). "PromptEM: Prompt-tuning for Low-resource Generalized Entity Matching." VLDB.
5. Wang, R., et al. (2023). "Sudowoodo: Contrastive Self-supervised Learning for Multi-purpose Data Integration and Preparation." ICDE.
6. Primpeli, A., & Bizer, C. (2020). "Profiling Entity Matching Benchmark Tasks." CIKM.
7. sentence-transformers documentation: https://www.sbert.net/docs/cross_encoder/
