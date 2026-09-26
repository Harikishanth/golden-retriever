# The BERT Family for Entity Resolution: A Model-by-Model Analysis

**Author:** Research artifact for Amazon ML Challenge 2026 — Da Big Three  
**Date:** 2026-09-26  
**Purpose:** Exhaustive survey of every BERT-family model relevant to entity resolution, ranked by suitability for our contest constraints (≤8B params, MIT/Apache 2.0 license, 72h time budget, multilingual French requirement). Covers architecture differences, ER-specific strengths, inference speed, and concrete recommendations.

---

## 1. How Transformers Are Used in ER

There are three paradigms for using transformer models in entity resolution:

### 1.1 Bi-Encoder (Embedding Model)

```
Entity A → Encoder → embedding_a ─┐
                                    ├─→ cosine(a, b) → similarity score
Entity B → Encoder → embedding_b ─┘
```

- Each entity encoded independently — embeddings can be **cached**
- Similarity = cosine/dot product of embeddings
- Used for **blocking** (find top-K nearest neighbors via FAISS)
- Fast: O(1) per query after index built
- Quality: moderate — no cross-attention between entities

### 1.2 Cross-Encoder (Pair Classifier)

```
[CLS] Entity A [SEP] Entity B [SEP] → Encoder → [CLS] embedding → classifier → match/no-match
```

- Both entities processed **together** — full cross-attention
- Used for **matching** (score candidate pairs)
- Slow: O(n) per candidate — cannot cache
- Quality: high — the model sees both entities simultaneously

### 1.3 Hybrid (Bi-Encoder Blocking → Cross-Encoder Scoring)

```
Bi-encoder → top-100 candidates → Cross-encoder → final matches
```

This is the standard modern ER pipeline and what we're building.

---

## 2. The BERT Family Tree

```
BERT (2018, 110M/340M, Google)
├── DistilBERT (2019, 66M, HuggingFace) — distilled, 60% faster
├── RoBERTa (2019, 125M/355M, Meta) — better pre-training
│   └── XLM-RoBERTa (2019, 125M/355M, Meta) — multilingual RoBERTa
├── ALBERT (2019, 12M/18M, Google) — parameter sharing
├── ELECTRA (2020, 14M/110M/335M, Google) — replaced token detection
├── DeBERTa (2020, 100M/350M/900M, Microsoft) — disentangled attention
│   ├── DeBERTa-v2 (2021, up to 1.5B)
│   └── DeBERTa-v3 (2021, 86M/184M/304M) — ELECTRA-style + disentangled
├── mBERT (2018, 110M, Google) — multilingual BERT, 104 languages
├── ModernBERT (2024, 149M/395M, Answer.AI) — modernized architecture
└── Specialized for ER:
    └── Ditto fine-tunes (2020, BERT-base/DistilBERT/RoBERTa)
```

---

## 3. Model-by-Model Deep Dive

### 3.1 BERT-base-uncased (The Original)

| Property | Value |
|----------|-------|
| **Params** | 110M |
| **Hidden dim** | 768 |
| **Layers** | 12 |
| **Heads** | 12 |
| **Max length** | 512 |
| **License** | Apache 2.0 |
| **Multilingual** | No (English only) |
| **Pre-training** | MLM + Next Sentence Prediction on BookCorpus + English Wikipedia |

**ER Strengths:**
- Well-studied; Ditto paper's primary model
- Strong English understanding of abbreviations ("Corp" = "Corporation")
- NSP pre-training gives it some pair-matching intuition

**ER Weaknesses:**
- English only — **cannot handle France (15% of test)**
- Slower than DistilBERT with marginal quality gain for short inputs
- NSP objective now considered suboptimal (RoBERTa dropped it)

**Speed (T4 GPU):** ~800 pairs/sec (batch=64, max_length=128)

**Verdict for our contest:** ✗ Skip. DistilBERT-multilingual is better on every axis.

---

### 3.2 DistilBERT (The Fast Workhorse)

| Property | Value |
|----------|-------|
| **Params** | 66M (uncased) / 134M (multilingual-cased) |
| **Hidden dim** | 768 |
| **Layers** | 6 |
| **Heads** | 12 |
| **Max length** | 512 |
| **License** | Apache 2.0 |
| **Multilingual** | `distilbert-base-multilingual-cased` covers 104 languages |

**Architecture:** Knowledge distillation from BERT-base. Removes every other layer (12 → 6), removes token-type embeddings and the pooler. Trained with triple loss: MLM + distillation + cosine embedding.

**ER Strengths:**
- 60% faster than BERT-base with 97% of BERT's quality on GLUE
- Multilingual variant handles French, English, Hindi natively
- Small enough to fine-tune on Kaggle T4 in ~25 min
- Apache 2.0 — fully contest-legal
- The Ditto paper showed DistilBERT achieves 95-98% of BERT-base's ER F1

**ER Weaknesses:**
- 6 layers means less capacity for subtle distinctions
- Multilingual variant dilutes per-language quality (104 languages share 134M params)

**Speed (T4 GPU):** ~1200 pairs/sec (batch=64, max_length=128)

**Verdict for our contest:** ✓ **Strong candidate.** `distilbert-base-multilingual-cased` is the default recommendation — fast, multilingual, Apache 2.0, proven in Ditto.

---

### 3.3 RoBERTa (The Better Pre-Trained BERT)

| Property | Value |
|----------|-------|
| **Params** | 125M (base) / 355M (large) |
| **Hidden dim** | 768 / 1024 |
| **Layers** | 12 / 24 |
| **Heads** | 12 / 16 |
| **Max length** | 512 |
| **License** | MIT |
| **Multilingual** | No (English only); see XLM-RoBERTa for multilingual |

**Architecture:** Same as BERT but with optimized pre-training:
- Removed NSP objective (just MLM)
- Dynamic masking (different mask each epoch)
- Larger batches (8K sequences)
- More data (160GB text vs BERT's 16GB)
- Byte-Pair Encoding (BPE) tokenizer instead of WordPiece

**ER Strengths:**
- Consistently outperforms BERT-base on NLU benchmarks (+2-3 points)
- Better representations for entity names due to more diverse pre-training data
- MIT license — even more permissive than Apache 2.0

**ER Weaknesses:**
- English only — **fails on France**
- Same speed as BERT-base (same architecture)
- Marginal gain may not justify the French limitation

**Speed (T4 GPU):** ~800 pairs/sec

**Verdict for our contest:** ✗ Skip base. English-only kills it. BUT see XLM-RoBERTa below.

---

### 3.4 XLM-RoBERTa (Multilingual RoBERTa)

| Property | Value |
|----------|-------|
| **Params** | 125M (base) / 355M (large) |
| **Hidden dim** | 768 / 1024 |
| **Layers** | 12 / 24 |
| **Heads** | 12 / 16 |
| **Max length** | 512 |
| **License** | MIT |
| **Multilingual** | 100 languages including French, English, Hindi |

**Architecture:** RoBERTa pre-trained on 2.5TB of CommonCrawl data in 100 languages. Uses SentencePiece tokenizer.

**ER Strengths:**
- **Best multilingual encoder in the BERT family** for its size
- French, English, Hindi all well-represented in pre-training
- Cross-lingual transfer: ER patterns learned in English transfer to French
- MIT license
- Well-supported in sentence-transformers ecosystem

**ER Weaknesses:**
- Same speed as BERT-base (12 layers)
- 125M params is 2x DistilBERT — longer fine-tuning
- SentencePiece tokenizer may handle business names slightly differently than WordPiece

**Speed (T4 GPU):** ~800 pairs/sec (base) / ~300 pairs/sec (large)

**Verdict for our contest:** ✓ **Top candidate.** If DistilBERT-multilingual isn't good enough, XLM-RoBERTa-base is the upgrade. MIT license, strong multilingual, proven in cross-lingual NLU.

---

### 3.5 DeBERTa-v3 (The Accuracy Champion)

| Property | Value |
|----------|-------|
| **Params** | 86M (small) / 184M (base) / 304M (large) |
| **Hidden dim** | 768 / 768 / 1024 |
| **Layers** | 6 / 12 / 24 |
| **Heads** | 12 / 12 / 16 |
| **Max length** | 512 |
| **License** | MIT |
| **Multilingual** | `mdeberta-v3-base` (276M, 100+ languages) |

**Architecture:** Two key innovations over BERT:
1. **Disentangled Attention:** Separates content and position embeddings. Content-to-content, content-to-position, and position-to-content attention are computed separately. This helps the model understand that "McDonald's at 123 Main St" and "123 Main St McDonald's" refer to the same entity despite different word positions.
2. **Enhanced Mask Decoder:** Uses absolute position in the decoding layer for better token prediction.

DeBERTa-v3 adds **ELECTRA-style pre-training** (replaced token detection instead of MLM) for more efficient use of training compute.

**ER Strengths:**
- **Highest accuracy among BERT-family models** on SuperGLUE, GLUE, and NLI tasks
- Disentangled attention is particularly good for ER: word order shouldn't matter for matching
- DeBERTa-v3-small (86M, 6 layers) is comparable to BERT-base quality at DistilBERT speed
- mDeBERTa-v3-base handles multilingual natively
- MIT license
- Amazon ML Challenge 2025 top-10 text-only approach used DeBERTa-v3

**ER Weaknesses:**
- Less community cross-encoder support than BERT/RoBERTa (but growing)
- `mdeberta-v3-base` at 276M is larger than XLM-RoBERTa-base
- Slightly more complex to fine-tune (disentangled attention has more hyperparameters)

**Speed (T4 GPU):** ~1100 pairs/sec (small) / ~700 pairs/sec (base)

**Verdict for our contest:** ✓ **Best quality option.** `microsoft/mdeberta-v3-base` for multilingual or `microsoft/deberta-v3-small` for speed. The disentangled attention is a genuine advantage for ER where word order is noise.

---

### 3.6 mBERT (Multilingual BERT)

| Property | Value |
|----------|-------|
| **Params** | 110M |
| **Hidden dim** | 768 |
| **Layers** | 12 |
| **Heads** | 12 |
| **Max length** | 512 |
| **License** | Apache 2.0 |
| **Multilingual** | 104 languages |

**Architecture:** Standard BERT-base trained on the concatenation of Wikipedia dumps in 104 languages. No explicit cross-lingual alignment — emergent cross-lingual ability from shared vocabulary.

**ER Strengths:**
- Handles French, English, Hindi
- Apache 2.0
- Well-studied for cross-lingual transfer

**ER Weaknesses:**
- **Superseded by XLM-RoBERTa** in every benchmark
- Worse cross-lingual transfer than XLM-R (trained on less multilingual data)
- Same speed as BERT-base

**Speed (T4 GPU):** ~800 pairs/sec

**Verdict for our contest:** ✗ Skip. XLM-RoBERTa is strictly better.

---

### 3.7 ALBERT (The Tiny Giant)

| Property | Value |
|----------|-------|
| **Params** | 12M (base) / 18M (large) / 60M (xlarge) / 235M (xxlarge) |
| **Hidden dim** | 768 / 1024 / 2048 / 4096 |
| **Layers** | 12 / 24 / 24 / 12 |
| **License** | Apache 2.0 |
| **Multilingual** | No |

**Architecture:** Two parameter-reduction techniques:
1. **Factorized embedding parameterization:** Decomposes embedding matrix into two smaller matrices
2. **Cross-layer parameter sharing:** All 12 layers share the same weights

ALBERT-xxlarge has fewer parameters than BERT-large but more compute (each layer is expensive despite weight sharing).

**ER Strengths:**
- Very small model files (12M for base)
- Competitive quality despite tiny parameter count

**ER Weaknesses:**
- English only
- Despite few params, inference is **not faster** (same number of layers, same FLOPs)
- Weight sharing limits capacity for learning ER-specific patterns
- Less community support for ER tasks

**Speed (T4 GPU):** ~800 pairs/sec (same FLOPs as BERT despite fewer params)

**Verdict for our contest:** ✗ Skip. No speed advantage, no multilingual, less capacity.

---

### 3.8 ELECTRA (The Efficient Pre-Trainer)

| Property | Value |
|----------|-------|
| **Params** | 14M (small) / 110M (base) / 335M (large) |
| **Hidden dim** | 256 / 768 / 1024 |
| **Layers** | 12 / 12 / 24 |
| **License** | Apache 2.0 |
| **Multilingual** | No |

**Architecture:** Instead of MLM (mask 15% of tokens, predict them), ELECTRA uses a **generator-discriminator** setup:
1. Small generator fills in masked tokens
2. Discriminator (the actual model) classifies **every** token as original or replaced

This means ELECTRA learns from every token (not just 15%), making it ~4x more sample-efficient.

**ER Strengths:**
- ELECTRA-small (14M) matches BERT-base on some tasks — extreme efficiency
- Better at detecting subtle word-level differences (its pre-training task is literally "is this token correct?")
- This is relevant for ER: detecting typos, substitutions, abbreviations

**ER Weaknesses:**
- English only
- Less community fine-tuning support for cross-encoder tasks
- The advantage is mainly in pre-training efficiency, not inference speed

**Speed (T4 GPU):** ~800 pairs/sec (base), ~2500 pairs/sec (small)

**Verdict for our contest:** ✗ Interesting architecture but English-only kills it. DeBERTa-v3 (which incorporates ELECTRA-style training) is the better choice.

---

### 3.9 ModernBERT (The 2024 Refresh)

| Property | Value |
|----------|-------|
| **Params** | 149M (base) / 395M (large) |
| **Hidden dim** | 768 / 1024 |
| **Layers** | 22 / 28 |
| **Heads** | 12 / 16 |
| **Max length** | 8192 |
| **License** | Apache 2.0 |
| **Multilingual** | Primarily English (some code) |

**Architecture:** A from-scratch redesign of BERT incorporating 6 years of transformer research:
1. **Rotary Position Embeddings (RoPE):** Instead of learned absolute positions — better generalization to different sequence lengths
2. **Flash Attention 2:** Hardware-efficient attention implementation
3. **GeGLU activations:** Replace GELU — better gradient flow
4. **Alternating attention:** Local attention on most layers, global on every 3rd
5. **Unpadding:** Removes padding tokens from computation — significant speedup on variable-length inputs
6. **8192 context length:** Much longer than BERT's 512

**ER Strengths:**
- ~2x faster than BERT-base at the same quality (Flash Attention + unpadding)
- 8192 context is overkill for ER (our pairs are ~80 tokens) but means no truncation ever
- Outperforms RoBERTa-base on most benchmarks
- Apache 2.0
- Already available as cross-encoder in sentence-transformers (`tomaarsen/reranker-modernbert-base-msmarco-margin-mse`)

**ER Weaknesses:**
- **English-focused** — not trained on multilingual data like XLM-R
- Very new (late 2024) — less community fine-tuning for ER specifically
- 22 layers (vs 12 for BERT-base) means longer per-sample computation despite Flash Attention gains
- No multilingual variant yet

**Speed (T4 GPU):** ~1500 pairs/sec (base, with Flash Attention)

**Verdict for our contest:** ⚠️ Best English model but no French support. Could be used for US+India entities only with a separate multilingual model for France. Or wait for multilingual ModernBERT.

---

### 3.10 Sentence-Transformers Cross-Encoder Pretrained Models

These are models already fine-tuned for pair scoring, ready to use:

| Model | Base | Params | License | NDCG@10 (TREC DL 2019) | Throughput (V100) |
|-------|------|--------|---------|------------------------|-------------------|
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | TinyBERT | 4.4M | Apache 2.0 | 69.84 | 9000 docs/s |
| `cross-encoder/ms-marco-MiniLM-L-2-v2` | MiniLM | 6.7M | Apache 2.0 | 71.01 | 7500 docs/s |
| `cross-encoder/ms-marco-MiniLM-L-4-v2` | MiniLM | 19M | Apache 2.0 | 73.04 | 4000 docs/s |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | MiniLM | 22M | Apache 2.0 | 74.30 | 3400 docs/s |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | MiniLM | 33M | Apache 2.0 | 74.31 | 1800 docs/s |
| `cross-encoder/ms-marco-electra-base` | ELECTRA | 110M | Apache 2.0 | 71.99 | 600 docs/s |
| `tomaarsen/reranker-modernbert-base-msmarco` | ModernBERT | 149M | Apache 2.0 | 74.50+ | ~1500 docs/s |
| `BAAI/bge-reranker-v2-m3` | XLM-R | 568M | MIT | — | ~200 docs/s |

**Key insight:** The MiniLM models are **distilled** from larger models (not just smaller architectures). `ms-marco-MiniLM-L-6-v2` at 22M params achieves 99.7% of the 12-layer model's quality.

**For our contest:** `cross-encoder/ms-marco-MiniLM-L-6-v2` is the fastest viable option for English. `BAAI/bge-reranker-v2-m3` handles multilingual but is 25x slower.

---

## 4. Head-to-Head Comparison for Our Task

### 4.1 Ranking by Suitability

| Rank | Model | Params | French? | License | Speed | Quality | Notes |
|------|-------|--------|---------|---------|-------|---------|-------|
| **1** | `microsoft/mdeberta-v3-base` | 276M | ✓ | MIT | 500/s | Highest | Disentangled attention helps ER |
| **2** | `xlm-roberta-base` | 125M | ✓ | MIT | 800/s | High | Best multilingual for size |
| **3** | `distilbert-base-multilingual-cased` | 134M | ✓ | Apache 2.0 | 1200/s | Good | Fast, proven in Ditto |
| **4** | `BAAI/bge-reranker-v2-m3` | 568M | ✓ | MIT | 200/s | Highest | Pre-trained reranker, but slow |
| **5** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | ✗ | Apache 2.0 | 3400/s | Good | Fastest, but English-only |
| **6** | `tomaarsen/reranker-modernbert-base` | 149M | ✗ | Apache 2.0 | 1500/s | High | Best English, no French |

### 4.2 Speed vs Quality Trade-off for Test Inference

Test set: 1.73M S1 × 50 candidates = 86.5M pairs

| Model | Pairs/sec | Total Time | Feasible in 72h? |
|-------|-----------|------------|-------------------|
| MiniLM-L-6-v2 | 3400 | 7.1 hours | ✓ Easily |
| DistilBERT-multilingual | 1200 | 20 hours | ✓ Tight |
| XLM-RoBERTa-base | 800 | 30 hours | ⚠️ Needs 2 GPUs |
| mDeBERTa-v3-base | 500 | 48 hours | ✗ Too slow for 50 cands |
| BGE-reranker-v2-m3 | 200 | 120 hours | ✗ Impossible |

**Key decision:** If we reduce to top-20 candidates (35M pairs):

| Model | Total Time (20 cands) | Feasible? |
|-------|----------------------|-----------|
| MiniLM-L-6-v2 | 2.9 hours | ✓ |
| DistilBERT-multilingual | 8.1 hours | ✓ |
| XLM-RoBERTa-base | 12.2 hours | ✓ |
| mDeBERTa-v3-base | 19.4 hours | ✓ Tight |

---

## 5. Architecture Details That Matter for ER

### 5.1 Tokenization Impact

Different tokenizers split entity names differently:

```
Input: "McDonald's Corporation"

WordPiece (BERT/DistilBERT):  ["mc", "##don", "##ald", "'", "s", "corporation"]
BPE (RoBERTa/XLM-R):         ["Mc", "Don", "ald", "'s", "ĠCorpor", "ation"]
SentencePiece (DeBERTa):      ["▁Mc", "Donald", "'s", "▁Corporation"]
```

SentencePiece (DeBERTa) keeps "Donald" as one token — better for matching "McDonald" vs "MacDonald". BPE and WordPiece fragment more aggressively.

For French: "Boulangerie Pâtisserie"
```
WordPiece (mBERT):     ["boulanger", "##ie", "pat", "##iss", "##erie"]
SentencePiece (XLM-R): ["▁Boulangerie", "▁Pâtisserie"]  
```

XLM-RoBERTa keeps full French words as single tokens much more often than mBERT — a significant advantage for understanding meaning.

### 5.2 Positional Encoding

| Model | Position Type | Impact on ER |
|-------|--------------|-------------|
| BERT/DistilBERT | Learned absolute | Position-sensitive — "Name Addr" ≠ "Addr Name" |
| RoBERTa/XLM-R | Learned absolute | Same as BERT |
| DeBERTa | **Relative (disentangled)** | Position-insensitive — "Name Addr" ≈ "Addr Name" |
| ModernBERT | **Rotary (RoPE)** | Smooth distance-based — good for local context |

**DeBERTa's disentangled attention is specifically helpful for ER** where the same information may appear in different positions ("John's Pizza, 123 Main St" vs "123 Main Street, John's Pizza").

### 5.3 Cross-Attention Pattern

In a cross-encoder, the model sees: `[CLS] Entity A [SEP] Entity B [SEP]`

The attention pattern reveals how the model compares entities:
- **BERT/RoBERTa:** All tokens attend to all other tokens (full attention). Name tokens from Entity A naturally attend to name tokens from Entity B.
- **ModernBERT:** Alternating local/global attention. Local layers (most) only see nearby tokens — but entity pairs are short enough (~80 tokens) that this rarely matters.
- **DeBERTa:** Content-to-content attention + content-to-position + position-to-content. The model can learn that "the 3rd token of Entity A should attend to the 3rd token of Entity B" (positional alignment) independently from "the word 'McDonald' should attend to 'MacDonald'" (content alignment).

### 5.4 [CLS] Token vs Mean Pooling

Cross-encoders typically use the `[CLS]` token embedding for classification:
- **BERT/DistilBERT:** `[CLS]` is trained via NSP to aggregate sentence-pair information
- **RoBERTa:** No NSP → `[CLS]` is less specialized; mean pooling sometimes better
- **DeBERTa:** Uses enhanced mask decoder → `[CLS]` is well-informed
- **ModernBERT:** Supports both `"cls"` and `"mean"` pooling (configurable)

For cross-encoder ER, `[CLS]` pooling is standard and works well with all models.

---

## 6. Fine-Tuning Recipes for ER

### 6.1 DistilBERT-Multilingual (Recommended Starting Point)

```python
from sentence_transformers.cross_encoder import (
    CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
import torch

model = CrossEncoder("distilbert-base-multilingual-cased", max_length=128)

args = CrossEncoderTrainingArguments(
    output_dir="./er-distilbert-multi",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
)

loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(5.0))
# ... train with CrossEncoderTrainer
```

### 6.2 mDeBERTa-v3-base (Higher Quality)

```python
model = CrossEncoder("microsoft/mdeberta-v3-base", max_length=128)

args = CrossEncoderTrainingArguments(
    output_dir="./er-mdeberta-v3",
    num_train_epochs=5,          # DeBERTa benefits from more epochs
    per_device_train_batch_size=32,  # Larger model → smaller batch
    learning_rate=2e-5,          # Slightly lower LR for DeBERTa
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=2,  # Effective batch = 64
)
```

### 6.3 Two-Model Ensemble (Best Quality)

```python
# Model 1: Fast English-focused
model_en = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# Further fine-tune on ER pairs...

# Model 2: Multilingual (handles France)
model_multi = CrossEncoder("microsoft/mdeberta-v3-base")
# Fine-tune on ER pairs...

# At inference:
def ensemble_score(pair, country):
    score_en = model_en.predict([pair])[0]
    score_multi = model_multi.predict([pair])[0]
    
    if country == "France":
        return score_multi  # Trust multilingual for French
    else:
        return 0.6 * score_en + 0.4 * score_multi  # Blend for US/India
```

---

## 7. Late Fusion: Transformer + Feature Engineering

Instead of replacing LightGBM, use the transformer as a **feature extractor**:

```python
# Extract [CLS] embedding from cross-encoder
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")

def extract_pair_embedding(text_a, text_b):
    inputs = tokenizer(text_a, text_b, return_tensors="pt",
                       truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
    return cls_embedding.numpy().flatten()

# Add as features to LightGBM
# Option 1: Full 768-dim embedding (may overfit with small training set)
# Option 2: PCA to 32 dims (more practical)
# Option 3: Just the cross-encoder score (1 feature — simplest)
```

**Recommendation:** Option 3 (just the score as 1 feature) is the right choice for our contest. The cross-encoder score captures semantic similarity; LightGBM handles the feature interactions with string-similarity features.

---

## 8. Practical Recommendations

### For Our Contest (72h, Kaggle T4, France 15%)

**Primary model:** `distilbert-base-multilingual-cased` fine-tuned as cross-encoder
- Fast enough for test inference (~20h for 86M pairs, or ~8h with top-20 candidates)
- Handles French natively
- Apache 2.0
- Proven in Ditto paper

**Upgrade if time allows:** `microsoft/mdeberta-v3-base`
- Higher quality due to disentangled attention
- Slower — only feasible with top-20 candidates (~19h)
- MIT license

**Speed hack:** `cross-encoder/ms-marco-MiniLM-L-6-v2` for US+India (3400 pairs/sec), `distilbert-base-multilingual-cased` for France only
- Reduces total inference time by ~60%
- Requires country-conditional model routing

**Do NOT use:** BERT-base, mBERT, ALBERT, ELECTRA (all superseded by better options)

---

## 9. Key Takeaways

1. **DeBERTa-v3 is the quality leader** — disentangled attention is genuinely helpful for ER where word order is noise
2. **DistilBERT-multilingual is the speed/quality sweet spot** — 97% of BERT quality at 60% compute, handles French
3. **XLM-RoBERTa is the multilingual standard** — if DistilBERT isn't good enough, go here
4. **ModernBERT is the future** — best English model but no multilingual variant yet
5. **Pre-trained cross-encoders** (ms-marco-MiniLM) save training time but need ER fine-tuning for best results
6. **Tokenization matters** — SentencePiece (DeBERTa/XLM-R) handles French and business names better than WordPiece (BERT)
7. **Late fusion (transformer score as LightGBM feature)** is simpler and often better than replacing LightGBM entirely
8. **Test inference time is the binding constraint** — choose model size based on how many pairs you need to score, not just quality

---

## 10. References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL.
2. Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." NeurIPS Workshop.
3. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv.
4. Conneau, A., et al. (2020). "Unsupervised Cross-lingual Representation Learning at Scale." ACL.
5. He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." ICLR.
6. He, P., et al. (2021). "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing." arXiv.
7. Clark, K., et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators." ICLR.
8. Lan, Z., et al. (2020). "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations." ICLR.
9. Warner, B., et al. (2024). "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference." arXiv (ModernBERT).
10. Li, Y., et al. (2020). "Deep Entity Matching with Pre-Trained Language Models." VLDB.
11. Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP.
