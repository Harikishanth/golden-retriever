# Reddit 004 — 2025 implementations discussion (r/learnmachinelearning)

- **URL:** https://www.reddit.com/r/learnmachinelearning/comments/1o696pn/amazon_ml_challenge_2025_implementations/
- **Subreddit:** r/learnmachinelearning
- **Author:** u/zarouz (OP)
- **Posted:** during/after 2025 contest, asking teams with SMAPE **< 45** to share methods
- **Flair:** Discussion
- **Score:** ~7 upvotes / 12 comments
- **Captured:** 2026-09-15 from user paste (thread may have more unreplied questions)

## OP (zarouz) — stalled at 48 local / 50 LB

Questions they wanted answered (good Day-1 checklist for any year):

1. Approach?
2. Feature engineering?
3. Failed experiments, and what transferred?
4. How to tell if the bottleneck is **features vs architecture**?
5. Performance on **sparse expensive items**?

**Their stack (did not break 45):**

- RNN on text
- Text **and** image embeddings
- Categorized food into sets with **BART**
- Local: **48 SMAPE** on a 15k holdout
- Leaderboard: **50**

Val 48 → LB 50 is an honest ~2 point gap (worse than the 0.1–0.8 gaps reported by stronger teams).

They later suspected **image embeddings added noise**, after seeing a 42 text-only result.

**Unanswered in this paste:** failed-experiment transfer, feature vs arch diagnosis, expensive-item tail. Still open research questions.

---

## yashBhaskar — **42 SMAPE, text only, almost no FE**

Simplest high-scoring recipe in the whole research pile so far:

1. Take a **good pretrained open-source embedding model**
2. Feed the **entire `catalog_content` as-is** — **no preprocessing**
3. Train a **regression head**
4. **Text only** — no images

- Embedding size: **~150M params** (not named)
- Someone asked “is it Mistral?” — **unanswered**. 150M is far below Mistral-7B; more likely an E5/BGE/GTE/BERT-scale encoder.
- Score: **42** (inside the public top-50 band)

**Implication:** aggressive FE (BART food clusters, RNNs) is not required and can lose to a strong frozen/finetuned sentence encoder + head. Images are optional; they can **hurt** if the visual encoder is weak or the fusion is noisy.

This lines up with:

- Unlucky_Chocolate_34 text-only DeBERTa **40.x** (reddit-002)
- WontLetMeKnow text-only LSTM **45** (weaker encoder)
- SPAM_LLMs: frozen strong embeddings beat rushed fine-tunes (but they *did* use images with better vis encoders)

---

## Apprehensive-Talk971 — log-target + triplet + kNN

Two separate ideas:

### 1. Target

> Main big thing imo was regress on **log of prices**; log prices follow a very good distribution.

OP agreed: log-price handled skew well. Recurs across 2023 length and 2025 price.

### 2. Architecture

- **3 language models + 1 vision model**
- Fine-tuned with **triplet loss**
- Then **kNN on those embeddings**
- Then a **regressor on top** of the kNN (and/or the embeddings)

The 3 text models, named:

1. **Qwen**
2. **multilingual DistilUSE** (`distiluse` — typically `sentence-transformers/distiluse-base-multilingual-cased-v*`)
3. **CLIP text** encoder

Vision: “+1 vis” not named (CLIP image is the obvious pair).

Also asked when the **final leaderboard** drops (same comms fog as reddit-002/003).

**Why this is interesting:** metric learning (triplet) + retrieval (kNN) treats price as “similar catalogs have similar prices,” which is a natural prior for pack-size / brand neighbors. Complements a global regressor. 2023 2nd place also rounded to nearest train lengths — same neighborhood idea.

Score **not stated** in this paste. They were answering in a **< 45** thread, so treat as “claimed competitive,” not a number.

---

## Transfer to playbook

| Do | Don't |
|---|---|
| Strong ~100–150M+ text embedding + log-price head, catalog dumped raw | Assume you need RNN / BART topic clusters |
| Ablate **images off** on Day 1; keep them only if holdout improves | Concatenate a random ViT/ResNet and hope |
| Log-price (or log1p) from the first baseline | Train SMAPE on raw price with a skewed target |
| Try embedding kNN / neighbor features as well as a regressor | Only one global MLP |
| Diagnose expensive tail separately (OP asked; nobody answered here) | Optimize mean SMAPE only and ignore high-price SKUs |

## Open questions this thread did not answer

- Named 150M embedding model
- Whether that 42 was frozen encoder or encoder+head FT
- Triplet/kNN actual SMAPE
- Feature vs architecture bottleneck method
- Sparse expensive-item tricks
