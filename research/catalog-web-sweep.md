# Web sweep — GitHub, blogs, LinkedIn, extra Reddit (15 Sep 2026)

What this pass could and could not do:

- **Could:** public GitHub READMEs, Medium titles/snippets, LinkedIn post snippets, Unstop fragments, extra Reddit threads.
- **Could not:** log into LinkedIn, scrape private Unstop, or fetch some Medium pages (timeout). Rank-1 **finale winner 2025 is still unnamed** in public writeups. Ranks 1–3 on the Unstop HTML table were also hidden in the scrape (table starts at rank 4).

---

## Extra Reddit (not in reddit-001…006)

| Thread | Year | Why it matters |
|---|---|---|
| https://www.reddit.com/r/learnmachinelearning/comments/1fhzz7a/solutions_of_amazon_ml_challenge/ | **2024** | Same post as reddit-006 (now we have the URL) |
| https://www.reddit.com/r/learnmachinelearning/comments/1ffddqt/amazon_ml_challenge/ | 2024 live | “0.8 F1 rumor”; OCR vs “is this even ML”; F1 showing 0 despite 0.43; stream images one-by-one and delete |
| https://www.reddit.com/r/learnmachinelearning/comments/1fhaamv/can_anyone_help_me_with_code_which_should_include/ | 2024 | Official F1 definition (TP if value+unit match allowed map); `sanity.py` / `constants.py` |
| https://www.reddit.com/r/hackathon/comments/1nzfikg/amazon_ml_challenge_2025_join_now/ | 2025 announce | Finale **17 Oct 2025 virtual**; prizes 1L/75k/50k |

---

## 2021 — browse-node classification

| Place | Team | Score / notes | Link |
|---|---|---|---|
| **1st runner-up** | DeVaSh.Ai | BERT/sentence-transformers/kNN stack; Twitch talk | https://github.com/DebarshiChanda/Amazon-ML-Challenge2021 — video https://www.twitch.tv/videos/1107585685?t=4h18m36s |
| 16 | Panaroma | multilingual BERT 3 epochs, 67.465 public | https://github.com/nikhil6041/AmazonMLChallenge2021 |
| ~top 5% | off-by-one-bit | RoBERTa 67.23% / DistilBERT 59% | https://github.com/jharkawat/Amazon_ml_challenge |
| 26 / 3290 | — | SBERT + RAPIDS kNN ensemble 66.7% | https://github.com/pranshurastogi29/Amazon_ml_challenge-solution |
| 296 / 3290 | — | browse-node starter dump | https://github.com/akshatprogrammer/Amazon-ML-Challenge |

HackerEarth. ~3k teams. Metric accuracy. 2.9M train / 9.9k classes.

---

## 2023 — product length (MAPE-style)

Official score formula from VectorNd: `max(0, 100*(1-MAPE))`.

| Place | Team | Notes | Link |
|---|---|---|---|
| **Winner** | ART (PJ, Ansh, Chaitanya, Harshit) | #1 in 1.5 days, Kaggle only, ~24–25k people | https://github.com/pj-mathematician/Amazon-ML-Challenge-2023 |
| **2nd** | Fishes | BERT+RoBERTa + type embedding, min-ensemble, nearest-length | https://github.com/TashvikDhamija/AmazonML https://github.com/greenfish8090/AmazonML Twitch 1:33:50 of https://www.twitch.tv/videos/1804684510 |
| — | VectorNd | char/word GRU + type emb, log+PowerTransformer | https://github.com/VectorNd/Amazon-ML-Challenge-2023 |

Kaggle data mirror: https://www.kaggle.com/datasets/vinayak4444/amazon-ml-challenge-2023

LinkedIn: Poojan, Ansh Tanwar, Chaitanya Giri winner posts.

---

## 2024 — entity from images (F1)

Scale: Medium/GitHub cite **~74,800 teams/participants** (Aman Prakash: 74,850). Earlier Reddit said ~18k — use **~75k registrations** as the cited figure.

| Place | Team | F1 | Stack | Link |
|---|---|---|---|---|
| Claims **winning code** | KhadgaA | 0.617 → **0.865** on curated 1.6k | Qwen2-VL-7B QLoRA LLaMA-Factory | https://github.com/KhadgaA/Amazon-ML-Challenge (fork: nachiketashunya) |
| **Finale 6th** | DBkaScam | **71.8** ensemble | MiniCPM-2.6 ZSP + Qwen2-VL-7B FSL + SFT; vote | https://github.com/arnav10goel/Amazon-ML-Challenge-24 — Canva: https://www.canva.com/design/DAGRau30tRI/06v7kPdBwb99GDjsiv1fcg/edit |
| **22 / 74,843** | Tensor Titans (BITS Goa) | **0.679** with only 39% rows filled | Paddle + custom line detect; empty depth | nirvan840/Product-Image-Attribute-and-Entity-Extraction |
| ~top 50 | — | 0.616 | Idefics-2 8B | https://github.com/hwaseem04/Amazon-ML-Challenge-2024 |
| — | — | 0.627 | Qwen2-VL-**2B** + postprocess, A6000 | VishalTheHuman/Amazon-ML-Challenge-2024 |
| 357 / 74,850 | Q* | low | MoonDream + regex; free GPU died | https://medium.com/@aman_prakash/how-we-secured-357th-spot-among-74-850-in-amazon-ml-challenge-2024-aef31f4e31b0 |
| — | Vishesh Rawal | blog | entity extraction overview | https://visheshrwl.medium.com/amazon-ml-challenge-2024-solution-ac3013a81a0e |
| disaster | smv et al. | **0.00016** | 1k/130k rows | https://medium.com/@the.smv/from-heroes-to-zeroes-my-amazon-ml-challenge-2024-experience-c31e29dd7245 |

DBkaScam ablation (F1):

- MiniCPM zero-shot 66.2 → +postproc **69.3**
- Qwen2 few-shot **70.9**
- Qwen2 SFT alone 64.8 (SFT *without* FSL/ZSP was worse than few-shot)
- Best: **SFT + FSL + ZSP = 71.8**

Postproc they named: fractions→decimal; `'`/`"` → feet/inches; ranges take **max**; illegal units stripped.

PJ (ART): Qwen-VL + Donut, LB #5, **runners-up after talks**.

Kaggle dump: https://www.kaggle.com/datasets/abhishekgautam12/amazon-ml-challenge-2024

Prep guide: https://github.com/KushalVijay/Amazon-ML-Challenge-Guide

---

## 2025 — smart product pricing (SMAPE)

**Public vs finale are different.** PJ’s 2023–24 “#1 LB wins the cup” **failed in 2025.**

| Public LB | Finale / claimed AIR | Team | SMAPE | Notes |
|---|---|---|---|---|
| **1st (39.19)** | **5th** | Test Data (IIT Patna) | 39.19 public | LinkedIn: Animesh Tripathy. Proof that talks reshuffle. |
| (hidden 2–3 in Unstop HTML) | ? | ? | ~39.2–39.7 “winning score” (ML Mavericks) | Rank-1 **finale winner still unnamed** publicly |
| 4 | | MessI (IITM) | **40.074** | Unstop table |
| 5 | | royal_recruits (IIITD) | 40.255 | |
| 6 public | **3rd finale** | **00_Team_Rocket** (IIITD) | **40.330** | Text-only DeBERTa-large-v3 + engineered feats + **cross-attention**. CLIP UMAP did not cluster by price → **dropped images**. Smooth L1. BERT 48 → DeBERTa 43 → hybrid **40.3**. XGB/CatBoost ~56. https://github.com/parthrastogicoder/Amazon-ML-Challenge-2025-3rd |
| 3 public / 5 private | (not 3rd finale) | **SPAM_LLMs** (IIT ISM) | tight to top | Frozen Qwen3-4B + SigLIP2 Giant + DINOv3 + modality nets. https://github.com/RudrakshSJoshi/amlc-multimodal-mlp Alok R. later Amazon AS intern. |
| 8 | **AIR 8** | CTRL + ALT + DEV | ~40.55 / public 40.777 CLIP+DistilBERT | 2024 they were AIR **43**. https://github.com/VishalTheHuman/Amazon-ML-Challenge-2025 + Google sheet in their README |
| 9–11 | | Abhimanyu / ML Amigos / Gradient Ascenders (IITJ) | 40.75–40.80 | reddit-003 allegation cluster |
| 17 | | Helios | 41.10 | https://github.com/siddeshrizwani/AmazonML-Price-Prediction-Transformer Medium: https://medium.com/@siddeshrizwani/from-noisy-data-to-top-17-how-team-helios-cracked-the-amazon-ml-challenge-2025-052a91ab520b |
| 20 | **AIR 20** + intern OA | IIITA (Prakhar, Tanay, Jot, Debjyoti) | — | **EmbeddingGemma** not BERT (numeric/abbrev). **SigLIP2-Giant** not CLIP (batch-norm scale drift). Cross-attn not concat. Small FFN. **Train on SMAPE** not MSE. AdamW cosine. College labs GPU. Interview blog: https://medium.com/nybles/amazon-applied-scientist-intern-interview-experience-ml-challenge-2025-5092441d0b2e |
| ~30 | | Akshat Jain et al. IITR | — | CLIP, NeoBERT, DeBERTaV3, RexBERT, Qwen3-emb 0.6B+MLP. https://medium.com/@akshaaat/our-journey-to-the-top-30-in-the-amazon-ml-challenge-predicting-product-prices-from-catalog-data-f8b374e25ec5 |
| 47 | | adityabagrii | 42.25 | DeBERTa-v3 Large gated multi-head, Huber+L1+SMAPE surrogate, Yeo–Johnson. https://github.com/adityabagrii/Price-Prediction-via-Fine-tuning-DeBERTa-Large |
| 80 / ~23k | | ML Mavericks | 43.28 | “winning score 39.7”. https://github.com/NeelDevenShah/Amazon-ML-Challenge-2025 |
| ~top 0.3% same repo | | | | CLIP+brand, Qwen2.5 FT, Granite Unsloth |

Unstop 2025 board: https://unstop.com/hackathons/crp-amazon-ml-challenge-2025-amazon-1560375/coding-challenge/287972

**Scale noise:** LinkedIn “80k+ participants”; Unstop/repos “20k+ teams”; ML Mavericks “23k teams”; one Helios README “3000+ teams”. Treat as **tens of thousands of teams**.

### 2025 methods that keep repeating

1. **log1p price** + Smooth L1 or direct SMAPE, not raw MSE.
2. **DeBERTa-v3** (even **text-only**) is enough for top-3 **finale** if features (organic, gourmet, pack, unit, value) go through **cross-attention**.
3. Strong frozen **SigLIP2 / DINOv3 / Qwen3** embeddings beat rushed small fine-tunes.
4. CLIP is mixed: rank 8 used CLIP+DistilBERT; AIR 20 **abandoned CLIP** for SigLIP2; Team Rocket **dropped vision**.
5. GBM-only ~51–56.
6. **Intern OA** (Debjyoti): they grilled **2024 VLM** (PaddleOCR + Qwen2-VL-2B LoRA 4-bit) in depth — “why this model” more than LeetCode trivia. Two weeks after top-50: interest form.

---

## LinkedIn (public snippets only)

- Poojan / Ansh / Chaitanya — 2023 win
- Angadjeet Singh, Abhishek Jha, Parth Rastogi — 00_Team_Rocket 3rd 2025
- Vishal S / Amritha — CTRL+ALT+DEV AIR 8 (43→8 YoY)
- Alok R. — SPAM_LLMs 6th claimed / 5 private; later Amazon AS intern
- Animesh Tripathy — Test Data IIT Patna **#1 public, 5th finale**
- Debjyoti Ray, Tanay Falor, Prakhar Shukla — AIR 20 + intern blog
- Jury named in a Rocket post: Ajay Srinivasamurthy, Arunita Das, Deepak Gupta

---

## Practice datasets

| Year | Kaggle |
|---|---|
| 2023 | vinayak4444/amazon-ml-challenge-2023 |
| 2024 | abhishekgautam12/amazon-ml-challenge-2024 ; sarthak4156/amazon-ml-challenge-2024 |
| 2025 | manav2805/amazon-ml-challenge-25 ; raghavdharwal/amazon-ml-challenge-2025 ; alienxc137/amazonml25 |

---

## Still missing (don’t invent)

- 2025 **finale 1st and 2nd** team names / repos
- 2024 **official winner** besides KhadgaA’s self-claim and ART runners-up
- 2022 edition (still no clean archive)
- Full Vishesh Rawal / Akshat Jain Medium bodies (fetch timed out)
- Intern coding-question list (Debjyoti post is resume + VLM grilling, not a DSA dump)
