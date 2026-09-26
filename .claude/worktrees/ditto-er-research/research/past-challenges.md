# Past Amazon ML Challenges (compiled)

Pattern across years: **Amazon catalog / product-attribute problems**, huge data, 72-hour Kaggle-style scoring, then a **presentation finale** for top teams. Hosted on HackerEarth (older) then **Unstop**.

---

## Format (stable across recent years)

- **Team:** 3–4 (PJ strongly prefers 4)
- **Duration:** ~72 hours once dataset drops
- **Submit:** predictions file + 1–2 page approach doc + zipped code
- **Live leaderboard** during the contest (public subset of test)
- **Private / full test** + documentation used for final ranking
- **Top 50:** Applied Scientist intern OA + interview (PPI), subject to eligibility
- **Top 10:** certificates, SWAG, grand finale presentation to Amazon scientists
- **Prizes (recent):** Winner ₹1,00,000 / 1st RU ₹75,000 / 2nd RU ₹50,000
- Cross-college teams: **allowed in 2024 listing**; some 2025 guides said same-institute — **always re-check Unstop**
- One person, one team
- Eligibility: full-time B.E/B.Tech/M.E/M.Tech/M.S/PhD at Indian engineering campuses; **graduating years change every edition**

---

## 2021 — Browse Node Classification (HackerEarth)

- **Task:** Multi-class text classification of products into Amazon browse nodes
- **Inputs:** `TITLE`, `DESCRIPTION`, `BULLET_POINTS`, `BRAND`
- **Target:** `BROWSE_NODE_ID` (~**9,919** classes)
- **Size:** train **2,903,024**; test **110,775**
- **Metric:** Accuracy
- **Duration:** ~3 days
- **What worked:** multilingual BERT / XLM-RoBERTa; Sentence-BERT embeddings + kNN/ensemble; heavy preprocessing of noisy catalog text
- **Example:** team Panaroma rank 16 / 3000+, multilingual BERT, ~67.5 public score
- **1st runner-up:** DeVaSh.Ai — https://github.com/DebarshiChanda/Amazon-ML-Challenge2021 Twitch https://www.twitch.tv/videos/1107585685?t=4h18m36s
- **Repos:**
  - https://github.com/akshatprogrammer/Amazon-ML-Challenge
  - https://github.com/shanmukh05/Amazon-ML-Challenge
  - https://github.com/nikhil6041/AmazonMLChallenge2021
  - https://github.com/pranshurastogi29/Amazon_ml_challenge-solution
  - https://github.com/jharkawat/Amazon_ml_challenge (RoBERTa 67.23%)

---

## 2022

No widely archived, clearly labeled “Amazon ML Challenge 2022” problem pack found in this pass. Treat 2021 → 2023 as the documented sequence unless a later source fills this gap.

---

## 2023 — Product Length Prediction (HackerEarth)

- **Winner:** ART in Artificial Intelligence (PJ, Ansh Tanwar, Chaitanya Giri, Harshit Kumar) — LB #1 and finale winner
- **Scale:** ~**24k participants**, ~**7k teams**
- **Task:** Predict **product length** (warehouse packaging + customer size)
- **Inputs:** title, description, bullet points, product type ID
- **Size:** ~**2.2 million** products
- **They hit #1 in ~1.5 days and held it**; won on **Kaggle GPUs only**
- **Finale VOD:** https://www.twitch.tv/videos/1804684510
- **Winner notebook (no data, T&C):** https://github.com/pj-mathematician/Amazon-ML-Challenge-2023

### 2nd place (Team Fishes) — well documented

Repo: https://github.com/TashvikDhamija/AmazonML  
Presentation on same Twitch VOD (~1:33:50)

1. EDA: heavily skewed target, many outliers
2. Concat title + description + bullets for BERT context
3. **Log target**, clip log at 12, normalize
4. Baseline: frozen BERT embeddings + shallow ANN
5. Then **end-to-end finetune BERT and RoBERTa** + learned **product-type embedding** + regressor
6. Postprocess: round prediction to **nearest train length** (lengths repeat); take **min(BERT, RoBERTa)** to counter high-side bias

Other 2023 notes: log + PowerTransformer on target; character-level text to handle typos (VectorNd repo).

**Meta:** NLP **regression** on noisy catalog text. Feature engineering + transformer fine-tune + smart postprocess beat “just a bigger model.”

---

## 2024 — Entity Extraction from Product Images (Unstop)

- **ART:** LB **#5**, **runners-up** after presentations
- **PJ models:** **Qwen VL** + **Donut**
- **Scale:** commenters cite **~18k teams**
- **Task:** given image + queried entity, extract value+unit (or empty)
- **Entities:** width, height, depth, item_weight, maximum_weight_recommendation, voltage, wattage, item_volume
- **Output format:** `"<value> <unit>"` with **allowed units only** (strict map). Invalid units fail.
- **Size:** train.csv **263,859** rows / **255,906** train images; test.csv **131,187** / **90,666** test images (~2 lakh images)
- **Metric:** **F1**
- **Community feel:** “pay to win” / bigger VLM better; GPU-hungry

### Approaches that scored

| Style | Models / stack | Notes |
|---|---|---|
| VLM fine-tune (winners / top) | Qwen2-VL-7B-Instruct, Idefics-2 8B, Donut | LoRA/QLoRA via LLaMA-Factory; prompt “What is the {entity_name}?” |
| OCR + regex | PaddleOCR, EasyOCR | Cheap baseline; unit mapping is the hard part |
| Data curation | Fix bad labels, drop invalid units, range → max or NA | KhadgaA: curated 1600 samples jumped score to **0.865** after weaker 20k-sample FT |

KhadgaA (claims winning code): https://github.com/KhadgaA/Amazon-ML-Challenge  
- Baseline Qwen2-VL-7B AWQ: **0.617**
- FT 10k–20k: ~**0.68**
- FT on **curated 1600**: **0.865**
- Tooling: LLaMA-Factory, QLoRA 8-bit

Idefics-2 top-50 writeup: https://github.com/hwaseem04/Amazon-ML-Challenge-2024  
- FT with custom prompts: F1 **0.616**
- Resize to 980px (Idefics limit); rotation aug for sideways text

Kaggle dataset: https://www.kaggle.com/datasets/sarthak4156/amazon-ml-challenge-2024

**Meta:** multimodal / OCR. **Label quality and unit postprocessing** mattered as much as the VLM. Sampling was OK if distribution held. **~75k** registrations cited on Medium/GitHub.

DBkaScam **finale rank 6**, F1 **71.8**: MiniCPM-2.6 zero-shot + Qwen2-VL-7B few-shot + Qwen SFT vote. Canva talk + repo: https://github.com/arnav10goel/Amazon-ML-Challenge-24  
SFT alone 64.8 **lost to** few-shot 70.9. Postproc: fractions, feet/inch quotes, range→max, illegal units out.

More 2024 links: `catalog-web-sweep.md`.

### 2024 Reddit post-mortem (reddit-006)

OCR+regex band without a fine-tuned VLM:

| Who | F1 | Notes |
|---|---|---|
| OP xayushman | **0.447**, rank **206** | EasyOCR + small LLaVA + **Pint**; 10h Kaggle upload; 4×P100 PaddleOCR never finished |
| mopasha1 | **0.489** | EasyOCR streamed (no download), 15 shards / 7 sessions / 3.5h; **30% rows blank**; spatial angles via RPN |
| Smooth_Loan_8851 | **~0.51 val / no submit** | Tesseract + H/W box sides; index bug at midnight |
| adithyab14 | **0.42 → 0.48** | Lightning $15; PaddleOCR 132k in **1.5h (~50ms)**; XGB picks which (value,unit) span; H=first/D=last regex |
| Terrible_Bar_1158 | **0.0016** | LLaVA; GPU died; only **1k rows** filled |
| Tensor Titans (GitHub) | **0.679**, rank **22 / 74,843 teams** | Paddle + custom line detect; only **39% filled**; leaving depth empty helped |

- Naive 0.5B VLM: 1–2 s/img → **40–80h** for 132k. OCR first, VLM only on hard residuals if time.
- Train **skewed to item_weight**; Tesseract looked fine until they val’d L/W/H.
- Hearsay top: fine-tuned **LLaVA 8B**. PJ: Qwen-VL + Donut.
- Scale note: one GitHub says **~75k teams**; earlier comments said ~18k — treat as uncertain, order-of-magnitude huge.

---

## 2025 — Smart Product Pricing (Unstop)

- **Task:** predict **price** from text catalog + product image
- **Columns:** `sample_id`, `catalog_content` (title + description + IPQ), `image_link`, `price` (train only)
- **Size:** **75k train + 75k test**; public LB on **25k** test slice; private on full 75k
- **Metric:** **SMAPE** (0–200, lower better)
- **Output:** `test_out.csv` with `sample_id`, `price` (float). Filename/columns must match or submit crashes.
- **Docs:** 1-page methodology (architecture, features, etc.)
- **Kaggle mirrors:**
  - https://www.kaggle.com/datasets/manav2805/amazon-ml-challenge-25
  - https://www.kaggle.com/datasets/raghavdharwal/amazon-ml-challenge-2025
  - https://www.kaggle.com/datasets/alienxc137/amazonml25

### What actually worked

**SPAM_LLMs (IIT ISM Dhanbad) — 3rd public / 5th private**  
https://github.com/RudrakshSJoshi/amlc-multimodal-mlp

- Frozen **pretrained embeddings** + modality-specific nets + regressor **beat many fine-tuners**
- Text: **Qwen3-4B**; image+text: **SigLIP2 Giant (~2B)**; image: **DINOv3 (~0.8B)**
- Distilled small LLMs gave **worse** embeddings; bigger frozen > small fine-tuned
- Need **both** image and text (same image, different catalogs/prices and vice versa)
- log1p on right-skewed price; unit normalization; numbers→words; trim long catalogs
- Train with log-MSE

**Other 2025 numbers**

- CLIP-L + DistilBERT fusion + contrastive: **~40.8 SMAPE** on public 25k (VishalTheHuman)
- GBM ensemble (LGBM + XGB + CatBoost) on text+image embeddings + regex quantity/unit/brand: **43.9 SMAPE** (Ashrith-Yathin)
- Weak text-only DistilBERT: ~50–51 SMAPE; image-only ViT failed badly (~190)

**Meta:** multimodal **regression**. Frozen strong embeddings + tabular/GBM or small MLP can beat rushed VLM fine-tunes. Feature extraction from catalog (pack quantity, unit, brand) is “golden.”

### 2025 live-thread score reality (reddit-002)

Self-reported public SMAPE while the contest was running:

- Top of LB seen around **low 30s**; “how did top guys get **39**?”
- **Top-50 cutoff guessed ~42.2**. A **42.5** 3-model average fell out of top 50 in the last 15 minutes.
- **42.1** last-hour weighted blend of old CSVs (ViT+BERT→NN), rank ~10–20 during contest; Kaggle P100 only
- **~42** EfficientNet-B7 + brand/category features + multi-GBM
- **41** tower model after GBMs overfit to 47
- **40.x text-only:** DeBERTa-v3 pretrain-for-regression → unfrozen DeBERTa + feature embeddings + cross-attention, log(p+1), non-MSE loss. Plain BERT 5-epoch FT = **48**
- **43** T5-base text embeddings (commenter said SBERT was not enough); public MLP fusion repos also ~43.4–43.9
- **45** text-only LSTM; **47** CLIP-ViT-B/16 FT; **50.9** Optuna XGB + ResNet50 + text; **56** plain LGBM
- Local 17–28 SMAPE that looked like a win was **overfit** (test came back ~42–51)
- Honest val vs LB gap ~**0.1–0.8**
- Unstop: **5 valid subs/day**, score in ~5 min, zip of main ipynb accepted
- Public LB ~33% of test; private moved ranks ~2–3 places
- Top 50 ≠ Applied Scientist intern — OA/interview is the next filter
- Extra public writeups: https://github.com/adityabagrii/Image-Text-Fusion-Model-for-Product-Price-Prediction (43.45), https://github.com/BhavyaGoyal777/AMAZON_ML_SOLUTION (44.8, rank 183)

### 2025 public vs finale (web sweep — important)

PJ’s “#1 LB wins the cup” **did not hold in 2025**.

| Public | After talks | Team | SMAPE |
|---|---|---|---|
| **#1** | **5th** | Test Data, IIT Patna | **39.19** |
| #6 | **3rd (2nd runner-up)** | 00_Team_Rocket, IIITD | **40.33** (text-only DeBERTa-v3 + flags + cross-attn; **images dropped**) |
| 3 pub / 5 priv | not 3rd finale | SPAM_LLMs, IIT ISM | frozen Qwen3-4B+SigLIP2+DINOv3 |
| #8 | AIR 8 | CTRL+ALT+DEV | ~40.55; 2024 they were 43rd |

Winning public score cited ~**39.2–39.7**. Finale **1st/2nd still unnamed** in public repos. Full table: `catalog-web-sweep.md`.

00_Team_Rocket repo: https://github.com/parthrastogicoder/Amazon-ML-Challenge-2025-3rd  
AIR 20 intern blog (EmbeddingGemma, SigLIP2, train on SMAPE): https://medium.com/nybles/amazon-applied-scientist-intern-interview-experience-ml-challenge-2025-5092441d0b2e

### 2025 public-LB snapshot + allegations (reddit-003)

Posted before the **final/private** board (OP expected that on the **17th**). Treat named teams as **alleged**, not convicted.

| Rank | Team | SMAPE | College |
|---|---|---|---|
| 10 | ML Amigos | 40.797 | IIT Jodhpur |
| 11 | Gradient Ascenders | 40.798 | IIT Jodhpur |
| 15 | (unnamed in table; a commenter) | — | — |
| 41 | Low Voltage | 42.106 | VIT Vellore |
| 42 | Interstellars | 42.106 | VIT Vellore |
| 43 | Optimizers | 42.107 | VIT Vellore |

- **~40.8 ≈ rank 10**, **~42.1 ≈ rank 40** on this public snapshot
- royal-retard: BERT-family (DeBERTa, ELECTRA) explained most **~44** scores; **ensembles / more models** explained 42–42.5; 50.1 without BERT is “pretty good”
- Rank 180–200 called very good for a 3rd year
- Counter: IIT Jodhpur pair later described (by Drake_18776, 10 months later) as BERT vs DistilBERT, different epochs, tiny unrounded SMAPE gap — similar models, not necessarily one CSV
- VIT triple at **42.106 / 42.106 / 42.107** is the harder coincidence
- Top 10/50 **emails were not out** yet; even a rank-15 team had no mail
- OP’s “82000” coincidence pool is **unverified** vs other 5k–20k team counts

### 2025 implementations thread (reddit-004)

- **zarouz (OP):** RNN + text/image embeddings + BART food clusters → **48 local / 50 LB**. Suspected images added noise.
- **yashBhaskar:** ~**150M** pretrained text embedding, **raw catalog, no preprocessing**, regression head, **no images** → **42 SMAPE**. (Not Mistral; 150M is encoder-scale.)
- **Apprehensive-Talk971:** log-price is “the main big thing.” Then **Qwen + multilingual DistilUSE + CLIP text + 1 vis**, fine-tuned with **triplet loss**, **kNN on embeddings + regressor**. Score not given; posted in a <45 thread.
- Lesson: a clean text-embedding baseline belongs on Day 1 **before** RNNs, topic models, or forced vision.

### 2025 approach thread (reddit-005)

- **filterkaapi44:** unfrozen ViT+BERT→NN, **20–22h** train + aug, path **45.5→43.8→43.3→42.9→42.1** (last step CSV blend). Not CLIP.
- **Technical_Scheme_933:** frozen-style CLIP+Qwen→NN **~47**
- **frankenstienAP — final rank 8**, finale 17 Oct 2025: SigLIP + second embedding, **separate DNNs**, `alpha` mix for SMAPE; GLiNER brand + 70 binaries + **price_per_unit** EDA + **drop ~6k outliers**; **≤900M** params; FE on RTX 4050 6GB; HPC useless; last submit 11:58 as public rank 9. Public 43→40, ranks 35→7 then 8. Code withheld for a paper.
- **CryptoDarth_:** late join, sentence-transformers+ConvNeXt, **55**; 2–4h to train 20% data
- **PrateekSingh007:** val 22 / LB **122** (leakage)
- Images ~16GB: process on the fly. Top-10 signal = **email**, not last public rank.

---

## 2026 — upcoming (updated 15 Sep 2026 from dept circular)

Primary: `sources/official-2026-dept-circular.md` (Amazon University Talent Acquisition via college). Secondary: Internshala times.

**Register:** https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604?ref=AMHbdgmy

| Item | Official circular | Extra from Internshala (verify on Unstop) |
|---|---|---|
| Register | **7–20 Sep 2026** | 20 Sep **11:59 PM IST** |
| Contest | **25–27 Sep 2026**, 72 hours | **SUPERSEDED.** Official pre-challenge mail: **25 Sep 12:00 AM IST – 27 Sep 11:59 PM IST** (`sources/official-2026-unstop-pre-challenge-email.md`). Internshala 9 AM–9 PM was wrong. |
| Top 50 list | not in circular | **2 Oct 2026** |
| Grand Finale | **7 Oct 2026** | 10:00 AM–3:00 PM IST |
| Who | **2027 & 2028** BE/B.Tech/ME/M.Tech, pre-final & final year | also listed PhD/MS — **Unstop wins if conflict** |
| Team | not restated here | 3–4, leader, cross-college allowed |
| Cash | **₹2,25,000** | 1L / 75k / 50k |
| Top 50 | PPI **Applied Scientist intern** | same |
| SWAG | top 10 **and** top 10 **women-only** teams | same |
| AWS | student-verify on **Builder Center** → **$579** pack (credits + $100 cert voucher + 12 mo Skill Builder). During contest: **$200 Free Tier credits** (“no expensive hardware required”) | — |
| Problem | real Amazon dataset, national leaderboard | **not public until Day 1** |

**Guessing 2026 topic:** still unknown. History is catalog/multimodal product understanding. Do not overfit to one year.

**This week:** register; AWS Builder Center verify **each** teammate; find how to attach the $200 credits to SageMaker/EC2 **before** 25 Sep; keep Kaggle/Lightning as backup.

---

## Year-over-year skill map

| Year | Modality | Task type | Metric | Compute pressure |
|---|---|---|---|---|
| 2021 | Text | 9.9k-way classification | Accuracy | High data volume |
| 2023 | Text | Regression (length) | error metric on length | High (2.2M rows) |
| 2024 | Image (+OCR/VLM) | Entity extraction | F1 | Very high (GPU / VLM) |
| 2025 | Text + image | Regression (price) | SMAPE | Medium-high (75k images) |
| 2026 | Unknown | Unknown | Unknown | Assume huge + multimodal |

Always prepare: **fast data pipeline, holdout, NLP, CV, multimodal fusion, ensembling, metric-aware postprocess, 1–2 page writeup.**
