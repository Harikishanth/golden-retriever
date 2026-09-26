# Reddit 002 — Live 2025 contest thread (r/learnmachinelearning)

- **URL:** https://www.reddit.com/r/learnmachinelearning/comments/1o4hezg/amazon_ml_challenge_2025/
- **Subreddit:** r/learnmachinelearning
- **Author:** u/Arceus918
- **Posted:** during Amazon ML Challenge 2025 (Unstop), ~2 days left, then comments continued after contest
- **Score:** ~32 upvotes / 204 comments
- **Captured:** 2026-09-15 from user paste. Direct Reddit fetch was blocked; user noted more comments exist beyond this paste. Treat as a large but not 100% complete thread.

## OP

> So Unstop competitors, how is your progress going? With only 2 days left I hope you have achieved something.

OP later: **56 SMAPE** mid-contest, planned to improve.

## Language note

Almost everyone said “43%” etc. They mean **SMAPE** (lower is better), not accuracy. iwashuman1 called this out. 61% / 75% in this thread are **bad** scores.

---

## Score ladder (public Unstop LB, 2025)

Lower SMAPE = better. These are **self-reported**.

| SMAPE | Who | Approach (short) | Notes |
|---|---|---|---|
| ~30s | “the top” (lilkidlj8 saw this) | unknown | made people nervous |
| ~39 | “top guys” (The-Cactus-Flower asked how) | unknown | |
| **~40.x** | Unlucky_Chocolate_34 | **text-only** DeBERTa-v3 two-stage + cross-attn | would have added images if time |
| **41** | ThicBones | tower model | after wasting 2 days on overfitting GBMs (~47) |
| **42.1** | filterkaapi44 | ViT + BERT → NN fusion, then **weighted blend of old CSVs** | last-hour clutch from 42.9; rank **10–20** during contest |
| **42.5** | royal-retard | simple average of **3** models | **fell out of top 50 in last 15 minutes** |
| **~42** | lilkidlj8 | EfficientNet-B7 + brand/category features + multi-GBM | tried B0–B7 |
| **43** | filterkaapi44 (earlier) / bbx_vansh-2587 / Automatic_Amount7392 | fusion NN; T5-base text emb; public MLP repo | T5-only already 43 before images |
| **43.9 / 43.45** | public GitHubs | GBM ensemble / DeBERTa+ViT fusion | see past-challenges.md |
| **44.8** | BhavyaGoyal repo | RoBERTa ensemble | rank 183 / claimed 20k+ teams |
| **45** | WontLetMeKnow | **text-only LSTM** | |
| **~47** | ThicBones traditional; SinDrafter CLIP-ViT-B/16 | GBMs / CLIP FT | |
| **48** | Unlucky_Chocolate_34 (ablation) | BERT FT 5 epochs | “shocked people couldn't cross 50” |
| **50–51** | many (SinDrafter 50.9, UnBracedFlyer 51) | XGB+ResNet50+text; various | common stall zone |
| **56** | OP; Commercial-Fly LGBM | plain LGBM | |
| **61** | God-D-Kaushik | text | |
| **75** | Potential-Fox-9487 | unknown | |

**Top-50 cutoff guess (filterkaapi44):** ~**42.2 SMAPE**. Consistent with 42.5 dropping out in the last 15 minutes.

**Scale guesses in-thread:** “~5k teams or more” (filterkaapi44). Other 2025 repos claim **20k+** teams — treat as uncertain.

---

## Approaches that people actually used

### filterkaapi44 — ~42.1, rank ~10–20

- Image: **ViT**
- Text: **BERT**
- Fuse outputs → **neural net** regressor
- Not “genius math”; just an approach that worked
- Last hour: **weighted average of previous best submissions by intuition** (42.9 → 42.1)
- Compute: **Kaggle P100 + 29 GB RAM** (no rented GPUs)
- Kaggle quota nearly exhausted
- Code: zip of **main ipynb**
- Would not share model names live; after contest posted the fusion sketch
- Train SMAPE **17.1** by **purposely overfitting**; test 42.1. Wanted late 39s / early 40s.

### lilkidlj8 — ~42

- Image: **EfficientNet-B7** (swept B0–B7)
- Text: **engineered features** (brand, category, …) not a transformer
- Head: **multiple gradient boosting** models
- Local SMAPE 35 but **couldn't finish training**; submitted a half-trained model
- Would not leak full pipeline mid-contest

### ThicBones — tower 41, GBM 47, **made top 50**

- First two days: classical GBMs **overfit like crazy** → ~47
- **Tower model** → ~41
- After top 50: “making it to Applied Scientist is the real challenge”

### Unlucky_Chocolate_34 — **40.x text-only**

Strongest *detailed* text-only recipe in this thread:

1. Pretrain **DeBERTa-v3** on all samples for regression
2. Final model: **CLS embedding + other feature embeddings + cross-attention**
3. **Did not freeze** DeBERTa
4. Loss **not MSE**; regression on **log(p+1)**
5. BERT 5-epoch FT alone → **48 SMAPE**
6. No classical GBM in the final

### SinDrafter101 — 50.9 then ~47

- XGBoost (Optuna) + **ResNet50** image emb + text emb → **50.9** Unstop, val about **0.8 better**
- Then trained **openai/clip-vit-base-patch16** → **~47**

### WontLetMeKnow — 45 text-only

- **LSTM**; incomplete architecture (teammates implemented)

### bbx_vansh-2587 — 43 text-only T5

- **T5-base embeddings** (not SBERT — “not enough”)
- Planned image embeddings + MLP next
- Attitude: “it is so easy” (mid-contest, likely overconfident)

### Commercial-Fly-6296 / Formal_Salt1020

- Plain **LGBM** ~56
- Text ~50, then adding images

### Automatic_Amount7392

- Repeated “Git repository — this gives **43**”, approach **MLP**
- Closest public matches:
  - https://github.com/adityabagrii/Image-Text-Fusion-Model-for-Product-Price-Prediction (DeBERTa frozen 44.1 → ViT/Swin fusion **43.45**)
  - https://github.com/Ashrith-Yathin/Amazon-ML-Challenge-2025 (GBM ensemble **43.9**; README also says MLP)

### Mother-Purchase-9447 (didn't compete; architecture takes)

- Hypothetical winner: small **VLM + Unsloth QLoRA 4-bit**, train attn (then FFN)
- Critiqued heavy `nn.Linear` stacks
- Prefer **RMSNorm** over LayerNorm, mixed precision, **no biases**, A100/bf16

### ufo-890einstein — “20 SMAPE yet to upload, I think I win”

- Combined text+image
- Thread consensus: **overfit**. Similar local 28.8 last-minute story; another overfit local that scored **51** on submit.

### Eastern-Jellyfish995 — local 28.8, submit 1 min before deadline

- Train ran 5–6 hours, finished 11:55, submit 11:59, **no LB score visible**
- Email “successful” at 12:05
- nemesisoflife: “Looks like you overfitted”

---

## Operations / Unstop mechanics (2025)

These are the highest-value bits for 2026.

### Submissions

- Official mail: **5 valid submissions per day**
- Confirmed by multiple people (`Large-Philosophy-922`, later filterkaapi44)
- Confusion: some said unlimited, some 15/team total, some hit a wall after **7** (including invalids) then had to wait until midnight
- Invalids (format errors) **do not** eat the valid quota the same way — filterkaapi submitted >5 total but some invalid
- Score back in **~5 minutes or less**
- Code upload: **zipped ipynb with main code** was accepted (filterkaapi44). Unclear if models/weights were required; they did not upload a full pipeline folder.

### Format error

`tuple indices must be integers or slices, not str`

- filterkaapi44: columns must match — they said **`sample_index` and `price`**
- Official 2025 problem statement (other sources): **`sample_id` and `price`**, file `test_out.csv`
- **Action:** copy the sample output file headers exactly; do not guess the id column name

### Leaderboard / deadline chaos

- Public LB is **~33% of test** (25k/75k). Private can move rank **2–3 places** (Disastrous-Leg7370)
- Rank **swings hard in last 15 minutes** (42.5 fell out of top 50)
- Submits in the last seconds may **not show a score**; LB goes down after round 1
- Public coding-challenge URL (still the public board): https://unstop.com/hackathons/amazon-ml-challenge-2025-amazon-1560375/coding-challenge/287972
- Notifications for top 50 were **slow / unclear**; people still asking days later (“ab to 19 ho gya”)
- ThicBones did eventually confirm **top 50**

### Val vs test SMAPE

- Disastrous-Leg7370: gap **~0.5** on most subs, **~0.1** on best
- SinDrafter: Unstop **0.8 worse** than val
- Train 17 or local 20–28 with test 42–51 = **overfit**. Do not trust train SMAPE.

### Compute

- Kaggle GPU quota is a real limiter mid-contest
- Multi-hour jobs that finish at 11:55 are a known failure mode

### Team / platform risk

- Only the **team leader** may be able to submit on Unstop/Kaggle. If they delete the group, others are stuck (`wtfis_spnbruh`)

### Cheating noise

- Offers to **sell CSVs** (~42–43), “pay 30k for approach”, “working code rank <500 for a price”
- Ignore. Also a signal that **blended prediction files** were a last-day meta.

---

## Quotes worth keeping

- Weighted last-hour blend of old submissions beat a new model for filterkaapi44.
- Simple average of 3 models at 42.5 was **not safe** for top 50.
- Text-only DeBERTa with log-target + unfrozen encoder + extra features hit **40.x** — images are not mandatory if text is done well, but fusion still won the actual top.
- BERT 5-epoch FT → 48 is the “you are doing real ML” floor; 56 LGBM is the “started late / tabular only” zone.
- Top 50 ≠ intern. OA/interview is the next wall.

## Problem restated in-thread

iwashuman1: catalog content + image → **predict price**. Catalog has item name, desc if any, quantity, unit. Dataset later on Kaggle.
