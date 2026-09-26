# Reddit 005 — “What approach did you take?” (r/learnmachinelearning)

- **URL:** https://www.reddit.com/r/learnmachinelearning/comments/1o5thcf/what_approach_did_you_take_in_the_amazon_ml/
- **Subreddit:** r/learnmachinelearning
- **Author:** u/CryptoDarth_
- **Posted:** after / near end of 2025 contest
- **Flair:** Question
- **Score:** ~8 upvotes / 33 comments
- **Captured:** 2026-09-15 from user paste (some comments deleted; “download images on the fly” advice survived)

## OP (CryptoDarth_)

New to ML, Colab user, upset with Colab during the hackathon, considering Kaggle next time.

Own result: joined **1.5 days late**. NN fusion of **sentence-transformers + ConvNeXt**, plus feature engineering similar to the rank-8 team. First (only) submit **55 SMAPE**. Training **2–4 hours on 20% of the data** — thought that was too slow.

Questions they wanted: preprocess/postprocess, models explored vs final stack, 3-day flow, **training times**, Kaggle vs Colab.

---

## filterkaapi44 — 42.1, public ~top 50, full fine-tune

Same person as reddit-002; this thread has the **score path** and train budget.

| Step | SMAPE |
|---|---|
| early | 45.5 |
| | 43.8 |
| | 43.3–43.4 |
| | 42.9 |
| last-hour **weighted blend of old CSVs** | **42.1** |

- Image **ViT** + text **BERT** → fuse → NN
- **Did not freeze** any layers; fine-tuned the **whole** architecture
- **~20–22 hours** of training; 4–5 hours of early play
- **Data augmentation** (not specified)
- Explicitly **not CLIP** — CLIP+Qwen→NN in this thread scored **~47** (Technical_Scheme_933)
- Resources for intuition: theory + experiment; **cs231n**, **Karpathy**

**Read:** unfrozen ViT+BERT for a full day can beat frozen CLIP+Qwen dumped into an NN. Blend still did the last 0.8.

---

## frankenstienAP — public rank 7–9, **final rank 8**, finale 17th

Highest-detail top-10 writeup in the Reddit pile. Code not shared (planning a paper); they answered everything else.

### Score path

- Feature + embedding work: **rank 35 → 7**
- SMAPE **43.… → 40.…**
- Last submit **11:58** while sitting **rank 9** (LB about to close)
- Mail confirmed **rank 8**; finals **17th**
- Unlucky_Chocolate_34: top-10 mails went out a few hours before that comment; **no mail ⇒ not top 10**

### Model

- **SigLIP** embeddings + a second embedding (unnamed)
- Train the **best DNN on each embedding separately**
- Infer with SMAPE-optimal mix:  
  `alpha * SigLIP_pred + (1-alpha) * other_pred`
- Teammate spent **two days** on **loss function + DNN hyperparameters**
- Total params (encoders + MLP) **≤ ~900M** — they say lighter than other top-10 VLMs
- Good GPU only on **last day**; college HPC was **queued by everyone** (used twice in 72h)

### Feature engineering (laptop, Lenovo LOQ **RTX 4050 6GB**)

- Brand via **GLiNER NER** (small LM extraction not feasible on 6GB; still had to clean GLiNER output)
- **~70 binary** presence features
- EDA: which flags raise **Price** and **Price_Per_Unit**
- **Dropped ~6000 outliers** on price and price-per-unit — “crucial”
- FE locally; heavy train later on a real GPU

**Read:** unit/pack-normalized **price per unit** is a first-class target for EDA, not only raw price. Outlier drops on 75k train (~8%) mattered. Light embeddings + a well-tuned DNN + **alpha blend** beat waiting for a giant VLM. **Do not depend on campus HPC** during this contest.

---

## Other scores / failures

| Who | Result | Stack |
|---|---|---|
| Technical_Scheme_933 | **~47** | CLIP + Qwen embeddings → NN (no long FT?) |
| CryptoDarth_ | **55** | sentence-transformers + ConvNeXt fusion, late join |
| PrateekSingh007 | val **22** / test **122** | leakage, tiny val, or catastrophic overfit |

Val 22 → LB 122 is the most extreme leakage example yet. Unlucky_Chocolate_34: leakage or tiny validation. Own_Math_5764: overfit.

---

## Compute / data logistics

- OP: Colab felt bad; wants Kaggle next time
- filterkaapi44 (reddit-002): Kaggle P100
- frankenstienAP: **6GB 4050** for FE; 900M-cap models; HPC almost useless
- YouCrazy6571: how to get **~16GB of images** onto Kaggle?
  - Deleted comment’s surviving lesson: **download and process on the fly** rather than uploading a 16GB dataset
- Mother-Purchase-9447: Unsloth QLoRA on a VLM; CLIP is cosine-sim not a price head; rumor **max < 8B params** allowed — **unverified**, check 2026 rules

---

## 3-day flow implied by rank 8

1. FE on the laptop immediately (NER brand, binaries, price/unit, outlier drop) — does not need A100
2. Extract two strong embeddings (SigLIP + other)
3. One person **lives in loss + DNN hparams** for 48h
4. Blend the two DNN heads with **alpha tuned on SMAPE**
5. Real GPU if/when it appears, last day
6. Don’t fire a new untested train at 11:58 unless it already beat holdout (they still did a last submit as rank 9)

## Playbook takeaways

- **Price_per_unit** + **~70 binary catalog flags** + **drop ~6k price outliers**
- **GLiNER** for brand when you cannot run an SLM
- Train **separate heads per embedding**, mix with **SMAPE-optimal alpha** (not just concat-then-one-MLP)
- CLIP+Qwen frozen→NN ≈ 47 ≠ “NN doesn’t work”; **train time and which ViT** matter (22h unfrozen ViT+BERT → 43s before blend)
- 2–4h to train on **20% data** is a red flag (OP); profile dataloaders / image I/O
- Campus GPU queues will be slammed — have Kaggle/Colab/local as the real plan
- Top-10 **email** is the signal; public rank 9 at 11:58 is not final
