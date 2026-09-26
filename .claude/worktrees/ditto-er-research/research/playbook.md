# Contest playbook (from PJ + past winners)

Living doc. Updated as more Reddit/research lands.

## Team

- **4 people**, all Python + NumPy/Pandas
- Suggested roles: data/I/O, modeling, inference+ensemble, docs/presentation
- Scarce skill historically: **DL / VLM / fine-tune**, not web
- Everyone must be able to defend every technique on slides

## Compute

- Kaggle T4 can win (2023 did). filterkaapi44’s 42.1 was **Kaggle P100**, **~22h** unfrozen ViT+BERT.
- Comfortable: **40–80 GB** VRAM (A100 40/80, or multi-GPU)
- 24 GB (3090) is workable, not comfortable
- **RTX 4050 6GB laptop was enough for rank 8 FE** (GLiNER, binaries, outlier EDA). They kept models **≤ ~900M params**.
- **One Colab Pro is enough**; buy after seeing the dataset. 2025 Colab users complained; Kaggle or local was the workaround.
- **Do not bet on college HPC.** In 2025 the queue was full of the same contest; rank 8 used it twice in 72h and got a “good GPU” only on day 3.
- Don't blow the clock training giant models on full data. If **20% of data takes 2–4 hours**, the dataloader/image pipeline is wrong.
- Images: **2025 ~16GB / 2024 ~1e5+ product shots.** Prefer **stream URLs or Lightning `download_images` (~20 min for 132k)**. One 2024 team burned **10 hours** downloading then **uploading to Kaggle** — that is how you lose Day 1.
- **Latency budget:** if the test set is ~1e5 images, you need **≲100 ms/image** to finish in ~4 hours. A 0.5B VLM at 1–2 s/image is **40–80 hours**. PaddleOCR at ~50 ms is ~**200×** faster. Time 100 images on hour one.
- Tools: LoRA/QLoRA, sampling with **same distribution**, frozen embeddings + small head — **or** a long unfrozen 20h fine-tune if that is the one bet (both showed up in top 50).
- **Lightning AI $15/month** credits were a 2024 life-saver. RunPod/Paperspace for a few hours is the honest alternative to farming extra Colab/Kaggle accounts (phone-verify + ToS risk).
- **2026 official (Builder blog 23 Sep, Jatin Mehrotra):** **$200** contest credits for all (signup **$100** + 5 Explore-AWS activities **$100**). **Extra $100 at the 48-hour mark for Top 500.** Student Rewards still **$579** (Skill Builder $449 + badge ladder 7=$10 / 14=$20 / 21=cert voucher; badges **lag**). Use **SageMaker Notebook Instance + local train/predict**, not Studio Domain and **not** a $0.12/h endpoint. You submit a **CSV + code zip + approach doc**. `ml.t3.medium` is the free-tier box (250h) — pandas/GBM only. Still keep Kaggle/Lightning/GPU elsewhere; “this is all you need” is the beginner path, not 2024–25 top-50.
- Rumor in 2025: **param cap < 8B**. Unverified; read the year’s rules.

## 72-hour clock (PJ)

1. **Day 1 — explore.** Parallel approaches. Always a **private holdout**. Kill losers fast.
2. **Day 2 — deepen the winner.** No new architecture tourism.
3. **Day 3 — ensemble + rank defense.** Submission format, docs, no last-minute untested ideas.

Submit **early** (format bugs are common) then iterate. **2026 official mail: 5 submissions per day × 3 days, then the button dies.** Same cap as 2025. Invalid format dumps may not count. Score return time unknown until Day 1. **Public and private LBs both count for shortlisting** — do not overfit the public board. Keep every submission file; they may demand source later.

Last 15 minutes move ranks. Do not start a 6-hour train hoping it finishes at 11:55. Keep a **blend of previous CSVs** ready (weighted average of best subs was a documented last-hour gain: 42.9 → 42.1). Simple average of 3 models at 42.5 fell **out of top 50** in that window.

Only the team leader may be able to submit. Don't let one person own every login.

## Modeling principles that keep showing up

1. **Match the metric.** SMAPE ≠ MSE. F1 with unit constraints ≠ “the VLM said 2g.”
2. **Postprocess is free score.** Nearest-neighbor rounding (2023), unit maps (2024), pack-quantity features (2025).
3. **Target transform** on skewed regression (log / log1p). 2025 commenters called log-price “the main big thing”; raw prices were badly skewed.
4. **Don't assume you must fine-tune.** 2025 top-5 froze big embedding models. 2023 top-2 *did* fine-tune BERT. Decide on Day 1 with holdout, not Twitter.
5. **Pick models from the closest public benchmark** (PJ): OCR → Qwen-VL + Donut; pricing → CLIP/SigLIP + text encoder + GBM/MLP.
6. **Label noise is real.** 2024 curated 1.6k clean samples >> 20k dirty.
7. **Multimodal when both signals exist — but ablate images.** 2025: same image, different catalogs/prices, so vision *can* help. A **150M text embedding + regression head, catalog dumped with zero preprocessing, no images, scored 42**. RNN + image emb + BART food clusters scored **50**. Weak vision can be **noise**. Keep images only if holdout improves vs text-only.
8. **AI copilots:** OK for boilerplate. Bad as the optimization loop. PJ: they produce *a* solution, not *the* score.
9. **2025 score bands (SMAPE, lower better):** public snapshot had **~40.80 at rank 10** and **~42.11 at rank 41–43** (cutoff guess **~42.2** still holds). Single BERT-family models (DeBERTa, ELECTRA, …) clustered **~44**; **more models in the ensemble** was the usual path to 42. 43 fusion/GBM is “good but not safe”; 47–51 is the stall; **50 without BERT** is decent, not top-50; 56 LGBM is weak; 48 is roughly BERT-FT-5-epochs **or** RNN+BART+images (OP of reddit-004). Text-only DeBERTa + log-target + extra features hit **40.x**; text-only 150M embedding+head hit **42**. ViT+BERT+NN and EfficientNet+GBM both landed ~42. Rank **180–200** was still called strong for a 3rd year.
10. **Don't waste Day 1–2 only on GBMs** if they overfit (ThicBones: 47 classical → 41 tower once they switched). Parallel a neural fusion from hour one. Also parallel a **dumb text-embedding baseline** before fancy FE.
11. Last-day **CSV blending** of diverse models beats training one more overfit net. Keep every decent submission file. Documented path: **45.5 → 43.8 → 43.3 → 42.9 → 42.1** (last step = weighted old CSVs). Rank 8 mixed two DNN heads as `alpha*SigLIP + (1-alpha)*other` **tuned for SMAPE**.
12. **Neighborhood / metric learning:** 3 text encoders (Qwen, multilingual DistilUSE, CLIP text) + 1 vis, **triplet loss**, then **kNN on embeddings + regressor**. Same spirit as 2023 “round to nearest train length.” Worth a Day-1 experiment for catalog pricing. Watch the **expensive / sparse tail** separately (asked in reddit-004, never answered).
13. **Catalog FE that moved a team from rank 35 → 7 (43 → 40):** **GLiNER** brand NER; **~70 binary** flags; EDA on **price and price_per_unit**; **drop ~6000** price/unit outliers. Spend a teammate on **loss + DNN hparams for 48h**.
14. Frozen CLIP+Qwen → NN **~47** ≠ “NNs don’t work.” Unfrozen ViT+BERT **20–22h + aug** reached the 43s before blending. Concat-and-hope is the failure mode.
15. **OCR years (2024):** cache OCR to disk; **Pint + allowed units** (drop `15kg` for height); empty is better than a guess; **stratify val by entity** (train is weight-skewed); H/W/D from **box geometry**; fill OCR-misses with `group_id`×entity priors. **F1 0.45–0.49** was the OCR+regex band; **0.62–0.86** needed a VLM or heavy curation. Don’t submit a partial CSV if the GPU dies — pad empties. 2024 rank-6 ensemble: MiniCPM zero-shot + Qwen2-VL few-shot + Qwen SFT vote = **71.8 F1**. SFT alone (64.8) lost to few-shot (70.9).
16. **2025 pricing stack that made top 10:** DeBERTa-v3 + pack/unit/quality flags + **cross-attention** (text-only **3rd finale**, images dropped after UMAP). Or frozen **SigLIP2 + Qwen3 + DINOv3**. CLIP is optional (rank 8 used it; AIR 20 dumped it for SigLIP2 because of batch-norm scale drift). **Train Smooth L1 or SMAPE**, not raw MSE. GBM-only ~56.
17. **Intern interview (AIR 20 writeup):** they asked **last year’s VLM** in depth (why Qwen, LoRA, VLM vs LLM), not only this year’s price model. Know every line on the 1–2 pager.

## Submission landmines

- Exact filename (`test_out.csv` in 2025)
- Exact columns from **sample_test_out.csv**, not from memory. Official 2025 text said `sample_id`,`price`; one 2025 competitor said `sample_index`,`price`. Copy the sample file.
- Allowed units / types only
- Zip of the **main notebook** was enough in 2025 (weights folder not always required — still include a reproducible pipeline)
- Error `tuple indices must be integers or slices, not str` → wrong column names / zip, not the model
- **Test ids are not row index.** 2024 shard merges labeled 0..n and failed; official sanity check + copying ids from `test.csv` saved a 0.489 submit. Another team had 0.51 val and **zero** valid submit for the same bug.
- Train SMAPE of 17–28 with LB 42–51 = overfit. Typical honest val↔LB gap in 2025: **0.1–0.8**. **Val 22 / test 122** = leakage or tiny val, not a good model.
- Public LB was ~33% of test; private shuffled ranks by ~2–3 places
- **Never share test CSVs or full notebooks across teams.** 2025 public LB had adjacent same-college scores identical to 3 decimals (42.106 twice; 40.797 vs 40.798). That was called out as possible plagiarism. Similar BERT + ChatGPT code can land *nearby* scores; **exact** matches look like the same file.
- Unstop **emails for top 10 / top 50 were late** in 2025 (rank 15 still had no mail; rumor that even top 10 lacked presentation invites). Screenshot the LB. Keep submission receipts. Final/private board came days later (OP expected the **17th**).

## Finale (top 10)

- End-to-end story + ablations with numbers
- What failed, latency, scaling, limits
- Do not name a model/idea you cannot explain in depth
- #1 LB often won the finale in **2023–2024**. **2025 broke that:** Test Data (IIT Patna) was **#1 public (SMAPE 39.19)** and finished **5th after presentations**. 00_Team_Rocket was **6th public / 3rd finale**. Prepare the talk as if score is not enough. Judges named in one post: Ajay Srinivasamurthy, Arunita Das, Deepak Gupta.

## Prep before 2026 drop

**Clock (official 24 Sep mail):** **25 Sep 2026 12:00 AM IST → 27 Sep 2026 11:59 PM IST.** Dataset + statement on Day 1. Finale **7 Oct**. Internshala 9 AM–9 PM is wrong.  
Eligibility in the dept mail: **2027 & 2028** BE/B.Tech/ME/M.Tech. Unstop: https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604?ref=AMHbdgmy

Before 25 Sep:

- [ ] Team already on Unstop (blog said 2–4; circular implied 3–4)
- [ ] Each person student-verified on AWS Builder Center
- [ ] Finish Explore AWS 5 activities if the second **$100** is still locked
- [ ] Billing alarm on; SageMaker notebook **stopped** when idle
- [ ] Know Notebook Instance + local XGBoost path (`official-2026-aws-builder-prep-guide.md`) — do not recreate the churn demo on Day 1
- [ ] Kaggle + Lightning / GPU as backup; `ml.t3.medium` cannot do VLMs

Practice on public past sets:

- 2023 length / 2024 OCR-from-image / 2025 price — all mirrored on Kaggle
- Kaggle NLP from “The Learning Lab”
- Have a **template repo**: loaders, image download, embedding cache, train/infer, submission writer, experiment log
- Pre-test OCR + CLIP/SigLIP + HuggingFace VLM LoRA on a small set so Day 1 is not env hell

## Career notes (PJ)

- Top 50 intern pipeline is the main career prize
- CGPA ~7.8 was fine
- Intern → FTE conversion is **low** and team-dependent
- Still the highest-leverage student ML contest in this circuit
