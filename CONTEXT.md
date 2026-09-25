# Amazon ML Challenge 2026 — team context (agent-safe)

**File role:** single shared source of truth for humans + AI agents.  
**Timezone for every stamp:** IST (UTC+5:30).  
**Last rewritten:** `2026-09-25 09:50 IST`  
**Rewrite reason:** Rule-baseline holdout scores + Claude handoff (`CLAUDE_HANDOFF.md`).

If you are an AI: read **§0 then §1 then §4** before doing anything else. Do not skip to “build a model.”

---

## 0. Agent contract — do not hallucinate

`updated: 2026-09-24 23:41 IST` · `type: RULE`

1. Treat every block as one of: **FACT** (official 2026), **FINDING** (past contest, sourced), **ADVICE** (our inference), **UNKNOWN**, **CHANGELOG**.
2. Never invent **column names, row counts, file sizes, or 2026 LB scores.** The **task + metric + submit filenames** are now FACT in §3.0 (video caption, 2026-09-25 05:11). Still do not invent TSV schemas until someone prints `df.columns`.
3. Never invent leaderboard numbers for 2026.
4. Never “fill in” a model because last year was pricing / OCR / length. History is a **prior**, not this year’s task.
5. If two sources conflict, use **§5 authority order**. Do not average them.
6. If a teammate pastes new official text, **append a changelog row + a FACT block with a new timestamp.** Do not silently overwrite old text; strike the old line and point to the new stamp.
7. Do not recommend: extra Unstop/AWS/Claude accounts, sharing test CSVs with other teams, SageMaker **endpoints**, phone as the contest machine, dual login as one person, skipping college / fighting parents, recreating Jatin’s telecom-churn demo as Day-1 work.
8. Code you write must match **this year’s sample submit file**, not 2025’s `sample_id,price` memory.
9. When unsure, output `UNKNOWN` and the next **observation** (open file, print columns). Do not pick XGBoost “because SageMaker blog.”
10. Night-of-drop protocol was §6. Problem is known: **entity resolution**. Data is on disk. Next action = **CAP=80 holdout** → `run.py test` → `validate_submission.py` → one Unstop upload. No geocoding APIs. Full memory: `CLAUDE_HANDOFF.md`.

---

## 1. Changelog (append-only)

`type: CHANGELOG` · newest first

| Stamp (IST) | Who | What changed | Where in this file |
|---|---|---|---|
| 2026-09-25 ~16:30 | Claude (this chat) | BM25 blocking written + LightGBM + rapidfuzz (15 feat). v4 running locally but RAM-bottlenecked (7047s to load). Cloud strategy decided: Kaggle/AWS/Vultr. Dense retrieval plan (sentence-transformers + FAISS) for tonight. `rapidfuzz 3.14.6` + `lightgbm 4.7.0` installed locally. model.json deleted (no submittable model on disk). | §3.12, `notes.md`, `CLAUDE_HANDOFF.md` |
| 2026-09-25 ~13:00 | Claude (this chat) | Pipeline fully rewritten: rules → learned LR. v1 holdout (CAP=80, 9 feat) = **0.738**. v2 (CAP=150, 10 feat + JW) val **0.7415**, ceil **0.834**. v3 (CAP=150, 11 feat + TF-IDF cos) val **0.7422**, ceil **0.835**, ceil_macro **0.9287**. Hash-key ceiling confirmed: cannot exceed 0.93. | §3.11, `notes.md`, `CLAUDE_HANDOFF.md` |
| 2026-09-25 09:50 | local (holdout + this chat) | Rule pipeline live. Holdout F0.5 **0.592 → 0.680** after ranked shortlist. Uncapped block recall **0.887**; hard cap 60 cut it to **0.693**. `CAP=80` in code **unscored**. No test TSV / no Unstop submit. | §3.10, `notes.md`, `CLAUDE_HANDOFF.md` |
| 2026-09-25 07:41 | local (streamed `dataset/student_resource/dataset`) | **S1 train 2.21M / test 1.73M.** Singleton rate **5.58%**. France 15% of test S1. All-empty F0.5 ≈ 0.056. Public 0.70/0.96 are real matchers, not empty-list. | §3.9, `src/eda_stats.md` |
| 2026-09-25 05:26 | local (Unstop LB screenshot) | Public snapshot ~5h in. Top ~0.96 / 0.94 / 0.92. A ~0.70 band also live. Our team header: **Da Big Three** (Harikishanth R +2). Refresh every 15 min. | §3.8 |
| 2026-09-25 05:25 | local (Unstop PDF / portal paste) | Full schema: 4 cols, France in **test only**, zip layout, 8B MIT/Apache cap, candidate_pairs = **last** pre-model set, Student Resource download. Dataset download in progress. | §3.0, §3.7 |
| 2026-09-25 05:11 | local (Unstop video + official caption paste) | **2026 task = business entity resolution.** Macro F0.5. `matching_results.tsv` only LB file. Blocking + `candidate_pairs.tsv` audited. No external lookup. | §3.0, §3.6, §4 |
| 2026-09-24 23:41 | local (this rewrite) | Team-share rewrite; FACT/FINDING/ADVICE split; Reshma night protocol; algorithm catalog | whole file |
| 2026-09-24 23:27 | local | First CONTEXT.md (personal schedule draft). **Superseded** by 23:41 rewrite. Keep only as history. | — |
| 2026-09-24 23:16 | Harikishanth R inbox → team | Official Unstop pre-challenge email pasted. Clock = 25 Sep 00:00 – 27 Sep 23:59 IST. 5 subs/day. Public+private. Top 100 + blocking writeup. | §3 |
| 2026-09-24 23:12 | local | Full Jatin Builder blog captured (credits, Notebook Instance, local XGBoost demo). | §3.4, §11 |
| 2026-09-15 | college circular | Register 7–20 Sep; contest 25–27 Sep; finale 7 Oct; $200 + $579; PPI top 50. No clock hours. | §3 |
| — | Internshala scrape (pre-email) | Claimed contest 25 Sep **09:00** – 27 Sep **21:00** IST. **SUPERSEDED** by 23:16 official mail. Do not use. | §5 |

**When Reshma (or anyone) opens Unstop tonight, add a row immediately**, even if the problem is “not up yet”:

```
| 2026-09-25 HH:MM | Reshma | Unstop status: <not live / live> · files: <names> · metric: <quote or UNKNOWN> | §3.0 Day-1 drop |
```

Then paste verbatim quotes into **§3.0**. Do not paraphrase the metric.

---

## 2. Team and availability

`updated: 2026-09-24 23:41 IST` · `type: FACT` (names) + `ADVICE` (roles)

| Person | Known facts | Availability 25 Sep | Suggested role until Day-1 files exist |
|---|---|---|---|
| **Reshma** | Teammate. May be **awake tonight** and will check Unstop. | Night of 24–25 Sep: **primary drop watch** if she is actually up. | §6 only: download, read, dummy CSV, write §3.0 |
| **Harikishanth R** | Received official Unstop mail (captured 23:16 IST 24 Sep). | UNKNOWN unless he says | Keep mail / Google Form URL |
| **Local author of this file** | Lin/logreg only. Parents: cannot stay up. College Fri: wake 05:00, bus 06:10, campus 07:10–15:00, home **16:30**. | Dead 00:00–16:30 Fri. Owns Fri 16:30 → bedtime + Sat/Sun house hours. | After drop: EDA / baseline / notes. Not night GPU. |
| Other members | **UNKNOWN** — fill names here | UNKNOWN | — |

**Submit button:** 2025 pattern was leader-only. **2026: UNKNOWN.** Confirm on Unstop. Do not have one person own every password.

**Team size already registered:** FACT = whatever Unstop shows. Blog said 2–4; older notes said 3–4. Do not re-register.

---

## 3. Official 2026 facts only

Do not mix Reddit into this section.

### 3.0 Day-1 drop

`updated: 2026-09-25 05:25 IST` · `type: FACT` from Unstop problem page + video · texts: `research/sources/official-2026-problem-statement.md`

```
STATUS: official text complete; Student Resource download in progress; row counts still UNKNOWN
UNSTOP_LIVE: yes
DOWNLOAD: Unstop problem page → "Download Data Set" / "Student Resource" (student_resource/ zip)
PROBLEM_TITLE: Business Entity Resolution Challenge
TASK_FAMILY: matching / ER
TASK_IN_ONE_SENTENCE: For every S1 entity, find all matching S2 and S3 records (zero / one / many).
METRIC_NAME: macro F0.5 (per S1 entity, then average; singletons included)
METRIC_DIRECTION: higher-better
METRIC_FORMULA: F0.5 = (1.25 * P * R) / (0.25 * P + R)
METRIC_BIAS: precision ~2x recall; official example P=2/3 R=1 → F0.5=0.714
SUBMIT_FILENAME: matching_results.tsv
SUBMIT_COLUMNS: source1_entity_id <TAB> matched_entity_ids
CANDIDATE_COLUMNS: source1_entity_id <TAB> candidate_entity_ids
SOURCE_COLUMNS: entity_id, business_name, business_address, country
GT_COLUMNS: source1_entity_id, matched_entity_ids
TRAIN_FILES:
  dataset/train/train_source1.tsv
  dataset/train/train_source2.tsv
  dataset/train/train_source3.tsv
  dataset/train/train_ground_truth.tsv
TEST_FILES:
  dataset/test/test_source1.tsv   # predict a row for EVERY id here
  dataset/test/test_source2.tsv
  dataset/test/test_source3.tsv
N_TRAIN: UNKNOWN
N_TEST: UNKNOWN
HAS_IMAGES: no
COUNTRY_TRAIN: US + India
COUNTRY_TEST: US + India + France (France absent from train)
COUNTRY_RULE: open-set string; do NOT hard-code / filter / one-hot only {US, India}
PARAM_OR_MODEL_CAP: <= 8B params; MIT or Apache 2.0 license
SUBMISSION_ZIP: <team>_submission.zip → output/ + code/business_entity_resolution/{src,README,requirements} + Documentation_template.md
FIRST_DUMMY_LB_SCORE: UNKNOWN
FILE_FORMAT: TSV because addresses and ID lists contain commas
VALIDATE: from student_resource/: python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
EXTERNAL_DATA: forbidden (ER APIs, gov registries, geocoders, internet augment)
FINAL_RANK: private LB (PDF). Email still said shortlist uses both — do not overfit public.
```

### 3.1 Clock

`updated: 2026-09-24 23:16 IST` · `type: FACT` · `source: official Unstop email to Harikishanth R`

- Challenge window: **25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST**
- Problem + dataset: **Day 1**
- Build + submit: through **Day 3**
- Internshala 09:00–21:00 is **void**

Secondary (weaker) dates, not in that email:

- Unstop timeline card 2: **2 Oct 2026 10:00–23:59 IST** (screenshot 2026-09-25; purpose not fully visible — do not invent)
- Top 50 list: 2 Oct 2026 — `source: Internshala` `type: SECONDARY`
- Finale: 7 Oct 2026 — `source: dept circular + Internshala + Jatin blog` `type: FACT` (date) / finale hours 10:00–15:00 IST = Internshala only
- Registration: 7–20 Sep (circular) vs 7–22 Sep (Jatin). **Closed either way.**

### 3.2 Rules from the same email

`updated: 2026-09-24 23:16 IST` · `type: FACT`

- Cheating / plagiarism / **multiple IDs** → instant DQ
- Live leaderboard during contest; **final** board after
- Queries: Google Form (**URL not in paste** — pull from original mail)
- Artefacts with best solution:
  - 1–2 page doc: approach, models, experiments, conclusion
  - Commented source: experiments, training, inference
- **5 submissions per day × 3 days**, then submit disabled
- Keep version history of submissions; shortlisting uses submitted solutions; **final source may be demanded later**
- **Public and private** LBs; evaluation + shortlisting use **both**
- After artefacts + score + **every member eligible** → **top 100** announced
- Top 100 then submit: methodology · **candidate generation / blocking strategy** · model architecture + FE · other
- Desktop/laptop **only**
- **One machine per participant**; simultaneous login can terminate the session
- Support: `support@unstop.com` + screenshot + registered email; they will not make modeling decisions

### 3.3 Stakes (mixed official sources — labeled)

`updated: 2026-09-24 23:12 IST`

| Item | Value | Source | Stamp |
|---|---|---|---|
| Winner / RU1 / RU2 | ₹1L / ₹75k / ₹50k | circular + Jatin | 2026-09 |
| PPI Applied Scientist intern | top 50 teams | dept circular | 2026-09-15 |
| Top 100 extra writeup | official mail | email | 2026-09-24 23:16 |
| SWAG | top 10 + top 10 women-only | circular + Jatin | 2026-09 |
| Registered students | 79,000+ | Jatin blog | 2026-09-21/23 |
| Contest AWS credits | $200 all participants | circular + Jatin | 2026-09 |
| Extra $100 at **48-hour mark** | top 500 teams | Jatin **only** | 2026-09-23 |
| Student Rewards pack | $579 claimed (Skill Builder + badge credits + cert voucher) | circular + Jatin | 2026-09 |
| Eligibility years | 2027 or 2028 | circular + Jatin + email context | 2026-09 |
| Cross-college | allowed | Jatin | 2026-09-23 |
| Submit shape | **LB file = `matching_results.tsv`** (not a generic CSV). Final zip + `candidate_pairs.tsv` + code + methodology | official video | 2026-09-25 05:11 |

**Do not collapse “top 50 PPI” and “top 100 docs” into one cutoff.** They are different documents.

### 3.4 Official compute / SageMaker (Jatin blog)

`updated: 2026-09-24 23:12 IST` · `type: FACT` (what AWS said) · not a winning recipe

- Builder Center profile was mandatory to register
- $200 = **$100 on signup** + **$100 after 5 Explore AWS activities** (EC2, Bedrock playground, Budgets, Lambda app, Aurora/RDS)
- Badge ladder: 7 → $10, 14 → $20, 21 → cert voucher; badges **lag** (Jatin comment)
- Recommended for this contest: **Notebook Instance + local train + local predict**
- Do **not** wait for SageMaker Studio Domain on a new account
- Endpoints ≈ **$0.12/hour even idle** — delete/stop
- Stop notebook instances when done (stop ≠ delete)
- Prefer **us-east-1**; set billing alerts
- Demo dataset = synthetic telecom churn ~5k rows, **not** the contest
- Demo model = local XGBoost, binary logloss; published snippet omitted `import xgboost as xgb`
- Free-tier notebook they used: **ml.t3.medium** (2 vCPU / 4 GB class)

Blog: https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo  
Full extract: `research/sources/official-2026-aws-builder-prep-guide.md`

### 3.6 Problem mechanics (official video)

`updated: 2026-09-25 05:11 IST` · `type: FACT`

- **S1** = Amazon-side reference, clean, already deduplicated.
- **S2, S3** = vendor dumps, noisy names/addresses, **no shared identifier**.
- Video said name+address only. **PDF adds `country`.** PDF wins. Phone/email still not in the files.
- Each record has its **own ID**.
- Output per S1 id: complete match set of S2/S3 ids, or **empty**.
- Two-stage pipeline they describe: **blocking** (cheap name+address key → candidate pairs, recall-heavy, includes lookalikes) then **matching model** (drop false pairs).
- Blocking sets the **recall ceiling**. You cannot match a record you never put in a bucket.
- Lookalikes they name: similar name / different address; different business / same address.
- **Singletons:** S1 with no S2/S3 match. Empty list → **1.0 on that entity**. Any predicted match → **0 on that entity**.
- **Macro F0.5:** false merge (precision error) costs ~**2×** a miss (recall error). Prefer not to merge when unsure.
- Region-specific name/address patterns matter.
- `candidate_pairs.tsv` is **audited, not LB-scored**. Still required in the final zip.

### 3.7 Portal PDF additions (2026-09-25 05:25)

`type: FACT` · supersedes video where they conflict

**Read:** `pd.read_csv(..., sep="\\t")`. No tab → one smashed column. Tabs exist because addresses and ID lists contain **commas**.

**Source row:** `entity_id` (`S1-` / `S2-` / `S3-`), `business_name`, `business_address`, `country`. No separate source column.

**Train countries:** US, India. **Test also has France** (not in train). Country is an **open string**. Do not hard-code, filter, or one-hot only `{US, India}`. Every test S1 row including France must appear in the submit file.

**Noise they name:** Corp/Corporation, Pvt/Private, Ltd/Limited, DBA, `&`/`and`, word-order swap, typos; Rd/Road, St/Street, transliteration, missing PIN/state, “Near SBI ATM”, number formats, reordered address parts.

**`candidate_pairs.tsv` definition (strict):** the set the **matching model actually scores** — last stage if you have many blockers, **not** an early wide pass you later filter. Every id in `matching_results.tsv` must appear in the candidate list for that S1. Validator **warns** if a match was never a candidate.

**Rejection (no score):** missing S1 rows, S1 self-match, IDs not in test, duplicate ids in a list, duplicate `source1_entity_id` rows. Good format → portal status **SCORED** + F0.5.

**Model license / size:** final model **MIT or Apache 2.0**, **≤ 8B params**. (2025 rumor is now official.)

**LB:** you always upload the **full** test. Public = subset. **Final ranking = private** (this PDF). Pre-challenge email said shortlist uses both — still do not chase public only.

**Self-score:** no test GT. Hold out from **train** and compute F0.5 yourself.

**Docs:** fill `Documentation_template.md`. **No page limit** (this PDF). Email’s “1–2 page” is weaker for the zip.

**Zip tree:**

```
<team_name>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/src/
├── code/business_entity_resolution/README.md
├── code/business_entity_resolution/requirements.txt
└── Documentation_template.md
```

**Official first features to try (their tips, not our invention):** Jaccard, Levenshtein, TF-IDF cosine on name and address; country-specific address patterns **without** locking the pipeline to two countries.

**Download path:** same Unstop problem page → **Download Data Set / Student Resource**. On this machine: `D:\ML challenge\dataset\student_resource\`.

### 3.9 Dataset on disk (streamed)

`updated: 2026-09-25 07:41 IST` · `type: FACT` · detail: `src/eda_stats.md`

On-disk root: `D:\ML challenge\dataset\student_resource\dataset\`

| File | Rows |
|---|---:|
| train S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 |
| train GT | 2,206,821 (1:1 with S1) |
| test S1 / S2 / S3 | 1,732,544 / 4,887,273 / 5,082,316 |

Headers confirmed. No empty names. Empty addresses: S2/S3 only (~3%).

Countries: train **US + India only**. Test S1: US 663,106 · India 809,986 · **France 259,452 (15.0%)**.

Train GT: **singletons 123,247 (5.58%)**. With matches: mean **3.67** ids, mode 3, max 11. S2 links 3.69M + S3 links 3.94M.

**Train diagnostic** `src/eval_exact_match_train.py` (lower+collapse space on name and address; skip empty addr):

- S1 with any exact hit: **59,633 / 2,206,821**
- **macro F0.5 = 0.073** (all-empty floor ≈ 0.056)

So a naive exact dummy is almost useless. Public 0.9x requires **real normalization + blocking + similarity**, done at **~12M-row** scale (hash joins, not nested loops).

### 3.12 BM25 + LightGBM pipeline (afternoon 2026-09-25)

`updated: 2026-09-25 ~16:30 IST` · `type: FINDING` · `code:` `code/business_entity_resolution/`

**Architecture (current code on disk):**

`blocking.py` — full BM25 token retrieval. Index keyed by `(country, 'N'|'A'|'H', token)`. Scoring: name token IDF×2.0, addr token IDF×1.2, house number fixed +8.0. MAX_POSTING=2000 guard prevents OOM. CAP=150.

`match.py` — 15 features: 11 original + rapidfuzz `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `addr_sort_ratio`.

`run.py` — LightGBM (n_estimators=500, num_leaves=63) with sklearn LR fallback. Batched test inference (20k entities/batch) to avoid OOM on 1.73M test set. Model saved as `model.lgb` + threshold in `model.json`.

**v4 holdout (running locally, ~16:00+):**
- BM25 index: 2,152,388 posting lists (vs 22.4M compound hash keys before)
- Local machine hit RAM bottleneck: index ~8GB + pool ~2GB → swapping. Pool loaded at 7047s (117 min!). Featurization not started yet.
- STATUS: still running. May finish overnight. **No model.json on disk right now.**

**Installed locally:** `rapidfuzz 3.14.6`, `lightgbm 4.7.0`

**Cloud plan (tonight):**
- Kaggle (29GB RAM, T4 GPU, free): run BM25+LightGBM in ~35 min
- OR AWS SageMaker ml.r5.2xlarge (64GB, ~$0.50/hr)
- OR Vultr GPU instance

**Dense retrieval (GPU, tonight):**
- `sentence-transformers/all-MiniLM-L6-v2` (22M params, MIT)
- Encode 12.3M records → 384-dim embeddings → FAISS top-100 per S1
- Expected recall ceiling: **>0.95**
- Combined BM25 + dense features → LightGBM
- Expected holdout F0.5: **0.88-0.93**

**No Unstop submit yet. model.json/model.lgb do not exist on disk.**

### 3.11 Learned LR pipeline (this chat, 2026-09-25)

`updated: 2026-09-25 ~13:00 IST` · `type: FINDING` (train holdout, not public LB)
`code:` `code/business_entity_resolution/` · `log:` `notes.md` · `handoff:` `CLAUDE_HANDOFF.md`

**Architecture change:** rules replaced by logistic regression on 11 pair features. IDF computed in the same blocking pass. No external dependencies — stdlib + numpy + sklearn at train time; dot product only at test time (model.json).

**Blocking changes:**
- `build_index_and_idf()` replaces `build_index()` — single pass builds index + IDF
- 8 blocking keys (added NT2, NLC, NP to the original 5)
- CAP raised to 150; ranked by key-type score before truncation

**Feature set (11):**
`idf_name_jac`, `name_contain`, `char_sim` (trigram Jaccard on squashed name), `house_agree`, `addr_jac`, `addr_exact`, `city_agree`, `name_rare`, `name_len_ratio`, `jaro_winkler` (on full folded name), `tfidf_cos`

**Learned coefficients (v2, retrained on 100k):**
`char_sim +9.45`, `addr_jac +6.69`, `jaro_winkler +4.37`, `house_agree +3.98`, `name_len_ratio -4.59`, `idf_name_jac -2.03`, `name_rare +0.52`, `name_contain +0.40`

| Run | CAP | Features | Val F0.5 | Ceil | Notes |
|---|---:|---:|---:|---:|---|
| v1 (11:00) | 80 | 9 | 0.738 | 0.814 | first LR; rules → model |
| v2 (OOM on report) | 150 | 10 (+JW) | **0.7415** | **0.834** | model saved; Phase 6 OOM fixed |
| v3 (running ~13:00) | 150 | 11 (+tfidf_cos, NP key) | UNKNOWN | ~0.84+ | NP key adds 226k extra blocking entries |

**Not uploaded.** No `output/` TSVs. Do not treat as LB score.

### 3.10 Rule baseline on this machine

`updated: 2026-09-25 09:50 IST` · `type: FINDING` (train holdout, not public LB)  
`code:` `code/business_entity_resolution/` · `log:` `notes.md` · `handoff:` `CLAUDE_HANDOFF.md`

**Not uploaded.** No `output/` TSVs yet. Do not treat 0.68 as a leaderboard number.

Pipeline: 5-key blocking (country + squashed name / house+street / house+city / full addr / first-token+city) → ranked shortlist → precision-first rules (house+name-sim, rare exact name, or exact addr + name evidence).

| Stamp | What | macro F0.5 | Blocking recall ceiling | Mean cands / S1 |
|---|---|---:|---:|---:|
| 07:44 | exact name+addr, **full train** | 0.073 | — | — |
| 08:36 | multi-key + **hard first-60** | 0.592 | 0.693 | 52.7 |
| 09:38 | ranked shortlist, **CAP=40** | **0.680** | 0.736 | 36.2 |
| diagnostic | same keys, **no cap** | — | **0.887** | (80.8% of S1 had >60 raw ids) |
| code now | `CAP = 80` in `blocking.py` | **UNKNOWN** | UNKNOWN | — |

Other measured constraints (same morning, samples / 100k holdout):

- Name-only blocking ceiling ~**0.51** — abandoned.
- Matcher recall **on true pairs already in the candidate set:** **0.756**.
- ~**63%** of singletons share a squashed name with some S2/S3 row → no name-only accept except rare names.
- House numbers agree ~**72.5%** true pairs vs ~**5.5%** singletons.

**ADVICE:** next observation is holdout at CAP=80, then `run.py test` + validator, then **one** Unstop submit. Raising CAP without ranking was the 0.887→0.693 failure; ranking without a large enough CAP left ceiling at 0.736.

### 3.8 Public LB snapshot

`updated: 2026-09-25 05:26 IST` · `type: FINDING` (public subset only) · `source: Unstop screenshot, OCR noisy`

Contest clock on page: ~**2d 18h** left (consistent with 00:00 start). Banner: board refreshes **every 15 min**.

Our team chrome: **Da Big Three** — Harikishanth R +2. No scored row visible in the crop.

| Public rank | Team | Score | Submit time (IST) | Notes |
|---|---|---|---|---|
| 1 | (PESU Bengaluru) | **0.964733** | — | |
| 2 | ExploitationVsExploitation | **0.938794** | — | IIT (ISM) Dhanbad |
| 3 | H5X | **0.924977** | — | MGIT Hyderabad |
| 4 | ISeeData | **0.9211** | 02:36 | IIEST Shibpur |
| 5 | Chooser | **0.6966** | 03:18 | Woxsen |
| 6 | (cut off) | ~0.69? | 01:46 | |

**ADVICE (not a fact):** ~0.70 is the shape of “almost all empty” if many S1 are singletons. ~0.92–0.96 is the shape of **normalized exact/near-exact + empty when unsure**, not a trained LLM. Private subset can move this. Do not treat 0.96 as the finale trophy.

**SUPERSEDED 07:41:** train singleton rate is **5.58%**, so all-empty ≈ **0.056**. Raw exact name+address on train = **0.073** (`src/eval_exact_match_train.py`). Public 0.70/0.96 are **not** empty-list and **not** raw exact match. See §3.9.

### 3.5 Links

`updated: 2026-09-24 23:41 IST`

- Unstop: https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604
- Support: support@unstop.com
- Builder: https://builder.aws.com
- Student rewards: https://aws.amazon.com/builder

---

## 4. Explicit UNKNOWNs (2026)

`updated: 2026-09-25 09:50 IST` · `type: UNKNOWN`

**Now known:** schema, filenames, metric + example, France rule, 8B + MIT/Apache, zip tree, validator CLI, candidate_pairs = last pre-model set, **row counts, singleton rate 5.58%, country strings `US`/`India`/`France`, holdout F0.5 0.680 at ranked CAP=40**.

**Now known:** v3 val=0.7422, ceil=0.835, **ceil_macro=0.9287** (hash-key ceiling). rapidfuzz + lightgbm installed. BM25 blocking code written. Cloud resources identified (Kaggle/AWS/Vultr). Dense retrieval plan ready.

Still UNKNOWN:
- v4 holdout F0.5 / ceil_macro with BM25 blocking (running locally, slow)
- This team’s **public LB** score — **zero Unstop submits so far**
- Dense retrieval ceiling (expected >0.95 with sentence-transformers + FAISS)
- Who clicks submit; whether failed validate burns a daily shot
- How many of Friday’s 5 shots already used
- Test singleton mix (no test GT)
- France failure modes on dense retrieval (no France train)

---

## 5. Authority order

`updated: 2026-09-24 23:41 IST` · `type: RULE`

1. Unstop page / Day-1 PDF / in-platform sample files **after they exist**
2. Official email (24 Sep 23:16 capture)
3. This file’s **FACT** blocks with the **newest** stamp
4. Jatin blog — AWS how-to and credit numbers only
5. Dept circular — eligibility, cash, PPI
6. Internshala — **hours already wrong**; other dates secondary
7. `research/` FINDINGS — past years only
8. ADVICE blocks — last

---

## 6. Reshma tonight — Unstop protocol

`updated: 2026-09-24 23:41 IST` · `type: ADVICE` for the human who is awake  
`goal:` files on Drive + one legal dummy + §3.0 filled  
`not the goal:` a trained model

### 6.1 If Unstop is still locked

1. Screenshot “not started” / countdown.
2. Changelog row: `Unstop not live`.
3. Recheck every 10–15 min until 00:20. If still dead at 00:30, write that and sleep or wait — **do not invent a problem.**
4. Ping the team either way.

### 6.2 If Unstop is live

Do **in this order**. Stop after step 8 unless you still have energy for 9.

1. **Download everything** to a shared Drive / zip. Official train, test, sample submit, PDF/statement. Do not rely on streaming later.
2. Screenshot the rules page (metric, filename, sub cap).
3. Open **sample submit file**. Write exact filename + exact columns into §3.0.
4. Open train + test: `shape`, column list, 5 rows, missing %, whether images/URLs exist, whether there are two id spaces / two tables.
5. Quote the metric sentence into §3.0. Do not reword “SMAPE-like” unless the PDF says SMAPE.
6. **Dummy submit** (1 of 5 Friday):
   - Copy **test ids from test file**, not `0..n`
   - Target = train mean / median / majority / empty / sample-row clone — whichever is **legal** for this metric
   - Columns + order + filename = sample file
7. Record dummy **public** score in §3.0 and `subs/20260925_dummy.csv`.
8. Paste into team chat: metric, filename, columns, n_train, n_test, has_images, dummy score, Drive link.
9. Optional if <45 min extra: target histogram / class balance + 10 dirty rows. That is EDA, not modeling.

**Do not:** SageMaker Studio, endpoints, geocoding APIs, Google Places, `pip` of a 7B model, train 2 hours before a dummy.

### 6.4 After the video (25 Sep morning)

`updated: 2026-09-25 05:11 IST` · `type: ADVICE`

1. Download the data zip. Read **every** table with `sep="\\t"`.
2. Print columns, 10 rows, counts per source, % of train S1 with empty GT (singleton rate).
3. Run `utils/validate_submission.py` on a dummy before Unstop upload.
4. **First dummy (precision-first, F0.5):** for each S1, emit a match only if normalized name **and** normalized address are **exact-equal** to an S2/S3 row (optionally require same `country` string). Everyone else = empty list. Do not drop France rows. Candidates for the dummy = that same exact-match set (so matches ⊆ candidates).
5. Save as `subs/20260925_dummy_exact.tsv`. Record LB score here.
6. Do **not** skip a real blocker later. Dummy exact-match is not the blocking strategy for the zip audit.

**Login:** one laptop, one browser profile. No phone submit. No second Unstop account.

### 6.3 Message template after the drop

```
DROP <HH:MM IST>
metric: <quote>
submit file: <name> cols: <a,b>
train/test: <n>/<n>
images: yes/no
two tables / pairs: yes/no
dummy score: <x>
Drive: <link>
§3.0 in CONTEXT.md updated: yes
```

---

## 7. Problem-statement identification (do this before choosing a model)

`updated: 2026-09-24 23:41 IST` · `type: ADVICE` · run only after files exist

**This is the most important modeling decision.** Wrong family = 72 hours on the wrong loss.

### 7.1 Classify the supervision

Walk the first matching row. Write the chosen family into §3.0 as `TASK_FAMILY`.

| If you observe… | Task family | Typical metric family | First legal dummy |
|---|---|---|---|
| One row, discrete label, predict class | **Classification** (binary / multi / extreme multi-class) | accuracy, F1, logloss, AUC | majority class |
| One row, numeric target | **Regression** | MSE/MAE/RMSE, **SMAPE**, MAPE | train mean or **median** |
| Image or text → value+unit / span | **Extraction / IE** | F1, exact match, unit-constrained F1 | empty or regex hit; **empty > illegal unit** (2024 FINDING) |
| Query + corpus / two catalogs | **Retrieval** | recall@k, mAP | popular-item / identity if ids overlap |
| Two records → same/different or link | **Matching / ER / linking** | F1, precision/recall, AUC on pairs | block on exact key, predict “same” if key equal |
| Ranked list / relative order | **Learning to rank** | NDCG, pairwise loss | baseline ranker (popularity) |
| Sequence / generate text | **Seq2seq** | token F1 / constrained decode | copy field / empty |
| Multi-label attributes | **Multi-label class** | micro-F1 | per-label majority |

**2026 prior (not a fact):** Amazon catalog problems 2021/23/24/25. Official top-100 field **“candidate generation / blocking”** is **ER vocabulary**. Use it as a **hypothesis to test on the files**, not as the answer.

### 7.2 Blocking / candidate generation (only if family = match / retrieve)

`type: ADVICE` grounded in standard ER, not in a 2026 PDF.

**Blocking** = cheap rule that produces a **small candidate pair set** so you do not score N×M.

Common blockers (scientific names):

- **Equality block:** same brand / asin / normalized title
- **Token overlap / Jaccard / cosine-on-TF-IDF**
- **Sorted neighborhood:** sort keys, window
- **Canopy / LSH / minhash**
- **Q-gram / phonetic** (Soundex) for typos
- **Embedding kNN** (FAISS) as a soft block — heavier

Then a **pair scorer** (logistic / GBM / cross-encoder) on the candidate pairs only.

If the data is a **single table with a price**, do **not** invent a blocking pipeline. Write “N/A — single-row regression” in the top-100 doc if you ever need that heading.

### 7.3 Metric → loss (scientific)

| Official metric | Train loss that usually **matches** | Train loss that usually **mismatches** |
|---|---|---|
| MSE / RMSE | MSE | MAE-only if outliers dominate — still check |
| MAE | MAE / Huber | MSE (outlier-dominated) |
| **SMAPE** | SMAPE or Smooth L1 on **log1p** target (2025 FINDING) | raw MSE on price |
| MAPE | log / relative | raw MSE |
| Accuracy | CE | MSE |
| F1 / PR | CE + threshold sweep; focal if rare class | accuracy-only selection |
| Unit-constrained F1 | generate only allowed units; empty if unsure | free-form VLM string |

**Target transforms (regression):** if `target` is right-skewed (2023 length, 2025 price FINDING), train on `log1p(y)` and invert. Clip insane preds.

### 7.4 Validation (non-negotiable)

- Hold out **before** model shopping. Stratify by class / entity / price bucket if those exist.
- Public LB is a **subset**. 2025 FINDING: ~33% public, private moved ~2–3 ranks. 2026 public fraction = UNKNOWN.
- Selection = holdout (and both LBs for shortlist). Not public alone.

---

## 8. Algorithm / training catalog (checkboxes for the real statement)

`updated: 2026-09-24 23:41 IST` · `type: ADVICE` + pointers to FINDINGS  
Use as a **menu**. Tick only what the files justify.

Legend: `C` class · `R` regress · `E` extract · `M` match · `V` vision · `T` text

### 8.1 Linear / classical (this team can actually run tonight)

| Method | Families | When it is the right tool | When it is a trap |
|---|---|---|---|
| Majority / mean / median dummy | C R | **Always first submit** | stopping here all weekend |
| Linear / Ridge / logistic | C R | small tabular, baseline, high-dim bag-of-words | raw skewed price; 9k-way softmax without features |
| Regularized GLM | C R | interpretability, writeup | images |
| kNN on raw rows | C R M | low-d, or **postprocess** (round to nearest train y — 2023 FINDING) | high-d raw pixels |
| Naive Bayes | C | bag-of-words text class | regression |
| Decision tree | C R | debug / splits you can explain | solo SOTA |
| Random forest | C R | medium tabular, noisy | very high-card text without hashing |
| **GBDT: LightGBM / XGBoost / CatBoost** | C R M | **default** for tabular + hashed text + pair features | 2025 price-only GBM ~56 SMAPE FINDING — weak vs text towers, still a required baseline |
| Isotonic / Platt / threshold sweep | C | F1 / PR contests | regression SMAPE (use different postproc) |

**Training notes (GBDT):** `max_depth` 4–8; early stop on holdout; CatBoost if many categoricals; class weights / `is_unbalance` if rare class; **never** early-stop on public LB.

### 8.2 Sparse text (CPU, Day 1)

| Method | Families | Use | Trap |
|---|---|---|---|
| Regex / unit maps / pack-qty flags | T E R | 2024 units, 2025 pack size FINDING | thinking regex is the whole 2024 contest |
| TF-IDF + SVD + linear/GBM | T C R | strong cheap baseline | 9k-class without hierarchical / proto |
| Hashing vectorizer | T | huge cardinality, no vocab freeze | need interpretability |
| Character n-grams | T | typos (2023 FINDING) | slow if n huge |
| GLiNER / NER | T | brand / entity flags (2025 rank-8 FINDING) | using it as the only model |

### 8.3 Dense embeddings + small head (frozen)

| Method | Families | Use | Trap |
|---|---|---|---|
| Frozen BERT-family + linear/MLP | T C R | 2023 first baseline; 2025 150M emb + head → **42 SMAPE** FINDING | calling 150M “LLM” |
| Sentence-Transformers / E5 / GTE | T C R M | retrieval, pair cosine, cheap text | assuming SBERT always enough (2025 T5-base comment) |
| CLIP / SigLIP text+image | T V R M | multimodal catalog | CLIP batch-norm scale drift vs SigLIP2 (2025 AIR20 FINDING) |
| DINOv3 / ViT frozen | V | image prior | vision when same image has many prices (2025: images can be **noise**) |
| Cross-encoder (two texts in) | M | pair matching after blocking | scoring all N×M |

**Training:** freeze encoder, train head 1–5 epochs, AdamW, cosine or constant small LR on head only. Cache embeddings to disk **once**.

### 8.4 Fine-tune encoders (needs GPU + time budget)

| Method | Families | Use | Trap |
|---|---|---|---|
| Unfrozen BERT / DeBERTa / RoBERTa | T C R | 2023 length winners; 2025 DeBERTa-v3 + FE **top-10 text-only** | 5-epoch BERT-only ~48 SMAPE 2025 |
| LoRA / QLoRA on LLM/VLM | T V E | 2024 Qwen2-VL | 1–2 s/image × 1e5 rows = **40–80 h** (FINDING) |
| Triplet / contrastive + kNN | T V R M | 2025 comment (Qwen+DistilUSE+CLIP+vis); 2023 nearest-length cousin | no holdout on the k |
| Unfrozen ViT+BERT joint | T V R | 2025 20–22 h → 43s then blend to 42.1 | concat-and-hope (~47) |

**Training:** LoRA rank 8–16; 4/8-bit; sample **same distribution** as full train; one long train **or** frozen+head — decide with holdout, not Twitter. Time 100 images before VLM commit.

### 8.5 Extraction / OCR / VLM

| Method | Families | Use | Trap |
|---|---|---|---|
| PaddleOCR / EasyOCR + Pint + allowed-unit map | E | 2024 F1 **0.45–0.49** band | 10 h download+Kaggle upload (FINDING) |
| Donut / Idefics / Qwen-VL | E | 2024 top band 0.62–0.86 | illegal units; GPU death + partial CSV (0.0016) |
| Label curation | E | 1.6k clean >> 20k dirty (2024 FINDING) | more dirty data |
| Box geometry for H/W/D | E | 2024 spatial FINDING | using it for weight |

### 8.6 Matching / ER stack (if §7.1 says match)

| Stage | Methods | Note |
|---|---|---|
| Normalize | lower, unicode, strip pack size, brand map | do this before blocking |
| Block | keys in §7.2 | report **reduction**: pairs kept / N×M |
| Score | Jaccard, TF-IDF cosine, GBM on pair feats, cross-encoder | train on labeled pairs if labels exist |
| Cluster | union-find / connected components | if “same product group” |
| Postproc | one-to-one constraint (Hungarian) if required | only if rules say unique match |

### 8.7 Ensembles / postprocess (Day 3)

| Method | Use | FINDING |
|---|---|---|
| Weighted average of **diverse** CSVs | last hours | 2025 45.5→42.1 last step = old CSVs |
| `alpha * model_a + (1-alpha) * model_b` tuned on **metric** | two heads | 2025 rank 8 SMAPE mix |
| Round to nearest train target | discrete-looking y | 2023 length |
| Unit allow-list / empty | extract | 2024 |
| Calibrate threshold | F1 | sweep on holdout only |

### 8.8 Training-job types (infra, not algorithms)

| Mode | When | Cost risk |
|---|---|---|
| Local laptop / notebook (`ml.t3.medium`) | pandas, GBDT, small sklearn | leave instance **InService** overnight |
| Kaggle / Colab GPU | frozen emb, one FT | session kill; 2025 Colab complaints |
| SageMaker **training job** | need a bigger box than the notebook | spins EC2; watch credits |
| SageMaker **endpoint** | **not for this contest** (CSV submit) | **$0.12/h idle** |
| Multi-account farm | **forbidden** | DQ |

---

## 9. Past-challenge FINDINGS (sourced, not 2026)

`type: FINDING` · do not treat as this year’s task

### 9.1 Format that stayed stable

`updated: 2026-09-15 / 2026-09-24`

- ~72 h, team, live public LB, private later, 1–2 pager + code zip
- Catalog / product-attribute flavor **every documented year**

### 9.2 2021 — browse-node class

- Task: product → `BROWSE_NODE_ID` (~9,919 classes)
- Inputs: TITLE, DESCRIPTION, BULLET_POINTS, BRAND
- Size: train ~2.90M, test ~111k · metric accuracy
- What scored: multilingual BERT / XLM-R; SBERT + kNN
- Sources: `research/past-challenges.md`

### 9.3 2023 — product length regression

- Winner ART (PJ et al.), Kaggle GPUs only; #1 in ~1.5 d
- ~2.2M products; skewed length
- **FINDING:** log target, clip, FT BERT+RoBERTa + product-type embedding; **round pred to nearest train length**; min(BERT, RoBERTa) vs high-side bias
- Character n-grams helped typos
- Sources: Team Fishes repo, winner notebook, `past-challenges.md`

### 9.4 2024 — image entity extract (F1)

- Image + entity → `"<value> <unit>"` with **allowed units only**
- ~264k train rows, ~1e5+ images
- **FINDING:** OCR+regex band **0.45–0.49**; VLM/curation **0.62–0.86**; 1.6k clean labels >> 20k dirty
- PaddleOCR ~50 ms/img; VLM 1–2 s/img can exceed the clock
- Empty > illegal unit; pad CSV if GPU dies
- DBkaScam finale 6, F1 71.8: MiniCPM zero-shot + Qwen2-VL few-shot + Qwen SFT **vote**; SFT 64.8 **lost to** few-shot 70.9
- Index bug (test ids ≠ 0..n) zeroed teams
- Sources: reddit-006, KhadgaA, DBkaScam repo

### 9.5 2025 — smart product pricing (SMAPE, lower better)

- Public cutoff guess **~42.2** for ~top 50; rank 10 ~40.80
- Public #1 Test Data **39.19** → **5th finale** (talks matter)
- **FINDING:** log-price “the main big thing”
- 150M text emb + head, **no images, no preprocess** → **42**
- Text-only DeBERTa-v3 + flags + cross-attn → **3rd finale**; images dropped after UMAP
- Frozen CLIP+Qwen→NN ~47; unfrozen ViT+BERT 20–22 h then **CSV blend** → 42.1
- Rank 8: SigLIP + second head, alpha mix, GLiNER + ~70 binaries, drop ~6k outliers, ≤~900M params, RTX 4050 6GB for FE
- GBM-only ~56; 5-epoch BERT FT ~48; RNN+images+BART clusters ~50
- Honest val↔LB **0.1–0.8**; val 22 / LB 122 = leakage
- 5 valid subs/day (2025 mail); score ~5 min; public ~33% test
- Do not share CSVs across teams; identical 3-decimal scores were called out
- Sources: reddit-002..005, catalog-web-sweep, 00_Team_Rocket repo

### 9.6 Score bands 2025 (public SMAPE) — FINDING only

| Band | Meaning in 2025 |
|---|---|
| ~39–40.8 | top 10 public |
| ~42.1–42.2 | ~rank 40 / cutoff talk |
| ~44 | single BERT-family |
| ~47–51 | stall / weak vision / RNN |
| ~56 | plain LGBM |

---

## 10. ADVICE distilled from §9 (not official 2026)

`updated: 2026-09-24 23:41 IST` · `type: ADVICE`

1. Dummy CSV in hour 1. Format zeros people.
2. Holdout before ideas.
3. Match the metric; log1p if target skewed.
4. Parallel a **dumb** model (median / TF-IDF / frozen emb) and one “real” idea. Kill with holdout.
5. FE / cleanup often > new architecture. 70% time on features is Jatin **ADVICE**, consistent with 2023–25 FINDINGS.
6. Ablate images. Keep only if holdout improves.
7. Time 100 images before VLM.
8. Keep every CSV; Day-3 blend.
9. 5/day — do not waste on broken format.
10. Write the 1–2 pager as you go.
11. Public #1 ≠ finale cup (2025 FINDING).
12. This team’s linreg person should not own a 22 h unfrozen train. They own dummy, EDA, GBDT/logreg, writeup glue.

**72 h shape (PJ ADVICE, still reasonable):** Day 1 explore + dummy; Day 2 deepen one winner; Day 3 ensemble + docs. Last 15 min move ranks — no 6 h train at 23:00 Sunday.

---

## 11. Compute on this team (status, stamped)

`updated: 2026-09-24 23:41 IST` · `type: FACT` (as last reported) / confirm if stale

| Resource | Last known | Stamp | Note |
|---|---|---|---|
| AMD credits | dead | 2026-09-24 chat | ignore |
| AWS contest credits | $200 official; Explore AWS **0/5** on one screenshot | 2026-09-24 | second $100 may still be locked for that account |
| SageMaker | `ml-challenge-notebook` `ml.t3.medium` InService 2026-09-23 04:57 | 2026-09-24 | **STOP when idle** |
| HF | $25 | 2026-09-24 | small |
| Claude/Builder keys | coding hours, not GPU | 2026-09-24 | no extra accounts |
| Kaggle / Lightning | should exist | — | first real GPU |

`ml.t3.medium` = pandas + GBDT. Not VLM FT.

---

## 12. Landmines (past FINDINGS → treat as likely)

`updated: 2026-09-24 23:41 IST`

- `pd.read_csv` without `sep="\\t"` (official: columns will not parse)
- Wrong filename (`matching_results.tsv`, not `test_out.csv`)
- Test id = row index
- Predicting any match on a singleton (that entity scores **0**)
- External geocoder / Maps / company DB (DQ)
- Empty or fake `candidate_pairs.tsv` in the final zip (audited)
- Partial CSV
- Endpoint left on
- Dual login
- Phone attempt
- Multi-ID
- Sharing preds across teams
- Early-stop on public LB
- Recreating churn demo

---

## 13. How to update this file (humans + agents)

`updated: 2026-09-24 23:41 IST` · `type: RULE`

1. Add a **changelog row first** (stamp, who, one line).
2. New official text → new **FACT** subsection with stamp + raw quote. Strike old FACT, do not delete.
3. New experiment → `notes.md` + one line here only if it changes strategy. New agent session → refresh `CLAUDE_HANDOFF.md` (do not silently fork a second handoff).
4. Never put a guessed 2026 metric in §3. Put guesses in a dated ADVICE note.
5. If Reshma fills §3.0, bump **Last rewritten** at the top.

Suggested disk layout (create after drop):

```
CONTEXT.md
data/          official files only
subs/          never overwrite CSVs
notes.md       running log with timestamps
src/
writeup/
research/      already exists
```

---

## 14. Local research paths (not required for Reshma tonight)

| Path | Contents |
|---|---|
| `CLAUDE_HANDOFF.md` | **2026-09-25 09:50** agent memory: convo, decisions, scores, next steps |
| `notes.md` | holdout experiment log (append-only) |
| `code/business_entity_resolution/` | current rule baseline |
| `research/playbook.md` | tactics dump |
| `research/past-challenges.md` | year-by-year |
| `research/sources/official-2026-problem-statement.md` | official video caption + named files |
| `research/sources/official-2026-unstop-pre-challenge-email.md` | full mail |
| `research/sources/official-2026-aws-builder-prep-guide.md` | full Jatin extract |
| `research/sources/official-2026-dept-circular.md` | college mail |
| `research/sources/reddit-001` … `006` | PJ + 2025 + 2024 OCR |

---

## 15. One-screen summary for an agent

`updated: 2026-09-25 ~13:00 IST`

- **Task:** 3-source ER. Cols: `entity_id, business_name, business_address, country`.
- **France is in test only.** Do not one-hot {US, India}.
- **LB:** `matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`). Macro F0.5. Final = **private**.
- **Candidates:** last set the model scores; must be a superset of matches.
- **Cap:** ≤8B, MIT/Apache 2.0. No external lookup / geocode.
- **Data on disk.** Train S1 2.21M, test S1 1.73M, singletons 5.58%. Exact dummy F0.5 = 0.073.
- **Pipeline (current):** BM25 token retrieval blocking + LightGBM + 15 features (incl. rapidfuzz). v4 running locally, RAM-bottlenecked. Best completed holdout: **val 0.7422** (v3, hash keys). Hash-key ceil_macro = **0.9287** — cannot exceed without BM25/dense retrieval.
- **No Unstop submit yet.** Immediate next: move to Kaggle or AWS (29-64GB RAM). BM25+LightGBM run in ~35 min on cloud. Dense retrieval (sentence-transformers+FAISS) tonight → expected 0.88-0.93. Full dump: `CLAUDE_HANDOFF.md`.
