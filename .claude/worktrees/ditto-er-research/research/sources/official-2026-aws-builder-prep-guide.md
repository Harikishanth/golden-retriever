# Official 2026 AWS Builder prep guide (Jatin Mehrotra)

- **Title:** Amazon ML Challenge 2026: Your Complete Prep Guide with Live Demo
- **Author:** Jatin Mehrotra — AWS Employee, Developer Advocate
- **URL:** https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo
- **Published:** 21 Sep 2026 · **Last modified:** 23 Sep 2026
- **Captured:** 24 Sep 2026 (full article text pasted by user after the Builder SPA blocked automated fetch)
- **Live session:** 21 Sep 2026, 5:00–6:00 PM IST · [Twitch recording](https://www.twitch.tv/) (link on the article; search “Amazon Student Programs” / Jatin)
- **Tags:** builder-center, aws-community, getting-started, aws-free-tier, machine-learning

This is an **AWS marketing + onboarding post**, not the problem statement. It does **not** leak the 2026 task, metric, or dataset. Treat Unstop as last word if this blog and the dept circular disagree.

**Related official source:** `official-2026-dept-circular.md` (college mail: register by **20 Sep**, team implied 3–4, no 48-hour bonus credits).

---

## Conflicts with the dept circular / earlier research

| Fact | Dept circular | This Builder blog | What to treat as true |
|---|---|---|---|
| Registration window | 7–**20** Sep 2026 | 7–**22** Sep 2026 | Registration is already closed either way (today is 24 Sep). Ignore. |
| Team size | not explicit; earlier Unstop/research said **3–4** | **2–4**, cross-college allowed | Use **Unstop team you already registered**. Blog is looser. |
| Degrees | BE/B.Tech/ME/M.Tech | also **PhD / M.S.** | Irrelevant if you already registered. |
| Contest credits | **$200** for all | **$200** all + **extra $100 at 48-hour mark for Top 500 teams** | New. Plan as if the extra $100 is real but **not Day-1 money**. |
| SWAG | top 10 + top 10 women-only | same | Same. |
| PPI | top 50 Applied Scientist intern | “Pre-Placement Interviews at Amazon” / top 50 | Same idea. |
| Must-have | AWS Builder Profile implied | **valid AWS Builder Center Profile ID is mandatory** | Already needed to register. |
| What you submit | not specified | **CSV + zip of code + approach document**. Not a running API. | Match this unless Day-1 rules say otherwise. |

---

## TL;DR of the article (author’s)

- **79,000+** students registered
- **$200** free AWS credits for every participant
- 72-hour ML hackathon
- Prizes **INR 2.25 lakh** + PPIs at Amazon
- Prep session 21 Sep; watch the Twitch recording
- Author socials: LinkedIn / Instagram (Jatin Mehrotra)

---

## What is the Amazon ML Challenge (as written)

Two-stage ML competition **only for engineering students in India** (B.E./B.Tech/M.Tech/M.S./PhD, graduating **2027 or 2028**).

- Team **2–4**, **cross-college allowed**
- Real-world dataset from Amazon
- Build an ML model for a business problem
- Need a **valid AWS Builder Center Profile ID**

### Stakes (blog table)

| Reward | Who |
|---|---|
| INR 1,00,000 + certificates + goodies | Winners |
| INR 75,000 + certificates + goodies | First runners-up |
| INR 50,000 + certificates + goodies | Second runners-up |
| Pre-Placement Interviews at Amazon | Top 50 teams |
| $200 AWS credits | All participants |
| Extra $100 credits at the **48-hour mark** | **Top 500 teams** |
| Certificates + Amazon SWAGs | Top 10 + top 10 women-only teams |

### Timeline (blog)

1. Registration → 7–22 Sep 2026
2. Prep virtual session → 21 Sep 2026, 5:00–6:00 PM IST (recording available)
3. 72-hour hackathon → **25–27 Sep 2026**
4. Top 50 results → **2 Oct 2026**
5. Grand finale → **7 Oct 2026** (top 10 present to Amazon scientists)

---

## What the live session covered

1. Cloud fundamentals (core AWS services)
2. Free resources worth **$579**
3. Live SageMaker demo: train + deploy a real model end-to-end

### 1) AWS Builder Center

Mandatory for registration. Also:

| Feature | Pitch |
|---|---|
| Free sandbox environments | Real AWS accounts, no card, **8 hours/week** |
| Hands-on workshops | Level 100–400 |
| Public builder profile | Portfolio URL |
| Student builder groups | University communities, 60+ countries |

Setup: builder.aws.com → Join → verify email → pick alias. No credit card. <5 minutes.

### 2) Student Rewards — up to $579

Launched **20 Aug 2026**. Card never required.

| Action | Reward |
|---|---|
| Verify student status | 12 months Skill Builder Premium (**$449**) |
| Earn **7** Builder Center badges | **$10** AWS credits |
| Earn **14** badges | **$20** AWS credits |
| Earn **21** badges | **$100** cert exam voucher |
| **Total claimed value** | **$579** |

Badges = daily activities (sign in, comment, publish, feature requests). Author: most students can hit 7 badges in week 1.

**Pro tip in article:** start badges now; $10/$20 arrive early and can be used on SageMaker during the challenge.

**Confirmed in comments (Jatin, ~27 min before capture):** same 7 / 14 / 21 ladder. Badge is **not instant** — “will show up in some time.” User AKSHAT KUMAR said adding a profile picture was mentioned on stream as giving a badge + credits.

### 3) AWS Free Tier + contest credits

- Every participant: **$200** contest credits for registering.
- New accounts also get Free Tier.
- **Important note in article:** on signup you get **$100** immediately; completing **5 “Explore AWS” activities** (EC2, Bedrock playground, Budgets, Lambda web app, Aurora/RDS) unlocks **another $100** → **$200 total**.

This matches the user’s screenshot: **Explore AWS = 0 of 5, $0 of $100**, EC2 activity “In progress.”

| Service | Free tier (as written) |
|---|---|
| SageMaker Notebooks | 250 hours on **ml.t3.medium** (2 months) |
| SageMaker Training | 50 hours on **ml.m5.xlarge** (2 months) |
| SageMaker Inference | 125 hours on **ml.m5.xlarge** (2 months) |
| S3 | 5 GB always free |
| Lambda | 1M requests/month always free |

**Caveats in article:**

- Set **billing alerts immediately**
- **Always delete endpoints** when done (~**$0.12/hour** even idle)
- Use **us-east-1** for compatibility

### 4) Cloud fundamentals (the only services they say you need)

| Service | What it is | Why they mention it |
|---|---|---|
| EC2 | VMs | SageMaker training jobs run on EC2 under the hood |
| S3 | Object storage | Data in, models out. Everything through S3 |
| DynamoDB | NoSQL | Results / metadata / feature store (you will almost certainly **not** need this in 72h) |
| IAM | Roles/permissions | The role that lets SageMaker read S3 and launch machines |
| VPC | Private network | SageMaker resources live in a VPC. Quick Setup creates one |

You do not need to be an expert. You need to know they exist.

---

## Stream vs blog — **do not skip** (author’s own IMP box)

| Stream (21 Sep) | This blog |
|---|---|
| SageMaker **Studio** (full workspace, needs a **Domain**, can be slow / quota-blocked on brand-new accounts) | **Notebook Instance** (plain Jupyter, no Domain) |
| Training on a **separate SageMaker training job** (pulls from S3, bigger machine / GPU) | **Local training inside the notebook** (CPU of ml.t3.medium, data already in memory) |
| Possibly **endpoint** deploy | **Local `model.predict()`** |

**For the ML Challenge, the author explicitly recommends:**

> Notebook Instance + local training + local prediction. You submit a **CSV**, not a running API.

Same code can run on Studio or a Notebook Instance. Use Training Jobs only when data is big or you need GPUs. Endpoints cost **$0.12/hour** even idle — do not leave one up.

If the Free Tier account is still “setting up behind the scenes,” **do not wait for Studio**. Use a Notebook Instance. Long-time AWS accounts can follow the stream’s Studio + training-job path.

---

## Demo: SageMaker Notebook Instance + local XGBoost churn

30-minute walkthrough. **Synthetic telecom churn**, ~5,000 rows, public AWS S3. Not the contest dataset.

### Step 0 — create the notebook

1. SageMaker **AI** console
2. Applications and IDEs → Notebook → Notebook instances
3. Create notebook instance
4. Name: e.g. `ml-challenge-notebook`
5. Instance: **ml.t3.medium** (free-tier 250h)
6. IAM: Create a new role → defaults → Create role
7. Wait 2–3 min → **InService**
8. Open **JupyterLab** → `+` → **conda_python 3** notebook

User screenshots match this exactly: notebook `ml-challenge-notebook`, `ml.t3.medium`, created **23 Sep 2026 4:57 AM**, status **InService**, Open Jupyter / Open JupyterLab.

### Step 1 — install

```python
!pip install xgboost scikit-learn -q
```

Kernel busy line at the bottom = it is working.

### Step 2 — session (optional for local training; needed if you later touch S3)

```python
import sagemaker
import boto3
import pandas as pd
import numpy as np
import time

session = sagemaker.Session()
role = sagemaker.get_execution_role()
region = session.boto_region_name
bucket = session.default_bucket()

print(f"Region: {region}")
print(f"Role: {role}")
print(f"Bucket: {bucket}")
```

- `Session()` → SageMaker client
- `get_execution_role()` → IAM role attached to the notebook
- `default_bucket()` → auto S3 bucket

### Step 3 — load public demo data

```python
data = pd.read_csv(
    f"s3://sagemaker-example-files-prod-{region}/datasets/tabular/synthetic/churn.txt"
)
df = data.copy()
print(f"Dataset shape: {df.shape}")
df.head()
```

Synthetic telecom churn (~5k customers: minutes, charges, service calls). AWS public bucket — no upload.

**For the real challenge:** upload *your* dataset to *your* S3 bucket, replace the URL, and give the notebook IAM permission to read it. Or skip S3 entirely and `pd.read_csv` a local file / Kaggle path.

### Step 4 — EDA

```python
print("Target variable distribution:")
print(df["Churn?"].value_counts())
print(f"\nChurn rate: {df['Churn?'].value_counts(normalize=True)['True.']:.1%}")

print(f"Features: {df.shape[1]} columns, {df.shape[0]} rows")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nColumn types:\n{df.dtypes.value_counts()}")
```

Author talking points:

- Target is `Churn?` (`True.` / `False.`)
- Demo set is ~50/50. **Challenge data may be heavily skewed** — that changes training and metrics.
- Demo: 21 columns, **zero missing**. Challenge data **will not** be this clean.
- Always check target distribution first.

### Step 5 — feature engineering

```python
from sklearn.model_selection import train_test_split

df = df.drop("Phone", axis=1)
df["Area Code"] = df["Area Code"].astype(object)
df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

model_data = pd.get_dummies(df)
model_data = pd.concat(
    [model_data["Churn?_True."],
     model_data.drop(["Churn?_False.", "Churn?_True."], axis=1)],
    axis=1
)
model_data = model_data.astype(float)
print(f"Processed: {model_data.shape}")
model_data.head()
```

Author talking points:

- Drop unique IDs (phone, serial, row number) — they do not predict
- Area code is categorical, not numeric
- Charges are redundant with minutes × rate
- One-hot encode Yes/No
- **SageMaker built-in XGBoost CSV convention:** no headers, **target in column 0**. Forgetting this is “the number one beginner mistake”
- “Feature engineering often matters more than algorithm. Spend **70%** of your time here.”
- “How do I know these rules? Documentation.”

### Step 6 — split 67 / 22 / 11

```python
train_data, validation_data = train_test_split(model_data, test_size=0.33, random_state=42)
validation_data, test_data = train_test_split(validation_data, test_size=0.33, random_state=42)

test_target = test_data['Churn?_True.']
test_data_no_target = test_data.drop(['Churn?_True.'], axis=1)

train_features = train_data.iloc[:, 1:]
train_labels = train_data.iloc[:, 0]
val_features = validation_data.iloc[:, 1:]
val_labels = validation_data.iloc[:, 0]

print(f"Training:   {train_data.shape[0]} rows")
print(f"Validation: {validation_data.shape[0]} rows")
print(f"Test:       {test_data.shape[0]} rows")
```

Train = textbook, val = practice test, test = final exam. Never touch test while experimenting.

### Step 7 — train XGBoost **locally** (missing `import xgboost as xgb` in the published snippet)

```python
dtrain = xgb.DMatrix(train_features, label=train_labels)
dval = xgb.DMatrix(val_features, label=val_labels)

params = {
    "max_depth": 5,
    "eta": 0.2,
    "gamma": 4,
    "min_child_weight": 6,
    "subsample": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

print("Training locally... (no separate machine needed)")
model = xgb.train(
    params, dtrain, num_boost_round=100,
    evals=[(dtrain, "train"), (dval, "validation")],
    verbose_eval=10,
)
print("\nTraining complete!")
```

No SageMaker training job, no extra machine, no quota. Author: “For the ML Challenge dataset, this is all you need.”

That last sentence is **marketing / beginner-path**. 2024–2025 top-50 was not “XGBoost on 5k rows in a notebook.” Treat this as the **format + EDA + GBM baseline**, not the winning stack.

### Step 8 — predict

```python
dtest = xgb.DMatrix(test_data_no_target)
predictions = model.predict(dtest)

print("PREDICTIONS (probability of churn):")
print("-" * 55)
for i in range(10):
    pred = predictions[i]
    actual = test_target.iloc[i]
    predicted = "CHURN" if pred > 0.5 else "STAY"
    actual_lbl = "CHURN" if actual == 1.0 else "STAY"
    status = "CORRECT" if predicted == actual_lbl else "WRONG"
    print(f"  Customer {i+1}: {pred:.3f} -> {predicted:5s} (Actual: {actual_lbl:5s}) {status}")
```

Threshold 0.5. Confidence matters (0.92 vs 0.51).

### Step 9 — metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

binary_preds = (predictions > 0.5).astype(int)
print("FULL TEST SET RESULTS:")
print(f"  Accuracy:  {accuracy_score(test_target, binary_preds):.1%}")
print(f"  Precision: {precision_score(test_target, binary_preds):.1%}")
print(f"  Recall:    {recall_score(test_target, binary_preds):.1%}")
print(f"  F1 Score:  {f1_score(test_target, binary_preds):.1%}")
```

Claimed result: **~92% accuracy**, no tuning, seconds. That is XGBoost on a **clean, balanced, tiny synthetic table**. It is not a contest score.

### Step 10 — save model

```python
model.save_model("xgboost_churn_model.json")
# reload:
# loaded_model = xgb.Booster()
# loaded_model.load_model("xgboost_churn_model.json")
```

**Submission as stated here:** zip of **code + approach document** (plus a predictions CSV). Not an endpoint.

### Cleanup

SageMaker console → Notebook instances → select → **Stop** (not Delete). Stopped notebooks do **not** bill. Files persist. Delete only after the challenge.

Endpoints are the opposite: they bill while idle.

---

## Author’s “do this now” list + links

- Twitch recording of 21 Sep session
- Builder Center: https://builder.aws.com
- Student Rewards: https://aws.amazon.com/builder
- Unstop registration page (already closed)
- SageMaker pricing: https://aws.amazon.com/sagemaker/pricing
- SageMaker SDK v3 tutorial: https://sagemaker.readthedocs.io

Disclaimer on article: opinions are the author’s, not necessarily AWS.

---

## Comments worth keeping (50 total; newest first at capture)

Most are “thanks.” Signal:

- **AKSHAT KUMAR:** added profile picture after stream said it gives a badge + credits.
- **Jatin reply:** 7 = $10, 14 = $20, 21 = cert voucher. Badge **lags**. Not instant.
- **Sreyajyoti Mondal, IIT BHU:** session helps with **challenge workflow**.
- Related article promoted: “AWS Student Rewards: Your Step-by-Step Guide to All 21 Badges” by Adidev V.N.

---

## What the user’s screenshots add (same blog/stream)

1. **Explore AWS** student-credit quest: 5 activities × $20 = $100. User is **0/5, $0**, EC2 “In progress.” Other four: Bedrock playground, Budgets, Lambda web app, Aurora/RDS.
2. Stream UI: Amazon Student Programs Twitch + SageMaker AI tile page (Studio, built-in algos, HPO, Autopilot, Experiments, Batch Transform).
3. Notebook instances console + create form (`ml-challenge-notebook`, `ml.t3.medium`, Amazon Linux 2025 / JupyterLab 4, SageMaker execution role, root access enabled).
4. Instance **InService** as of 23 Sep 2026 4:57 AM.

---

## Honest read for this team (not in the article)

This blog is **AWS onboarding**, written so 79k beginners can open a notebook and train *something*. It is useful for:

- Claiming credits without lighting money on fire
- Knowing **Studio vs Notebook Instance** (use Notebook Instance)
- Knowing **do not deploy an endpoint**
- A copy-paste **tabular GBM baseline** if Day 1 is structured data
- Confirming submit shape: **CSV + code zip + writeup**

It is **not** useful as a winning recipe. Past years that paid PPI were:

- 2021 browse-node classification
- 2023 product-length regression
- 2024 image entity extraction (F1; OCR vs VLM)
- 2025 smart product pricing (SMAPE; text emb + GBM/DNN, some vision)

An `ml.t3.medium` (2 vCPU, 4 GB RAM) **cannot** fine-tune a VLM or even comfortably embed 100k images. Use this box for **pandas + XGBoost/LightGBM + small sklearn**. Put heavy jobs on **Kaggle / HF / a real GPU** (Vultr, AWS `g4dn`/`g5` only if you have credits **and** a kill switch).

The **$200 + optional extra $100 at hour 48** is real money if you treat SageMaker like a rented GPU, not a 72-hour always-on Studio domain.

**Do tonight (24 Sep), from this article only:**

1. Finish Explore AWS 5 activities if you still need the second $100 (user screenshot was 0/5).
2. **Stop** the `ml-challenge-notebook` when not using it.
3. Set a **billing alarm**.
4. Confirm region (**us-east-1** per blog) and that the notebook role can read S3 **if** you will use S3.
5. Watch the Twitch recording at 1.5× for Studio/training-job bits you will skip.
6. Do **not** spend Day 1 recreating this churn demo. Recreate the **workflow**: EDA → drop IDs → holdout → metric → dumb baseline CSV.

Missing import in their Step 7: add `import xgboost as xgb` or the cell dies.
