# Reddit 003 — 2025 public-LB plagiarism allegation (r/Btechtards)

- **URL:** https://www.reddit.com/r/Btechtards/comments/1o77kk6/amazon_ml_challenge_2025/
- **Subreddit:** r/Btechtards
- **Author:** u/VastThen1742 (OP)
- **Posted:** after 2025 public leaderboard, before final/private LB (OP expected final on the **17th**)
- **Flair:** Serious
- **Score:** ~94 upvotes
- **Captured:** 2026-09-15 from user paste (some deleted/mod-removed comments; “1 more reply” not included)

## OP claim

Public leaderboard clusters with **almost equal SMAPE, same college, adjacent ranks**:

| Rank | Team | SMAPE | College |
|---|---|---|---|
| 10 | ML Amigos | **40.797** | IIT Jodhpur |
| 11 | Gradient Ascenders | **40.798** | IIT Jodhpur |
| 41 | Low Voltage | **42.106** | VIT Vellore |
| 42 | Interstellars | **42.106** | VIT Vellore |
| 43 | Optimizers | **42.107** | VIT Vellore |

OP’s argument: identical SMAPE to 3 decimals is not a coincidence unless the **same CSV / same code + same epochs + same dropout + same ensemble**. “No one else from 82000 had that coincidence.” **82k is unverified** (other sources said ~5k–20k **teams**; this may mean participants or be inflated).

OP denied targeting “IITs” as a class; said they were already at “a better IIT.”

This is an **allegation**. Not proven in-thread. Store the scores; do not treat named teams as confirmed cheaters.

## Counter-claim

**Drake_18776:** nearby scores happen if people use the same BERT-family model and ChatGPT-similar code; epochs / bagging / boosting explain small gaps. Later (10 months): said they asked the IIT Jodhpur side — **minor SMAPE difference without rounding**; one team **DistilBERT**, other **BERT**; different epochs; ensemble/bagging/boosting changes.

**royal-retard** pushed back: the discussion was **same submissions**, and the named colleges were VIT Vellore + IIT Jodhpur, not “IITs in general.”

**Read for 2026:** 0.001 SMAPE (40.797 vs 40.798) can be similar models. **Exact 42.106 twice** is much harder to explain without shared predictions. Amazon/Unstop may look at this; **do not share CSVs or notebooks across teams.**

## Confirmed public-LB numbers (useful even if cheating is false)

- Rank **10–11** sat at **~40.80 SMAPE**
- Rank **41–43** sat at **~42.106**
- Rank **15** existed (Extreme-Apple-9576’s team) — still no mail
- So the earlier “top-50 cutoff ~42.2” guess is **consistent**: 42.106 was still inside ~top 45 on this snapshot

This is the **public** board. OP: **final leaderboard on the 17th.** Someone asked if they would **remove** suspected teams. Unknown.

## Score / method comments

- **detox_retarded:** 50.1 SMAPE — OP said “Good”
- **royal-retard:** 50.1 without BERT is actually **pretty good**. Most **~44** scores were BERT-family (**DeBERTa, ELECTRA**, …). Better than ~44 came from **ensembles; more models → better**. They themselves hit **42–42.5** and “missed out” on heavier ensembling.
- Someone’s rank **180–200**: OP called that **very good for a 3rd year**
- OP also answered **“50–55”** to a deleted question (could be their rank or SMAPE; not clear)

## Ops / comms (same pain as reddit-002)

- Final / private LB not out yet; people still asking “kb ayega”
- **No emails** yet for top 50 or even, per OP’s later rumor, **top 10 presentations**
- Rank 15 and a top-50 friend: **no mail**
- Presentation / “second round” rumored for the next day while LB still hidden
- Do not count on Unstop mail being on time. Screenshot ranks and keep submission receipts.

## Thread tone

Most comments are national-shame / “cheating culture” (Hacktoberfest, GSoC, CP, MLH). Low technical content besides the table and royal-retard’s BERT vs ensemble split. Several comments removed by mods.

## What to take into the playbook

1. Public LB snapshot: **~40.8 ≈ rank 10**, **~42.1 ≈ rank 40**
2. Single BERT-like model cluster **~44**; ensembles were the gap to **42**
3. 50 SMAPE with classical ML is respectable, not competitive for top 50
4. Identical scores at adjacent ranks will be called out publicly — **no collusion, no shared test CSVs**
5. Amazon’s shortlist is **not instant**; rank on public ≠ email in inbox
