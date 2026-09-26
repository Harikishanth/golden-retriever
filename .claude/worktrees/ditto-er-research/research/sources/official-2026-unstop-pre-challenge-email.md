# Official 2026 pre-challenge email (Unstop / Team Amazon ML Challenge)

- **To:** Harikishanth R (user’s teammate / inbox)
- **From:** Team Amazon ML Challenge 2026
- **Captured:** 24 Sep 2026 ~23:16 IST (user paste)
- **Authority:** higher than Internshala scrape and the Builder blog for **clock, submit caps, artefacts, LB rules**
- **“Prep before you start” blog** = `official-2026-aws-builder-prep-guide.md` (Jatin SageMaker demo)

---

## Clock (this mail wins)

**25 September 2026, 12:00 AM IST → 27 September 2026, 11:59 PM IST.**

Not Internshala’s 25 Sep 9:00 AM – 27 Sep 9:00 PM. Not “~9 AM.” Midnight start.

Problem statement + dataset drop on **day 1**. Build and submit through **day 3**.

---

## Full text as received

> Hi Harikishanth R,
>
> We appreciate your participation in the Amazon ML Challenge 2026!!
>
> With the upcoming ML Challenge, we urge you to review the following key instructions and guidelines meticulously. Your attention to detail and adherence to these guidelines will greatly contribute to your success in this endeavour.
>
> Prep before you start: Walk through the blog for the ML Challenge on best practices and the live demo.
>
> Note: Any form of cheating, plagiarism, or unfair practices, such as registering and attempting the challenge via multiple IDs, will not be tolerated and will lead to instant disqualification of the participant.
>
> Kindly read through the following vital instructions and important guidelines for this round:
>
> Key Instructions:
>
> Challenge Window: 25th September 2026, 12:00 AM IST to 27th September 2026, 11:59 PM IST.
> All teams will get access to the problem statement and the dataset on day 1 and will have time to build and submit solutions till day 3.
> Teams can track their performance through the leaderboard, which will reflect team rankings live over the course of this challenge. After the challenge, the final leaderboard will be revealed.
> Please use this Google Form to ask any queries during the hackathon.
> The below-mentioned artefacts need to be shared for the best solution submitted by the team:
> 1-2-page document explaining the ML approach, ML models used, experiments and conclusion.
> Source code used for experiments, training and inference, with proper comments describing the functions.
> Each team can make a maximum of 5 submissions per day for over 3 days of the hackathon, after which the submit button will be disabled.
> Maintain the version history of all your submissions, as shortlisting will be based on the submitted solutions. Participants may also be required to submit the final source code at a later stage.
> There will be two leaderboards - Private and Public. Evaluation and shortlisting will be based on performance across both leaderboards.
> After successful submission of the artefacts, leaderboard score and each team member satisfying the eligibility criteria, the top 100 teams will be announced.
> The Top 100 teams will then be required to submit the following details/documents:
> Methodology used
> Candidate generation/ Blocking strategy
> Model Architecture and feature engineering
> Any other relevant information about the approach.
>
> Simultaneous Logins and Accessibility:
>
> You can attempt the ML Challenge on a desktop or laptop only and not on a mobile device.
> Simultaneous logins are not allowed; i.e. you can only attempt the ML Challenge from one laptop or desktop per participant.
> In case simultaneous logins are detected, the system may terminate the ML Challenge altogether, and you may only get error messages.
>
> Other instructions:
>
> If you face any technical problem, clear your browser's cache or try it on a different browser or in incognito mode.
> You may also try changing your internet - mobile hotspot, wifi, etc.
> Please send an email to support@unstop.com with a screenshot of the page where you are facing a problem and your registered email ID. Please note that we won't be helping you make decisions, and any email asking us to make decisions will not be entertained.
>
> All the best!
>
> Regards,
> Team Amazon ML Challenge 2026

Google Form URL was not in the paste (“Please use this Google Form…”). Pull it from the original mail if queries come up.

---

## Rules that change how you play

| Rule | Detail |
|---|---|
| Submit cap | **5 per day × 3 days**, then button dies. Same as 2025 mail. |
| Keep every CSV | Version history; shortlisting uses submitted solutions; they may demand **final source later** |
| Artefacts (with best solution) | **1–2 page** approach + **commented** train/infer code |
| LBs | **Public and private**. Shortlist uses **both** — do not chase only public |
| Cut for next docs | **Top 100** (circular/blog said top 50 PPI; this mail’s next gate is 100) |
| Top-100 extra writeup | methodology, **candidate generation / blocking**, architecture + FE |
| Device | **Desktop/laptop only.** No phone. |
| Logins | **One machine per participant.** Dual login can **kill the session** |
| Multi-ID / plagiarism | Instant DQ |
| Support | `support@unstop.com` + screenshot + registered email. They will not make modeling decisions. |

---

## Problem-type leak (not confirmed, but the wording is specific)

Top-100 required field: **“Candidate generation / Blocking strategy.”**

That is entity-resolution / record-linkage language (pair candidates, block on brand/title/asin, then score matches). It can also be leftover template text from an Amazon-internal matching brief. **Do not lock the whole team to ER before the files drop.** Do open Day 1 expecting:

- pairing / dedup / catalog match, **or**
- retrieval + rerank, **or**
- generic tabular/text and that bullet is copy-paste.

If the data looks like two tables / left-right records / “match or not,” blocking is the official Day-1 job, not a VLM.
