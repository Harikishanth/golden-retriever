# Reddit 001 — PJ (Poojan) on Amazon ML Challenge 2025

- **URL:** https://www.reddit.com/r/Btechtards/comments/1ntmvrr/amazon_ml_challange_is_back_for_2025_i_won_it_2/
- **Subreddit:** r/Btechtards
- **Author:** u/DotOtherwise1256 (PJ / Poojan Vachharajani)
- **Posted:** ~1 year before Sep 2026 (around 2025 challenge)
- **Score:** ~168 upvotes
- **Captured:** 2026-09-15

## Who PJ is

- Team: **ART in Artificial Intelligence**
- 2023: leaderboard **#1** and **winner** after finalist presentations
- 2024: leaderboard **#5** and **runners-up** after finalist presentations
- Interned at Amazon as Applied Scientist twice (3rd and 4th year B.Tech)
- Now full-time Applied Scientist at Amazon
- GitHub: https://github.com/pj-mathematician
- 2023 winner notebook: https://github.com/pj-mathematician/Amazon-ML-Challenge-2023/blob/main/amazon-ml-challenge-2023-winner-solution.ipynb
- 2023 finale Twitch: https://m.twitch.tv/videos/1804684510
- LinkedIn (from 2023 win post): Poojan Vachharajani; teammates Ansh Tanwar, Chaitanya Giri, Harshit Kumar
- Explicitly asked people **not** to DM / LinkedIn him; replies only in comments

## Core nature of the contest (PJ)

- 3-day, Kaggle-like **core ML** contest, **not** a full-stack hackathon
- Dataset + problem statement + metric; optimize the metric
- Huge train/test volume → efficiency matters as much as model quality
- JavaScript / product-building skills are useless here
- Team of **4**, all strong in Python, NumPy, Pandas
- Top 50 → OA + interview for Applied Scientist internship
- Top 10 → final presentation
- Trend 2023–2024: **#1 on leaderboard also won the finale**
- “Relatively easy to get top 50” because the problem is hard; basic ML can get you there
- Aim for **top 3 / #1**

## PJ's timeline

| Day | What to do |
|---|---|
| Day 1 | Everyone tries different approaches. Test on a **holdout validation set**. Find what works. |
| Day 2 | Refine the working approach. **Do not try anything new.** |
| Day 3 | Ensemble. Protect leaderboard rank. |

## Compute / modeling advice (PJ)

- Don't sleep, or minimize sleep
- Practice Kaggle playgrounds if new to core ML
- GPU: Kaggle T4 is usable (they won 2023 on Kaggle GPUs only) but be frugal
- Colab Pro or college GPU cluster is better
- Comfortable GPU memory: **40–80 GB**
- Don't waste time training huge models on full huge data
- Use **LoRA**, sampling, smaller task-specific models
- ChatGPT / AI tools **suck at optimizing** Kaggle-like scores. They can produce a working baseline but are weak at iterating to the best score. Do **not** 100% rely on them.
- One Colab Pro is enough; buy **after** dataset drop if Kaggle T4 is insufficient
- PJ personally did **not** downsample in 2023/2024; other top-10 teams did and it worked **if distribution was preserved**
- How he picks models: look at the **closest public benchmark** to the task

## 2024 (PJ's team)

- Problem: structured text extraction from images (OCR / entity extraction)
- Models: **Qwen VL** + **Donut**
- Reason: Qwen VL was best open-source visual LLM then; Donut is a Kaggle-standard OCR-finetune model
- Other teams mentioned in comments: LLaVA 8B, multithreading, PaddleOCR, EasyOCR

## 2023

- Finale presentations: Twitch VOD above
- Winner notebook linked above
- ~24k participants; they hit #1 in ~1.5 days and held it

## Presentation tips (top 10)

- End-to-end solution: what worked, what didn't, with metrics
- Limitations, scaling, latency
- Every teammate must deeply understand every tech mentioned
- If you mention VAEs, judges will grill VAEs — don't include what you don't understand

## Career / eligibility answers from PJ

| Question | PJ's answer |
|---|---|
| Worth it if you want research, but only internships offered? | Full-time conversion is low; depends on team + intern performance. Internship itself has more research opportunities. |
| Intern → PPO conversion criteria (CGPA, papers)? | (He didn't answer the later conversion-rate comment.) CGPA: his was **~7.8**, they didn't care. No CGPA cutoff mentioned for PPI. |
| Beginner with basic ML? | Need **NLP and CV fundamentals at least** |
| Coursera vs Kaggle? | Python then Kaggle. Coursera doesn't help that much. |
| 2nd year participate? | Participate even if ineligible, to get a feel. |
| Classical ML vs DL/NLP/CV? | Contest is Kaggle-like score chasing. “Classical” here meant TF-IDF / CNNs as easy-to-learn NLP/CV baselines — not tabular-only contests. Years vary. |
| 2025 topic guess (at the time of post)? | Something **multimodal**: image/audio/video → text |

## Comment-thread operational intel

### Compute reality (2024 especially)

- Commenters called 2024 “pay to win”: bigger model → better score
- Dataset: ~2 lakh images; teams begged friends for Lightning AI $15 GPU credits
- RTX 3090 (24 GB): “okish”, not comfortable; people claimed 50 GB+ ideal
- GTX/RTX 3050: still participate with grit
- Top 50 often had school GPUs or Colab Pro

### Submissions (2024/2025 era)

- Cooldown: **~5–10 submissions/day** (memory of commenters)
- Many teams struggled to submit even once
- Output filename and columns must match spec exactly
- 2025 pricing task: `test_out.csv` with columns `sample_id` and `price`
- Common error: `tuple indices must be integers or slices, not str` — often caused by wrong filename/columns/zip structure (per u/Lazy-Reputation8960)

### Practice resources PJ recommended

- 2023 and 2024 datasets are on Kaggle — rerun them
- Kaggle NLP comps from **The Learning Lab**

### Teammate hunting (historical, from this thread)

Many comments were “looking for teammates” (LLM/finetune people, last-year top 300, M.Tech + Colab Pro). Not operationally useful now except: **DL/LLM people are the scarce slot**.

### Later comments (near 2026)

- PJ (6 days before capture): “goodluck for 2026!”
- Neon297 asked whether modern AI helps with **ideation** (not 100% AI solutions). PJ had not answered that in the captured thread.

## Notable community takes (not PJ)

- “Vibe coding chal jayegi kya /s” — joke, but aligns with PJ: AI-only is weak
- “Chatgpt se toh mat likhvata”
- 2024 was OCR-heavy; 2023 was NLP regression on catalog text
- HuggingFace VLMs were the 2024 meta

## Links PJ posted

- 2023 finale: https://m.twitch.tv/videos/1804684510
- 2023 winner notebook: https://github.com/pj-mathematician/Amazon-ML-Challenge-2023/blob/main/amazon-ml-challenge-2023-winner-solution.ipynb
