# Reddit 006 — 2024 solutions post-mortem (OCR + LLaVA)

- **URL:** https://www.reddit.com/r/learnmachinelearning/comments/1fhzz7a/solutions_of_amazon_ml_challenge/
- **Author:** u/xayushman (OP)
- **Captured:** 2026-09-15 from user paste (“1 more reply” missing; some Colab-speed replies truncated)

This is **not** the 2025 pricing contest. Rank/F1 numbers below are **2024**.

---

## OP — rank 206, F1 **0.447**

Pipeline evolution:

1. **10 hours** just to download test data and **upload to Kaggle** (anti-pattern; others avoided this)
2. Pretrained image-text-to-text: answers bad
3. Idea: OCR the image, feed **image + OCR text + query** into a small VLM
4. **PaddleOCR:** quality good, speed terrible. **4× P100**, 6 wall hours = 24 GPU-hours, **still not finished**
5. **EasyOCR:** worse OCR, much faster. **~10 GPU-hours** to finish extract
6. Small **LLaVA** for the actual prediction (sentence-form)
7. Postprocess: **Pint** + regex. Drop wrong-unit answers (query=height, pred=`15kg`)

Could not fine-tune a big VLM: download + compute already eaten the weekend. Asks others not to just say “train your model.”

LinkedIn rumor they heard: **top teams fine-tuned LLaVA 8B (~20GB)**. (PJ’s team used Qwen-VL + Donut — both can be true for different top-10 teams.)

---

## mopasha1 — F1 **0.489**, one submit 11:47, first ML challenge

- Started PaddleOCR, switched EasyOCR
- **Zero images downloaded.** `requests.get` + multithreaded dataloader
- Test split into **15 shards**, run on **7 Colab+Kaggle sessions**, OCR done in **~3.5 hours**
- **42,000 blank rows (~30% of test)** where EasyOCR extracted nothing
- Still **0.489 F1** → they estimate **>70% of detected rows were correct**. Filling the 42k blanks (e.g. PaddleOCR) might push **>0.6**
- Fallback idea (unbuilt): **KMeans** on image embedding + `group_id` + `entity_name`; if OCR empty, assign **train cluster-center value** (soap ≈ 50g prior)
- Spatial idea (built): **ResNet RPN** product box → vectors from product center to OCR boxes → angle vs x-axis:
  - ~0°/180° → height
  - ~90°/270° → width
  - ~45°/135°/225°/315° → depth
  - sort boxes by relevant angle, take **largest value**
  - Viz: https://imgur.com/HSKRx0l https://imgur.com/PiqzEs0
- Multi-product images: take the **largest product region**, not full-frame (full-frame OCR boxes can latch onto a smaller item)
- **Index bug:** merging shards labeled 0..n, but **test ids ≠ row order**. Official **sanity check** caught it. Fixed in Excel by copying `test.csv` indices. Almost ate the only submit.
- Tesseract: OK until they filtered **length/width/height**; train is **skewed to item_weight**. Abandoned Tesseract for EasyOCR.
- Compute takeaway they wrote: rent **RunPod / Paperspace** for a few hours
- Extra Kaggle accounts need **phone verify**; they used **6 Colab accounts** that morning instead
- LinkedIn: https://www.linkedin.com/in/mopasha/

**Do not copy the multi-account farm** as a plan — it fights Colab/Kaggle ToS. Legitimate equivalent: Lightning $15, RunPod, several *owned* machines, shard the work.

---

## Smooth_Loan_8851 — Tesseract, val F1 **~0.51** on 5k, **no valid submit**

- Own machine, 12 worker threads, **~4.5 hours**, no teammates
- Index mess at midnight; first failed submit; panicked past basic pandas
- Spatial H/W without RPN: compare OCR box `start_x/start_y` — height left or right of width; width above or below height. Got most H/W right that way
- Argues RPN is overkill if the dimension infographic is a clean diagram; mopasha1 disagrees because of **multi-product** shots
- Custom NER: <20% of spans correct; **OCR and train at the same time** = worst idea; used BytesIO, no download
- Tesseract for some users = “weird symbols”

---

## adithyab14 — F1 **0.42 → 0.48**, Lightning AI

GitHub: https://github.com/adithya04dev  
LinkedIn: https://www.linkedin.com/in/adithya-balagoni-78082b168/

### Time math (print this on Day 1 if the task is image-over-100k)

- **LLaVA-OneVision-Qwen2-0.5B:** 1–2 s/image → **~1.5 days** for 132k (batching/futures failed for them)
- Need **<100 ms/image** to finish 132k in ~4 hours. 1–2 s VLM ⇒ **40–80 hours**
- **PaddleOCR:** **~50 ms/image**, **1.5 hours for 132k** on Lightning ≈ **200× faster** than that 0.5B VLM

### What they actually ran

- Lightning **$15/month** credits; `download_images` utils: **132k test images in 20 minutes** (vs OP’s 10h Kaggle upload)
- Days 1–2 wasted: image embeddings → NN/XGB, loss **865890** (wrong problem framing)
- Then OCR + regex + unit map (`g/gm` → gram) + **XGBoost classifier** that picks **which (value, unit) pair** in the OCR list is the answer
- 40k train: OCR contained the gold entity_value in **26k**; classifier only recovered **16k**
- Spatial regex on Paddle reading order: height=first, depth=last, width=skip-1 → **0.39 → 0.48**
- NER idea arrived too late; 20 min / 26k labels → gibberish

---

## Other

| Who | Result | Note |
|---|---|---|
| Terrible_Bar_1158 | F1 **0.0016** | LLaVA + postprocess; Kaggle GPU died; submitted **only 1k labeled rows** |
| Colab T4 user | — | ~**1000 images/hour** OCR, gave up |
| Top teams (hearsay) | — | Fine-tuned **LLaVA 8B / 20GB** |
| Harshill09 | — | Looking for finale Twitch; 2024 stream unclear (2023 was Twitch) |

Related public OCR writeup not in-thread: Tensor Titans **22 / 74,843 teams**, F1 **0.6793** with only **38.85% rows filled** (empty is better than wrong). Custom line detection for H/W; leaving **depth empty** helped. ~0.2 s/image, no heavy VLM. https://github.com/nirvan840/Product-Image-Attribute-and-Entity-Extraction

---

## Playbook (2024-style / any OCR year)

1. **Never spend 10h uploading images to Kaggle.** Stream URLs, or Lightning `download_images` (~20 min).
2. **Time the first 100 images.** If >~100 ms and you have 1e5 images, you cannot VLM the full test. OCR first.
3. PaddleOCR > EasyOCR quality; EasyOCR >> Paddle on Kaggle-speed. **Cache OCR text to disk** once.
4. **Pint + allowed-unit map + drop mismatches.** Wrong unit is an F1 false positive.
5. Blanks are OK. F1 rewards leaving empties rather than guessing (0.679 with 39% fill; 0.489 with 30% EasyOCR blanks).
6. **Fill empties with group_id × entity_name priors** (median/mode/kMeans), not random VLM.
7. Use **OCR boxes** for H/W/D (angles or start_x/y). Train is weight-heavy — val must be **stratified by entity_name**.
8. **Sanity-check indices before submit.** Shard merges + sequential ids will fail. Official checker exists.
9. Don’t submit a 1k-row file. GPU death → still need a full-length CSV (empties allowed).
10. Lightning $15 / RunPod for a few hours beats 7 ToS-dodgy Colab logins.
