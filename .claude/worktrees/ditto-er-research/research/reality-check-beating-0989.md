# Reality Check: Beating 0.9889 (Current #1)

**Why this file exists:** Original winning strategy artifact assumed top score was 0.9647. Actual current #1 is **0.9889**. This changes the difficulty calculus dramatically.

**Date:** 2026-09-26  
**Your baseline:** 0.7422  
**Current #1:** 0.9889  
**Target:** 0.99  
**Gap to #1:** +0.0011 (11 basis points)  
**Gap from baseline:** +0.2478  

---

## The New Reality

### What 0.9889 Means

**Current #1 is near-perfect:**
- 0.9889 = roughly 19,000 errors across 1.73M entities (1.1% error rate)
- 0.99 = 17,300 errors allowed
- **You need 1,700 FEWER errors than them**

**Error breakdown at 0.9889:**
- Singletons (96k): ~1,100 errors (1.14% error rate)
- Non-singletons (1.63M): ~17,900 errors (1.10% error rate)
- France (260k): ~3,500 errors (1.35% error rate)

**To beat them to 0.99:**
- Singletons: <950 errors (reduce by 150)
- Non-singletons: <16,350 errors (reduce by 1,550)
- France: <2,900 errors (reduce by 600)

---

## What They Have (Reverse-Engineered)

**To hit 0.9889, top team likely has:**

### 1. Blocking (0.99+ recall)
- Triple union: Dense (BGE-m3 or GTE-Qwen2) + BM25 + MinHash/LSH
- Possibly ColBERT late interaction for ultra-high recall
- Top-50 to top-100 candidates per entity

### 2. Matcher (Ensemble)
- **7-10 models**, not 3-5
- At least one fine-tuned on contest data (16+ hours training)
- BGE-reranker-v2.5-gemma-9B (9B params, violates 8B limit BUT highest quality)
- Multiple cross-encoders: Jina-v2, mxbai, XLM-RoBERTa, custom fine-tuned
- Stacked ensemble (meta-model on top)

### 3. France (Fine-Tuned)
- NOT zero-shot multilingual
- Fine-tuned on back-translated French pairs (synthetic data)
- Possibly separate model just for France (not shared with US/India)
- Expected France F0.5: 0.96-0.97

### 4. Singleton Detection (0.998+ accuracy)
- Three-stage cascade: LightGBM → Cross-encoder → LLM
- LLM verification on ALL borderline cases (not just 5k)
- Qwen2.5-7B or Llama-3.1-8B with careful prompting
- Per-entity confidence-based threshold (not global)

### 5. Threshold Optimization
- Trained meta-model for per-entity thresholds
- Adaptive based on: confidence, match count, country, entity features
- Calibrated on 100k+ holdout with grid search + Bayesian optimization

### 6. Post-Processing
- Graph-based transitive closure (union-find)
- Correlation clustering to fix inconsistent predictions
- Manual rules for edge cases (e.g., "Inc" vs "LLC" normalization)

### 7. Compute Investment
- **$1,000-2,000** over **72+ hours**
- Multiple fine-tuning runs (16h each × 3-5 models)
- Large-scale inference on 100-150 candidates/entity (150M+ pairs)
- Extensive hyperparameter tuning

---

## Your Path From 0.7422 → 0.99

### Phase 1: Reach 0.92-0.94 (24 hours, $94)

**Execute the 24-hour emergency plan** (artifact #18):
- Dense retrieval (BGE-m3 512-dim) → 0.96 recall
- 3-model pretrained ensemble (Jina-v2, mxbai, LightGBM)
- Top-30 candidates (52M pairs)
- Basic two-threshold singleton detection
- Per-country thresholds

**Expected:** F0.5 = 0.92-0.94  
**Rank:** Top 50-100

---

### Phase 2: Submit & Assess (1 hour)

**Submit Phase 1 output:**
- Generate predictions.tsv
- Validate locally
- Upload to Unstop
- **Wait for leaderboard score**

**Decision point:**
- **If 0.92-0.93:** You're in the game, proceed to Phase 3
- **If 0.90-0.91:** Blocking recall issue, revisit dense retrieval
- **If <0.90:** Critical bug, debug before continuing

---

### Phase 3: Push to 0.96-0.97 (48 hours, $300)

**Only if Phase 1 hits 0.92+**

#### 3.1 Fine-Tune Cross-Encoder (16 hours, $80)

```python
# finetune_jina.py
"""Fine-tune Jina-reranker-v2 on contest data"""

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
import pandas as pd

# Load model
model = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", max_length=128)

# Prepare training data (100k pairs: 50k matches, 50k non-matches)
# Include hard negatives from Phase 1 (high scores but wrong)
train_pairs = load_contest_train_pairs()  # Your 100k holdout

# Fine-tune
model.fit(
    train_samples=train_pairs,
    epochs=3,
    warmup_steps=5000,
    optimizer_params={'lr': 2e-5},
    evaluator=CEBinaryAccuracyEvaluator(dev_pairs),
    evaluation_steps=1000,
    output_path="./models/jina-v2-contest-finetuned",
)

# Expected gain: +0.02-0.03 F0.5
```

**Cost:** A40 48GB × 16h = $80

#### 3.2 Add More Models (12 hours, $72)

Expand ensemble from 3 → 5 models:
1. Fine-tuned Jina-v2 (560M)
2. Fine-tuned mxbai (278M) 
3. XLM-RoBERTa-large (560M) - fine-tune separately
4. BGE-reranker-v2-m3 (568M) - pretrained multilingual
5. LightGBM (17 features)

**Inference:** Top-50 candidates (86M pairs) on 5× A40  
**Cost:** $72

#### 3.3 France Fine-Tuning (8 hours, $40)

```python
# finetune_france.py
"""Separate model for France with synthetic data"""

# Generate French training data via back-translation
from transformers import MarianMTModel, MarianTokenizer

# Translate 50k English pairs → French
en_pairs = train_pairs[:50000]
fr_pairs = back_translate_to_french(en_pairs)

# Fine-tune XLM-RoBERTa on French pairs
france_model = CrossEncoder("xlm-roberta-large", max_length=128)
france_model.fit(train_samples=fr_pairs, epochs=4)

# Expected France F0.5: 0.94-0.96 (vs 0.90-0.92 zero-shot)
```

**Cost:** A40 48GB × 8h = $40

#### 3.4 Three-Stage Singleton Cascade (4 hours, $8)

```python
# singleton_cascade.py
"""LightGBM → XLM-R → Qwen2.5-7B"""

# Stage 1: LightGBM (fast filter)
if max(lgbm_scores) < 0.60:
    prediction = []  # High-confidence singleton
elif max(lgbm_scores) > 0.85:
    # High-confidence match, proceed
    pass
else:
    # Stage 2: Cross-encoder (10k entities reach here)
    xlm_score = fine_tuned_xlm.predict(s1, best_candidate)
    
    if xlm_score < 0.65:
        prediction = []  # Medium-confidence singleton
    elif xlm_score > 0.90:
        # Match confirmed
        pass
    else:
        # Stage 3: LLM (~2k entities reach here)
        llm_decision = qwen_singleton_check(s1, best_candidate)
        if llm_decision == "SINGLETON":
            prediction = []

# Expected singleton accuracy: 0.997+ (vs 0.99 without LLM)
```

**Cost:** A40 48GB × 4h = $8

#### 3.5 Graph Post-Processing (2 hours, CPU)

```python
# graph_postprocess.py
"""Transitive closure + correlation clustering"""

from collections import defaultdict

# Phase 1: Union-find on high-confidence edges
uf = UnionFind()
for s1_id, matches in predictions.items():
    for match_id in matches:
        if scores[s1_id][match_id] > 0.95:  # Only high confidence
            uf.union(s1_id, match_id)

components = uf.get_components()

# Phase 2: Correlation clustering on medium-confidence
# Fix inconsistent predictions within components
for component in components:
    # If S1_A → S2_X (score 0.70) and S1_B → S2_X (score 0.95)
    # and A, B in same component, boost S1_A → S2_X score
    refine_component_predictions(component)

# Expected gain: +0.01-0.015 F0.5
```

---

### Phase 3 Expected Performance

| Component | Contribution | Cumulative |
|-----------|-------------|------------|
| Phase 1 baseline | 0.92-0.94 | 0.92-0.94 |
| Fine-tuned Jina-v2 | +0.02 | 0.94-0.96 |
| 5-model ensemble | +0.01 | 0.95-0.97 |
| France fine-tuning | +0.01 | 0.96-0.98 |
| Singleton cascade | +0.005 | 0.965-0.985 |
| Graph post-processing | +0.005 | **0.97-0.99** |

**Best case:** 0.98-0.99 (top 3-5)  
**Realistic:** 0.97-0.98 (top 5-10)  
**Total cost:** $94 (Phase 1) + $300 (Phase 3) = **$394**

---

### Phase 4: The Final Push to 0.99 (If Needed)

**Only if Phase 3 hits 0.97-0.98 and you want to go for #1**

#### 4.1 What's Left to Gain

At 0.98, you have ~34,600 errors. To reach 0.99, you need to fix ~17,000 of them.

**Where are the remaining errors?**

Run error analysis:
```python
# error_analysis.py
"""Find patterns in Phase 3 mistakes"""

holdout_errors = []
for s1_id, true_matches in holdout_ground_truth.items():
    pred_matches = predictions[s1_id]
    
    if set(pred_matches) != set(true_matches):
        error = {
            's1_id': s1_id,
            's1_data': s1_records[s1_id],
            'true_matches': true_matches,
            'pred_matches': pred_matches,
            'scores': scores[s1_id],
            'error_type': classify_error(true_matches, pred_matches)
        }
        holdout_errors.append(error)

# Cluster errors by type
error_types = defaultdict(list)
for err in holdout_errors:
    error_types[err['error_type']].append(err)

# Print error distribution
for err_type, errors in error_types.items():
    print(f"{err_type}: {len(errors)} errors ({len(errors)/len(holdout_errors)*100:.1f}%)")
```

**Common error types at 0.97-0.98:**
1. **France false negatives** (30-40% of errors) - multilingual transfer failures
2. **Singleton false merges** (20-30%) - over-aggressive matching
3. **Abbreviation mismatches** (10-15%) - "Corp" vs "Corporation", "LLC" vs "Ltd"
4. **Address format variations** (10-15%) - "123 Main St" vs "123 Main Street Suite 200"
5. **Typographical variations** (5-10%) - "McDonald's" vs "McDonalds"

#### 4.2 Targeted Fixes

**Fix 1: France-Specific Ensemble (8 hours, $64)**

```python
# france_ensemble.py
"""3 multilingual models just for France"""

france_models = [
    "BAAI/bge-reranker-v2-m3",  # 568M, MIT, best multilingual
    "jinaai/jina-reranker-v2-base-multilingual",  # Fine-tuned
    "xlm-roberta-large",  # Fine-tuned on French
]

# Run all 3 on France subset only (260k entities)
for model in france_models:
    france_scores[model] = model.predict(france_pairs)

# Weighted ensemble
france_final = 0.4 * france_scores['bge-m3'] + \
               0.35 * france_scores['jina-v2'] + \
               0.25 * france_scores['xlm']

# Expected: Fix 1,500-2,000 France errors → +0.008-0.012 F0.5
```

**Fix 2: Rule-Based Normalization (2 hours, CPU)**

```python
# normalize_rules.py
"""Handle known variations before matching"""

def normalize_business_name(name):
    """Canonical form for business names"""
    # Legal entity type
    name = re.sub(r'\bIncorporated\b', 'Inc', name, flags=re.I)
    name = re.sub(r'\bCorporation\b', 'Corp', name, flags=re.I)
    name = re.sub(r'\bLimited Liability Company\b', 'LLC', name, flags=re.I)
    name = re.sub(r'\bLimited\b', 'Ltd', name, flags=re.I)
    
    # Punctuation
    name = name.replace("'", "")  # "McDonald's" → "McDonalds"
    name = re.sub(r'[,\.]', '', name)  # Remove commas, periods
    
    # Whitespace
    name = ' '.join(name.split())
    
    return name.lower().strip()

def normalize_address(addr):
    """Canonical form for addresses"""
    # Street types
    addr = re.sub(r'\bStreet\b', 'St', addr, flags=re.I)
    addr = re.sub(r'\bAvenue\b', 'Ave', addr, flags=re.I)
    addr = re.sub(r'\bBoulevard\b', 'Blvd', addr, flags=re.I)
    addr = re.sub(r'\bRoad\b', 'Rd', addr, flags=re.I)
    
    # Suite/Unit (remove - often missing in one source)
    addr = re.sub(r'Suite \d+|Ste \d+|#\d+', '', addr, flags=re.I)
    
    return addr.lower().strip()

# Apply before computing features AND before cross-encoder
# Expected: Fix 1,000-1,500 abbreviation/format errors → +0.006-0.009 F0.5
```

**Fix 3: LLM on ALL Uncertain Pairs (16 hours, $128)**

Instead of only singleton borderline cases, run LLM on ALL uncertain predictions:

```python
# llm_all_uncertain.py
"""Qwen2.5-7B on score 0.4-0.6 range"""

uncertain_pairs = []
for s1_id, cands in predictions.items():
    for cand_id in cands:
        if 0.4 <= scores[s1_id][cand_id] <= 0.6:
            uncertain_pairs.append((s1_id, cand_id))

# ~500k uncertain pairs (0.6% of 86M)
# LLM at 10 pairs/sec = 50k seconds = 14 hours

llm_scores = qwen_batch_score(uncertain_pairs)

# Override ensemble scores with LLM for uncertain region
for s1_id, cand_id in uncertain_pairs:
    if llm_scores[(s1_id, cand_id)] > 0.7:
        final_scores[s1_id][cand_id] = 0.9  # Boost to match
    else:
        final_scores[s1_id][cand_id] = 0.2  # Push to non-match

# Expected: Fix 2,000-3,000 borderline errors → +0.012-0.018 F0.5
```

**Fix 4: Stacked Meta-Model (4 hours, $8)**

```python
# stacked_metamodel.py
"""Train LightGBM on top of ensemble scores"""

# Features for meta-model
meta_features = []
for s1_id, cands in predictions.items():
    for cand_id in cands:
        feats = {
            'jina_score': jina_scores[s1_id][cand_id],
            'mxbai_score': mxbai_scores[s1_id][cand_id],
            'xlm_score': xlm_scores[s1_id][cand_id],
            'lgbm_score': lgbm_scores[s1_id][cand_id],
            'ensemble_std': np.std([jina, mxbai, xlm, lgbm]),  # Disagreement
            'max_score': max([jina, mxbai, xlm, lgbm]),
            'min_score': min([jina, mxbai, xlm, lgbm]),
            'country': s1_records[s1_id]['country'],
            'name_similarity': name_sim(s1_id, cand_id),
            # ... 20+ meta-features
        }
        meta_features.append(feats)

# Train meta-model
meta_lgbm = lgb.LGBMClassifier(n_estimators=500, max_depth=8)
meta_lgbm.fit(meta_features, meta_labels)

# Expected: Fix 500-1,000 ensemble errors → +0.003-0.006 F0.5
```

#### 4.3 Phase 4 Total

| Fix | Errors Fixed | F0.5 Gain | Cost |
|-----|-------------|-----------|------|
| France ensemble | 1,500-2,000 | +0.009-0.012 | $64 |
| Normalization rules | 1,000-1,500 | +0.006-0.009 | $0 |
| LLM uncertain pairs | 2,000-3,000 | +0.012-0.018 | $128 |
| Stacked meta-model | 500-1,000 | +0.003-0.006 | $8 |
| **Total** | **5,000-7,500** | **+0.03-0.045** | **$200** |

**Starting from 0.97:** 0.97 + 0.03 = **1.00** ❌ (exceeds 1.0)  
**Starting from 0.98:** 0.98 + 0.012 = **0.992** ✓

**Realistic outcome:** 0.985-0.995 (top 1-3)

---

## Complete Timeline: 0.7422 → 0.99

| Phase | Duration | Cost | Target | Rank |
|-------|----------|------|--------|------|
| Phase 1: Emergency plan | 24h | $94 | 0.92-0.94 | Top 50-100 |
| Phase 2: Submit & assess | 1h | $0 | — | — |
| Phase 3: Fine-tune & scale | 48h | $300 | 0.97-0.98 | Top 5-10 |
| Phase 4: Final push | 28h | $200 | 0.985-0.995 | Top 1-3 |
| **Total** | **101h** | **$594** | **0.99** | **#1-3** |

---

## Probability Assessment

**To hit 0.99 (beat current 0.9889):**

| Scenario | Probability | What Happens |
|----------|------------|--------------|
| **Everything works perfectly** | 10-15% | Phase 1 → 0.94, Phase 3 → 0.98, Phase 4 → 0.99 |
| **Phase 3 plateaus at 0.97** | 30-40% | Can't break through with fine-tuning alone, Phase 4 hits 0.985 |
| **Phase 1 only hits 0.90-0.91** | 20-25% | Blocking recall issue, need to revisit dense retrieval |
| **Critical bug discovered** | 10-15% | Data leak, feature bug, threshold miscalibration |
| **Top team has secret sauce** | 10-15% | They have something we haven't reverse-engineered |

**Expected outcome:** 0.97-0.985 (top 3-10) ✓  
**Stretch outcome:** 0.99+ (beat #1) if everything breaks right

---

## What If You Can't Beat 0.9889?

**Being top 3-5 is STILL a huge win:**

✅ Prize money  
✅ PPI (Pre-Placement Interview) consideration  
✅ Proof of advanced ML skills  
✅ Strong portfolio piece  

**Don't get anchored on #1.** If you hit 0.97-0.98, you're in an elite tier.

---

## Start Command

**Phase 1 (right now):**
```bash
# Rent A100 80GB immediately
vultr-cli compute create --plan vhf-16c-80gb-a100 --qty 1

# Start dense retrieval (critical path, 8 hours)
ssh a100-1 "python encode_fast.py > encode.log 2>&1 &"

# After 8h: Rent 3× A40 for inference
vultr-cli compute create --plan vhf-8c-48gb-a40 --qty 3
```

**Execute Phase 1, submit, THEN decide whether to continue to Phase 3.**

---

## Key Takeaways

1. **Current #1 is 0.9889** (not 0.9647) - near-perfect execution
2. **Gap is +0.0011** (11 basis points) - requires finding 1,700 errors they missed
3. **Phase 1 → 0.92-0.94** in 24h ($94) gets you in the game
4. **Phase 3 → 0.97-0.98** in 48h ($300) makes you competitive for podium
5. **Phase 4 → 0.99** in 28h ($200) is a 10-15% shot at #1
6. **Total investment:** 101 hours, $594 for realistic shot at top 3
7. **Don't anchor on #1** - top 5 is already exceptional
8. **Start with Phase 1 NOW** - every hour costs you

**YOUR MOVE: Execute the 24-hour emergency plan, submit, then reassess.**
