# Advanced Threshold Optimization for Entity Resolution

**Author:** Research artifact for Amazon ML Challenge 2026 — Da Big Three  
**Date:** 2026-09-26  
**Purpose:** Beyond brute-force sweeping from 0.05 to 0.95 — calibration, per-entity adaptive thresholds, confidence-based multi-threshold strategies, gradient-based optimization, and how to exploit the macro F0.5 structure for threshold decisions. Shows how to squeeze out the final +0.01-0.03 F0.5 from smarter thresholding.

---

## 1. Why Threshold Optimization Matters

Entity resolution reduces to: **score(pair) ≥ threshold → predict match**

The threshold choice has massive impact:
- Too low → false positives → precision crash → F0.5 drops
- Too high → false negatives → recall crash → F0.5 drops

For **macro F0.5**, the threshold interacts with:
- Singleton rate (5.58% in our data)
- Match set size distribution (mean 3.67, max 11)
- Score distribution per entity (some S1s have many high-scoring candidates, others have none)

A single global threshold is suboptimal. We can do better.

---

## 2. The Baseline: Grid Search

```python
def sweep_threshold_f05(scores, labels, entity_boundaries, gold_sets, cand_lists):
    """Brute force sweep over candidate thresholds."""
    best_f05, best_t = 0.0, 0.5
    
    for t in np.arange(0.05, 0.95, 0.01):
        entity_scores = []
        for ei, (start, end) in enumerate(entity_boundaries):
            gold = gold_sets[ei]
            cands = cand_lists[ei]
            pred = {cands[j] for j in range(end - start) if scores[start + j] >= t}
            entity_scores.append(f05_one(pred, gold))
        
        macro = np.mean(entity_scores)
        if macro > best_f05:
            best_f05, best_t = macro, t
    
    return best_t, best_f05
```

**Cost:** 90 threshold evaluations × (compute F0.5 for 100k entities) ≈ 10 seconds. Fast enough.

**Problem:** This finds the optimal **global** threshold. But different entities have different optimal thresholds:
- Easy entities (one clear match): low threshold OK
- Ambiguous entities (many medium-scoring candidates): high threshold needed
- Singletons: very high threshold (or dedicated singleton detector)

---

## 3. Calibration: Making Scores Interpretable

### 3.1 The Calibration Problem

LightGBM outputs "probabilities" but they're not **calibrated**:
- A score of 0.7 doesn't mean "70% chance this is a match"
- LightGBM tends to output overconfident scores (too close to 0 or 1)

**Test:**
```python
def calibration_curve(y_true, y_score, n_bins=10):
    """Plot predicted probability vs actual match rate."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_score, bins) - 1
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        avg_pred = y_score[mask].mean()
        actual_rate = y_true[mask].mean()
        print(f"Bin {i}: pred={avg_pred:.3f}, actual={actual_rate:.3f}")
```

**Typical LightGBM calibration:**
- Predicted [0.0-0.1] → Actual 0.02 (underconfident on negatives)
- Predicted [0.5-0.6] → Actual 0.45 (close)
- Predicted [0.9-1.0] → Actual 0.85 (overconfident on positives)

### 3.2 Calibration Methods

#### Platt Scaling (Logistic Calibration)

Fit a logistic regression `P(y=1 | score)` on validation set:

```python
from sklearn.linear_model import LogisticRegression

def platt_scaling(train_scores, train_labels):
    """Learn P(match | raw_score) mapping."""
    lr = LogisticRegression()
    lr.fit(train_scores.reshape(-1, 1), train_labels)
    return lr

def calibrate_scores(raw_scores, calibrator):
    """Apply calibration."""
    return calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
```

**Pros:** Simple, fast, works well if scores are monotonic with true probability
**Cons:** Assumes sigmoid shape (not always true)

#### Isotonic Regression (Non-Parametric)

Learns a **monotonic** but non-parametric mapping:

```python
from sklearn.isotonic import IsotonicRegression

def isotonic_calibration(train_scores, train_labels):
    """Non-parametric monotonic calibration."""
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(train_scores, train_labels)
    return iso
```

**Pros:** More flexible than Platt, handles non-sigmoid shapes
**Cons:** Needs more data (1000+ pairs), can overfit on small validation

#### Temperature Scaling (For Deep Models)

For Ditto cross-encoder, scale logits before softmax:

```python
def temperature_scaling(val_logits, val_labels):
    """Find temperature T that minimizes NLL."""
    from scipy.optimize import minimize
    
    def nll(T):
        scaled_probs = torch.softmax(val_logits / T, dim=1)
        loss = F.nll_loss(torch.log(scaled_probs), val_labels)
        return loss.item()
    
    result = minimize(nll, x0=1.0, bounds=[(0.1, 10.0)])
    return result.x[0]

# At inference:
calibrated_probs = torch.softmax(logits / T_optimal, dim=1)
```

**Pros:** Single parameter, works well for transformers
**Cons:** Requires access to logits (not just scores)

### 3.3 Why Calibration Helps Thresholding

After calibration, a threshold of **0.5 actually means "50% confidence"** — a principled default. Without calibration, the optimal threshold might be 0.23 or 0.87 with no clear interpretation.

**Expected gain:** +0.005-0.015 F0.5 from calibration + threshold=0.5 vs uncalibrated sweep.

---

## 4. Per-Entity Adaptive Thresholds

### 4.1 The Insight

Macro F0.5 averages **per-entity** F0.5 scores. Each entity can have its own threshold:

```python
def per_entity_threshold_optimization(scores, entity_boundaries, gold_sets, cand_lists):
    """Find optimal threshold for EACH entity independently."""
    entity_thresholds = []
    
    for ei, (start, end) in enumerate(entity_boundaries):
        gold = gold_sets[ei]
        cands = cand_lists[ei]
        entity_scores = scores[start:end]
        
        # Try thresholds specific to this entity's score distribution
        best_f05, best_t = 0.0, 0.5
        for t in np.unique(entity_scores):  # Only try thresholds at score boundaries
            pred = {cands[j] for j in range(end - start) if entity_scores[j] >= t}
            f05 = f05_one(pred, gold)
            if f05 > best_f05:
                best_f05, best_t = f05, t
        
        entity_thresholds.append(best_t)
    
    return entity_thresholds
```

**Problem:** This is **oracle** — we don't know gold at test time. But it reveals patterns:
- Entities with gold=∅ (singletons) → optimal threshold is very high (0.8-0.95)
- Entities with many matches → optimal threshold is lower (0.4-0.6)
- Entities with max(scores) < 0.3 → optimal is to predict empty

### 4.2 Learning Adaptive Thresholds

Train a **meta-model** to predict the optimal threshold for each entity:

```python
def learn_entity_threshold(entity_features, optimal_thresholds):
    """
    entity_features: [max_score, score_gap, n_above_03, mean_score, name_rarity]
    optimal_thresholds: oracle thresholds from per-entity optimization
    """
    from sklearn.ensemble import RandomForestRegressor
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=5)
    rf.fit(entity_features, optimal_thresholds)
    return rf

# At test time:
def predict_entity_threshold(entity_features, meta_model):
    return meta_model.predict([entity_features])[0]
```

**Entity features for threshold prediction:**
```python
def extract_entity_threshold_features(scores, cands):
    """Features that predict optimal threshold for this entity."""
    if len(scores) == 0:
        return [0, 0, 0, 0, 0]
    
    sorted_scores = np.sort(scores)[::-1]
    return [
        sorted_scores[0],  # max score
        sorted_scores[0] - (sorted_scores[1] if len(scores) > 1 else 0),  # gap to 2nd
        (scores >= 0.3).sum(),  # number of candidates above baseline
        scores.mean(),  # average score
        scores.std() if len(scores) > 1 else 0,  # score variance
    ]
```

**Expected gain:** +0.01-0.02 F0.5 vs global threshold.

---

## 5. Multi-Threshold Strategies

### 5.1 Two-Threshold Approach

Use **two thresholds** — one for match inclusion, one for singleton detection:

```python
def two_threshold_predict(scores, match_threshold=0.45, singleton_threshold=0.70):
    """
    match_threshold: include candidates above this
    singleton_threshold: if max(scores) < this, predict empty
    """
    if len(scores) == 0 or scores.max() < singleton_threshold:
        return []  # Singleton
    
    return [i for i, s in enumerate(scores) if s >= match_threshold]
```

**Rationale:**
- `match_threshold` = 0.45: Permissive — include borderline candidates
- `singleton_threshold` = 0.70: Conservative — only predict non-empty if confident

**Optimization:**
```python
def optimize_two_thresholds(scores, entity_boundaries, gold_sets, cand_lists):
    """Grid search over (match_t, singleton_t) space."""
    best_f05, best_pair = 0.0, (0.5, 0.5)
    
    for match_t in np.arange(0.2, 0.8, 0.05):
        for singleton_t in np.arange(match_t, 0.95, 0.05):
            entity_scores = []
            for ei, (start, end) in enumerate(entity_boundaries):
                entity_s = scores[start:end]
                if len(entity_s) == 0 or entity_s.max() < singleton_t:
                    pred = set()
                else:
                    pred = {cand_lists[ei][j] for j in range(len(entity_s))
                           if entity_s[j] >= match_t}
                entity_scores.append(f05_one(pred, gold_sets[ei]))
            
            macro = np.mean(entity_scores)
            if macro > best_f05:
                best_f05, best_pair = macro, (match_t, singleton_t)
    
    return best_pair, best_f05
```

**Expected gain:** +0.01-0.02 F0.5 vs single threshold.

### 5.2 Confidence-Based Routing

Different thresholds for **high-confidence** vs **low-confidence** candidates:

```python
def confidence_routing_threshold(scores):
    """
    High-confidence (> 0.8): accept
    Medium (0.4-0.8): accept if multiple agree
    Low (< 0.4): reject
    """
    high = scores >= 0.8
    medium = (scores >= 0.4) & (scores < 0.8)
    
    matches = []
    matches.extend(np.where(high)[0].tolist())
    
    if medium.sum() >= 2:  # At least 2 medium candidates
        matches.extend(np.where(medium)[0].tolist())
    
    return matches
```

### 5.3 Threshold Per Score Bin

Divide score range into bins, optimize threshold per bin:

```python
def binned_threshold_optimization(scores, labels, n_bins=5):
    """Find optimal threshold for each score quantile."""
    quantiles = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    bin_thresholds = []
    
    for i in range(n_bins):
        mask = (scores >= quantiles[i]) & (scores < quantiles[i+1])
        if mask.sum() < 10:
            bin_thresholds.append(quantiles[i])
            continue
        
        # Optimize F0.5 for this bin
        bin_scores = scores[mask]
        bin_labels = labels[mask]
        best_f05, best_t = 0.0, quantiles[i]
        for t in np.linspace(quantiles[i], quantiles[i+1], 20):
            preds = (bin_scores >= t).astype(int)
            f05 = compute_f05(preds, bin_labels)
            if f05 > best_f05:
                best_f05, best_t = f05, t
        
        bin_thresholds.append(best_t)
    
    return bin_thresholds, quantiles
```

---

## 6. Gradient-Based Threshold Optimization

### 6.1 Differentiable Thresholding

Standard thresholding is non-differentiable:
```python
pred = (score >= threshold).astype(int)  # Discontinuous jump
```

Replace with **soft** thresholding during optimization:

```python
def soft_threshold(scores, threshold, temperature=0.01):
    """Sigmoid approximation of hard threshold."""
    return torch.sigmoid((scores - threshold) / temperature)
```

### 6.2 Directly Optimize F0.5

```python
import torch
from torch.optim import Adam

def optimize_threshold_gradient(scores, entity_boundaries, gold_sets, cand_lists):
    """Gradient-based threshold optimization."""
    threshold = torch.tensor([0.5], requires_grad=True)
    optimizer = Adam([threshold], lr=0.01)
    
    scores_t = torch.tensor(scores, dtype=torch.float32)
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Soft predictions
        entity_f05_sum = 0.0
        for ei, (start, end) in enumerate(entity_boundaries):
            entity_scores = scores_t[start:end]
            soft_preds = soft_threshold(entity_scores, threshold, temperature=0.01)
            
            # Soft F0.5 (differentiable approximation)
            gold = gold_sets[ei]
            cands = cand_lists[ei]
            
            # True positives (soft)
            tp = sum(soft_preds[j] for j in range(len(cands)) if cands[j] in gold)
            # False positives (soft)
            fp = sum(soft_preds[j] for j in range(len(cands)) if cands[j] not in gold)
            # False negatives
            fn = len(gold) - tp
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (len(gold) + 1e-8)
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-8)
            entity_f05_sum += f05
        
        loss = -entity_f05_sum / len(entity_boundaries)
        loss.backward()
        optimizer.step()
        
        # Clip threshold to [0, 1]
        with torch.no_grad():
            threshold.clamp_(0, 1)
    
    return threshold.item()
```

**Expected gain:** Marginal over grid search (both find similar thresholds), but faster convergence.

---

## 7. Exploiting Macro F0.5 Structure

### 7.1 The Macro Average Property

Macro F0.5 = mean of per-entity F0.5 scores.

**Key insight:** Improving score on **low-performing entities** has higher marginal value than optimizing already-high entities.

```python
def prioritize_low_scorers(scores, entity_boundaries, gold_sets, cand_lists, global_threshold):
    """Identify entities with low F0.5, adjust their thresholds."""
    entity_f05 = []
    for ei, (start, end) in enumerate(entity_boundaries):
        pred = {cand_lists[ei][j] for j in range(end - start)
               if scores[start + j] >= global_threshold}
        entity_f05.append((ei, f05_one(pred, gold_sets[ei])))
    
    # Sort by F0.5 ascending (worst performers first)
    entity_f05.sort(key=lambda x: x[1])
    
    # Re-optimize threshold for bottom 10%
    bottom_10pct = int(0.1 * len(entity_f05))
    for ei, current_f05 in entity_f05[:bottom_10pct]:
        start, end = entity_boundaries[ei]
        entity_scores = scores[start:end]
        
        # Try different thresholds for this entity
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            pred = {cand_lists[ei][j] for j in range(end - start)
                   if entity_scores[j] >= t}
            new_f05 = f05_one(pred, gold_sets[ei])
            if new_f05 > current_f05:
                print(f"Entity {ei}: F0.5 {current_f05:.3f} → {new_f05:.3f} with t={t}")
                break
```

### 7.2 Singleton-Weighted Optimization

Singletons contribute **1.0 or 0.0** (binary). Non-singletons contribute fractional scores. This means singletons have **disproportionate impact** on macro average.

```python
def singleton_aware_threshold(scores, entity_boundaries, gold_sets, cand_lists):
    """Weight optimization by entity type."""
    singleton_entities = [ei for ei, g in enumerate(gold_sets) if not g]
    nonsingleton_entities = [ei for ei, g in enumerate(gold_sets) if g]
    
    # Optimize two thresholds: one for singletons, one for non-singletons
    # (Requires detecting singletons at test time — use singleton classifier)
    
    best_f05, best_pair = 0.0, (0.5, 0.5)
    for t_sing in np.arange(0.6, 0.95, 0.05):
        for t_non in np.arange(0.3, 0.8, 0.05):
            entity_scores = []
            
            for ei in singleton_entities:
                start, end = entity_boundaries[ei]
                if len(scores[start:end]) == 0 or scores[start:end].max() < t_sing:
                    pred = set()
                else:
                    pred = {cand_lists[ei][j] for j in range(end-start)
                           if scores[start+j] >= t_non}
                entity_scores.append(f05_one(pred, gold_sets[ei]))
            
            for ei in nonsingleton_entities:
                start, end = entity_boundaries[ei]
                pred = {cand_lists[ei][j] for j in range(end-start)
                       if scores[start+j] >= t_non}
                entity_scores.append(f05_one(pred, gold_sets[ei]))
            
            macro = np.mean(entity_scores)
            if macro > best_f05:
                best_f05, best_pair = macro, (t_sing, t_non)
    
    return best_pair
```

---

## 8. Dynamic Thresholds Based on Context

### 8.1 Per-Country Thresholds

France (15% of test) has different score distribution due to zero-shot transfer:

```python
def country_adaptive_threshold(country, base_threshold=0.5):
    """Adjust threshold based on country."""
    adjustments = {
        "US": 0.0,      # Base threshold
        "India": -0.05,  # Slightly lower (more noise in addresses)
        "France": +0.10,  # Higher (zero-shot uncertainty)
    }
    return base_threshold + adjustments.get(country, 0.0)
```

### 8.2 Threshold Based on Candidate Count

Entities with many candidates → higher threshold (avoid false positives).

```python
def candidate_count_threshold(n_candidates, base_threshold=0.5):
    """Raise threshold if many candidates (reduce false positives)."""
    if n_candidates > 50:
        return base_threshold + 0.1
    elif n_candidates > 20:
        return base_threshold + 0.05
    else:
        return base_threshold
```

### 8.3 Threshold Based on Max Score Margin

If best candidate is **much better** than second-best → lower threshold OK. If tight race → higher threshold.

```python
def margin_based_threshold(scores, base_threshold=0.5):
    """Adjust based on score gap between top-2 candidates."""
    if len(scores) < 2:
        return base_threshold
    
    sorted_scores = np.sort(scores)[::-1]
    margin = sorted_scores[0] - sorted_scores[1]
    
    if margin > 0.3:  # Clear winner
        return base_threshold - 0.05
    elif margin < 0.05:  # Close race
        return base_threshold + 0.1
    else:
        return base_threshold
```

---

## 9. Practical Implementation for Our Contest

### 9.1 Recommended Strategy

**Phase 1 (Baseline):** Grid search for global threshold
```python
best_t, best_f05 = sweep_threshold_f05(lgbm_scores, ...)
```

**Phase 2 (Calibration):** Apply isotonic regression, use threshold=0.5
```python
calibrator = isotonic_calibration(val_scores, val_labels)
cal_scores = calibrator.transform(test_scores)
# Now use threshold=0.5
```

**Phase 3 (Two-threshold):** Optimize match + singleton thresholds
```python
(match_t, singleton_t), f05 = optimize_two_thresholds(...)
```

**Phase 4 (Per-country):** Adjust threshold for France
```python
if country == "France":
    threshold = match_t + 0.10
else:
    threshold = match_t
```

### 9.2 Expected Gains

| Strategy | Effort | Expected Gain vs Baseline | Cumulative F0.5 |
|----------|--------|---------------------------|-----------------|
| Grid search (baseline) | 5 min | 0.0 (reference) | 0.7422 (current v3) |
| Isotonic calibration | 10 min | +0.005-0.01 | 0.748-0.752 |
| Two-threshold | 20 min | +0.01-0.02 | 0.758-0.772 |
| Per-country adjustment | 5 min | +0.005-0.01 | 0.763-0.782 |
| Adaptive entity thresholds | 1 hour | +0.01-0.02 | 0.773-0.802 |

**Total potential:** +0.03-0.06 F0.5 from advanced thresholding alone.

---

## 10. Key Takeaways

1. **Calibration makes thresholds interpretable** — threshold=0.5 means "50% confidence" after calibration
2. **Two-threshold strategy is the lowest-hanging fruit** — match threshold + singleton threshold exploits the binary nature of singleton scoring
3. **Per-entity thresholds can boost macro F0.5** — entities with different characteristics need different thresholds
4. **Singletons dominate the optimization** — each singleton error costs 1.0 in the macro average
5. **Per-country thresholds matter** — France (zero-shot) should have a higher threshold than US/India
6. **Margin-based adjustment is cheap and effective** — if top-2 candidates are close, raise threshold
7. **Grid search is often good enough** — advanced methods give marginal gains (+0.01-0.02) but can be worth it in contests
8. **Macro F0.5 rewards conservative predictions** — precision matters ~4× more than recall in F0.5 weighting

---

## 11. References

1. Zadrozny, B., & Elkan, C. (2001). "Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers." ICML.
2. Platt, J. (1999). "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods." Advances in Large Margin Classifiers.
3. Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML.
4. Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting good probabilities with supervised learning." ICML.
5. Kull, M., et al. (2019). "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration." NeurIPS.
