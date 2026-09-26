# Singleton Detection and F0.5 Precision-Recall Trade-offs

**Author:** Research artifact for Amazon ML Challenge 2026 — Da Big Three  
**Date:** 2026-09-26  
**Purpose:** Deep analysis of how singletons (S1 entities with no S2/S3 matches) interact with the F0.5 metric, and strategies for detecting and protecting singletons. Each false merge on a singleton drops that entity's score from 1.0 to 0.0 — the single most expensive error in the macro average.

---

## 1. The Singleton Problem

### 1.1 What Are Singletons?

A singleton is an S1 entity that has **no matching S2/S3 entity** in the ground truth. The correct prediction for a singleton is an **empty list** `[]`.

Train statistics:
- Total S1 entities: 2,206,821
- Singletons: 123,247 (**5.58%**)
- Non-singletons: 2,083,574 (94.42%)

### 1.2 Scoring Impact

The F0.5 formula for one entity:
```
F0.5 = (1.25 * P * R) / (0.25 * P + R)
```

For a **singleton** (gold = empty set):
- Predict empty → P undefined, but **score = 1.0** (perfect)
- Predict anything → P = 0 (all predictions are false positives) → **score = 0.0** (total failure)

This is a **binary outcome** — there's no partial credit. One wrong prediction destroys the entire entity's contribution.

### 1.3 Cost Analysis

In the macro average with 100k holdout entities:
- ~5,580 singletons, each contributing 1.0 if empty, 0.0 if any match predicted
- Each singleton false merge costs: 1.0 / 100,000 = **0.00001** to the macro average
- But: 100 false merges on singletons = **0.001** loss
- 1000 false merges on singletons = **0.01** loss

For comparison, one **missed** true match on a non-singleton with 4 matches:
- Gold = {A, B, C, D}, Pred = {A, B, C}: P=1.0, R=0.75, F0.5 = 0.962
- Loss vs perfect: 1.0 - 0.962 = 0.038 → macro loss = 0.038 / 100k = 0.00000038

**One false merge on a singleton is ~26× more expensive than one missed match.**

This asymmetry is why precision matters so much in F0.5, and why singleton protection is critical.

---

## 2. Singleton Detection Strategies

### 2.1 Score-Based Thresholding

The simplest approach: if the highest-scoring candidate for an S1 entity is below a "singleton threshold," predict empty.

```python
def singleton_guard(candidate_scores, singleton_threshold=0.7):
    """If no candidate exceeds singleton_threshold, predict empty."""
    if not candidate_scores:
        return True  # No candidates → singleton
    max_score = max(candidate_scores)
    return max_score < singleton_threshold
```

**Key insight:** The singleton threshold should be **higher** than the match threshold. A match threshold of 0.4 might catch all true matches, but a singleton threshold of 0.7 ensures we only predict non-empty when we're confident.

```
match_threshold = 0.40  # Accept individual candidates above this
singleton_threshold = 0.70  # Only predict non-empty if best candidate > this
```

### 2.2 Feature-Based Singleton Classifier

Train a separate classifier to predict whether an S1 entity is a singleton, using features derived from the candidate set:

| Feature | Description | Singleton Signal |
|---------|-------------|-----------------|
| `max_score` | Highest candidate match score | Low → likely singleton |
| `score_gap` | Gap between best and 2nd-best score | Small gap → ambiguous → maybe singleton |
| `n_above_threshold` | How many candidates score > 0.3 | Few → likely singleton |
| `mean_top5_score` | Average score of top-5 candidates | Low → likely singleton |
| `name_frequency` | How common is this name in S1? | Very common → more likely singleton (many "similar" but different entities) |
| `has_address` | Does S1 have a non-empty address? | No address → harder to match → more likely singleton |
| `country` | S1's country | Country-specific singleton rates differ |
| `best_cand_country` | Best candidate's country | Cross-country → less reliable |

```python
def singleton_features(s1_id, candidate_scores, s1_records, freq):
    """Features for singleton detection."""
    name, addr, country, _ = s1_records[s1_id]
    scores = sorted(candidate_scores, reverse=True) if candidate_scores else []

    return [
        scores[0] if scores else 0.0,  # max_score
        (scores[0] - scores[1]) if len(scores) > 1 else scores[0] if scores else 0.0,  # score_gap
        sum(1 for s in scores if s > 0.3),  # n_above_threshold
        np.mean(scores[:5]) if scores else 0.0,  # mean_top5_score
        freq.get((country.strip(), name_squash(name)), 1),  # name_frequency
        1.0 if addr and addr.strip() else 0.0,  # has_address
    ]
```

### 2.3 Calibrated Confidence

Instead of using raw model scores, calibrate them so that `score = 0.5` actually means 50% match probability:

```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate_model(clf, X_val, y_val):
    """Calibrate classifier probabilities using isotonic regression."""
    calibrated = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
    calibrated.fit(X_val, y_val)
    return calibrated
```

After calibration, the singleton threshold has a clear probabilistic interpretation: "predict non-empty only if there's a >70% chance the best candidate is a true match."

### 2.4 Ensemble Singleton Detection

Combine multiple signals:

```python
def ensemble_singleton_detection(
    max_lgbm_score, max_ditto_score, 
    name_rarity, n_candidates_above_03,
    weights=(0.4, 0.4, 0.1, 0.1),
    threshold=0.55
):
    """Weighted combination of singleton signals."""
    combined = (
        weights[0] * max_lgbm_score +
        weights[1] * max_ditto_score +
        weights[2] * (1.0 - min(name_rarity / 10.0, 1.0)) +  # Rare names are less likely singletons
        weights[3] * min(n_candidates_above_03 / 5.0, 1.0)     # More strong candidates → not singleton
    )
    return combined < threshold  # True = predict as singleton
```

---

## 3. The Precision-Recall Trade-off Under F0.5

### 3.1 F0.5 Isocurves

F0.5 weights precision 4× more than recall in the harmonic mean:
```
F0.5 = (1.25 * P * R) / (0.25 * P + R)
```

At different operating points:

| Precision | Recall | F0.5 | Notes |
|-----------|--------|------|-------|
| 1.00 | 0.50 | 0.833 | Perfect precision, half recall |
| 0.95 | 0.70 | 0.921 | Good balance |
| 0.90 | 0.80 | 0.887 | Slight precision loss |
| 0.85 | 0.90 | 0.859 | Precision too low for F0.5 |
| 0.80 | 1.00 | 0.833 | Dropping precision hurts more than perfect recall helps |
| 0.70 | 1.00 | 0.758 | Even lower |
| 0.60 | 1.00 | 0.682 | Precision crash |

**Key insight:** Going from P=0.95 to P=0.90 (losing 5% precision) while gaining R=0.70→0.80 (gaining 10% recall) actually **decreases** F0.5 from 0.921 to 0.887.

### 3.2 Optimal Operating Point

For F0.5, the optimal threshold is where:
```
dF0.5/dthreshold = 0
```

This occurs where the marginal precision gain from raising the threshold equals the marginal recall loss, weighted by F0.5's precision preference.

In practice, sweep the threshold on holdout and pick the peak:

```python
def sweep_threshold_f05(scores, labels, entity_boundaries, gold_sets, cand_lists):
    """Find optimal threshold for F0.5."""
    best = (0.0, 0.5)  # (F0.5, threshold)

    for t in np.arange(0.05, 0.95, 0.01):
        f05_sum = 0.0
        for ei, (start, end) in enumerate(entity_boundaries):
            gold = gold_sets[ei]
            cands = cand_lists[ei]
            pred = {cands[j] for j in range(end - start) if scores[start + j] >= t}

            if not pred and not gold:  # Singleton correct
                f05_sum += 1.0
            elif not pred and gold:  # Missed all matches
                f05_sum += 0.0
            elif pred and not gold:  # False merge on singleton
                f05_sum += 0.0
            else:
                p = len(pred & gold) / len(pred)
                r = len(pred & gold) / len(gold)
                f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
                f05_sum += f05

        macro = f05_sum / len(entity_boundaries)
        if macro > best[0]:
            best = (macro, t)

    return best
```

### 3.3 Two-Threshold Strategy

Use different thresholds for different confidence levels:

```
High confidence (score > 0.8):  → always predict as match
Medium confidence (0.4-0.8):    → predict only if multiple candidates agree
Low confidence (< 0.4):         → never predict as match

Singleton decision: if max score < singleton_threshold → predict empty
```

```python
def two_threshold_predict(scores, cands, high_t=0.8, low_t=0.4, singleton_t=0.7):
    """Two-threshold prediction with singleton guard."""
    max_score = max(scores) if scores else 0.0

    if max_score < singleton_t:
        return set()  # Singleton

    matches = set()
    for cand, score in zip(cands, scores):
        if score >= high_t:
            matches.add(cand)
        elif score >= low_t:
            # Only include if multiple medium-confidence candidates
            n_medium = sum(1 for s in scores if s >= low_t)
            if n_medium >= 2:
                matches.add(cand)

    return matches
```

---

## 4. Singleton Patterns in the Data

### 4.1 Why Entities Are Singletons

From our EDA (eda_stats.md):
- 63% of singletons share a squashed name with some S2/S3 entity
- Only 37% are truly unique names

This means **name-only matching is insufficient** to detect singletons — most have name-lookalikes in S2/S3 that are different businesses.

Singletons are entities that:
1. **Don't exist in S2/S3** — new businesses not yet in vendor dumps
2. **Have common names** — "ABC Enterprises" with no distinguishing address match
3. **Have unique addresses** — the business exists but at a location not in S2/S3
4. **Are in a different source** — S2 has it but mapped to a different S1

### 4.2 Country Distribution of Singletons

Train statistics suggest singleton rate varies by country (UNKNOWN exact split). Possible patterns:
- US: many large chains, more duplicates → lower singleton rate
- India: more unique small businesses → higher singleton rate
- France (test only): entirely new source → singleton rate UNKNOWN

**Risk:** If France has a higher singleton rate, our model (trained on US+India distribution) may over-predict matches for French entities.

---

## 5. The Confidence Margin Approach

Rather than a single threshold, use the **margin** between the best candidate's score and the decision boundary:

```python
def confidence_margin_predict(scores, cands, base_threshold=0.45,
                               margin_for_singleton=0.25):
    """Predict matches with confidence margin for singleton detection."""
    if not scores:
        return set()

    max_score = max(scores)
    margin = max_score - base_threshold

    if margin < margin_for_singleton:
        return set()  # Not confident enough → treat as singleton

    # Standard thresholding for individual candidates
    return {c for c, s in zip(cands, scores) if s >= base_threshold}
```

This naturally adapts to the score distribution — if the best candidate barely exceeds the threshold, it's treated as a potential singleton.

---

## 6. Holdout Analysis Framework

To tune singleton detection, measure separately:

```python
def analyze_singleton_performance(predictions, gold, s1_records):
    """Break down performance by singleton/non-singleton."""
    sing_correct = sing_wrong = 0
    nonsingleton_f05_sum = nonsingleton_count = 0

    for s1_id in gold:
        pred = predictions.get(s1_id, set())
        true_matches = gold[s1_id]

        if not true_matches:  # Singleton
            if not pred:
                sing_correct += 1
            else:
                sing_wrong += 1
        else:  # Non-singleton
            p = len(pred & true_matches) / len(pred) if pred else 0
            r = len(pred & true_matches) / len(true_matches)
            f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
            nonsingleton_f05_sum += f05
            nonsingleton_count += 1

    total = len(gold)
    singleton_contribution = sing_correct / total  # Perfect singletons
    singleton_loss = sing_wrong / total  # Lost singletons
    nonsingleton_contribution = nonsingleton_f05_sum / total

    print(f"Singletons: {sing_correct} correct, {sing_wrong} wrong "
          f"(contribution: +{singleton_contribution:.4f}, loss: -{singleton_loss:.4f})")
    print(f"Non-singletons: avg F0.5={nonsingleton_f05_sum/nonsingleton_count:.4f} "
          f"(contribution: +{nonsingleton_contribution:.4f})")
    print(f"Total macro F0.5: {singleton_contribution + nonsingleton_contribution:.4f}")
```

---

## 7. Practical Recommendations

### 7.1 Minimum Viable Singleton Protection

1. **Sweep the threshold** on holdout — the optimal F0.5 threshold is typically higher than 0.5 for F0.5 metrics
2. **Add a singleton guard** — if max candidate score < `singleton_threshold`, predict empty
3. **singleton_threshold = match_threshold + 0.15** is a reasonable starting point

### 7.2 Advanced (if time permits)

1. **Train a singleton classifier** on holdout using candidate score features
2. **Per-country thresholds** — France may need a higher threshold due to zero-shot transfer uncertainty
3. **Two-pass scoring** — first pass identifies likely singletons, second pass re-scores borderline cases

### 7.3 What NOT To Do

- Don't use a very low threshold hoping to catch all matches — the precision loss from false merges on singletons outweighs the recall gain
- Don't predict empty for all entities with common names — 63% of singletons have name-lookalikes, but so do most non-singletons
- Don't ignore singletons in evaluation — they're 5.58% of entities but contribute disproportionately to the macro average

---

## 8. Expected Impact

| Strategy | Effort | Expected F0.5 Gain | Risk |
|----------|--------|-------------------|------|
| Optimized threshold sweep | 5 min | +0.005-0.01 | None |
| Singleton guard (max score threshold) | 10 min | +0.005-0.02 | None |
| Per-country threshold | 15 min | +0.005-0.01 | Low |
| Feature-based singleton classifier | 1 hour | +0.005-0.015 | Medium |
| Confidence margin approach | 10 min | +0.005-0.01 | None |

**Total potential:** +0.02-0.05 F0.5 from better singleton handling alone.

---

## 9. Key Takeaways

1. **Each false merge on a singleton costs 1.0** in the entity-level score (binary: perfect or zero)
2. **One singleton false merge is ~26× more expensive** than one missed true match
3. **F0.5 weights precision ~4× more than recall** — conservative prediction is rewarded
4. **Two-threshold strategy** (match threshold + singleton guard) is the minimum viable approach
5. **63% of singletons have name-lookalikes** — name alone cannot detect singletons
6. **Per-country thresholds** are important because France has unknown singleton rate
7. **Calibrated confidence** allows principled threshold setting with probabilistic interpretation
8. **Holdout analysis should separate singleton and non-singleton performance** for targeted debugging

---

## 10. References

1. Fellegi, I.P., & Sunter, A.B. (1969). "A Theory for Record Linkage." JASA.
2. Elmagarmid, A.K., Ipeirotis, P.G., & Verykios, V.S. (2007). "Duplicate Record Detection: A Survey." IEEE TKDE.
3. Christen, P. (2012). "Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection." Springer.
4. Kopcke, H., & Rahm, E. (2010). "Frameworks for Entity Matching: A Comparison." Data & Knowledge Engineering.
