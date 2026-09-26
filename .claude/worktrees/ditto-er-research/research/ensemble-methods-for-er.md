# Ensemble Methods for Entity Resolution: Combining Multiple Matchers

**Author:** Research artifact for Amazon ML Challenge 2026 — Da Big Three  
**Date:** 2026-09-26  
**Purpose:** Comprehensive guide to ensemble techniques for ER — stacking, blending, cascades, confidence-based routing, and multi-matcher fusion. Shows how to combine BM25 blocking + LightGBM features + Ditto cross-encoder + graph signals into a unified system that outperforms any single component.

---

## 1. Why Ensemble ER?

No single matcher is perfect:
- **String similarity features** (Jaccard, Jaro-Winkler) catch lexical overlap but miss semantics
- **LightGBM** learns feature interactions but is limited by feature engineering
- **Cross-encoders** (Ditto) understand semantics but are compute-heavy and can overfit
- **Graph features** propagate evidence but are noisy on weak edges

Ensembles exploit the strengths of each:
- When LightGBM and Ditto **agree** → high confidence
- When they **disagree** → investigate further (examine graph structure, re-weight)
- When string features are weak but Ditto is strong → semantic match (e.g., "Corp" = "Corporation")

Research shows **3-7% absolute F1 gain** from well-tuned ensembles vs single best model in ER benchmarks.

---

## 2. Taxonomy of Ensemble Methods for ER

```
Ensemble Methods
├── Voting (simple combination)
│   ├── Majority vote
│   ├── Weighted vote
│   └── Threshold vote (k-of-n)
├── Score fusion (combine scores)
│   ├── Linear weighting
│   ├── Rank-based fusion
│   └── Learned weighting
├── Stacking (meta-classifier)
│   ├── Logistic regression on scores
│   ├── LightGBM on scores
│   └── Neural network on scores
├── Cascades (sequential filtering)
│   ├── Fast filter → slow reranker
│   ├── Confidence-based routing
│   └── Active learning selection
└── Hybrid (task-specific combinations)
    ├── Blocking + matching ensemble
    ├── Per-country model routing
    └── Per-difficulty model selection
```

---

## 3. Voting Ensembles

### 3.1 Majority Vote

Each model predicts match/no-match (binary). Final prediction = majority.

```python
def majority_vote_ensemble(models, pairs, threshold=0.5):
    """Simple majority vote over model predictions."""
    votes = []
    for model in models:
        scores = model.predict(pairs)
        votes.append((scores >= threshold).astype(int))
    
    votes = np.array(votes)  # shape: (n_models, n_pairs)
    final = (votes.sum(axis=0) > len(models) / 2).astype(int)
    return final
```

**Pros:** Simple, robust to individual model errors
**Cons:** Loses information (score → binary), treats all models equally

**For our contest:** Not recommended. We have confidence scores — use them.

### 3.2 Weighted Vote

Each model gets a weight based on its validation performance:

```python
def weighted_vote_ensemble(models, weights, pairs, threshold=0.5):
    """Weighted vote where weights sum to 1.0."""
    weighted_votes = []
    for model, weight in zip(models, weights):
        scores = model.predict(pairs)
        weighted_votes.append(weight * (scores >= threshold))
    
    combined = np.sum(weighted_votes, axis=0)
    return (combined > 0.5).astype(int)
```

**Weight optimization:**
```python
from scipy.optimize import minimize

def find_optimal_weights(models, val_pairs, val_labels):
    """Find weights that maximize validation F0.5."""
    predictions = [m.predict(val_pairs) for m in models]
    
    def objective(weights):
        weights = weights / weights.sum()  # normalize
        combined = sum(w * p for w, p in zip(weights, predictions))
        # Compute F0.5 on validation set
        return -compute_f05(combined, val_labels)
    
    w0 = np.ones(len(models)) / len(models)
    result = minimize(objective, w0, bounds=[(0, 1)] * len(models))
    return result.x / result.x.sum()
```

**Pros:** Uses validation performance to weight models
**Cons:** Ignores correlation between models (redundant predictions get over-weighted)

---

## 4. Score Fusion Methods

### 4.1 Linear Weighting (Alpha Blending)

Combine raw scores with learned weights:

```python
def linear_fusion(score_a, score_b, alpha=0.5):
    """Combine two model scores with weight alpha."""
    return alpha * score_a + (1 - alpha) * score_b
```

For our pipeline (LightGBM + Ditto):
```python
def fuse_lgbm_ditto(lgbm_probs, ditto_scores, alpha=0.6):
    """
    alpha=0.6 means 60% LightGBM, 40% Ditto.
    Sweep alpha on holdout to optimize F0.5.
    """
    return alpha * lgbm_probs + (1 - alpha) * ditto_scores
```

**Optimal alpha finding:**
```python
def sweep_alpha_f05(lgbm_scores, ditto_scores, labels, 
                     entity_boundaries, gold_sets):
    """Find alpha that maximizes macro F0.5 on holdout."""
    best_f05, best_alpha = 0.0, 0.5
    
    for alpha in np.arange(0.0, 1.01, 0.05):
        fused = alpha * lgbm_scores + (1 - alpha) * ditto_scores
        # Compute entity-level F0.5
        f05 = evaluate_entity_f05(fused, labels, entity_boundaries, gold_sets)
        if f05 > best_f05:
            best_f05, best_alpha = f05, alpha
    
    return best_alpha, best_f05
```

**Expected result:** Optimal alpha is typically 0.4-0.7 (balanced) unless one model is much stronger.

### 4.2 Rank-Based Fusion

Combine based on **ranks** instead of raw scores (more robust to different score scales):

```python
def rank_fusion(scores_a, scores_b, k=60):
    """Reciprocal Rank Fusion (used in information retrieval)."""
    rank_a = np.argsort(np.argsort(-scores_a))  # Higher score → lower rank
    rank_b = np.argsort(np.argsort(-scores_b))
    
    # RRF formula: 1 / (k + rank)
    rrf_a = 1.0 / (k + rank_a)
    rrf_b = 1.0 / (k + rank_b)
    
    return rrf_a + rrf_b
```

**Pros:** Robust to score scale differences (LightGBM outputs [0,1] probabilities, Ditto outputs raw logits)
**Cons:** Loses absolute confidence information

**For our contest:** Use if LightGBM and Ditto have very different score distributions. Otherwise linear fusion is simpler.

### 4.3 Borda Count (Rank Voting)

Each model ranks candidates, final score = sum of ranks:

```python
def borda_count_fusion(models, pairs):
    """Each candidate gets points based on its rank in each model."""
    all_ranks = []
    for model in models:
        scores = model.predict(pairs)
        # Lower rank = higher score
        ranks = len(scores) - np.argsort(np.argsort(scores))
        all_ranks.append(ranks)
    
    return np.sum(all_ranks, axis=0)  # Sum of ranks across models
```

**Use case:** When you have 3+ matchers with different score scales and want a voting-like approach that uses score ordering.

---

## 5. Stacking (Meta-Learning)

Train a **meta-classifier** that learns to combine base model predictions:

```python
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

def stacking_ensemble(base_models, val_pairs, val_labels,
                      meta_classifier='lr'):
    """
    Phase 1: Get predictions from base models on validation set
    Phase 2: Train meta-classifier on those predictions
    """
    # Collect base predictions
    base_preds = np.column_stack([
        m.predict(val_pairs) for m in base_models
    ])
    
    # Train meta-model
    if meta_classifier == 'lr':
        meta = LogisticRegression()
    elif meta_classifier == 'lgbm':
        meta = lgb.LGBMClassifier(n_estimators=50, num_leaves=15)
    
    meta.fit(base_preds, val_labels)
    return meta

def predict_stacked(base_models, meta, test_pairs):
    """Make predictions using stacked ensemble."""
    base_preds = np.column_stack([
        m.predict(test_pairs) for m in base_models
    ])
    return meta.predict_proba(base_preds)[:, 1]
```

**Key insight:** The meta-classifier learns **when** to trust each base model:
- If LightGBM says 0.9 but Ditto says 0.3 → meta learns this is often a false positive (string similarity misleading)
- If LightGBM says 0.3 but Ditto says 0.9 → meta learns this is a semantic match (e.g., abbreviation)

### 5.1 Feature-Enriched Stacking

Don't just use base model scores — add metadata:

```python
def stacking_with_metadata(lgbm_score, ditto_score, pair_metadata):
    """
    pair_metadata: [country, name_length_diff, has_house_number, ...]
    """
    stacking_features = [
        lgbm_score,
        ditto_score,
        lgbm_score - ditto_score,  # Agreement feature
        abs(lgbm_score - ditto_score),  # Disagreement magnitude
        lgbm_score * ditto_score,  # Product (both high = very confident)
        max(lgbm_score, ditto_score),  # Best individual
        min(lgbm_score, ditto_score),  # Worst individual
        *pair_metadata,  # Entity-level features
    ]
    return stacking_features
```

**Expected gain:** +0.005 to +0.015 F0.5 over simple linear weighting.

---

## 6. Cascade Ensembles (Sequential Filtering)

Fast models filter first, slow models refine:

```
All candidate pairs (260M)
  ↓
LightGBM (fast) → score > 0.3 → 50M pairs (keep top 20 per S1)
  ↓
Ditto (slow) → final scoring → 8.7M matches
```

```python
def cascade_ensemble(s1_id, candidates, lgbm_model, ditto_model,
                     fast_threshold=0.3, top_k=20):
    """Two-stage cascade: fast filter → slow reranker."""
    # Stage 1: LightGBM scores all candidates
    lgbm_scores = lgbm_model.predict(candidates)
    
    # Keep only candidates above threshold, up to top_k
    kept_indices = np.where(lgbm_scores >= fast_threshold)[0]
    if len(kept_indices) > top_k:
        top_indices = np.argsort(lgbm_scores)[-top_k:]
        kept_indices = top_indices
    
    if len(kept_indices) == 0:
        return []  # Singleton
    
    # Stage 2: Ditto rescores survivors
    kept_candidates = [candidates[i] for i in kept_indices]
    ditto_scores = ditto_model.predict(kept_candidates)
    
    # Final threshold on Ditto scores
    final_matches = [
        kept_candidates[i] for i, s in enumerate(ditto_scores)
        if s >= 0.7
    ]
    return final_matches
```

**Pros:** Reduces compute by ~10-50× (Ditto only scores high-confidence pairs)
**Cons:** Recall limited by fast filter (if LightGBM misses a match, Ditto never sees it)

**Optimal thresholds:**
- Fast threshold = low (0.2-0.4) to preserve recall
- top_k = 20-50 balancing recall vs compute
- Final threshold = higher (0.6-0.8) because Ditto is more confident

---

## 7. Confidence-Based Routing

Route pairs to different models based on difficulty:

```python
def confidence_routing_ensemble(lgbm_model, ditto_model, pairs,
                                uncertainty_band=(0.3, 0.7)):
    """
    Certain pairs (lgbm < 0.3 or > 0.7): trust LightGBM
    Uncertain pairs (0.3 <= lgbm <= 0.7): use Ditto
    """
    lgbm_scores = lgbm_model.predict(pairs)
    
    certain_negative = lgbm_scores < uncertainty_band[0]
    certain_positive = lgbm_scores > uncertainty_band[1]
    uncertain = ~(certain_negative | certain_positive)
    
    final_scores = lgbm_scores.copy()
    
    # Only run Ditto on uncertain pairs
    if uncertain.sum() > 0:
        uncertain_pairs = [pairs[i] for i in np.where(uncertain)[0]]
        ditto_scores = ditto_model.predict(uncertain_pairs)
        final_scores[uncertain] = ditto_scores
    
    return final_scores
```

**Expected speedup:** ~3-5× (Ditto only sees ~30% of pairs) with <1% quality loss.

---

## 8. Multi-Model Fusion for Our Pipeline

### 8.1 Three-Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│ BM25 Blocking → CAP=150 candidates per S1              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ LightGBM (17 features) → score all candidates          │
│   - String features (Jaccard, JW, trigrams)            │
│   - IDF features (TF-IDF cosine)                        │
│   - Structural features (house, city, address exact)    │
│   - Metadata (name rarity, phonetic)                    │
└─────────────────────────────────────────────────────────┘
                        ↓
           ┌────────────┴────────────┐
           │                         │
    Score > 0.3                 Score < 0.3
    Keep top-20                 → Singleton
           │
           ↓
┌─────────────────────────────────────────────────────────┐
│ Ditto Cross-Encoder → rescore top-20                   │
│   - Semantic understanding                              │
│   - Cross-lingual (French)                              │
│   - Abbreviation/synonym awareness                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Fusion Layer → combine LightGBM + Ditto scores         │
│   Option A: Linear (alpha * lgbm + (1-alpha) * ditto)  │
│   Option B: Stacking (LR on [lgbm, ditto, metadata])   │
│   Option C: Confidence routing (use Ditto only if lgbm │
│             in uncertainty band)                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Graph Post-Processing → transitive closure (optional)   │
│   - Union-find on high-confidence edges (>0.9)          │
│   - Propagate matches through shared S2/S3 entities     │
└─────────────────────────────────────────────────────────┘
                        ↓
                 Final Matches
```

### 8.2 Implementation

```python
class EnsembleERPipeline:
    def __init__(self, bm25_index, idf, lgbm_model, ditto_model,
                 fusion='linear', alpha=0.6):
        self.bm25_index = bm25_index
        self.idf = idf
        self.lgbm = lgbm_model
        self.ditto = ditto_model
        self.fusion = fusion
        self.alpha = alpha
        self.lgbm_threshold = 0.3
        self.top_k = 20
        self.final_threshold = 0.65
    
    def predict_entity(self, s1_name, s1_addr, s1_country, pool):
        # Stage 1: Blocking
        candidates, _ = candidates_for(
            self.bm25_index, s1_name, s1_addr, s1_country, self.idf
        )
        
        if not candidates:
            return []
        
        # Stage 2: LightGBM features
        cand_names = [pool[c][0] for c in candidates]
        cand_addrs = [pool[c][1] for c in candidates]
        cand_countries = [pool[c][2] for c in candidates]
        
        X = batch_features(s1_name, s1_addr, s1_country,
                          cand_names, cand_addrs, self.idf, self.freq)
        lgbm_scores = self.lgbm.predict_proba(X)[:, 1]
        
        # Keep top-K above threshold
        keep = np.where(lgbm_scores >= self.lgbm_threshold)[0]
        if len(keep) > self.top_k:
            keep = np.argsort(lgbm_scores)[-self.top_k:]
        
        if len(keep) == 0:
            return []
        
        # Stage 3: Ditto rescoring
        kept_candidates = [candidates[i] for i in keep]
        ditto_pairs = [
            make_pair(s1_name, s1_addr, s1_country,
                     cand_names[i], cand_addrs[i], cand_countries[i])
            for i in keep
        ]
        ditto_scores = self.ditto.predict(ditto_pairs)
        
        # Stage 4: Fusion
        lgbm_kept = lgbm_scores[keep]
        
        if self.fusion == 'linear':
            final_scores = self.alpha * lgbm_kept + (1 - self.alpha) * ditto_scores
        elif self.fusion == 'max':
            final_scores = np.maximum(lgbm_kept, ditto_scores)
        elif self.fusion == 'product':
            final_scores = lgbm_kept * ditto_scores  # Both must be high
        elif self.fusion == 'stacking':
            stacked = np.column_stack([lgbm_kept, ditto_scores,
                                       np.abs(lgbm_kept - ditto_scores)])
            final_scores = self.stacking_meta.predict_proba(stacked)[:, 1]
        
        # Stage 5: Final thresholding
        matches = [
            kept_candidates[i] for i, s in enumerate(final_scores)
            if s >= self.final_threshold
        ]
        return matches
```

---

## 9. Ensemble Diversity and Correlation

### 9.1 Measuring Model Correlation

```python
def model_correlation(model_a, model_b, val_pairs):
    """Pearson correlation between model scores."""
    scores_a = model_a.predict(val_pairs)
    scores_b = model_b.predict(val_pairs)
    return np.corrcoef(scores_a, scores_b)[0, 1]
```

**Expected correlations:**
- LightGBM vs LightGBM (different seed): 0.85-0.95 (high — redundant)
- LightGBM vs Ditto: 0.60-0.75 (medium — complementary)
- Ditto-DistilBERT vs Ditto-DeBERTa: 0.80-0.90 (high — similar architectures)
- String features vs Ditto: 0.50-0.65 (low — very different mechanisms)

**Key insight:** Ensemble gain is highest when models have **low correlation but high individual quality**. Adding a second LightGBM with different hyperparameters gives minimal gain (high correlation). Adding Ditto to LightGBM gives large gain (low correlation, both high quality).

### 9.2 Diversity-Promoting Training

Train models to be **deliberately different**:

```python
def train_diverse_lgbm_ensemble(X, y, n_models=3):
    """Train multiple LightGBM models with different feature subsets."""
    models = []
    n_features = X.shape[1]
    
    for i in range(n_models):
        # Each model sees 70% of features (different 70% each time)
        feature_mask = np.random.rand(n_features) > 0.3
        X_subset = X[:, feature_mask]
        
        model = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,  # Further randomization
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=i * 42,
        )
        model.fit(X_subset, y)
        models.append((model, feature_mask))
    
    return models
```

**For our contest:** Not worth the complexity. One LightGBM + one Ditto is sufficient diversity.

---

## 10. Error Analysis: When Does Ensemble Help?

### 10.1 Categories of Pairs

| Pair Type | LightGBM | Ditto | Ensemble Advantage |
|-----------|----------|-------|-------------------|
| **Easy positive** (exact name+addr) | ✓ High | ✓ High | No gain (both correct) |
| **Easy negative** (completely different) | ✓ Low | ✓ Low | No gain (both correct) |
| **Lexical match, semantic mismatch** ("ABC Corp" same address, different business) | ✗ High (false pos) | ✓ Low (correct) | Ditto rescues |
| **Lexical mismatch, semantic match** ("McDonald's" vs "McDonald Corp") | ✗ Medium | ✓ High | Ditto rescues |
| **Typo variant** ("Chcago" vs "Chicago") | ✗ Low (miss) | ✓ Medium | Ditto rescues |
| **French entity, zero-shot** | ✗ Low (IDF wrong) | ✓ High | Ditto rescues |
| **Common name, same city, different addr** | ✗ High (false pos) | ✗ Medium (uncertain) | Ensemble lowers score |

**Expected error breakdown:**
- 70% of pairs: Both models agree (easy)
- 20% of pairs: One model correct, other wrong (ensemble helps)
- 10% of pairs: Both models wrong (ensemble cannot help)

### 10.2 Confusion Matrix Ensemble

```python
def analyze_ensemble_value(lgbm_preds, ditto_preds, true_labels, threshold=0.5):
    """
    Measure how often ensemble resolves disagreements correctly.
    """
    lgbm_binary = (lgbm_preds >= threshold).astype(int)
    ditto_binary = (ditto_preds >= threshold).astype(int)
    
    agree_correct = ((lgbm_binary == ditto_binary) & 
                     (lgbm_binary == true_labels)).sum()
    agree_wrong = ((lgbm_binary == ditto_binary) & 
                   (lgbm_binary != true_labels)).sum()
    
    disagree_lgbm_right = ((lgbm_binary != ditto_binary) & 
                           (lgbm_binary == true_labels)).sum()
    disagree_ditto_right = ((lgbm_binary != ditto_binary) & 
                            (ditto_binary == true_labels)).sum()
    
    print(f"Both correct:      {agree_correct:,} ({100*agree_correct/len(true_labels):.1f}%)")
    print(f"Both wrong:        {agree_wrong:,} ({100*agree_wrong/len(true_labels):.1f}%)")
    print(f"LGBM right, Ditto wrong: {disagree_lgbm_right:,}")
    print(f"Ditto right, LGBM wrong: {disagree_ditto_right:,}")
    print(f"Ditto win rate on disagreements: {100*disagree_ditto_right/(disagree_lgbm_right+disagree_ditto_right):.1f}%")
```

**If Ditto wins 70%+ of disagreements → weight Ditto higher (alpha < 0.5)**

---

## 11. Practical Recommendations for Our Contest

### 11.1 Recommended Fusion Strategy

**Phase 1 (Day 1-2):** LightGBM only, optimize features and threshold
**Phase 2 (Day 2-3):** Add Ditto cross-encoder, use **linear fusion**
```python
alpha = 0.5  # Start with equal weighting
final_score = alpha * lgbm_prob + (1 - alpha) * ditto_score
```

**Phase 3 (Day 3, if time):** Sweep alpha on holdout:
```python
for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
    scores = alpha * lgbm + (1-alpha) * ditto
    f05 = evaluate_f05(scores, holdout)
    print(f"alpha={alpha:.1f} → F0.5={f05:.4f}")
```

**Phase 4 (Final hours, if compute allows):** Try **confidence routing**:
```python
# Use Ditto only on uncertain LightGBM predictions
if lgbm_score < 0.3 or lgbm_score > 0.7:
    final_score = lgbm_score  # Trust LightGBM
else:
    final_score = ditto_score  # Use Ditto for uncertain pairs
```

### 11.2 Expected Gains

| Strategy | Effort | Expected Gain vs LightGBM-only | Compute Cost |
|----------|--------|-------------------------------|--------------|
| Linear fusion (alpha=0.5) | 5 min | +0.02-0.04 | Ditto inference time |
| Optimized alpha | 30 min | +0.03-0.05 | +holdout sweep |
| Confidence routing | 1 hour | +0.025-0.045 | 0.7× Ditto inference |
| Stacking with metadata | 2 hours | +0.04-0.06 | +meta-model train |

### 11.3 What NOT To Do

- **Don't ensemble LightGBM with LightGBM** (same features, different seeds) → high correlation, low gain
- **Don't use majority vote** → throws away score information
- **Don't optimize ensemble on public LB** → overfitting risk
- **Don't add a 3rd matcher unless it's very different** → diminishing returns, compute cost

---

## 12. Key Takeaways

1. **Ensembles work best when models are diverse** — LightGBM (features) + Ditto (semantic) is ideal
2. **Linear fusion is the simplest effective method** — alpha sweep on holdout finds optimal weighting
3. **Cascades save compute** — fast filter → slow reranker reduces Ditto inference by 5-10×
4. **Stacking adds ~0.01-0.02 F0.5** over linear fusion — worth it if you have time
5. **Confidence routing is compute-optimal** — only use expensive model on uncertain pairs
6. **Model correlation predicts ensemble gain** — low correlation = high gain
7. **Disagreement analysis reveals which model to trust** — if Ditto wins 70%+ disagreements, weight it higher
8. **Don't over-ensemble** — 2 diverse models is enough; 3+ has diminishing returns

---

## 13. References

1. Kuncheva, L.I. (2014). "Combining Pattern Classifiers: Methods and Algorithms." Wiley.
2. Dietterich, T.G. (2000). "Ensemble Methods in Machine Learning." MCS.
3. Elmagarmid, A.K., et al. (2007). "Duplicate Record Detection: A Survey." IEEE TKDE.
4. Kopcke, H., et al. (2010). "Evaluation of Entity Resolution Approaches on Real-World Match Problems." VLDB.
5. Mudgal, S., et al. (2018). "Deep Learning for Entity Matching: A Design Space Exploration." SIGMOD.
6. Ebraheem, M., et al. (2018). "Distributed Representations of Tuples for Entity Resolution." VLDB.
7. Li, Y., et al. (2020). "Deep Entity Matching with Pre-Trained Language Models." VLDB.
8. Kasai, J., et al. (2019). "Low-resource Deep Entity Resolution with Transfer and Active Learning." ACL.
