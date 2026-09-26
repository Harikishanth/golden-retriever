# Multilingual Entity Resolution: The France Problem

**Author:** Research artifact for Amazon ML Challenge 2026 — Da Big Three  
**Date:** 2026-09-26  
**Purpose:** Deep dive into handling French business entities in ER when training data contains zero French examples. France is 15% of test S1 (259,452 entities) — getting it wrong could cost ~0.15 × penalty in macro F0.5.

---

## 1. The Problem

Our training set contains only US and India entities. The test set introduces **France (15%)** — a language and address convention the model has never seen in labeled pairs.

This is a **zero-shot cross-lingual transfer** problem for ER:
- French business names: "Boulangerie Patisserie Jean-Pierre", "Société Générale", "SARL Les Trois Mousquetaires"
- French addresses: "12 Rue de la Paix, 75002 Paris", "ZI des Portes de Dordogne, 24660 Coulounieix-Chamiers"
- French legal suffixes: SARL, SAS, SASU, EURL, SA, SCI, SNC, EI

String similarity features (Jaccard, Jaro-Winkler) work identically across languages — they're language-agnostic by design. But:
- **IDF weights** will be wrong for French tokens (trained on US+India corpus)
- **Name normalization** may miss French-specific patterns
- **Address parsing** has different conventions (commune before postal code, cedex)

---

## 2. French Business Name Patterns

### 2.1 Legal Suffixes

| Suffix | Expansion | Equivalent |
|--------|-----------|------------|
| SARL | Société à Responsabilité Limitée | LLC/Ltd |
| SAS | Société par Actions Simplifiée | Corp |
| SASU | Société par Actions Simplifiée Unipersonnelle | Single-member Corp |
| EURL | Entreprise Unipersonnelle à Responsabilité Limitée | Single-member LLC |
| SA | Société Anonyme | PLC |
| SCI | Société Civile Immobilière | Real estate entity |
| SNC | Société en Nom Collectif | General partnership |
| EI | Entreprise Individuelle | Sole proprietor |
| GIE | Groupement d'Intérêt Économique | Economic interest group |
| EARL | Exploitation Agricole à Responsabilité Limitée | Agricultural LLC |

**Action:** These are already in `normalize.py`'s `SUFFIXES` set — verify they're being stripped from French names during blocking.

### 2.2 Common Name Structures

French business names often follow patterns different from English:
- **Activity + Location:** "Boulangerie de Paris", "Pharmacie du Centre"
- **Personal Name + Activity:** "Jean-Pierre Menuiserie", "Cabinet Dupont"
- **Activity + Qualifier:** "Le Petit Bistrot", "La Grande Épicerie"
- **Brand + Legal:** "Carrefour SA", "Total Energies SAS"

The articles "Le", "La", "Les", "L'", "Du", "De", "Des" are function words that should be treated like English "The" — stripped during normalization or given low IDF weight.

### 2.3 Accent and Diacritic Handling

French uses: é, è, ê, ë, à, â, ä, ù, û, ü, ô, ö, ï, ÿ, ç, œ, æ

Our `fold()` function already strips accents via `unicodedata.normalize("NFKD")` + combining character removal. This means:
- "Café" → "cafe" ✓
- "Épicerie" → "epicerie" ✓
- "François" → "francois" ✓
- "Ça" → "ca" ✓ (cedilla removed)
- "Cœur" → "coeur" ✓ (ligature decomposed)

**Good:** Our normalization already handles French diacritics correctly.

---

## 3. French Address Conventions

### 3.1 Structure

A French address typically follows:
```
[Number] [Street Type] [Street Name]
[Complement: Building, Floor, etc.]
[Postal Code] [Commune]
[CEDEX if applicable]
```

Example: `24 Boulevard du Général de Gaulle, 93110 Rosny-sous-Bois`

### 3.2 Key Differences from US/India

| Feature | US | India | France |
|---------|-----|-------|--------|
| Postal code | 5 digits, before city | 6 digits (PIN), after city | 5 digits, BEFORE city |
| Street prefix | Number first | Number often missing | Number first |
| City naming | Simple | Simple or compound | Often hyphenated ("Saint-Germain-en-Laye") |
| Region | State (2 letters) | State (varies) | Département (2-digit code) |
| Common words | St, Ave, Blvd | Marg, Nagar, Colony | Rue, Boulevard, Avenue, Place |
| Cedex | N/A | N/A | Distribution code for businesses |

### 3.3 French Address Tokens Already in normalize.py

Our `ADDR_STOP` set already includes: `rue, allee, quai, impasse, imp, bis, chemin, route, place, cours, passage, residence`

**Missing tokens to add:**
- `boulevard` (already there as `blvd` but French uses full word)
- `avenue` (already there)
- `cedex` — mail routing code, should be stopped
- `batiment`, `bat` — building
- `etage` — floor
- `escalier`, `esc` — staircase
- `porte` — door
- `lotissement`, `lot` — subdivision
- `zone`, `zi` — zone industrielle
- `lieu`, `dit` — "lieu-dit" (hamlet name)
- `commune` — township
- `departement`, `dept` — department
- `arrondissement` — district

### 3.4 French Postal Codes

French postal codes are 5 digits. The first two digits are the département number:
- 75 = Paris
- 13 = Bouches-du-Rhône (Marseille)
- 69 = Rhône (Lyon)
- 31 = Haute-Garonne (Toulouse)
- 33 = Gironde (Bordeaux)

Postal codes can be useful as blocking keys (same postal code → likely same area), but they might appear as regular numbers in our `house_numbers()` extraction. Since house numbers are typically 1-4 digits and postal codes are 5 digits, we could filter: **numbers ≥ 5 digits in French addresses are likely postal codes, not house numbers**.

---

## 4. Cross-Lingual Transfer for ER

### 4.1 Why String Similarity Still Works

The core insight: **ER features that measure structural similarity are language-agnostic.**

| Feature | Language Dependent? | Works on French? |
|---------|-------------------|-----------------|
| Jaccard on name tokens | No — bags of tokens | ✓ |
| Jaro-Winkler | No — character-level | ✓ |
| Trigram Jaccard | No — character n-grams | ✓ |
| House number agreement | No — numeric | ✓ |
| Address Jaccard | Partially — depends on tokenization | ✓ if stopwords are right |
| TF-IDF cosine | Yes — IDF weights from train | ⚠️ IDF values wrong |
| Name rarity | Yes — frequency from S1 | ⚠️ French names unseen |
| Phonetic match (Soundex) | Yes — English phonetics | ✗ Wrong for French |

### 4.2 Features That Need Adaptation

**TF-IDF cosine:** IDF computed from training pool (US+India). French tokens like "boulangerie", "pharmacie" will get IDF=1.0 (default for unseen tokens). This is actually not terrible — rare tokens get high IDF, and all French tokens are "rare" in the US+India corpus.

**Name rarity:** Uses frequency from S1. French business names will appear rare (low frequency), making them more "matchable." This could slightly increase false positives on French entities but also increase recall. Net effect: approximately neutral.

**Phonetic match (Soundex):** Soundex encodes English phonetics. "Beaumont" in English Soundex = B553, but the French pronunciation is completely different. Soundex will still match identical spellings, but won't catch French phonetic equivalences (e.g., "Beaumont" vs "Beaumon").

**Recommendation:** For French, phonetic features should be weighted down or replaced with language-aware phonetics (not practical in contest time).

### 4.3 Multilingual Transformer Advantage

This is where the Ditto cross-encoder shines. `distilbert-base-multilingual-cased` was pre-trained on:
- 104 languages including French, English, Hindi
- Wikipedia text in all languages
- Learned cross-lingual representations

The model understands:
- "Boulangerie" = "Bakery" (from multilingual pre-training)
- "Rue" = "Street" (cross-lingual alignment)
- "SAS" = corporate suffix (from French Wikipedia)
- French name structures and address patterns

**Even without French training pairs**, the model transfers ER patterns learned from English/Hindi:
- "Same name tokens + similar address → match" works in any language
- The attention mechanism aligns "COL name VAL Boulangerie X" with "COL name VAL Boulangerie X" regardless of language

### 4.4 Embedding Models for French Retrieval

For dense retrieval (blocking), multilingual models are critical:

| Model | French MTEB Retrieval | English MTEB | Params | License |
|-------|----------------------|-------------|--------|---------|
| `paraphrase-multilingual-MiniLM-L12-v2` | ~45 | ~48 | 118M | Apache 2.0 |
| `BAAI/bge-m3` | ~58 | ~62 | 568M | MIT |
| `intfloat/multilingual-e5-large` | ~55 | ~59 | 560M | MIT |
| `sentence-transformers/all-MiniLM-L6-v2` | ~32 | ~49 | 22M | Apache 2.0 |

**Key insight:** `all-MiniLM-L6-v2` (English-only) drops ~17 points on French retrieval vs English. `paraphrase-multilingual-MiniLM-L12-v2` is 5x bigger but maintains quality across languages.

---

## 5. Mitigation Strategies

### 5.1 Normalize French-Specific Noise

Add to `normalize.py`:
```python
FRENCH_STOP_NAME = {"le", "la", "les", "du", "de", "des", "au", "aux", "et"}
FRENCH_ADDR_STOP = {"cedex", "batiment", "bat", "etage", "escalier", "esc",
                    "porte", "lotissement", "lot", "zone", "zi", "lieu", "dit",
                    "commune", "departement", "dept", "arrondissement"}
```

### 5.2 Country-Conditioned Thresholds

French entities may have systematically different score distributions than US/India (different IDF, different name patterns). A single global threshold may not be optimal.

Strategy: After training on US+India holdout, sweep a separate threshold for country="France" using a synthetic validation set (if possible) or simply tighten the threshold (higher precision, lower recall) for unseen distributions.

### 5.3 French-Aware Blocking Keys

Current blocking uses BM25 on name+address tokens. For French:
- Add `cedex` stripping from addresses
- Add French postal code as a blocking token (same postal code → same area)
- Consider adding département code (first 2 digits of postal code) as a blocking dimension

### 5.4 Singleton Safety for France

If the model is uncertain about French entities (lower confidence due to zero-shot transfer), the F0.5 metric rewards conservative behavior:
- Empty prediction on a true singleton = 1.0 (perfect)
- Wrong prediction on a true singleton = 0.0 (total failure)
- Empty prediction on a non-singleton = miss, but weighted 2x less than false positive

**Strategy:** Raise the match threshold for French entities. Accept slightly lower recall for much higher precision.

---

## 6. Experimental Plan

1. **Measure baseline France performance** — run holdout with French-like entity names/addresses (none in train, but can check how the model behaves on unseen token distributions)
2. **Add French stop words** to normalize.py
3. **Switch to multilingual embedding model** for dense retrieval
4. **Train Ditto on English+Hindi** — evaluate zero-shot transfer to French
5. **Per-country threshold sweep** — optimize French threshold separately
6. **Compare:** string-similarity-only vs multilingual-Ditto on French subset

---

## 7. Key Takeaways

1. **String similarity features work across languages** — Jaccard, JW, trigram Jaccard are language-agnostic
2. **IDF and phonetic features degrade** on unseen languages — but not catastrophically
3. **Multilingual transformers are the biggest win** — they transfer ER patterns cross-lingually
4. **Conservative thresholding for France** — F0.5 rewards precision; don't force matches on uncertain French pairs
5. **French stop words and address conventions** need explicit handling in normalize.py
6. **Postal codes (5 digits)** should not be confused with house numbers in French addresses

---

## 8. References

1. Conneau, A., et al. (2020). "Unsupervised Cross-lingual Representation Learning at Scale." ACL.
2. Reimers, N., & Gurevych, I. (2020). "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation." EMNLP.
3. Ebraheem, M., et al. (2018). "Distributed Representations of Tuples for Entity Resolution." VLDB.
4. Fu, C., et al. (2021). "Hierarchical Matching Network for Heterogeneous Entity Resolution." IJCAI.
5. Chen, X., et al. (2023). "Entity Matching using Large Language Models." arXiv.
6. INSEE (French National Statistics): Business registration conventions and address standards.
