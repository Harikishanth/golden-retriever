# Dataset stats

`computed: 2026-09-25 07:35–07:41 IST` · streamed, not pandas  
`path:` `D:\ML challenge\dataset\student_resource\dataset\`

## Sizes on disk

| File | MB | Rows |
|---|---:|---:|
| train/train_source1.tsv | 200 | 2,206,821 |
| train/train_source2.tsv | 467 | 5,034,616 |
| train/train_source3.tsv | 480 | 5,285,603 |
| train/train_ground_truth.tsv | 121 | 2,206,821 |
| test/test_source1.tsv | 167 | 1,732,544 |
| test/test_source2.tsv | 486 | 4,887,273 |
| test/test_source3.tsv | 483 | 5,082,316 |

~24M records. Validator comment (~1.7M test entities) = **S1 test only** (1,732,544).

## Schema (confirmed)

`entity_id`, `business_name`, `business_address`, `country`  
GT: `source1_entity_id`, `matched_entity_ids`  
Prefixes match files (`S1`/`S2`/`S3`). No empty names. Empty **addresses** only on S2/S3 (~3%).

## Country

| Split | US | India | France |
|---|---:|---:|---:|
| train S1 | 1,323,633 | 883,188 | 0 |
| train S2 | 3,016,817 | 2,017,799 | 0 |
| train S3 | 3,170,056 | 2,115,547 | 0 |
| test S1 | 663,106 | 809,986 | **259,452** (15.0%) |
| test S2 | 1,871,330 | 2,312,565 | 703,378 |
| test S3 | 1,945,701 | 2,405,000 | 731,615 |

## Train labels

- Singletons: **123,247 / 2,206,821 = 5.58%**
- All-empty dummy F0.5 on train ≈ **0.056** (not 0.70)
- With matches: 2,083,574 entities, mean **3.67** S2/S3 ids
- Size hist: 3 (531k), 4 (484k), 2 (375k), 5 (322k), 6 (165k), 0 (123k), 1 (119k), … max **11**
- Link counts: 3.69M S2 + 3.94M S3

**Correction:** public 0.70 / 0.96 cannot be “all empty” unless the public slice is a totally different singleton mix. Default: those teams **found real matches**.

## Train exact-match dummy (2026-09-25 07:44 IST)

Script: `src/eval_exact_match_train.py`  
Key = `lower(name) + collapsed spaces` + same for address. Empty addresses skipped.

| | |
|---|---|
| index keys (S2+S3) | 9,895,074 |
| S1 with ≥1 exact hit | 59,633 / 2,206,821 (2.7%) |
| **macro F0.5** | **0.073089** |
| all-empty floor | **0.0558** |

Raw exact match is almost the empty baseline. Next lever is suffix/address abbreviation + token sort + Jaccard, not “upload exact.”
