# Business Entity Resolution — rule baseline

Multi-key blocking (normalized name, house-number+street, house-number+city,
full address, name-token+city) unioned, then a precision-first rule matcher that
requires address evidence unless the name is rare in Source 1.

## Run

From this directory:

```
python src/run.py holdout     # macro F0.5 on 100k held-out train Source-1 ids
python src/run.py test        # writes ../../output/matching_results.tsv and candidate_pairs.tsv
```

Validate before uploading (from the student_resource directory):

```
python utils/validate_submission.py ^
    --matching "..\..\..\output\matching_results.tsv" ^
    --candidate "..\..\..\output\candidate_pairs.tsv" ^
    --test-dir dataset\test
```

Data is read from `D:\ML challenge\dataset\student_resource\dataset`.
