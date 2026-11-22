# Intra-run finalist behavioural redundancy audit

- Symbol/timeframe: EURUSD / H4
- Seeds: 2033, 2034, 2035
- Final test evaluated: no
- Finalists: 60
- Within-seed pairs: 570

## Per-seed summary

| Seed | Finalists | Qualified | Unique train signals | Unique validation signals | Unique train trades | Unique validation trades |
|---:|---:|---:|---:|---:|---:|---:|
| 2033 | 20 | 3 | 13 | 12 | 19 | 19 |
| 2034 | 20 | 3 | 12 | 10 | 19 | 19 |
| 2035 | 20 | 0 | 14 | 13 | 20 | 15 |

## Highest training signal overlap

| Seed | Candidate A | Candidate B | Active Jaccard | Long | Short | Both qualified |
|---:|---:|---:|---:|---:|---:|---|
| 2033 | 2 | 4 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 2 | 5 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 2 | 18 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 4 | 5 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 4 | 18 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 5 | 18 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 8 | 12 | 1.000 | 1.000 | 1.000 | True |
| 2033 | 10 | 11 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 13 | 15 | 1.000 | 0.000 | 1.000 | False |
| 2033 | 13 | 20 | 1.000 | 0.000 | 1.000 | False |
| 2033 | 15 | 20 | 1.000 | 0.000 | 1.000 | False |
| 2034 | 2 | 7 | 1.000 | 0.000 | 1.000 | False |
| 2034 | 2 | 9 | 1.000 | 0.000 | 1.000 | False |
| 2034 | 2 | 10 | 1.000 | 0.000 | 1.000 | False |
| 2034 | 2 | 11 | 1.000 | 0.000 | 1.000 | False |

## Highest validation signal overlap

| Seed | Candidate A | Candidate B | Active Jaccard | Long | Short | Both qualified |
|---:|---:|---:|---:|---:|---:|---|
| 2033 | 2 | 4 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 2 | 5 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 2 | 18 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 3 | 7 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 4 | 5 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 4 | 18 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 5 | 18 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 8 | 12 | 1.000 | 1.000 | 1.000 | True |
| 2033 | 10 | 11 | 1.000 | 1.000 | 1.000 | False |
| 2033 | 13 | 15 | 1.000 | 0.000 | 1.000 | False |
| 2033 | 14 | 16 | 1.000 | 1.000 | 1.000 | False |
| 2034 | 2 | 7 | 1.000 | 0.000 | 1.000 | False |
| 2034 | 2 | 9 | 1.000 | 0.000 | 1.000 | False |
| 2034 | 2 | 10 | 1.000 | 0.000 | 1.000 | False |
| 2034 | 2 | 11 | 1.000 | 0.000 | 1.000 | False |

## Exact duplicate signal groups

### Seed 2033
- Training: ranks [2, 4, 5, 18] (n=4); ranks [13, 15, 20] (n=3); ranks [8, 12] (n=2); ranks [10, 11] (n=2)
- Validation: ranks [2, 4, 5, 18] (n=4); ranks [3, 7] (n=2); ranks [8, 12] (n=2); ranks [10, 11] (n=2); ranks [13, 15] (n=2); ranks [14, 16] (n=2)
### Seed 2034
- Training: ranks [2, 7, 9, 10, 11, 12, 15, 20] (n=8); ranks [13, 14] (n=2)
- Validation: ranks [2, 7, 9, 10, 11, 12, 15, 20] (n=8); ranks [3, 6] (n=2); ranks [5, 17] (n=2); ranks [13, 14] (n=2)
### Seed 2035
- Training: ranks [3, 8, 13] (n=3); ranks [4, 9, 14] (n=3); ranks [1, 2] (n=2); ranks [10, 12] (n=2)
- Validation: ranks [3, 8, 13] (n=3); ranks [4, 9, 14] (n=3); ranks [1, 2] (n=2); ranks [7, 11] (n=2); ranks [10, 12] (n=2)

## Limitations

- This is a finalist-only audit; it does not measure redundancy in the full evolving population.
- Validation has been repeatedly examined during development and is descriptive, not independent performance evidence.
- Exact signal agreement can be inflated by neutral bars; active-signal Jaccard is reported separately.
- Trade-entry overlap depends on execution and risk assumptions. EURUSD source data is bid-only, so spread is estimated.
- The final chronological holdout is never evaluated.
- No audit result changes fitness, selection, migration, validation gates or final-test access.
