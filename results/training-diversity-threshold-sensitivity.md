# Training-only diversity threshold sensitivity

- Seeds: 2033, 2034, 2035
- Thresholds: 0.80, 0.85, 0.90, 0.95
- Maximum shortlist size: 20
- Selection inputs: training fitness and training signals only
- Validation used for selection: no
- Final test evaluated: no

| Seed | Threshold | Diversity selected | Fitness-only same-size unique | Diversity unique | Strongest fitness retained |
|---:|---:|---:|---:|---:|:---:|
| 2033 | 0.80 | 5 | 3 | 5 | yes |
| 2034 | 0.80 | 6 | 6 | 6 | yes |
| 2035 | 0.80 | 5 | 4 | 5 | yes |
| 2033 | 0.85 | 5 | 3 | 5 | yes |
| 2034 | 0.85 | 6 | 6 | 6 | yes |
| 2035 | 0.85 | 6 | 5 | 6 | yes |
| 2033 | 0.90 | 6 | 4 | 6 | yes |
| 2034 | 0.90 | 6 | 6 | 6 | yes |
| 2035 | 0.90 | 6 | 5 | 6 | yes |
| 2033 | 0.95 | 8 | 6 | 8 | yes |
| 2034 | 0.95 | 6 | 6 | 6 | yes |
| 2035 | 0.95 | 7 | 6 | 7 | yes |

## Threshold 0.80

- Seed 2033: selected ranks [1, 8, 9, 10, 13]; rejected as redundant 15; lowest selected training fitness 1.765692.
- Seed 2034: selected ranks [1, 2, 3, 13, 16, 19]; rejected as redundant 14; lowest selected training fitness 1.126179.
- Seed 2035: selected ranks [1, 4, 5, 6, 16]; rejected as redundant 15; lowest selected training fitness 1.434495.

## Threshold 0.85

- Seed 2033: selected ranks [1, 8, 9, 10, 13]; rejected as redundant 15; lowest selected training fitness 1.765692.
- Seed 2034: selected ranks [1, 2, 3, 13, 16, 19]; rejected as redundant 14; lowest selected training fitness 1.126179.
- Seed 2035: selected ranks [1, 4, 5, 6, 7, 16]; rejected as redundant 14; lowest selected training fitness 1.434495.

## Threshold 0.90

- Seed 2033: selected ranks [1, 6, 8, 9, 10, 13]; rejected as redundant 14; lowest selected training fitness 1.765692.
- Seed 2034: selected ranks [1, 2, 3, 13, 16, 19]; rejected as redundant 14; lowest selected training fitness 1.126179.
- Seed 2035: selected ranks [1, 4, 5, 6, 7, 16]; rejected as redundant 14; lowest selected training fitness 1.434495.

## Threshold 0.95

- Seed 2033: selected ranks [1, 6, 8, 9, 10, 13, 14, 19]; rejected as redundant 12; lowest selected training fitness 1.639142.
- Seed 2034: selected ranks [1, 2, 3, 13, 16, 19]; rejected as redundant 14; lowest selected training fitness 1.126179.
- Seed 2035: selected ranks [1, 4, 5, 6, 7, 16, 17]; rejected as redundant 13; lowest selected training fitness 1.411400.

## Limitations

- This is an offline finalist-selection simulation, not a change to production evolution.
- It operates only on the existing finalist pool and cannot recover candidates discarded earlier.
- Training fitness and training signals are used for selection; validation is not read by the selection algorithm.
- The fitness-only baseline is truncated to the number of candidates actually retained by the diversity selector.
- The validation partition remains development data and is not independent performance evidence.
- The final chronological holdout is never evaluated.
- Bid-only source data requires an estimated spread for execution-based audits; this signal-only simulation does not model fills.
