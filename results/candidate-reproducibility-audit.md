# Candidate reproducibility audit

- Symbol/timeframe: EURUSD / H4
- Seeds: 2033, 2034, 2035
- Final test evaluated: no

## Run summaries

| Seed | Status | Qualified | Unique signals |
|---:|---|---:|---:|
| 2033 | validated_candidate | 3 | 72 |
| 2034 | validated_candidate | 3 | 61 |
| 2035 | no_robust_candidate | 0 | 67 |

## Cross-seed exact matches

- Exact rule-structure pairs: 0
- Exact validation entry-set pairs: 0

## Candidate motif counts

| Seed | Candidates | Qualified | Motif counts |
|---:|---:|---:|---|
| 2033 | 20 | 3 | atr_vs_rolling_atr: 3, ema_direction_with_atr_regime_gate: 2, ema_pair_comparison: 1, price_vs_ema_filter: 13, rsi_threshold_cross: 20 |
| 2034 | 20 | 3 | atr_vs_rolling_atr: 15, ema_direction_with_atr_regime_gate: 2, ema_pair_comparison: 11, price_vs_ema_filter: 6, rsi_threshold_cross: 17 |
| 2035 | 20 | 0 | atr_vs_rolling_atr: 6, ema_direction_with_atr_regime_gate: 4, ema_pair_comparison: 1, price_vs_ema_filter: 4, rsi_threshold_cross: 14 |

## Highest detailed structural similarities

| Seed A/rank | Seed B/rank | Rule similarity | Long | Short | Motif Jaccard | Risk match |
|---|---|---:|---:|---:|---:|---:|
| 2033/11 | 2034/13 | 0.400 | 0.611 | 0.138 | 1.000 | 0.000 |
| 2033/10 | 2034/16 | 0.311 | 0.625 | 0.138 | 1.000 | 0.000 |
| 2034/18 | 2035/6 | 0.301 | 0.556 | 0.240 | 0.250 | 0.000 |
| 2033/13 | 2034/14 | 0.277 | 0.333 | 0.138 | 0.500 | 0.000 |
| 2034/20 | 2035/5 | 0.253 | 0.000 | 1.000 | 0.500 | 0.000 |
| 2034/7 | 2035/17 | 0.228 | 0.328 | 0.000 | 0.250 | 0.000 |
| 2034/16 | 2035/11 | 0.228 | 0.000 | 1.000 | 1.000 | 0.000 |
| 2033/9 | 2034/16 | 0.215 | 0.625 | 0.082 | 1.000 | 0.000 |
| 2034/20 | 2035/16 | 0.208 | 0.000 | 1.000 | 0.500 | 0.000 |
| 2034/15 | 2035/5 | 0.206 | 0.000 | 1.000 | 0.333 | 0.000 |
| 2034/2 | 2035/17 | 0.196 | 0.262 | 0.000 | 0.250 | 0.000 |
| 2034/7 | 2035/5 | 0.185 | 0.000 | 1.000 | 0.500 | 0.333 |
| 2034/10 | 2035/5 | 0.180 | 0.000 | 1.000 | 0.333 | 0.000 |
| 2034/17 | 2035/5 | 0.180 | 0.000 | 0.667 | 0.333 | 0.000 |
| 2033/11 | 2034/14 | 0.176 | 0.194 | 0.138 | 1.000 | 0.000 |

## Interpretation constraints

- Validation entry events are represented by SHA-256 fingerprints in the source reports. Equality detects exact matches only; approximate cross-seed Jaccard overlap cannot be reconstructed.
- Motif labels are coarse structural indicators and do not establish equivalent trading behaviour.
- Detailed rule similarity is a weighted Jaccard score over path-aware operator, feature, threshold and lookback tokens. It is a structural heuristic, not behavioural equivalence.
- Exact threshold values and rolling lookbacks are retained in taxonomy v2; long and short entry trees are scored separately. Risk parameters are compared separately from entry rules.
- The validation partition has been repeatedly examined during development and is not an independent performance estimate.
- A structural match across seeds is descriptive evidence only and does not justify changing selection or validation thresholds.
