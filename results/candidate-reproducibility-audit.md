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

## Interpretation constraints

- Validation entry events are represented by SHA-256 fingerprints in the source reports. Equality detects exact matches only; approximate cross-seed Jaccard overlap cannot be reconstructed.
- Motif labels are coarse structural indicators and do not establish equivalent trading behaviour.
- The validation partition has been repeatedly examined during development and is not an independent performance estimate.
- A structural match across seeds is descriptive evidence only and does not justify changing selection or validation thresholds.
