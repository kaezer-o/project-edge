# Project EDGE: ecological generation-loop review

## Scope

Reviewed the ecological initialization, population evaluation, survivor selection, stagnation tracking, reproduction, and validation-candidate handoff. Changes are limited to those paths and their tests. Strategy rules, execution simulation, risk settings, and market-data processing were not changed.

## Changes

1. **Initial population quality**
   - Added `initialize_ecological_population()`.
   - Initial genomes must be unique and produce at least one exclusive long signal and one exclusive short signal on the training frame.
   - Initialization is deterministic for a given seed and fails clearly after a bounded number of attempts rather than silently filling the population with inactive genomes.

2. **Offspring activity**
   - The ecological generation loop now checks initial genomes before evaluation.
   - Reproduction receives the training frame and repairs inactive crossover offspring and immigrants through bounded mutation attempts.
   - Existing duplicate-genome fallback behavior is retained, provided the accepted genome is directionally active.

3. **Training sample size in selection**
   - Added `trade_count` to `OrganismFitness`.
   - Ecological research passes `ResearchConfig.minimum_train_trades` into survivor selection.
   - Adequately sampled organisms receive priority over under-sampled organisms when choosing species champions and filling global competition slots.
   - Under-sampled organisms remain eligible as an exploration fallback when qualified candidates are insufficient. They cannot become habitat champions or reset stagnation.

4. **Stagnation eligibility**
   - Stagnation can reset only from organisms meeting the minimum total training-trade count and having qualified observations in at least two distinct chronological periods.
   - Period indices are deduplicated across habitats, preventing two habitat observations in one time window from counting as two separate periods.
   - History now records the minimum trade threshold and number of stagnation-eligible candidates.

5. **Validation shortlist**
   - If `validation_pool` is smaller than the survivor pool, adequately sampled organisms are shortlisted before under-sampled organisms; Pareto rank and crowding then order candidates within those groups.
   - Final organism reports now include total trade count.

## Verification

- `python -m compileall -q src tests`: passed.
- `PYTHONPATH=src python -m pytest -q`: **148 passed**.
- Added regression tests for unique/directionally active initialization, inactive initial-population rejection, offspring repair, minimum-sample survivor priority, sample-qualified stagnation, and validation-shortlist ordering.

## Remaining design consideration

Stagnation still measures progress by the best eligible training expectancy. It does not require that the same candidate also improve drawdown, profit factor, or another composite robustness score. The multi-objective survivor selection continues to account for drawdown and the other fitness objectives, but the mutation-rate controller is expectancy-led. Changing that controller would require an explicit definition of what counts as meaningful multi-objective progress, so it was left unchanged.

## Experiment boundary

No market-scale research run was performed. The supplied ZIP contains source, scripts, tests, and package metadata, but no `data/EURUSD_M1.csv` or local virtual environment. The passing suite verifies code behavior; it does not establish that any strategy is profitable or ready for live trading.
