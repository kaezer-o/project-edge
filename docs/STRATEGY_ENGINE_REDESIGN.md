# Project EDGE: strategy-engine redesign

Status: proposed implementation plan  
Branch: `research/open-ended-strategy-engine`  
Baseline: `research/sample-aware-fitness`

## Decision

Retain the existing data ingestion, causal feature pipeline, quote-aware backtester, risk controls, ecological population model, and temporal validation framework. Replace the current fixed-condition genome as the long-term strategy representation with a bounded, typed expression-tree program. Introduce a batch evaluator behind a backend interface, with CPU as the reference implementation and GPU as an optional, benchmark-gated accelerator.

The objective is autonomous hypothesis discovery. No architecture or fitness function can guarantee a profitable strategy.

## Findings from the current repository

1. `StrategyGenome` currently contains separate long and short tuples of `Condition(feature, operator, threshold)`, plus stop distance, reward/risk, and holding duration.
2. Conditions are canonicalized to a small conjunction of feature thresholds. The feature vocabulary is selected from a fixed set in `evolution.py`.
3. The evolutionary engine has ecological speciation, habitat profiling, novelty, mutation, survivor selection, and persistent memory. These are useful foundations, but they evolve a restricted rule representation.
4. `evaluate_ecological_population` loops over organisms and calls the Python backtest path individually. The project has no GPU dependency or batch-evaluation backend.
5. The backtester includes next-bar execution, bid/ask-side handling when quote data exists, spread estimates otherwise, slippage, commission, rollover, risk limits, gap handling, and conservative same-bar stop ordering. Preserve this path as the correctness oracle.
6. The current repository explicitly constrains Python to >=3.11,<3.15 and uses NumPy/Pandas. Keep optional acceleration isolated from core installation.

## Target architecture

### 1. Typed strategy program

Represent a strategy as a bounded program tree, not arbitrary Python and not unrestricted code execution.

Initial node families:
- Inputs: causal market features and constants.
- Numeric operators: add, subtract, multiply, protected divide, absolute value, min/max, rolling mean, rolling standard deviation, lag, and rank/normalization operators with explicit lookback limits.
- Comparisons: greater-than, less-than, cross-above, cross-below.
- Boolean operators: AND, OR, NOT.
- Actions: long entry, short entry, exit, stop distance, target distance, and maximum holding period.

Enforce node type correctness, maximum depth, maximum node count, finite outputs, bounded lookbacks, and deterministic serialization/fingerprints. No future indexing, arbitrary imports, filesystem access, network access, or dynamic code execution.

### 2. Evolution and ecology

Keep population/species/habitat concepts. Add tree-aware operations:
- Subtree mutation and subtree crossover.
- Point/operator mutation and parameter perturbation.
- Structural novelty and behavioral novelty as separate measurements.
- Explicit complexity pressure and duplicate elimination.
- Reproducible seeded runs and checkpointable RNG state.

Do not select organisms on one scalar score alone. Use Pareto selection over net expectancy, drawdown, trade sufficiency, stability across chronological blocks, complexity, and novelty. Novelty helps exploration; it must never qualify a strategy for validation.

### 3. Evaluation backends

Define one evaluator contract with identical inputs and outputs for:
- CPU reference backend.
- Optional vectorized CPU batch backend.
- Optional CUDA backend.

First accelerate feature/program evaluation and signal generation across many organisms. Keep order-dependent fills, stops, position sizing, and accounting in the existing simulator until a separately verified batch simulator can reproduce it. A GPU backend is accepted only after numerical parity tests and a measured speedup on the user's actual GPU. CPU-only operation remains fully supported.

### 4. Fitness and validation boundaries

Separate discovery fitness from evidence required to promote a candidate.

Discovery may reward diversity and promising training behavior. Promotion must use chronological validation, adequate trade counts, cost stress, parameter perturbation, and untouched final testing. Preserve the final-test boundary: test observations must not influence evolution, feature thresholds, habitat fitting, candidate ranking, or repeated parameter selection.

Report raw expectancy and uncertainty separately. A confidence interval crossing zero is uncertainty, not proof of a negative edge; it should affect promotion confidence without silently erasing exploration diversity.

### 5. Research records and reproducibility

Every run records:
- Data file fingerprint, time range, quote side, gap audit, and timeframe.
- Feature/program language version and all settings.
- Random seed and checkpoint state.
- Backend, package versions, hardware, and parity status.
- Train/validation/test boundaries and all candidate rejection reasons.

Persist genomes and lineage, not stale fitness as permanent truth. Historical fitness must be recomputed on the current data split before influencing selection.

## Implementation sequence

1. **Baseline and safety:** capture current test baseline; audit tracked cache/data artifacts; do not delete local data.
2. **Program language:** implement typed AST, validation, serialization, interpreter, and exhaustive unit tests. Keep the old genome path operational during migration.
3. **Genetic operators:** implement bounded subtree mutation/crossover, validity repair, diversity metrics, and deterministic tests.
4. **Reference evaluator:** evaluate ASTs using causal feature arrays and compare generated signals against hand-computed fixtures. Add look-ahead leakage tests.
5. **Ecological integration:** introduce a versioned genome protocol and migrate population evaluation without changing the existing backtest semantics.
6. **Batch acceleration:** profile first; implement vectorized CPU evaluation; then assess CUDA support for Pascal/compute capability 6.1. Do not add a heavy GPU dependency until a benchmark proves it is useful and compatible.
7. **Validation and reports:** widen candidate evaluation using diversity-aware sampling, while preserving hard promotion gates and untouched test data.
8. **End-to-end research:** run small deterministic tests, medium smoke runs, then full EURUSD research with quote-quality caveats clearly reported.

## Acceptance criteria

- Existing tests continue to pass during migration.
- AST programs are deterministic, bounded, serializable, and incapable of accessing future data.
- CPU interpreter and optimized backends produce matching signals within declared numeric tolerances.
- Genetic operators preserve type validity and constraints.
- Backtest outputs remain unchanged for the same signals and settings.
- Final-test data is inaccessible to evolution and selection code by construction.
- GPU is optional and used only when parity and speed benchmarks pass.
- No profitability claim is made without genuinely untouched out-of-sample evidence.

## Explicit non-goals

- No live broker execution or unattended trading.
- No claims that GPU use improves strategy quality.
- No automatic relaxation of statistical promotion gates to manufacture winners.
- No deletion of local datasets, caches, results, or backups as part of repository cleanup.
