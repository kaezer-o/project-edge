# Autonomous training runtime: implementation plan

## Objective

Replace short, manually launched genetic-programming runs with a resumable,
multi-population training process. The training process must be able to run for
hours, preserve its state across interruption, and search for behaviourally
distinct strategies.

This work does not change acceptance thresholds or grant the optimizer access
to validation or final-test outcomes.

## Architecture

1. **Training state and checkpoints**
   - Persist generation, all population genomes, RNG state, run signature and
     generation history.
   - Write checkpoints atomically in the destination directory.
   - Verify a SHA-256 checksum and exact run signature before resume.
   - Include the source-data fingerprint and every search/evaluation setting
     in the run signature. A changed dataset or setting starts a new run.

2. **Island populations**
   - Run multiple populations with separate random streams.
   - Each island evolves independently and maintains its own diversity.
   - Periodically migrate a small number of novel organisms between islands.
   - Keep migration and replacement deterministic under a fixed seed.

3. **Behavioural diversity**
   - The integrated descriptor records long-signal frequency, short-signal
     frequency and executed-trade frequency per bar in each of three training
     eras, plus every executed trade's UTC entry timestamp and direction.
   - Novelty combines normalized signal-rate distance with Jaccard distance
     over exact directional trade-entry events. Identical trade sets have
     zero trade distance; disjoint sets have distance one.
   - Each island maintains a bounded nearest-neighbour novelty archive.
     Migration greedily selects behaviourally distinct migrants rather than
     taking the highest-novelty rows without checking overlap.
   - Descriptor/checkpoint versioning rejects checkpoints created by the
     earlier aggregate-only descriptor.
   - Expression trees are canonicalized after generation, mutation and
     crossover. Safe rewrites include double-negation removal, duplicate
     logical operands, identical min/max operands, commutative logical operand
     ordering and absolute value of raw positive FX price-level features.
   - A semantic fingerprint hashes the exact per-bar directional signal stream
     on training data, allowing syntactically different rules with identical
     signals to be recognized. A separate behaviour fingerprint also includes
     executed entry events and risk parameters, so risk variants are not
     automatically collapsed.
   - Next-generation construction rejects duplicate behavioural fingerprints
     while alternatives are available. Migration skips behaviours already
     present in the receiving island. Retries are bounded.
   - Generation telemetry records unique genomes, unique full behaviours,
     unique signal streams, per-island uniqueness and semantically classified
     strategy-family counts, both globally and separately for each island.
     Multi-label family counts include a hybrid's component families and can
     therefore exceed population size. Separate exclusive class counts assign
     each genome once (pure family, hybrid, other or random fallback) and sum
     exactly to population size. Per-island counts expose local family collapse
     that aggregate counts can hide.
   - For genomes classified as `other`, generation telemetry records
     value-normalized expression-tree fingerprints, long/short AST node counts
     and depths, operator counts, feature sets, rolling lookbacks and
     long/short-only features. Global and per-island summaries retain the most
     common structural archetypes. Constant thresholds are normalized so
     structurally equivalent rules with different threshold values group
     together. These diagnostics are descriptive only and do not affect
     fitness, selection, validation or execution.
   - Structural motif labels separately identify recurring primitives such as
     EMA-pair comparisons, price-versus-EMA filters, RSI threshold crosses,
     ATR-versus-rolling-ATR gates, Donchian breakouts and combined EMA/ATR
     regime filters. Motif counts are reported for the entire population and
     separately for each island, with motif co-occurrence broken down by
     mutually exclusive strategy-family class. The older unclassified-only
     archetype summaries remain available for structural inspection. Motifs
     are descriptive labels, not strategy-family classifications, and do not
     affect evolution or validation.
   - Each island retains a configurable minimum number of top-fitness elites
     from every recognizable family. A capped additional allocation gives one
     extra protected slot to the strongest half of families, ranked by the mean
     of their top three training-only combined fitness/novelty selection scores.
     Hybrid/other programs do not satisfy a named-family quota. Remaining
     population slots use normal novelty/fitness-guided reproduction.

4. **Search grammar**
   - Expand strategy primitives by family: trend, mean reversion, breakout,
     volatility expansion/contraction, session/time filters and long/short
     asymmetry.
   - Keep expressions typed and causal. No future bars or validation-derived
     features may enter training.

5. **Evaluation backends**
   - Keep the current CPU backtester as the correctness reference.
   - Profile feature calculation, signal evaluation and trade simulation
     separately before porting kernels.
   - Add an optional CUDA backend only for operations whose numerical results
     can be shown to match the CPU reference within explicit tolerances.
   - Benchmark the actual GTX 1060 before choosing batch size or device policy.
     GPU discovery alone does not mean the strategy backtester uses the GPU.

6. **Research controls**
   - Train and rank on training partitions only.
   - Use the existing chronological validation gates unchanged.
   - Never use the final holdout during evolution or repeated development.
   - Keep the final holdout disabled until the methodology is frozen and a
     candidate passes the pre-existing validation rules.

## Milestones

- **M1: Runtime foundation.** Atomic checkpoints, integrity checks, strict run
  signatures, deterministic resume tests.
- **M2: Persistent island evolution.** Independent populations, migration,
  recovery after interruption and bounded resource use.
- **M3: Behavioural novelty archive.** Signal/trade similarity metrics and
  diversity-aware selection.
- **M4: Profiling and GPU feasibility.** CPU baseline, CUDA capability check,
  kernel parity tests and GTX 1060 benchmark.
- **M5: Expanded strategy families.** Add and test primitives without
  introducing lookahead or weakening research gates.
- **M6: Integrated autonomous runner.** CLI controls for start, resume,
  checkpoint interval, duration/generation budget and graceful stop.

## Current status

- **M1 complete:** atomic, checksummed checkpoints with strict run signatures.
- **M2 complete at runtime level:** independent island populations, per-island
  random streams, rank-based fitness/novelty selection, bounded novelty
  archives, ring migration and deterministic generation-boundary resume.
- **M2 integration added:** `edge evolve-program-islands` connects typed
  StrategyProgram genomes to the existing CPU backtester. Its evaluator closes
  over training data only; validation is used only after evolution to rank the
  final shortlist. The final holdout remains disabled unless explicitly
  requested with `--final-test`.
- **M3 semantic diversity implementation:** expression canonicalization,
  exact per-bar training-signal fingerprints, behavioural duplicate rejection
  during reproduction, duplicate-aware migration, trade-overlap novelty and
  per-generation diversity/family telemetry are implemented. Validation and
  final-test data do not enter semantic fingerprints or evolutionary selection.
- **M6 partial:** the new command supports generation budgets, checkpoints and
  resume. Duration limits and graceful signal handling are not implemented.

### Running the island trainer

Start a new run:

```powershell
edge evolve-program-islands --input data\EURUSD_M1.csv --symbol EURUSD --source-timeframe M1 --timeframe H4 --islands 4 --population-per-island 32 --generations 100 --checkpoint results\island-training\checkpoint.json --output results\island-training
```

Resume the same run and extend its total target to 200 generations:

```powershell
edge evolve-program-islands --input data\EURUSD_M1.csv --symbol EURUSD --source-timeframe M1 --timeframe H4 --islands 4 --population-per-island 32 --generations 200 --checkpoint results\island-training\checkpoint.json --output results\island-training --resume
```

The generation argument is the total target, not an additional-generation count.
Do not add `--final-test` during development.
- No GPU speedup is claimed. Profiling and CPU/GPU parity work are still
  required before a CUDA evaluator can be considered.

## Cross-seed finalist reproducibility audit

The offline audit compares finalist structures and validation diagnostics from
independent evolutionary runs. It is descriptive only: it does not change
fitness, selection, migration, validation gates or final-test access.

Run the script directly from the repository root:

~~~powershell
.\.venv\Scripts\python.exe scripts\audit_candidate_reproducibility.py --report 2033=results\motif-repro-h4-seed-2033\island_program_evolution_report.json --report 2034=results\motif-repro-h4-seed-2034\island_program_evolution_report.json --report 2035=results\motif-repro-h4-seed-2035\island_program_evolution_report.json
~~~

It writes JSON and Markdown summaries under `results`. Candidate rule
structures are compared after normalizing numeric constants and ordering
commutative operands; risk parameters are retained separately. Motif labels
are coarse structural descriptors, not proof of equivalent behaviour.


### Motif taxonomy v2: parameter-aware finalist comparison

The offline finalist audit also preserves the detailed entry-rule structure:
operator direction, exact numeric thresholds, feature identities, rolling/lag
lookbacks, and the separate long/short expression trees. Commutative operands
are canonicalized; directional comparisons and ordered operands are preserved.
It computes weighted Jaccard similarity over path-aware structural tokens,
reports long-side and short-side similarity separately, and compares risk
parameters separately from entry logic. The score is a structural heuristic,
not a measure of equivalent signals or profitability.

Re-run the same command shown above to produce the expanded report. The JSON
includes the top cross-seed detailed structural similarities; the Markdown
report includes a ranked table. This analysis is observational only and does
not feed back into evolution, candidate selection, validation gates or
final-test access.

The reports store validation entry-event SHA-256 fingerprints rather than
event sets. Cross-seed exact equality can therefore be detected, but
approximate Jaccard overlap cannot be reconstructed. The audit rejects
reports with mismatched symbol, timeframe, partition metadata, or any
indication that the final test was evaluated. Validation results remain
development diagnostics because this partition has already been examined.


## Cross-seed behavioural similarity audit

Replay finalists from independent runs on the exact same chronological training
and validation partitions. The audit compares per-bar directional signals
(long, short and active-signal overlap) and executed trade-entry events. It
reports training and validation similarities separately, includes the
execution/risk assumptions in the output, and does not evaluate the final
holdout. The validation partition has already been examined during development;
its similarity results are descriptive only.

Example for the three 30-generation H4 runs:

~~~powershell
.\.venv\Scripts\python.exe scripts\audit_candidate_behaviour.py `
  --input data\EURUSD_M1.csv `
  --report 2033=results\motif-repro-h4-seed-2033\island_program_evolution_report.json `
  --report 2034=results\motif-repro-h4-seed-2034\island_program_evolution_report.json `
  --report 2035=results\motif-repro-h4-seed-2035\island_program_evolution_report.json `
  --price-basis bid --spread-pips 0.8 --slippage-pips 0.1
~~~

The script defaults to the execution and risk settings used in these runs.
Override them when auditing reports produced with different settings. It
rejects final-test-evaluated reports and mismatched partition metadata. Signal
similarity is computed on aligned bars; exact agreement is reported alongside
active-signal Jaccard because long stretches of no-signal bars can inflate
exact agreement. Trade-entry overlap is execution-dependent. All outputs are
observational and do not feed back into evolution or candidate selection.


## Intra-run finalist behavioural redundancy audit

Use this audit after the cross-seed audit to measure redundancy among the
finalists retained by each individual evolutionary run. It replays each
finalist on the same training and validation partitions and reports exact
signal/trade duplicates, near-duplicate pair counts, validation-qualified
pair overlap, and redundancy grouped by exclusive strategy-family class.

Example:

~~~powershell
.\\.venv\\Scripts\\python.exe scripts\\audit_candidate_redundancy.py `
  --input data\\EURUSD_M1.csv `
  --report 2033=results\\motif-repro-h4-seed-2033\\island_program_evolution_report.json `
  --report 2034=results\\motif-repro-h4-seed-2034\\island_program_evolution_report.json `
  --report 2035=results\\motif-repro-h4-seed-2035\\island_program_evolution_report.json `
  --price-basis bid --spread-pips 0.8 --slippage-pips 0.1
~~~

The report is finalist-only and cannot establish redundancy in the full
evolving population. It is observational: it does not alter fitness,
selection, migration, validation gates or final-test access. Validation has
been repeatedly examined during development and is not independent evidence.
The final chronological holdout is never evaluated.


## Training-only diversity selection simulation

Use `scripts/audit_training_diversity_selection.py` to compare a fitness-only
shortlist with a greedy shortlist that rejects candidates whose active
directional training signals overlap an already selected candidate at or above
a configurable Jaccard threshold. Ranking uses each report's
`training_fitness`. The selection routine reads only training signals and
training fitness; validation and final-test data do not influence it. This is
an offline finalist-pool diagnostic and does not change production selection,
fitness, validation gates, or checkpoint compatibility.

Example:

~~~powershell
.\.venv\Scripts\python.exe scripts\audit_training_diversity_selection.py `
  --input data\EURUSD_M1.csv `
  --report 2033=results\motif-repro-h4-seed-2033\island_program_evolution_report.json `
  --report 2034=results\motif-repro-h4-seed-2034\island_program_evolution_report.json `
  --report 2035=results\motif-repro-h4-seed-2035\island_program_evolution_report.json `
  --similarity-threshold 0.90 `
  --limit 20 `
  --output results\training-diversity-selection-audit.json
~~~

The default threshold is 0.90. This is a diagnostic choice, not a tuned
production parameter. The output records selected and rejected ranks, unique
training signal counts, and a fitness-only baseline. It never reads validation
results for selection and never evaluates the final chronological holdout.


### Diversity-selection threshold sensitivity benchmark

Use the same finalist reports to compare several active-signal Jaccard
thresholds in one pass. The default thresholds are 0.80, 0.85, 0.90 and 0.95.
At each threshold, the fitness-only baseline is truncated to exactly the
number of candidates retained by diversity selection. This controls for
shortlist size when comparing the number of unique training signal streams.
The report also records whether the highest-training-fitness finalist remains
selected and the lowest selected training fitness.

Example:

~~~powershell
.\.venv\Scripts\python.exe scripts\audit_training_diversity_selection.py `
  --input data\EURUSD_M1.csv `
  --report 2033=results\motif-repro-h4-seed-2033\island_program_evolution_report.json `
  --report 2034=results\motif-repro-h4-seed-2034\island_program_evolution_report.json `
  --report 2035=results\motif-repro-h4-seed-2035\island_program_evolution_report.json `
  --thresholds 0.80 0.85 0.90 0.95 `
  --limit 20 `
  --output results\training-diversity-threshold-sensitivity.json
~~~

This is still an offline finalist-pool simulation. It does not change
production selection, use validation to choose candidates, or evaluate the
final chronological holdout. A threshold that increases unique signals is not
automatically superior; fitness retention and the number of candidates kept
must be considered together.



## Optional training-signal finalist filtering

The island trainer can optionally reduce redundancy in its final validation
shortlist after evolution has finished. The existing fitness/structure-ranked
shortlist is formed first; the optional filter then greedily rejects a lower
ranked finalist only when both its directional training-signal stream and its
executed training trade paths have Jaccard similarity greater than or equal to
the configured threshold with a finalist already retained. Trade-path events
include entry time, direction, exit time and exit reason. Risk-profile similarity
also compares stop distance, reward/risk and maximum holding bars. This
distinguishes programs that share entry signals but use materially different
stop, target or holding-time behaviour.

The filter is opt-in and does not alter evolutionary fitness, population
selection, migration, checkpoint state, validation gates or final-test access.
It evaluates each unique program fingerprint once, stores trade events as
compact UTC nanosecond tuples, and avoids evaluating the remaining similarity
components when they cannot improve the current best pairwise score. The
selection order, minimum-of-three similarity score, threshold comparison and
reported winning comparison are preserved. Performance counters record unique
program evaluations, training backtests and short-circuited pairwise comparisons.

It evaluates directional signals and executed trade paths on the training
partition only. The report retains the original fitness-ranked shortlist,
selected original fitness ranks, and separate signal, trade-path and risk-profile
similarity diagnostics so the effect is auditable. The filter may return fewer
than 20 finalists when the pool is highly redundant.

Example using the threshold sensitivity experiment's provisional 0.95 value:

~~~powershell
edge evolve-program-islands --symbol EURUSD --source-timeframe M1 --timeframe H4 --pip-size 0.0001 --price-basis bid --spread-pips 0.8 --slippage-pips 0.1 --population-per-island 32 --islands 4 --minimum-family-elites 2 --generations 30 --migration-interval 2 --migrants-per-island 2 --novelty-weight 0.25 --seed 2041 --checkpoint results\training-signal-shortlist-h4\checkpoint.json --output results\training-signal-shortlist-h4 --finalist-similarity-threshold 0.95
~~~

Omit `--finalist-similarity-threshold` to preserve the previous shortlist
behaviour. The 0.95 value is an experimental starting point based on a small
offline finalist-pool benchmark, not a validated production optimum. Validation
remains post-selection only and the final chronological holdout must remain
untouched.


### Configurable island-program finalist limit

`evolve-program-islands` accepts `--finalist-limit` (default `50`) to control the maximum training-ranked shortlist size. The default of 50 is provisional, based on the seed-2044 finalist-limit benchmark; use an explicit value when reproducing older experiments. The same limit is applied before optional training-only diversity filtering. Structural diversity can make the actual pool smaller than the requested limit when fewer unique rule structures are available.

Before ranking finalists, the trainer now applies the existing three-era training eligibility gate. A candidate must have eligible trade counts in all three chronological training windows and a finite training fitness above the ineligible sentinel. This removes candidates that already failed the training robustness requirements before they consume finalist slots or trigger validation backtests. The report records evaluated, eligible and rejected counts with reasons. This gate uses training data only; validation and final-test partitions remain excluded from finalist selection.
