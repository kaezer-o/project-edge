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
