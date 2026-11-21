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
   - The first integrated descriptor records long-signal frequency, short-signal
     frequency and executed-trade frequency in each of three training eras.
   - Each island maintains a bounded nearest-neighbour novelty archive.
   - This is a coarse behavioural descriptor, not yet a full signal/position
     fingerprint or pairwise trade-overlap model.

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
- **M1 complete:** atomic, checksummed checkpoints with strict run signatures.
- **M2 complete at runtime level:** independent island populations, per-island
  random streams, rank-based fitness/novelty selection, bounded novelty
  archives, ring migration and deterministic generation-boundary resume.
- **M2 integration added:** `edge evolve-program-islands` connects typed
  StrategyProgram genomes to the existing CPU backtester. Its evaluator closes
  over training data only; validation is used only after evolution to rank the
  final shortlist. The final holdout remains disabled unless explicitly
  requested with `--final-test`.
- **M3 initial implementation:** novelty descriptors use long/short signal
  rates and executed-trade rates across three training eras. Full signal
  fingerprints and pairwise trade-overlap similarity remain future work.
- **M6 partial:** the new command supports generation budgets, checkpoints and
  resume. Duration limits and graceful signal handling are not implemented.
- No GPU speedup is claimed. Profiling and CPU/GPU parity work are still
  required before a CUDA evaluator can be considered.
