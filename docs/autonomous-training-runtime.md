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
   - Record signal/position fingerprints and trade-event summaries.
   - Measure overlap or similarity between candidates.
   - Penalize duplicate behaviour during parent selection and archive admission.
   - Retain structurally different and behaviourally different candidates.

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
- **M2 runtime foundation added:** independent island populations, per-island
  random streams, rank-based fitness/novelty selection, bounded novelty
  archives, ring migration and generation-boundary resume.
- The island runtime is callback-driven and is not yet wired into
  edge evolve-program; the existing strategy evaluator remains CPU-only.
- The runtime's evaluator interface receives genomes only. The caller must
  construct it around training data only; validation and final-test partitions
  must remain outside the callback.
- No GPU speedup is claimed. Profiling and CPU/GPU parity work are still
  required before a CUDA evaluator can be considered.
