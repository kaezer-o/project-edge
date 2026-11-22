# CPU pipeline profiling

Run from the repository root after installing the project in its virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\profile_pipeline.py `
  --input data\EURUSD_M1.csv `
  --symbol EURUSD `
  --source-timeframe M1 `
  --timeframe H4 `
  --price-basis bid `
  --spread-pips 0.8 `
  --slippage-pips 0.1 `
  --samples 3 `
  --output results\pipeline-profile-h4.json
```

The harness measures CSV loading, gap annotation/resampling/feature preparation,
signal evaluation, one OHLC backtest, three-era training fitness, and behavioural
descriptor construction. It warms each measured path before collecting timings
and reports median/minimum/maximum durations.

It also estimates the time for a 128-genome population from the measured
per-genome training-fitness and descriptor costs. That estimate is linear and
excludes reproduction, migration, checkpoint I/O, and validation. It is not a
measured full generation.

The script does not run evolution, does not evaluate validation or final-test
performance, and does not invoke a GPU. Run it on the target machine to obtain
the actual CPU baseline before choosing kernels for acceleration. Preserve the
JSON report with machine and software details when comparing later runs.

## CUDA capability probe

Run:

```powershell
.\.venv\Scripts\python.exe scripts\probe_acceleration.py
```

This records NVIDIA driver/GPU details, installed optional packages, PyTorch CUDA
availability and compiled GPU architectures, and Numba CUDA availability.
It does not install dependencies, benchmark a kernel, or establish numerical
parity. For a GTX 1060, verify that the installed framework build supports
compute capability 6.1 (sm_61) before attempting a CUDA implementation.
