from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from edge.backtest import BacktestSettings, run_backtest
from edge.config import ExecutionConfig, RiskConfig
from edge.data import validate_source_interval
from edge.features import add_features
from edge.island_program_research import (
    _behaviour_descriptor,
    _prepare_frames,
)
from edge.program import evaluate_program
from edge.program_evolution import random_expression
from edge.program import StrategyProgram
from edge.program_research import _evaluate, _training_fitness
from edge.providers import CSVOHLCProvider


def timed(fn, repeats: int) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return {
        "runs": len(values),
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile Project EDGE CPU research stages; does not run evolution or alter results."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--source-timeframe", default="M1")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--pip-size", type=float, default=0.0001)
    parser.add_argument("--price-basis", choices=("bid", "mid"), default="bid")
    parser.add_argument("--spread-pips", type=float, default=0.8)
    parser.add_argument("--slippage-pips", type=float, default=0.1)
    parser.add_argument("--commission-pips", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", default="results/pipeline-profile.json")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    raw = CSVOHLCProvider(args.input).load(args.symbol, args.source_timeframe)
    validate_source_interval(raw, args.source_timeframe)
    load_seconds = time.perf_counter() - load_start

    execution = ExecutionConfig(
        spread_pips=args.spread_pips,
        slippage_pips_per_side=args.slippage_pips,
        commission_round_trip_pips=args.commission_pips,
    )
    risk = RiskConfig()
    prep_start = time.perf_counter()
    train, validation, test, features = _prepare_frames(
        raw, timeframe=args.timeframe, source_timeframe=args.source_timeframe
    )
    preparation_seconds = time.perf_counter() - prep_start

    rng = np.random.default_rng(20261122)
    programs = []
    for _ in range(args.samples):
        long_rule = random_expression("bool", features, rng)
        short_rule = random_expression("bool", features, rng)
        programs.append(StrategyProgram(long_rule, short_rule))

    program = programs[0]
    signal = evaluate_program(program, train)
    backtest_settings = BacktestSettings(
        pip_size=args.pip_size,
        max_holding_bars=program.max_holding_bars,
        price_basis=args.price_basis,
    )

    def signal_stage():
        evaluate_program(program, train)

    def backtest_stage():
        run_backtest(
            train, signal, stop_atr=program.stop_atr,
            reward_risk=program.reward_risk, execution=execution,
            risk=risk, settings=backtest_settings,
        )

    def fitness_stage():
        _training_fitness(
            train, program, execution=execution, risk=risk,
            pip_size=args.pip_size, price_basis=args.price_basis,
            min_train_trades=30,
        )

    def descriptor_stage():
        _behaviour_descriptor(
            train, program, execution=execution, risk=risk,
            pip_size=args.pip_size, price_basis=args.price_basis,
        )

    def shared_genome_stage():
        shared_signal = evaluate_program(program, train)
        aggregate_result = _evaluate(
            train, program, execution=execution, risk=risk,
            pip_size=args.pip_size, price_basis=args.price_basis,
            signal=shared_signal,
        )
        _training_fitness(
            train, program, execution=execution, risk=risk,
            pip_size=args.pip_size, price_basis=args.price_basis,
            min_train_trades=30, aggregate_result=aggregate_result,
        )
        _behaviour_descriptor(
            train, program, execution=execution, risk=risk,
            pip_size=args.pip_size, price_basis=args.price_basis,
            signal=shared_signal, trades=aggregate_result[1],
        )

    # Warm each measured path once to reduce one-time interpreter/library effects.
    signal_stage()
    backtest_stage()
    fitness_stage()
    descriptor_stage()
    shared_genome_stage()

    timings = {
        "signal_evaluation": timed(signal_stage, args.samples),
        "single_backtest": timed(backtest_stage, args.samples),
        "training_fitness_three_eras": timed(fitness_stage, args.samples),
        "behaviour_descriptor": timed(descriptor_stage, args.samples),
        "shared_genome_evaluation": timed(shared_genome_stage, args.samples),
    }

    per_genome_seconds = timings["shared_genome_evaluation"]["median_seconds"]
    population = 128
    report = {
        "project": "Project EDGE",
        "profile_version": 1,
        "hardware_note": "CPU timings only. No GPU kernel is invoked.",
        "dataset": {
            "input": str(Path(args.input)),
            "symbol": args.symbol,
            "source_timeframe": args.source_timeframe.upper(),
            "timeframe": args.timeframe.upper(),
            "source_rows": int(len(raw)),
            "train_bars": int(len(train)),
            "validation_bars": int(len(validation)),
            "final_test_bars": int(len(test)),
            "feature_count": len(features),
        },
        "configuration": {
            "samples_per_stage": args.samples,
            "seed": 20261122,
            "spread_pips": args.spread_pips,
            "slippage_pips_per_side": args.slippage_pips,
            "commission_round_trip_pips": args.commission_pips,
            "final_test_evaluated": False,
        },
        "timings": {
            "csv_load_and_interval_validation": {
                "runs": 1, "median_seconds": load_seconds,
                "min_seconds": load_seconds, "max_seconds": load_seconds,
            },
            "gap_annotation_resampling_feature_preparation": {
                "runs": 1, "median_seconds": preparation_seconds,
                "min_seconds": preparation_seconds, "max_seconds": preparation_seconds,
            },
            **timings,
        },
        "estimated_evolution_cost": {
            "measured_genome_evaluation_seconds": per_genome_seconds,
            "population_128_seconds_estimate": per_genome_seconds * population,
            "population_128_minutes_estimate": per_genome_seconds * population / 60.0,
            "basis": "median training fitness plus behaviour descriptor for one genome; estimate excludes reproduction, migration, checkpoint I/O, and final validation",
        },
        "elapsed_seconds": time.perf_counter() - total_start,
        "interpretation": [
            "Training fitness evaluates the aggregate training frame and three chronological training eras.",
            "Shared genome evaluation reuses aggregate signals and trades between fitness and behaviour descriptors.",
            "The population estimate uses the shared evaluation timing and remains a linear estimate, not a measured full generation.",
            "Validation and final holdout are not used in the measured evolution workload.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(output.resolve()),
        "train_bars": len(train),
        "feature_count": len(features),
        "timings": report["timings"],
        "population_128_minutes_estimate": report["estimated_evolution_cost"]["population_128_minutes_estimate"],
        "final_test_evaluated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
