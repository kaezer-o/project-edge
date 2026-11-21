from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestSettings, run_backtest
from .config import ExecutionConfig, RiskConfig, TIMEFRAME_SECONDS
from .data import annotate_gaps, resample_ohlc
from .features import FEATURE_COLUMNS, add_features
from .metrics import calculate_metrics
from .program import Expr, StrategyProgram, evaluate_program
from .program_evolution import (
    crossover_programs, mutate_program, random_expression,
)


def _random_program(features: tuple[str, ...], rng: np.random.Generator) -> StrategyProgram:
    return StrategyProgram(
        long_entry=random_expression("bool", features, rng, max_depth=4),
        short_entry=random_expression("bool", features, rng, max_depth=4),
        stop_atr=float(rng.choice((0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0))),
        reward_risk=float(rng.choice((0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0))),
        max_holding_bars=int(rng.choice((5, 10, 20, 40, 80, 160))),
    )


def _program_complexity(program: StrategyProgram) -> int:
    def count(node: Expr) -> int:
        return 1 + sum(count(child) for child in node.args)
    return count(program.long_entry) + count(program.short_entry)


def _evaluate(
    frame: pd.DataFrame,
    program: StrategyProgram,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    signal = evaluate_program(program, frame)
    trades, equity = run_backtest(
        frame, signal,
        stop_atr=program.stop_atr,
        reward_risk=program.reward_risk,
        execution=execution,
        risk=risk,
        settings=BacktestSettings(
            pip_size=pip_size,
            max_holding_bars=program.max_holding_bars,
            price_basis=price_basis,
        ),
    )
    return calculate_metrics(trades, equity), trades


def _fitness(metrics: dict[str, object], program: StrategyProgram, min_trades: int) -> float:
    count = int(metrics.get("trades") or 0)
    expectancy = metrics.get("expectancy_r")
    drawdown = metrics.get("max_realized_drawdown_pct")
    profit_factor = metrics.get("profit_factor")
    if count < min_trades or expectancy is None or drawdown is None:
        return -1e9
    expectancy = float(expectancy)
    drawdown = float(drawdown)
    if not np.isfinite(expectancy) or not np.isfinite(drawdown):
        return -1e9
    # Training fitness is a ranking signal only; it is not evidence of robustness.
    pf_bonus = min(float(profit_factor or 0.0), 3.0) * 0.02
    return (
        expectancy * np.sqrt(count) / (1.0 + 4.0 * drawdown)
        + pf_bonus
        - 0.0015 * _program_complexity(program)
    )


def _validation_check(
    frame: pd.DataFrame,
    program: StrategyProgram,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    min_trades: int,
) -> dict[str, object]:
    chunks = np.array_split(np.arange(len(frame)), 3)
    windows = []
    for i, positions in enumerate(chunks, start=1):
        window = frame.iloc[positions]
        metrics, _ = _evaluate(
            window, program, execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
        trades = int(metrics.get("trades") or 0)
        exp = metrics.get("expectancy_r")
        pf = metrics.get("profit_factor")
        dd = metrics.get("max_realized_drawdown_pct")
        passed = (
            trades >= min_trades
            and exp is not None and float(exp) > 0
            and pf is not None and float(pf) > 1.0
            and dd is not None and float(dd) <= 0.20
        )
        windows.append({
            "window": i,
            "start_utc": window.index.min().isoformat() if len(window) else None,
            "end_utc": window.index.max().isoformat() if len(window) else None,
            "metrics": metrics,
            "passed": bool(passed),
        })
    full_metrics, _ = _evaluate(
        frame, program, execution=execution, risk=risk,
        pip_size=pip_size, price_basis=price_basis,
    )
    cost_stress = []
    for multiplier in (1.5, 2.0):
        stressed_execution = replace(
            execution,
            spread_pips=execution.spread_pips * multiplier,
            slippage_pips_per_side=execution.slippage_pips_per_side * multiplier,
            commission_round_trip_pips=execution.commission_round_trip_pips * multiplier,
            swap_long_pips_per_rollover=(
                execution.swap_long_pips_per_rollover * multiplier
                if execution.swap_long_pips_per_rollover < 0
                else execution.swap_long_pips_per_rollover
            ),
            swap_short_pips_per_rollover=(
                execution.swap_short_pips_per_rollover * multiplier
                if execution.swap_short_pips_per_rollover < 0
                else execution.swap_short_pips_per_rollover
            ),
        )
        stressed_frame = frame.copy()
        if "spread_pips" in stressed_frame.columns:
            stressed_frame["spread_pips"] = stressed_frame["spread_pips"].astype(float) * multiplier
        stressed_metrics, _ = _evaluate(
            stressed_frame, program, execution=stressed_execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
        expectancy = stressed_metrics.get("expectancy_r")
        profit_factor = stressed_metrics.get("profit_factor")
        cost_stress.append({
            "cost_multiplier": multiplier,
            "metrics": stressed_metrics,
            "pass": bool(
                expectancy is not None and float(expectancy) > 0
                and profit_factor is not None and float(profit_factor) > 1.0
            ),
        })
    passed_windows = sum(bool(row["passed"]) for row in windows)
    stress_pass = bool(cost_stress and cost_stress[0]["pass"])
    return {
        "metrics": full_metrics,
        "windows": windows,
        "passed_windows": passed_windows,
        "cost_stress": cost_stress,
        "one_and_half_x_cost_pass": stress_pass,
        "pass": bool(passed_windows >= 2 and len(windows) == 3 and stress_pass),
    }


def run_program_research(
    raw_frame: pd.DataFrame,
    *,
    timeframe: str,
    source_timeframe: str,
    symbol: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    population_size: int = 32,
    generations: int = 12,
    seed: int = 2026,
    min_train_trades: int = 30,
    min_validation_trades: int = 30,
    output_dir: str | Path = "results/program-evolution",
    run_final_test: bool = False,
) -> dict[str, object]:
    """Evolve expression-tree strategies on training data, then freeze for validation."""
    timeframe = timeframe.upper()
    source_timeframe = source_timeframe.upper()
    if timeframe not in TIMEFRAME_SECONDS or source_timeframe not in TIMEFRAME_SECONDS:
        raise ValueError("Unsupported timeframe")
    if TIMEFRAME_SECONDS[timeframe] < TIMEFRAME_SECONDS[source_timeframe]:
        raise ValueError("Target timeframe cannot be lower than source timeframe")
    if population_size < 8 or generations < 2:
        raise ValueError("population_size must be >= 8 and generations >= 2")

    annotated = annotate_gaps(raw_frame, TIMEFRAME_SECONDS[source_timeframe])
    market = annotated if timeframe == source_timeframe else resample_ohlc(annotated, timeframe)
    featured = add_features(market)
    n = len(featured)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    if train_end < 300 or validation_end - train_end < 100 or n - validation_end < 100:
        raise ValueError("Insufficient bars for chronological train/validation/test partitions")

    train = featured.iloc[:train_end].copy()
    validation = featured.iloc[train_end:validation_end].copy()
    test = featured.iloc[validation_end:].copy()
    # Only market indicators belong in the strategy search space.
    # Gap flags, segment IDs, timestamps, and execution metadata describe
    # the dataset; they are not tradable market signals.
    features = tuple(
        column for column in FEATURE_COLUMNS
        if column in train.columns
        and pd.api.types.is_numeric_dtype(train[column])
        and train[column].notna().any()
    )
    if len(features) < 4:
        raise ValueError("Fewer than four usable numeric features in training data")

    rng = np.random.default_rng(seed)
    population: list[StrategyProgram] = []
    seen: set[str] = set()
    while len(population) < population_size:
        candidate = _random_program(features, rng)
        if candidate.fingerprint() not in seen:
            seen.add(candidate.fingerprint())
            population.append(candidate)

    generation_rows: list[dict[str, object]] = []
    evaluated: list[tuple[float, StrategyProgram, dict[str, object]]] = []
    elite_count = max(2, min(4, population_size // 4))
    for generation in range(generations):
        evaluated = []
        for program in population:
            metrics, trades = _evaluate(
                train, program, execution=execution, risk=risk,
                pip_size=pip_size, price_basis=price_basis,
            )
            score = _fitness(metrics, program, min_train_trades)
            evaluated.append((score, program, metrics))
        evaluated.sort(key=lambda item: item[0], reverse=True)
        generation_rows.append({
            "generation": generation + 1,
            "best_training_fitness": float(evaluated[0][0]),
            "median_training_fitness": float(np.median([row[0] for row in evaluated])),
            "unique_programs": len({row[1].fingerprint() for row in evaluated}),
            "best_training_metrics": evaluated[0][2],
        })
        if generation == generations - 1:
            break

        next_population = [row[1] for row in evaluated[:elite_count]]
        next_seen = {program.fingerprint() for program in next_population}
        while len(next_population) < population_size:
            if rng.random() < 0.12:
                child = _random_program(features, rng)
            else:
                parent_a = evaluated[int(rng.integers(0, max(elite_count, population_size // 2)))][1]
                if rng.random() < 0.55:
                    parent_b = evaluated[int(rng.integers(0, max(elite_count, population_size // 2)))][1]
                    child = crossover_programs(parent_a, parent_b, rng)
                else:
                    child = mutate_program(parent_a, features, rng)
            fingerprint = child.fingerprint()
            if fingerprint in next_seen:
                child = _random_program(features, rng)
                fingerprint = child.fingerprint()
            if fingerprint not in next_seen:
                next_seen.add(fingerprint)
                next_population.append(child)
        population = next_population

    # Only the training-ranked shortlist is exposed to validation.
    shortlist = evaluated[:min(10, len(evaluated))]
    candidate_reports = []
    accepted: list[tuple[float, StrategyProgram, dict[str, object]]] = []
    for score, program, train_metrics in shortlist:
        validation_result = _validation_check(
            validation, program, execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
            min_trades=max(5, min_validation_trades // 3),
        )
        row = {
            "program": program.to_dict(),
            "fingerprint": program.fingerprint(),
            "training_fitness": float(score),
            "training_metrics": train_metrics,
            "validation": validation_result,
        }
        candidate_reports.append(row)
        if validation_result["pass"]:
            accepted.append((score, program, row))

    winner = max(accepted, key=lambda item: item[0]) if accepted else None
    final_test = None
    if winner is not None and run_final_test:
        test_metrics, _ = _evaluate(
            test, winner[1], execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
        final_test = {
            "metrics": test_metrics,
            "warning": "Single historical holdout; not prospective or live-trading evidence.",
        }

    report: dict[str, object] = {
        "project": "Project EDGE",
        "engine": "open_ended_expression_program_evolution",
        "version": 1,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_timeframe": source_timeframe,
        "data": {
            "bars": n,
            "start_utc": featured.index.min().isoformat(),
            "end_utc": featured.index.max().isoformat(),
            "train_bars": len(train),
            "validation_bars": len(validation),
            "final_test_bars": len(test),
            "features": list(features),
        },
        "evolution": {
            "population_size": population_size,
            "generations": generations,
            "seed": seed,
            "generation_history": generation_rows,
            "selection": "training fitness only; validation never feeds back into evolution",
        },
        "execution_assumptions": {
            "price_basis": price_basis,
            "spread_pips": execution.spread_pips,
            "slippage_pips_per_side": execution.slippage_pips_per_side,
            "commission_round_trip_pips": execution.commission_round_trip_pips,
            "signal_timing": "signal at candle close; backtester enters no earlier than next candle open",
        },
        "candidate_count": len(candidate_reports),
        "validated_candidate_count": len(accepted),
        "winner": winner[2] if winner else None,
        "final_test": final_test,
        "final_test_skipped": bool(winner is None or not run_final_test),
        "status": "validated_candidate" if winner else "no_robust_candidate",
        "limitations": [
            "Expression evolution currently uses CPU evaluation.",
            "OHLC execution assumptions remain approximate where quote-side data is unavailable.",
            "Historical validation is not evidence of future profitability.",
        ],
        "candidates": candidate_reports,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "program_evolution_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report
