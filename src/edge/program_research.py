from __future__ import annotations

import hashlib
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
    signal: pd.Series | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    if signal is None:
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


def _fitness(
    metrics: dict[str, object],
    program: StrategyProgram,
    min_trades: int,
    *,
    sample_multiplier: int = 1,
) -> float:
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
        expectancy * np.sqrt(count * sample_multiplier) / (1.0 + 4.0 * drawdown)
        + pf_bonus
        - 0.0015 * _program_complexity(program)
    )


def _minimum_window_trades(total_minimum: int, window_count: int) -> int:
    """Distribute the total training trade minimum across chronological windows."""
    if total_minimum < 1:
        raise ValueError("total_minimum must be at least one")
    if window_count < 1:
        raise ValueError("window_count must be at least one")
    return max(5, int(np.ceil(total_minimum / window_count)))


def _combine_training_fitness(
    aggregate_score: float,
    window_scores: list[float],
) -> float:
    """Reward training performance that persists across chronological regimes."""
    scores = np.asarray(window_scores, dtype=float)
    if (
        len(scores) != 3
        or not np.isfinite(scores).all()
        or (scores <= -1e8).any()
        or not np.isfinite(aggregate_score)
        or aggregate_score <= -1e8
    ):
        return -1e9
    return float(
        0.35 * aggregate_score
        + 0.45 * np.median(scores)
        + 0.20 * np.min(scores)
        - 0.10 * np.std(scores)
    )


def _expression_structure(expr: Expr) -> tuple[object, ...]:
    """Return a threshold-independent expression structure for diversity control."""
    if expr.op == "const":
        value: object = "scalar"
    elif expr.op == "feature":
        value = expr.value
    elif expr.op in {"lag", "rolling_mean", "rolling_std"}:
        value = expr.value
    else:
        value = None
    return (
        expr.op,
        value,
        tuple(_expression_structure(child) for child in expr.args),
    )


def _program_structure(program: StrategyProgram) -> tuple[object, ...]:
    """Identify a rule family independently of numeric thresholds and risk genes."""
    return (
        _expression_structure(program.long_entry),
        _expression_structure(program.short_entry),
    )


def _diverse_ranked(
    evaluated: list[tuple[float, StrategyProgram, dict[str, object], dict[str, object]]],
    limit: int,
    *,
    max_per_structure: int = 1,
) -> list[tuple[float, StrategyProgram, dict[str, object], dict[str, object]]]:
    """Keep high-scoring programs while limiting near-identical rule structures."""
    selected = []
    counts: dict[tuple[object, ...], int] = {}
    for row in sorted(evaluated, key=lambda item: item[0], reverse=True):
        structure = _program_structure(row[1])
        if counts.get(structure, 0) >= max_per_structure:
            continue
        selected.append(row)
        counts[structure] = counts.get(structure, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _training_fitness(
    frame: pd.DataFrame,
    program: StrategyProgram,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    min_train_trades: int,
    aggregate_result: tuple[dict[str, object], pd.DataFrame] | None = None,
) -> tuple[float, dict[str, object], dict[str, object]]:
    """Score aggregate training results and three unseen-within-training eras."""
    if aggregate_result is None:
        aggregate_metrics, _ = _evaluate(
            frame, program, execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
    else:
        aggregate_metrics, _ = aggregate_result
    aggregate_score = _fitness(aggregate_metrics, program, min_train_trades)
    positions = np.array_split(np.arange(len(frame)), 3)
    # Require the total minimum trade budget to be distributed across all
    # three chronological training eras, with a floor of five per era.
    minimum_window_trades = _minimum_window_trades(
        min_train_trades, len(positions)
    )
    window_rows = []
    window_scores = []
    for index, window_positions in enumerate(positions, start=1):
        window = frame.iloc[window_positions]
        metrics, _ = _evaluate(
            window, program, execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
        score = _fitness(
            metrics,
            program,
            minimum_window_trades,
            sample_multiplier=len(positions),
        )
        window_scores.append(score)
        window_rows.append({
            "window": index,
            "start_utc": window.index.min().isoformat() if len(window) else None,
            "end_utc": window.index.max().isoformat() if len(window) else None,
            "metrics": metrics,
            "fitness": float(score),
            "eligible": bool(score > -1e8),
        })
    robust_score = _combine_training_fitness(aggregate_score, window_scores)
    report = {
        "aggregate_fitness": float(aggregate_score),
        "window_fitness": [float(value) for value in window_scores],
        "median_window_fitness": (
            float(np.median(window_scores))
            if all(value > -1e8 for value in window_scores) else None
        ),
        "worst_window_fitness": (
            float(np.min(window_scores))
            if all(value > -1e8 for value in window_scores) else None
        ),
        "window_fitness_std": (
            float(np.std(window_scores))
            if all(value > -1e8 for value in window_scores) else None
        ),
        "minimum_window_trades": minimum_window_trades,
        "all_windows_eligible": bool(all(value > -1e8 for value in window_scores)),
        "windows": window_rows,
    }
    return robust_score, aggregate_metrics, report


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
    def entry_events(trades: pd.DataFrame) -> set[str]:
        if trades.empty or not {"entry_time", "direction"}.issubset(trades.columns):
            return set()
        events: set[str] = set()
        for trade in trades.itertuples(index=False):
            timestamp = pd.Timestamp(trade.entry_time)
            timestamp = (
                timestamp.tz_localize("UTC")
                if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
            )
            events.add(f"{timestamp.isoformat()}|{str(trade.direction).lower()}")
        return events

    def event_diagnostics(trades: pd.DataFrame) -> dict[str, object]:
        events = entry_events(trades)
        encoded = "\n".join(sorted(events)).encode("utf-8")
        directions = [event.rsplit("|", 1)[-1] for event in events]
        return {
            "entry_event_fingerprint": hashlib.sha256(encoded).hexdigest(),
            "long_entries": directions.count("long"),
            "short_entries": directions.count("short"),
            "events": sorted(events),
        }

    chunks = np.array_split(np.arange(len(frame)), 3)
    windows = []
    for i, positions in enumerate(chunks, start=1):
        window = frame.iloc[positions]
        metrics, window_trades = _evaluate(
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
            "trade_event_diagnostics": event_diagnostics(window_trades),
            "passed": bool(passed),
        })
    full_metrics, baseline_trades = _evaluate(
        frame, program, execution=execution, risk=risk,
        pip_size=pip_size, price_basis=price_basis,
    )

    baseline_events = entry_events(baseline_trades)
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
        stressed_metrics, stressed_trades = _evaluate(
            stressed_frame, program, execution=stressed_execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
        stressed_events = entry_events(stressed_trades)
        union_events = baseline_events | stressed_events
        expectancy = stressed_metrics.get("expectancy_r")
        profit_factor = stressed_metrics.get("profit_factor")
        baseline_expectancy = full_metrics.get("expectancy_r")
        baseline_profit_factor = full_metrics.get("profit_factor")
        cost_stress.append({
            "cost_multiplier": multiplier,
            "metrics": stressed_metrics,
            "trade_set_diagnostics": {
                "baseline_trade_count": len(baseline_trades),
                "stressed_trade_count": len(stressed_trades),
                "shared_entry_events": len(baseline_events & stressed_events),
                "baseline_only_entry_events": len(baseline_events - stressed_events),
                "stress_only_entry_events": len(stressed_events - baseline_events),
                "entry_event_jaccard_similarity": (
                    len(baseline_events & stressed_events) / len(union_events)
                    if union_events else 1.0
                ),
                "expectancy_delta_r": (
                    float(expectancy) - float(baseline_expectancy)
                    if expectancy is not None and baseline_expectancy is not None else None
                ),
                "profit_factor_delta": (
                    float(profit_factor) - float(baseline_profit_factor)
                    if profit_factor is not None and baseline_profit_factor is not None else None
                ),
            },
            "pass": bool(
                expectancy is not None and float(expectancy) > 0
                and profit_factor is not None and float(profit_factor) > 1.0
            ),
        })
    passed_windows = sum(bool(row["passed"]) for row in windows)
    stress_pass = bool(cost_stress and cost_stress[0]["pass"])
    return {
        "metrics": full_metrics,
        "trade_event_diagnostics": event_diagnostics(baseline_trades),
        "windows": windows,
        "passed_windows": passed_windows,
        "cost_stress": cost_stress,
        "one_and_half_x_cost_pass": stress_pass,
        "pass": bool(passed_windows >= 2 and len(windows) == 3 and stress_pass),
        "strict_pass": bool(passed_windows == 3 and len(windows) == 3 and stress_pass),
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
    evaluated: list[
        tuple[float, StrategyProgram, dict[str, object], dict[str, object]]
    ] = []
    elite_count = max(2, min(4, population_size // 4))
    for generation in range(generations):
        evaluated = []
        for program in population:
            score, metrics, training_robustness = _training_fitness(
                train, program, execution=execution, risk=risk,
                pip_size=pip_size, price_basis=price_basis,
                min_train_trades=min_train_trades,
            )
            evaluated.append((score, program, metrics, training_robustness))
        evaluated.sort(key=lambda item: item[0], reverse=True)
        generation_rows.append({
            "generation": generation + 1,
            "best_training_fitness": float(evaluated[0][0]),
            "best_aggregate_training_fitness": float(
                evaluated[0][3]["aggregate_fitness"]
            ),
            "median_training_fitness": float(np.median([row[0] for row in evaluated])),
            "unique_programs": len({row[1].fingerprint() for row in evaluated}),
            "unique_rule_structures": len({
                _program_structure(row[1]) for row in evaluated
            }),
            "best_training_metrics": evaluated[0][2],
            "best_training_robustness": evaluated[0][3],
        })
        if generation == generations - 1:
            break

        elite_rows = _diverse_ranked(
            evaluated, elite_count, max_per_structure=1
        )
        next_population = [row[1] for row in elite_rows]
        next_seen = {program.fingerprint() for program in next_population}
        parent_pool = _diverse_ranked(
            evaluated, max(elite_count, population_size // 2),
            max_per_structure=1,
        )
        if not parent_pool:
            parent_pool = evaluated[:max(1, elite_count)]

        def select_parent() -> StrategyProgram:
            tournament_size = min(3, len(parent_pool))
            indices = rng.choice(len(parent_pool), size=tournament_size, replace=False)
            winner_row = max(
                (parent_pool[int(index)] for index in indices),
                key=lambda row: row[0],
            )
            return winner_row[1]

        while len(next_population) < population_size:
            if rng.random() < 0.20:
                child = _random_program(features, rng)
            else:
                parent_a = select_parent()
                if rng.random() < 0.55:
                    parent_b = select_parent()
                    child = crossover_programs(parent_a, parent_b, rng)
                else:
                    child = mutate_program(parent_a, features, rng)
            fingerprint = child.fingerprint()
            attempts = 0
            while fingerprint in next_seen and attempts < 8:
                child = _random_program(features, rng)
                fingerprint = child.fingerprint()
                attempts += 1
            if fingerprint in next_seen:
                child = mutate_program(select_parent(), features, rng)
                fingerprint = child.fingerprint()
            if fingerprint not in next_seen:
                next_seen.add(fingerprint)
                next_population.append(child)
        population = next_population

    # A threshold-independent structural cap prevents near-identical variants
    # from occupying the entire validation shortlist.
    shortlist = _diverse_ranked(
        evaluated, min(10, len(evaluated)), max_per_structure=1
    )
    candidate_reports = []
    accepted: list[tuple[float, StrategyProgram, dict[str, object]]] = []
    for score, program, train_metrics, training_robustness in shortlist:
        validation_result = _validation_check(
            validation, program, execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
            min_trades=max(5, min_validation_trades // 3),
        )
        row = {
            "program": program.to_dict(),
            "fingerprint": program.fingerprint(),
            "training_fitness": float(score),
            "aggregate_training_fitness": float(
                training_robustness["aggregate_fitness"]
            ),
            "training_metrics": train_metrics,
            "training_robustness": training_robustness,
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
            "selection": "aggregate plus three chronological training-era fitness scores; validation never feeds back into evolution; structural diversity cap applied to elites, parents, and validation shortlist",
            "training_robustness": {
                "chronological_windows": 3,
                "minimum_trades_per_window": max(
                    3, int(np.ceil(min_train_trades / 6))
                ),
                "fitness_formula": (
                    "0.35 * aggregate + 0.45 * median(window_scores) "
                    "+ 0.20 * worst(window_scores) - 0.10 * std(window_scores)"
                ),
                "window_scores_scaled_to_full_training_sample": True,
                "validation_used_during_evolution": False,
            },
            "structural_diversity": {
                "thresholds_and_risk_parameters_ignored_in_structure_key": True,
                "max_per_structure_in_elite_pool": 1,
                "max_per_structure_in_parent_pool": 1,
                "max_per_structure_in_validation_shortlist": 1,
                "random_immigrant_probability": 0.20,
            },
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
        "strictly_validated_candidate_count": sum(
            bool(candidate["validation"].get("strict_pass"))
            for candidate in candidate_reports
        ),
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
