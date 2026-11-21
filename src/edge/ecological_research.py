
from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path

from .data import (
    annotate_gaps,
    build_gap_audit,
    data_quality_summary,
    resample_ohlc,
    validate_source_timeframe,
)

from .config import (
    ExecutionConfig,
    ResearchConfig,
    RiskConfig,
    TIMEFRAME_SECONDS,
)
from .ecology import (
    OrganismFitness,
    classify_habitats,
    fit_habitat_model,
    measure_habitat_performance,
)
from .ecological_engine import (
    initialize_ecological_population,
    run_ecological_generations,
)
from .evolution import (
    EvolutionSettings,
    _thresholds,
    available_features,
)
from .evolutionary_research import (
    _run_genome,
    _validation_robustness,
)
from .features import add_features
from .pattern_recognition import (
    add_causal_motif_dynamics,
    add_morphology_features,
    fit_pattern_model,
)


def build_walk_forward_splits(
    n_bars: int,
    *,
    folds: int = 3,
    initial_train_fraction: float = 0.60,
    step_fraction: float = 0.10,
) -> list[dict[str, int]]:
    """Build expanding-window train/validation/test partitions.

    Each fold expands training by one step. The previous fold's test
    period becomes historical data for the next fold.
    """
    if n_bars < 1:
        raise ValueError("n_bars must be positive")
    if folds < 1:
        raise ValueError("folds must be >= 1")
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be in (0, 1)")
    if not 0 < step_fraction < 1:
        raise ValueError("step_fraction must be in (0, 1)")

    final_fraction = initial_train_fraction + (folds + 1) * step_fraction

    if final_fraction > 1.0 + 1e-9:
        raise ValueError("Walk-forward folds exceed available observations")

    splits = []

    for number in range(folds):
        train_end = int(
            n_bars * (initial_train_fraction + number * step_fraction)
        )
        validation_end = int(
            n_bars * (
                initial_train_fraction + (number + 1) * step_fraction
            )
        )
        test_end = int(
            n_bars * (
                initial_train_fraction + (number + 2) * step_fraction
            )
        )

        if train_end < 300:
            raise ValueError("Each fold requires at least 300 training bars")

        if validation_end - train_end < 100:
            raise ValueError("Each fold requires at least 100 validation bars")

        if test_end - validation_end < 100:
            raise ValueError("Each fold requires at least 100 test bars")

        splits.append({
            "fold": number + 1,
            "train_end": train_end,
            "validation_end": validation_end,
            "test_end": test_end,
        })

    return splits

def _candidate_rejection_reasons(
    training_metrics: dict[str, object],
    validation_metrics: dict[str, object],
    robustness: dict[str, object],
    research: ResearchConfig,
) -> list[str]:
    reasons = []

    try:
        training_trades = int(training_metrics.get("trades") or 0)
        training_expectancy = float(training_metrics["expectancy_r"])
        validation_trades = int(validation_metrics.get("trades") or 0)
        expectancy = float(validation_metrics["expectancy_r"])
        profit_factor = float(validation_metrics["profit_factor"])
        drawdown = float(validation_metrics["max_realized_drawdown_pct"])
    except (KeyError, TypeError, ValueError):
        return ["invalid_or_missing_metrics"]

    if training_trades < research.minimum_train_trades:
        reasons.append("insufficient_training_trades")
    if not np.isfinite(training_expectancy) or training_expectancy <= 0:
        reasons.append("non_positive_training_expectancy")
    if validation_trades < research.minimum_validation_trades:
        reasons.append("insufficient_validation_trades")
    if not np.isfinite(expectancy) or expectancy <= 0:
        reasons.append("non_positive_validation_expectancy")
    if not np.isfinite(profit_factor) or profit_factor < 1.05:
        reasons.append("validation_profit_factor_below_1.05")
    if not np.isfinite(drawdown) or drawdown > research.max_drawdown_filter:
        reasons.append("validation_drawdown_exceeded")

    windows = robustness.get("chronological_validation_windows", [])
    if not (
        len(windows) == 3
        and all(
            bool(window.get("enough_trades"))
            and bool(window.get("pass"))
            for window in windows
        )
    ):
        reasons.append("chronological_validation_windows_failed")

    if not robustness.get("one_and_half_x_cost_pass", False):
        reasons.append("execution_cost_stress_failed")

    return reasons

def _candidate_passes_validation(
    training_metrics: dict[str, object],
    validation_metrics: dict[str, object],
    robustness: dict[str, object],
    research: ResearchConfig,
) -> bool:
    return not _candidate_rejection_reasons(
        training_metrics,
        validation_metrics,
        robustness,
        research,
    )


def _candidate_rank_key(
    validation_metrics: dict[str, object],
    robustness: dict[str, object],
    complexity: int,
) -> tuple[float, ...]:
    windows = robustness["chronological_validation_windows"]

    window_expectancies = [
        float(window["metrics"]["expectancy_r"])
        for window in windows
    ]

    bootstrap = robustness.get("expectancy_block_bootstrap") or {}

    lower = bootstrap.get("ci_90_low")
    upper = bootstrap.get("ci_90_high")

    bootstrap_lower = (
        float(lower) if lower is not None else float("-inf")
    )
    bootstrap_width = (
        float(upper) - float(lower)
        if lower is not None and upper is not None
        else float("inf")
    )

    stress = next(
        item for item in robustness["cost_stress"]
        if float(item["cost_multiplier"]) == 1.5
    )

    stress_expectancy = float(stress["metrics"]["expectancy_r"])

    return (
        min(window_expectancies),
        bootstrap_lower,
        -bootstrap_width,
        float(robustness["median_window_expectancy_r"]),
        stress_expectancy,
        float(validation_metrics["profit_factor"]),
        -float(validation_metrics["max_realized_drawdown_pct"]),
        -float(complexity),
    )


def _candidate_pool_order_key(
    organism_id: str,
    fitness: OrganismFitness,
    pareto_rank: int,
    crowding_distance: float,
    *,
    minimum_train_trades: int,
) -> tuple[bool, int, float, str]:
    """Prefer adequately sampled organisms before Pareto ordering."""
    return (
        fitness.trade_count < minimum_train_trades,
        pareto_rank,
        -crowding_distance,
        organism_id,
    )


def _trade_group_summary(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": None,
            "expectancy_r": None,
            "profit_factor": None,
            "total_net_r": 0.0,
            "average_gross_r": None,
            "average_separately_deducted_cost_r": None,
            "gap_exits": 0,
        }

    net = pd.to_numeric(trades["net_r"], errors="coerce").dropna()
    winners = net[net > 0]
    losers = net[net < 0]

    gross = pd.to_numeric(
        trades["gross_r"], errors="coerce"
    ).dropna()
    costs = pd.to_numeric(
        trades["cost_r"], errors="coerce"
    ).dropna()

    return {
        "trades": int(len(trades)),
        "win_rate": float((net > 0).mean()) if len(net) else None,
        "expectancy_r": float(net.mean()) if len(net) else None,
        "profit_factor": (
            float(winners.sum() / abs(losers.sum()))
            if len(losers) and abs(losers.sum()) > 0
            else None
        ),
        "total_net_r": float(net.sum()),
        "average_gross_r": float(gross.mean()) if len(gross) else None,
        "average_cost_r": float(costs.mean()) if len(costs) else None,
        "gap_exits": int(
            trades["gap_exit"].fillna(False).astype(bool).sum()
        ),
    }


def _grouped_trade_summary(
    trades: pd.DataFrame,
    column: str,
) -> list[dict[str, object]]:
    if trades.empty or column not in trades.columns:
        return []

    rows = []

    for key, group in trades.groupby(column, dropna=False, sort=True):
        rows.append({
            column: None if pd.isna(key) else str(key),
            **_trade_group_summary(group),
        })

    return rows


def _final_test_diagnostics(
    test_frame: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, object]:
    index = pd.DatetimeIndex(pd.to_datetime(test_frame.index, utc=True))
    if trades.empty:
        positions = np.array([], dtype=int)
    else:
        if "entry_time" not in trades.columns:
            raise ValueError("Trades missing entry_time column")

        entry_times = pd.to_datetime(
            trades["entry_time"], utc=True, errors="coerce"
        )

        if entry_times.isna().any():
            raise ValueError("Trade entry timestamps contain invalid values")

        positions = index.get_indexer(
            pd.DatetimeIndex(entry_times)
        )

    windows = []

    for number, chunk in enumerate(
        np.array_split(np.arange(len(index)), 4),
        start=1,
    ):
        if len(chunk) == 0:
            continue

        mask = (
            (positions >= int(chunk[0]))
            & (positions <= int(chunk[-1]))
        )

        window_trades = trades.loc[mask]

        windows.append({
            "window": number,
            "start_utc": index[int(chunk[0])].isoformat(),
            "end_utc": index[int(chunk[-1])].isoformat(),
            **_trade_group_summary(window_trades),
        })

    cost_columns = (
        "spread_cost_pips",
        "commission_cost_pips",
        "slippage_cost_pips",
        "swap_cost_pips",
    )

    cost_totals = {}

    for column in cost_columns:
        if column in trades.columns:
            values = pd.to_numeric(
                trades[column], errors="coerce"
            ).fillna(0)
            cost_totals[column] = float(values.sum())

    return {
        "window_method": (
            "Four equal chronological groups of observed holdout candles; "
            "trades assigned by entry candle"
        ),
        "unmatched_entry_timestamps": int((positions < 0).sum()),
        "chronological_windows": windows,
        "by_direction": _grouped_trade_summary(trades, "direction"),
        "by_exit_reason": _grouped_trade_summary(trades, "exit_reason"),
        "costs": {
            "separately_deducted_cost_r": (
                float(pd.to_numeric(
                    trades["cost_r"], errors="coerce"
                ).sum())
                if "cost_r" in trades.columns else None
            ),
            "component_totals_pips": cost_totals,
        },
        "overall": _trade_group_summary(trades),
        "spread_accounting": (
             "Embedded in synthetic bid/ask execution prices; "
             "not included in separately deducted cost_r."
        ),
    }

def research_ecological_timeframe(
    raw_frame: pd.DataFrame,
    *,
    timeframe: str,
    symbol: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    research: ResearchConfig,
    evolution: EvolutionSettings,
    pip_size: float,
    price_basis: str,
    run_final_test: bool = True,
    minimum_habitat_trades: int = 5,
    minimum_period_trades: int = 5,
    partition_ends: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Run ecological evolution and chronological candidate validation."""

    featured = add_morphology_features(add_features(raw_frame))
    n = len(featured)

    if partition_ends is None:
        train_end = int(n * research.train_fraction)
        validation_end = int(
            n * (research.train_fraction + research.validation_fraction)
        )
    else:
        train_end, validation_end = partition_ends

        if not 0 < train_end < validation_end < n:
            raise ValueError("Invalid explicit research partition boundaries")

    gap_mask = (
        featured["edge_gap_contaminated"].fillna(False).astype(bool)
        if "edge_gap_contaminated" in featured.columns
        else pd.Series(False, index=featured.index)
    )

    gap_diagnostics = {
        "contaminated_bars": int(gap_mask.sum()),
        "usable_bars": int((~gap_mask).sum()),
        "usable_fraction": float((~gap_mask).mean()) if n else 0.0,
    }

    if (
        train_end < 300
        or validation_end - train_end < 100
        or n - validation_end < 100
    ):
        return {
            "timeframe": timeframe,
            "status": "insufficient_data",
            "bars": n,
            "train_bars": train_end,
            "validation_bars": validation_end - train_end,
            "test_bars": n - validation_end,
            "reason": "Insufficient chronological training, validation or test data.",
            "winner": None,
            "gap_diagnostics": gap_diagnostics,
        }

    # Fit pattern recognition exclusively on the training prefix.
    featured, pattern_model = fit_pattern_model(
        featured,
        fit_end=train_end,
        n_clusters=evolution.pattern_clusters,
        random_state=evolution.seed,
    )

    featured = add_causal_motif_dynamics(featured, pattern_model)

    train_frame = featured.iloc[:train_end].copy()
    validation_frame = featured.iloc[train_end:validation_end].copy()
    test_frame = featured.iloc[validation_end:].copy()

    usable_features = available_features(train_frame)

    if len(usable_features) < 4:
        return {
            "timeframe": timeframe,
            "status": "insufficient_features",
            "bars": n,
            "usable_features": list(usable_features),
            "reason": "Fewer than four usable training features.",
            "winner": None,
        }

    # Fit habitat boundaries using training data only.
    habitat_model = fit_habitat_model(
        featured,
        fit_end=train_end,
    )

    # Apply the frozen model to all partitions.
    classified_habitats = classify_habitats(
        featured,
        habitat_model,
    )

    train_habitats = classified_habitats.iloc[:train_end].copy()

    feature_ranges = _thresholds(
        train_frame,
        usable_features,
    )

    if not feature_ranges:
        raise ValueError("Training data contains no usable feature ranges")

    # Deterministic, unique initial genomes with both trade directions active.
    initial_population = initialize_ecological_population(
        train_frame,
        population_id=f"{symbol.upper()}-{timeframe.upper()}-ecology",
        population_size=evolution.population_size,
        feature_ranges=feature_ranges,
        seed=evolution.seed,
    )

    # Evolution receives training data only.
    ecological_run = run_ecological_generations(
        initial_population,
        train_frame,
        train_habitats,
        execution=execution,
        risk=risk,
        pip_size=pip_size,
        price_basis=price_basis,
        timeframe_seconds=TIMEFRAME_SECONDS[timeframe.upper()],
        settings=evolution,
        minimum_habitat_trades=minimum_habitat_trades,
        minimum_period_trades=minimum_period_trades,
        minimum_total_trades=research.minimum_train_trades,
    )

    # Rank surviving organisms by their existing training Pareto results.
    selection = ecological_run.final_selection

    front_rank = {
        organism_id: rank
        for rank, front in enumerate(selection.pareto_fronts)
        for organism_id in front
    }

    crowding = dict(selection.crowding_distance)

    candidates = sorted(
        selection.population.organisms,
        key=lambda organism: _candidate_pool_order_key(
            organism.organism_id,
            ecological_run.final_evaluation.fitness[organism.organism_id],
            front_rank.get(organism.organism_id, len(front_rank)),
            crowding.get(organism.organism_id, 0.0),
            minimum_train_trades=research.minimum_train_trades,
        ),
    )[:evolution.validation_pool]

    candidate_rows = []
    eligible = []

    minimum_window_trades = max(
        8,
        research.minimum_validation_trades // 4,
    )

    # Validation is used only after evolution has finished.
    for organism in candidates:
        organism_id = organism.organism_id
        training_evaluation = ecological_run.final_evaluation.evaluations[
            organism_id
        ]

        training_metrics = training_evaluation.metrics

        validation_metrics, validation_trades, _ = _run_genome(
            validation_frame,
            organism.genome,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
        )

        validation_habitat_summary = measure_habitat_performance(
            validation_trades,
            classified_habitats.iloc[train_end - 1:validation_end].copy(),
            timeframe_seconds=TIMEFRAME_SECONDS[timeframe.upper()],
        )

        validation_habitat_diagnostics = {
            "classified_trades": validation_habitat_summary.classified_trades,
            "unclassified_trades": validation_habitat_summary.unclassified_trades,
            "outcomes": [
                {
                    "habitat_id": outcome.habitat_id,
                    "trade_count": outcome.trade_count,
                    "expectancy_r": outcome.expectancy_r,
                }
                for outcome in validation_habitat_summary.outcomes
            ],
        }

        robustness = _validation_robustness(
            validation_frame,
            organism.genome,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            min_window_trades=minimum_window_trades,
            max_drawdown_filter=research.max_drawdown_filter,
        )

        rejection_reasons = _candidate_rejection_reasons(
            training_metrics,
            validation_metrics,
            robustness,
            research,
        )
        passed = not rejection_reasons

        complexity = (
            len(organism.genome.long_conditions)
            + len(organism.genome.short_conditions)
        )

        validation_rank_key = (
            _candidate_rank_key(
                validation_metrics,
                robustness,
                complexity,
            )
            if passed
            else None
        )

        row = {
            "organism_id": organism_id,
            "species_id": organism.species_id,
            "genome": organism.genome.to_dict(),
            "training_metrics": training_metrics,
            "training_confidence_adjusted_expectancy_r": ecological_run.final_evaluation.fitness[organism_id].confidence_adjusted_expectancy_r,
            "training_temporal_persistence": ecological_run.final_evaluation
                .fitness[organism_id].temporal_persistence_score,
            "validation_metrics": validation_metrics,
            "validation_robustness": robustness,
            "eligible": passed,
            "rejection_reasons": rejection_reasons,
            "validation_rank_key": (
                list(validation_rank_key)
                if validation_rank_key is not None
                else None),
            "validation_habitat_performance": validation_habitat_diagnostics,              
        }

        candidate_rows.append(row)

        if passed:
            assert validation_rank_key is not None
            eligible.append((validation_rank_key, organism, row))

    # Serialize final generation information.
    final_organisms = []

    for organism in ecological_run.final_population.organisms:
        organism_id = organism.organism_id
        fitness = ecological_run.final_evaluation.fitness[organism_id]
        evaluation = ecological_run.final_evaluation.evaluations[organism_id]

        final_organisms.append({
            "organism_id": organism_id,
            "species_id": organism.species_id,
            "generation": organism.generation,
            "parent_ids": organism.parent_ids,
            "genome": organism.genome.to_dict(),
            "fitness": {
                "expectancy_r": fitness.expectancy_r,
                "confidence_adjusted_expectancy_r": fitness.confidence_adjusted_expectancy_r,
                "trade_count": fitness.trade_count,
                "max_drawdown_pct": fitness.max_drawdown_pct,
                "complexity": fitness.complexity,
                "novelty_score": fitness.novelty_score,
                "temporal_persistence_score": fitness.temporal_persistence_score,
                "habitat_outcomes": [
                    {
                        "habitat_id": item.habitat_id,
                        "expectancy_r": item.expectancy_r,
                        "trade_count": item.trade_count,
                    }
                    for item in fitness.habitat_outcomes
                ],
            },
            "temporal_habitat_persistence": [
                {
                    "habitat_id": item.habitat_id,
                    "qualified_periods": item.qualified_periods,
                    "profitable_periods": item.profitable_periods,
                    "persistence_ratio": item.persistence_ratio,
                    "weighted_expectancy_r": item.weighted_expectancy_r,
                    "worst_period_expectancy_r": item.worst_period_expectancy_r,
                }
                for item in evaluation.temporal_habitat_persistence
            ],
        })

    base_result = {
        "timeframe": timeframe.upper(),
        "bars": n,
        "train_bars": len(train_frame),
        "validation_bars": len(validation_frame),
        "test_bars": len(test_frame),
        "gap_diagnostics": gap_diagnostics,
        "pattern_model": pattern_model.to_dict(),
        "habitat_model": habitat_model.to_dict(),
        "usable_features": list(usable_features),
        "generation_history": ecological_run.history,
        "final_species": [
            {
                "species_id": item.species_id,
                "founder_organism_id": item.founder_organism_id,
                "member_ids": item.member_ids,
                "origin_generation": item.origin_generation,
                "last_active_generation": item.last_active_generation,
            }
            for item in ecological_run.final_species
        ],
        "species_habitat_profiles": [
            {
                "species_id": item.species_id,
                "preferred_habitat": item.preferred_habitat,
                "specialization_score": item.specialization_score,
                "sampled_habitats": item.sampled_habitats,
                "habitat_expectancy": dict(item.habitat_expectancy),
            }
            for item in ecological_run.final_species_profiles
        ],
        "final_organisms": final_organisms,
        "survivor_ids": selection.survivor_ids,
        "candidates": candidate_rows,
    }

    if not eligible:
        return {
            **base_result,
            "status": "no_robust_candidate",
            "winner": None,
            "final_test": None,
            "final_test_skipped": True,
            "reason": "No survivor passed the validation and execution-cost gates.",
        }

    eligible.sort(key=lambda item: item[0], reverse=True)
    _, winner, winning_row = eligible[0]
    if run_final_test:
        # Evaluate the selected candidate on the final holdout once.
        test_metrics, test_trades, test_equity = _run_genome(
            test_frame,
            winner.genome,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
        )

        test_diagnostics = _final_test_diagnostics(
            test_frame,
            test_trades,
        )

        test_habitat_summary = measure_habitat_performance(
            test_trades,
            classified_habitats.iloc[validation_end - 1:].copy(),
            timeframe_seconds=TIMEFRAME_SECONDS[timeframe.upper()],
        )

        test_habitat_diagnostics = {
            "classified_trades": test_habitat_summary.classified_trades,
            "unclassified_trades": test_habitat_summary.unclassified_trades,
            "outcomes": [
                {
                    "habitat_id": outcome.habitat_id,
                    "trade_count": outcome.trade_count,
                    "expectancy_r": outcome.expectancy_r,
                }
                for outcome in test_habitat_summary.outcomes
            ],
        }

        test_passed = (
            int(test_metrics.get("trades") or 0)
            >= research.minimum_validation_trades
            and test_metrics.get("expectancy_r") is not None
            and float(test_metrics["expectancy_r"]) > 0
            and test_metrics.get("profit_factor") is not None
            and float(test_metrics["profit_factor"]) > 1.0
            and test_metrics.get("max_realized_drawdown_pct") is not None
            and float(test_metrics["max_realized_drawdown_pct"])
            <= research.max_drawdown_filter
        )

        status = (
            "test_passed"
            if test_passed
            else "test_failed_or_inconclusive"
        )
    else:
        test_metrics = None
        test_diagnostics = None
        test_trades = None
        test_equity = None
        test_passed = None
        test_habitat_diagnostics = None
        status = "candidate_selected_research_only"

    return {
        **base_result,
        "status": status,
        "winner": winner.genome.to_dict(),
        "winner_id": winner.organism_id,
        "winner_validation": winning_row["validation_metrics"],
        "winner_validation_robustness": winning_row["validation_robustness"],
        "final_test": test_metrics,
        "final_test_diagnostics": test_diagnostics,
        "final_test_habitat_performance": test_habitat_diagnostics,
        "_final_test_trades": test_trades,
        "_final_test_equity": test_equity,
        "final_test_passed": test_passed,
        "final_test_skipped": not run_final_test,
    }


def run_ecological_research(
    source_frame: pd.DataFrame,
    *,
    source_timeframe: str,
    timeframes: list[str],
    symbol: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    research: ResearchConfig,
    evolution: EvolutionSettings,
    pip_size: float,
    price_basis: str,
    output_dir: str | Path,
    run_final_test: bool = True,
    minimum_habitat_trades: int = 5,
    minimum_period_trades: int = 5,
) -> dict[str, object]:
    """Run ecological research across multiple timeframes and save reports."""

    validate_source_timeframe(source_timeframe, timeframes)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_frame = annotate_gaps(
        source_frame,
        TIMEFRAME_SECONDS[source_timeframe.upper()],
    )

    source_quality = data_quality_summary(source_frame)

    source_gap_audit = build_gap_audit(
        source_frame,
        TIMEFRAME_SECONDS[source_timeframe.upper()],
    )

    gap_audit_path = output / "source_gap_audit.csv"
    source_gap_audit.to_csv(gap_audit_path, index=False)

    results = []

    for timeframe in timeframes:
        bars = resample_ohlc(source_frame, timeframe)

        result = research_ecological_timeframe(
            bars,
            timeframe=timeframe,
            symbol=symbol,
            execution=execution,
            risk=risk,
            research=research,
            evolution=evolution,
            pip_size=pip_size,
            price_basis=price_basis,
            run_final_test=run_final_test,
            minimum_habitat_trades=minimum_habitat_trades,
            minimum_period_trades=minimum_period_trades,
        )

        test_trades = result.pop("_final_test_trades", None)
        test_equity = result.pop("_final_test_equity", None)

        diagnostics = result.get("final_test_diagnostics")

        if isinstance(test_trades, pd.DataFrame):
            trades_path = output / f"{timeframe.upper()}_final_test_trades.csv"
            test_trades.to_csv(trades_path, index=False)

            if isinstance(diagnostics, dict):
                diagnostics["trade_log_file"] = trades_path.name

        if isinstance(test_equity, pd.DataFrame):
            equity_path = output / f"{timeframe.upper()}_final_test_equity.csv"
            test_equity.to_csv(equity_path, index=False)

            if isinstance(diagnostics, dict):
                diagnostics["equity_curve_file"] = equity_path.name

        results.append(result)

    report = {
        "project": "Project EDGE",
        "engine": "ecological evolutionary strategy research",
        "symbol": symbol.upper(),
        "source_timeframe": source_timeframe.upper(),
        "timeframes": [item.upper() for item in timeframes],
        "research_warning": (
            "Experimental research only. Evolution uses training data. "
            "Validation is used for candidate selection and is subject to "
            "model-selection bias. The final chronological holdout is "
            "evaluated once after selection. Previously inspected holdout "
            "data does not become independent by rerunning this engine. "
            "No live-trading claim is made."
        ),
        "evolution_settings": {
            "population_size": evolution.population_size,
            "generations": evolution.generations,
            "pattern_clusters": evolution.pattern_clusters,
            "seed": evolution.seed,
            "adaptive_mutation": evolution.adaptive_mutation,
            "mutation_rate": evolution.mutation_rate,
            "random_injection_rate": evolution.random_injection_rate,
            "minimum_habitat_trades": minimum_habitat_trades,
            "minimum_period_trades": minimum_period_trades,
            "minimum_train_trades": research.minimum_train_trades,
        },
        "temporal_evaluation": {
            "windows": 4,
            "method": "Equal-sized chronological groups of observed training candles",
            "persistence_objective": (
                "Fraction of sufficiently sampled habitat-period combinations "
                "with positive expectancy"
            ),
        },
        "habitat_model": {
            "fitting_policy": "Training prefix only",
            "classification_policy": "Frozen thresholds applied to all partitions",
        },
        "validation_design": {
            "candidate_selection": (
                "Training Pareto survivors followed by base validation, "
                "three chronological validation windows and 1.5x cost stress"
            ),
            "final_test_run": bool(run_final_test),
            "final_test_policy": "One evaluation after candidate selection",
        },
        "execution_assumptions": {
            "price_basis": price_basis,
            "spread_pips": execution.spread_pips,
            "slippage_pips_per_side": execution.slippage_pips_per_side,
            "commission_round_trip_pips": execution.commission_round_trip_pips,
            "swap_long_pips_per_rollover": execution.swap_long_pips_per_rollover,
            "swap_short_pips_per_rollover": execution.swap_short_pips_per_rollover,
            "rollover_timezone": execution.rollover_timezone,
            "rollover_local_time": execution.rollover_local_time,
            "triple_swap_weekday": execution.triple_swap_weekday,
            "triple_swap_multiplier": execution.triple_swap_multiplier,
        },
        "source_data_quality": source_quality,
        "source_gap_audit": {
            "file": gap_audit_path.name,
            "events": int(len(source_gap_audit)),
            "synthetic_bars_created": False,
        },
        "results": results,
    }

    report_path = output / "ecological_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    summary_rows = []

    for result in results:
        test = result.get("final_test") or {}
        history = result.get("generation_history") or []
        latest = history[-1] if history else {}

        summary_rows.append({
            "timeframe": result["timeframe"],
            "status": result["status"],
            "winner_id": result.get("winner_id"),
            "train_bars": result.get("train_bars"),
            "validation_bars": result.get("validation_bars"),
            "test_bars": result.get("test_bars"),
            "test_trades": test.get("trades"),
            "test_expectancy_R": test.get("expectancy_r"),
            "test_profit_factor": test.get("profit_factor"),
            "test_drawdown": test.get("max_realized_drawdown_pct"),
            "species_count": len(result.get("final_species", [])),
            "mean_temporal_persistence": latest.get(
                "mean_temporal_persistence_score"
            ),
        })

    summary_path = output / "ecological_timeframe_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    lines = [
        f"# Project EDGE ecological research: {symbol.upper()}",
        "",
        "> Experimental research only. Results do not establish a tradable edge.",
        "",
        f"Population: {evolution.population_size}",
        f"Generations: {evolution.generations}",
        f"Adaptive mutation: {evolution.adaptive_mutation}",
        f"Final test enabled: {run_final_test}",
        "",
        "| Timeframe | Status | Winner | Test trades | Test expectancy (R) | Species |",
        "|---|---|---|---:|---:|---:|",
    ]

    for row in summary_rows:
        def fmt(value: object) -> str:
            return "n/a" if value is None else str(value)

        lines.append(
            f"| {row['timeframe']} | {row['status']} | "
            f"{fmt(row['winner_id'])} | {fmt(row['test_trades'])} | "
            f"{fmt(row['test_expectancy_R'])} | "
            f"{row['species_count']} |"
        )

    lines.extend([
        "",
        "## Ecological mechanisms",
        "",
        "- Genetic speciation and lineage tracking.",
        "- Pareto-based multi-objective survival.",
        "- Habitat-specific survival and specialization.",
        "- Trade-weighted species habitat profiles.",
        "- Habitat-guided interspecies reproduction.",
        "- Chronological habitat persistence.",
        "- Adaptive mutation using stagnation and diversity.",
        "",
        "## Data and evaluation limitations",
        "",
        "- Habitat thresholds are fitted on training data only.",
        "- Validation influences candidate selection and is not an independent test.",
        "- The final holdout is evaluated once after selection.",
        "- Temporal persistence is descriptive and does not prove predictive stability.",
        "- OHLC execution simulations cannot reproduce all tick-level market conditions.",
        "",
    ])

    summary_path_md = output / "ecological_summary.md"
    summary_path_md.write_text("\n".join(lines), encoding="utf-8")

    report["report_files"] = {
        "json": report_path.name,
        "csv": summary_path.name,
        "markdown": summary_path_md.name,
    }

    # Rewrite JSON to include its own report-file manifest.
    report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    return report

def run_ecological_walk_forward_research(
    source_frame: pd.DataFrame,
    *,
    source_timeframe: str,
    timeframes: list[str],
    symbol: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    research: ResearchConfig,
    evolution: EvolutionSettings,
    pip_size: float,
    price_basis: str,
    output_dir: str | Path,
    folds: int = 3,
    initial_train_fraction: float = 0.60,
    step_fraction: float = 0.10,
    minimum_habitat_trades: int = 5,
    minimum_period_trades: int = 5,
) -> dict[str, object]:
    """Run independent ecological selection and testing for each walk-forward fold."""

    validate_source_timeframe(source_timeframe, timeframes)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_frame = annotate_gaps(
        source_frame,
        TIMEFRAME_SECONDS[source_timeframe.upper()],
    )

    source_quality = data_quality_summary(source_frame)

    source_gap_audit = build_gap_audit(
        source_frame,
        TIMEFRAME_SECONDS[source_timeframe.upper()],
    )

    gap_audit_path = output / "source_gap_audit.csv"
    source_gap_audit.to_csv(gap_audit_path, index=False)

    timeframe_results = []
    summary_rows = []

    for timeframe in timeframes:
        bars = resample_ohlc(source_frame, timeframe)

        splits = build_walk_forward_splits(
            len(bars),
            folds=folds,
            initial_train_fraction=initial_train_fraction,
            step_fraction=step_fraction,
        )

        fold_results = []
        test_expectancies = []

        for split in splits:
            number = split["fold"]
            test_end = split["test_end"]
            train_end = split["train_end"]
            validation_end = split["validation_end"]

            # The fold receives no observations beyond its test endpoint.
            fold_frame = bars.iloc[:test_end].copy()

            fold_result = research_ecological_timeframe(
                fold_frame,
                timeframe=timeframe,
                symbol=symbol,
                execution=execution,
                risk=risk,
                research=research,
                evolution=evolution,
                pip_size=pip_size,
                price_basis=price_basis,
                run_final_test=True,
                minimum_habitat_trades=minimum_habitat_trades,
                minimum_period_trades=minimum_period_trades,
                partition_ends=(train_end, validation_end),
            )

            test_trades = fold_result.pop("_final_test_trades", None)
            test_equity = fold_result.pop("_final_test_equity", None)

            diagnostics = fold_result.get("final_test_diagnostics")

            suffix = f"fold_{number:02d}"

            if isinstance(test_trades, pd.DataFrame):
                trades_path = output / (
                    f"{timeframe.upper()}_{suffix}_test_trades.csv"
                )
                test_trades.to_csv(trades_path, index=False)

                if isinstance(diagnostics, dict):
                    diagnostics["trade_log_file"] = trades_path.name

            if isinstance(test_equity, pd.DataFrame):
                equity_path = output / (
                    f"{timeframe.upper()}_{suffix}_test_equity.csv"
                )
                test_equity.to_csv(equity_path, index=False)

                if isinstance(diagnostics, dict):
                    diagnostics["equity_curve_file"] = equity_path.name

            test_metrics = fold_result.get("final_test") or {}
            expectancy = test_metrics.get("expectancy_r")

            if expectancy is not None and np.isfinite(float(expectancy)):
                test_expectancies.append(float(expectancy))

            fold_result["walk_forward"] = {
                "fold": number,
                "train_start_utc": bars.index[0].isoformat(),
                "train_end_utc": bars.index[train_end - 1].isoformat(),
                "validation_start_utc": bars.index[train_end].isoformat(),
                "validation_end_utc": bars.index[validation_end - 1].isoformat(),
                "test_start_utc": bars.index[validation_end].isoformat(),
                "test_end_utc": bars.index[test_end - 1].isoformat(),
                "train_end_position": train_end,
                "validation_end_position": validation_end,
                "test_end_position": test_end,
            }

            fold_results.append(fold_result)


            candidates = fold_result.get("candidates") or []
            rejection_counts = {}

            for candidate in candidates:
                for reason in candidate.get("rejection_reasons", []):
                    rejection_counts[reason] = (
                        rejection_counts.get(reason, 0) + 1
                    )

            eligible_count = sum(
                1 for candidate in candidates
                if candidate.get("eligible", False)
            )


            summary_rows.append({
                "timeframe": timeframe.upper(),
                "fold": number,
                "status": fold_result["status"],
                "winner_id": fold_result.get("winner_id"),
                "candidates_evaluated": len(candidates),
                "eligible_candidates": eligible_count,
                "rejected_candidates": len(candidates) - eligible_count,
                "rejection_reason_counts": json.dumps(
                    rejection_counts, sort_keys=True
                ),
                "train_bars": fold_result.get("train_bars"),
                "validation_bars": fold_result.get("validation_bars"),
                "test_bars": fold_result.get("test_bars"),
                "test_trades": test_metrics.get("trades"),
                "test_expectancy_R": expectancy,
                "test_profit_factor": test_metrics.get("profit_factor"),
                "test_drawdown": test_metrics.get(
                    "max_realized_drawdown_pct"
                ),
            })

        timeframe_results.append({
            "timeframe": timeframe.upper(),
            "status": "walk_forward_completed",
            "bars": len(bars),
            "folds": fold_results,
            "aggregate_test_summary": {
                "fold_count": len(fold_results),
                "folds_with_selected_candidate": sum(
                    1 for item in fold_results
                    if item.get("winner_id") is not None
                ),
                "positive_expectancy_folds": sum(
                    1 for value in test_expectancies if value > 0
                ),
                "mean_test_expectancy_r": (
                    float(np.mean(test_expectancies))
                    if test_expectancies else None
                ),
                "median_test_expectancy_r": (
                    float(np.median(test_expectancies))
                    if test_expectancies else None
                ),
                "candidate_selected_across_test_folds": False,
            },
        })

    summary_path = output / "ecological_walk_forward_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    report = {
        "project": "Project EDGE",
        "engine": "ecological evolutionary walk-forward research",
        "symbol": symbol.upper(),
        "source_timeframe": source_timeframe.upper(),
        "timeframes": [item.upper() for item in timeframes],
        "research_warning": (
            "Experimental research only. Each fold performs training, "
            "validation-based selection and subsequent chronological testing. "
            "Test results are never used to select candidates across folds. "
            "Expanding training windows overlap, so fold results are not "
            "statistically independent. Historical observations may have "
            "appeared in earlier research reports; therefore this experiment "
            "does not constitute fresh independent prospective confirmation. "
            "No live-trading claim is made."
        ),
        "walk_forward_design": {
            "folds": folds,
            "initial_train_fraction": initial_train_fraction,
            "step_fraction": step_fraction,
            "training_window": "Expanding",
            "validation_window": "Fixed chronological step",
            "test_window": "Fixed chronological step",
            "feature_and_pattern_fitting": "Training prefix only",
            "selection": "Independent within each fold",
            "cross_fold_test_selection": False,
        },
        "evolution_settings": {
            "population_size": evolution.population_size,
            "generations": evolution.generations,
            "pattern_clusters": evolution.pattern_clusters,
            "seed": evolution.seed,
            "adaptive_mutation": evolution.adaptive_mutation,
            "validation_pool": evolution.validation_pool,
        },
        "execution_assumptions": {
            "price_basis": price_basis,
            "spread_pips": execution.spread_pips,
            "slippage_pips_per_side": execution.slippage_pips_per_side,
            "commission_round_trip_pips": execution.commission_round_trip_pips,
            "swap_long_pips_per_rollover": execution.swap_long_pips_per_rollover,
            "swap_short_pips_per_rollover": execution.swap_short_pips_per_rollover,
        },
        "source_data_quality": source_quality,
        "source_gap_audit": {
            "file": gap_audit_path.name,
            "events": int(len(source_gap_audit)),
            "synthetic_bars_created": False,
        },
        "results": timeframe_results,
        "report_files": {
            "csv": summary_path.name,
            "json": "ecological_walk_forward_report.json",
        },
    }

    report_path = output / "ecological_walk_forward_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    return report

def test_walk_forward_splits_expand_training_without_future_overlap():
    from edge.ecological_research import build_walk_forward_splits

    splits = build_walk_forward_splits(
        1000,
        folds=3,
        initial_train_fraction=0.60,
        step_fraction=0.10,
    )

    assert splits == [
        {
            "fold": 1,
            "train_end": 600,
            "validation_end": 700,
            "test_end": 800,
        },
        {
            "fold": 2,
            "train_end": 700,
            "validation_end": 800,
            "test_end": 900,
        },
        {
            "fold": 3,
            "train_end": 800,
            "validation_end": 900,
            "test_end": 1000,
        },
    ]

    for split in splits:
        assert split["train_end"] < split["validation_end"]
        assert split["validation_end"] < split["test_end"]

    assert splits[0]["test_end"] == splits[1]["validation_end"]
    assert splits[1]["test_end"] == splits[2]["validation_end"]


def test_walk_forward_split_rejects_insufficient_data():
    import pytest

    from edge.ecological_research import build_walk_forward_splits

    with pytest.raises(ValueError):
        build_walk_forward_splits(100, folds=3)


def test_walk_forward_split_rejects_excessive_fold_count():
    import pytest

    from edge.ecological_research import build_walk_forward_splits

    with pytest.raises(ValueError):
        build_walk_forward_splits(
            1000,
            folds=5,
            initial_train_fraction=0.60,
            step_fraction=0.10,
        )