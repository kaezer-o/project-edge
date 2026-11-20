from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import BacktestSettings, run_backtest
from .config import ExecutionConfig, ResearchConfig, RiskConfig, TIMEFRAME_SECONDS
from .data import (
    annotate_gaps, build_gap_audit, data_quality_summary,
    gap_annotation_summary, resample_ohlc, validate_source_timeframe,
)
from .evolution import (
    EvolutionSettings, StrategyGenome, available_features, evolve_population,
    fitness_from_metrics, generate_genome_signal, genome_from_dict,
)
from .features import add_features
from .memory import load_genome_memory, update_genome_memory
from .metrics import calculate_metrics
from .pattern_recognition import add_causal_motif_dynamics, add_morphology_features, fit_pattern_model
from .validation import block_bootstrap_expectancy, split_chronological_windows


def _run_genome(
    frame: pd.DataFrame,
    genome: StrategyGenome,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    signal = generate_genome_signal(frame, genome)
    trades, equity = run_backtest(
        frame,
        signal,
        stop_atr=genome.stop_atr,
        reward_risk=genome.reward_risk,
        execution=execution,
        risk=risk,
        settings=BacktestSettings(
            pip_size=pip_size,
            max_holding_bars=genome.max_holding_bars,
            price_basis=price_basis,
        ),
    )
    return calculate_metrics(trades, equity), trades, equity


def _selection_score(metrics: dict[str, object], genome: StrategyGenome) -> float:
    expectancy = metrics.get("expectancy_r")
    drawdown = metrics.get("max_realized_drawdown_pct")
    count = int(metrics.get("trades") or 0)
    if expectancy is None or drawdown is None or count <= 0:
        return float("-inf")
    complexity = len(genome.long_conditions) + len(genome.short_conditions)
    return float(expectancy) * (count ** 0.5) / (1.0 + 4.0 * float(drawdown)) - 0.015 * complexity


def _validation_robustness(
    validation_frame: pd.DataFrame,
    genome: StrategyGenome,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    min_window_trades: int,
    max_drawdown_filter: float,
) -> dict[str, object]:
    windows = split_chronological_windows(validation_frame, windows=3)
    window_rows: list[dict[str, object]] = []
    for number, window in enumerate(windows, start=1):
        metrics, _, _ = _run_genome(
            window, genome, execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
        trades = int(metrics.get("trades") or 0)
        expectancy = metrics.get("expectancy_r")
        profit_factor = metrics.get("profit_factor")
        drawdown = metrics.get("max_realized_drawdown_pct")
        enough = trades >= min_window_trades
        profitable = expectancy is not None and float(expectancy) > 0 and profit_factor is not None and float(profit_factor) > 1.0
        window_rows.append({
            "window": number,
            "start_utc": window.index.min().isoformat() if len(window) else None,
            "end_utc": window.index.max().isoformat() if len(window) else None,
            "metrics": metrics,
            "enough_trades": bool(enough),
            "profitable": bool(profitable),
            "pass": bool(enough and profitable and drawdown is not None and float(drawdown) <= max_drawdown_filter),
        })

    enough_windows = [row for row in window_rows if row["enough_trades"]]
    positive_windows = sum(1 for row in enough_windows if row["profitable"])
    passing_windows = sum(1 for row in enough_windows if row["pass"])
    chronological_windows_pass = (
        len(enough_windows) >= 2
        and passing_windows >= int(np.ceil(len(enough_windows) * 2 / 3))
    )
    window_expectancies = [
        float(row["metrics"]["expectancy_r"])
        for row in enough_windows if row["metrics"].get("expectancy_r") is not None
    ]

    stress_rows: list[dict[str, object]] = []
    for multiplier in (1.5, 2.0):
        stressed = replace(
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
        stress_frame = validation_frame.copy()
        if "spread_pips" in stress_frame.columns:
            # Honour observed per-bar spreads and widen those observations too;
            # otherwise the stress run would silently leave that cost unchanged.
            stress_frame["spread_pips"] = stress_frame["spread_pips"].astype(float) * multiplier
        metrics, _, _ = _run_genome(
            stress_frame, genome, execution=stressed, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
        expectancy = metrics.get("expectancy_r")
        pf = metrics.get("profit_factor")
        stress_rows.append({
            "cost_multiplier": multiplier,
            "metrics": metrics,
            "positive_expectancy": bool(expectancy is not None and float(expectancy) > 0),
            "profit_factor_above_one": bool(pf is not None and float(pf) > 1.0),
        })

    _, validation_trades, _ = _run_genome(
        validation_frame, genome, execution=execution, risk=risk,
        pip_size=pip_size, price_basis=price_basis,
    )
    bootstrap = block_bootstrap_expectancy(validation_trades, seed=2026)
    stress_15 = stress_rows[0]["metrics"] if stress_rows else {}
    stress_15_expectancy = stress_15.get("expectancy_r")
    stress_15_pf = stress_15.get("profit_factor")
    stress_15_pass = (
        stress_15_expectancy is not None and float(stress_15_expectancy) > 0
        and stress_15_pf is not None and float(stress_15_pf) > 1.0
    )

    return {
        "chronological_validation_windows": window_rows,
        "chronological_windows_with_minimum_trades": len(enough_windows),
        "chronological_positive_windows": positive_windows,
        "chronological_passing_windows": passing_windows,
        "chronological_windows_pass": bool(chronological_windows_pass),
        "median_window_expectancy_r": float(np.median(window_expectancies)) if window_expectancies else None,
        "cost_stress": stress_rows,
        "one_and_half_x_cost_pass": bool(stress_15_pass),
        "expectancy_block_bootstrap": bootstrap,
        "bootstrap_warning": (
            "Descriptive moving-block interval only. It does not adjust for candidate selection, "
            "multiple testing, regime change or feed/execution uncertainty."
        ),
    }


def research_evolutionary_timeframe(
    raw_frame: pd.DataFrame,
    *,
    timeframe: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    research: ResearchConfig,
    evolution: EvolutionSettings,
    pip_size: float,
    price_basis: str,
    seed_genomes: Iterable[StrategyGenome] = (),
    run_final_test: bool = True,
) -> dict[str, object]:
    """Discover genomes and test robustness on validation only before final holdout."""
    featured = add_morphology_features(add_features(raw_frame))
    n = len(featured)
    contaminated_mask = (
        featured["edge_gap_contaminated"].fillna(False).astype(bool)
        if "edge_gap_contaminated" in featured.columns
        else pd.Series(False, index=featured.index)
    )
    gap_diagnostics = {
        "contaminated_bars": int(contaminated_mask.sum()),
        "usable_bars": int((~contaminated_mask).sum()),
        "usable_fraction": float((~contaminated_mask).mean()) if n else 0.0,
        **gap_annotation_summary(featured),
    }
    train_end = int(n * research.train_fraction)
    validation_end = int(n * (research.train_fraction + research.validation_fraction))
    if train_end < 300 or validation_end - train_end < 100 or n - validation_end < 100:
        return {
            "timeframe": timeframe, "status": "insufficient_data", "bars": n,
            "gap_diagnostics": gap_diagnostics,
            "reason": "Need sufficient bars in training, validation and final chronological partitions.",
            "winner": None, "evolution": None, "_training_archive": [],
        }

    # The morphology prototypes are fitted on the training prefix only.
    featured, pattern_model = fit_pattern_model(
        featured, fit_end=train_end, n_clusters=evolution.pattern_clusters,
        random_state=evolution.seed,
    )
    featured = add_causal_motif_dynamics(featured, pattern_model)
    train_frame = featured.iloc[:train_end].copy()
    validation_frame = featured.iloc[train_end:validation_end].copy()
    test_frame = featured.iloc[validation_end:].copy()
    usable_features = available_features(train_frame)
    if len(usable_features) < 4:
        return {
            "timeframe": timeframe, "status": "insufficient_features", "bars": n,
            "gap_diagnostics": gap_diagnostics,
            "train_bars": len(train_frame), "validation_bars": len(validation_frame),
            "test_bars": len(test_frame), "usable_feature_count": len(usable_features),
            "usable_features": list(usable_features), "pattern_model": pattern_model.to_dict(),
            "reason": "Fewer than four usable training features remain after gap-aware feature calculation.",
            "winner": None, "evolution": None, "_training_archive": [],
        }

    allowed_features = set(usable_features)
    usable_seed_genomes = [
        genome for genome in seed_genomes
        if {condition.feature for condition in genome.long_conditions + genome.short_conditions}.issubset(allowed_features)
        and genome.long_conditions and genome.short_conditions
    ]
    min_train_trades = research.minimum_train_trades

    def evaluate_training(genome: StrategyGenome) -> dict[str, object]:
        metrics, trades, _ = _run_genome(
            train_frame, genome, execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
        fitness, descriptor = fitness_from_metrics(metrics, trades, len(train_frame), genome, min_train_trades)
        return {"fitness": fitness, "behavior": descriptor, "metrics": metrics}

    seeds = list(seed_genomes)
    candidates, evolution_report = evolve_population(
        train_frame, evaluate=evaluate_training, settings=evolution, seed_genomes=usable_seed_genomes,
    )
    # The persistent memory receives only top genomes from training fitness.
    training_archive = [genome.to_dict() for _, genome, _ in candidates[:50]]
    candidate_pool = candidates[: evolution.validation_pool]

    candidate_rows: list[dict[str, object]] = []
    eligible: list[tuple[float, StrategyGenome, dict[str, object]]] = []
    min_window_trades = max(8, research.minimum_validation_trades // 4)
    for train_fitness, genome, train_eval in candidate_pool:
        validation_metrics, validation_trades, _ = _run_genome(
            validation_frame, genome, execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
        validation_count = int(validation_metrics.get("trades") or 0)
        validation_expectancy = validation_metrics.get("expectancy_r")
        validation_pf = validation_metrics.get("profit_factor")
        validation_dd = validation_metrics.get("max_realized_drawdown_pct")
        robustness = _validation_robustness(
            validation_frame, genome, execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
            min_window_trades=min_window_trades,
            max_drawdown_filter=research.max_drawdown_filter,
        )
        base_pass = (
            int(train_eval["metrics"].get("trades") or 0) >= research.minimum_train_trades
            and validation_count >= research.minimum_validation_trades
            and validation_expectancy is not None and float(validation_expectancy) > 0
            and validation_pf is not None and float(validation_pf) >= 1.05
            and validation_dd is not None and float(validation_dd) <= research.max_drawdown_filter
        )
        passed = bool(
            base_pass
            and robustness["chronological_windows_pass"]
            and robustness["one_and_half_x_cost_pass"]
        )
        val_score = _selection_score(validation_metrics, genome)
        window_median = robustness.get("median_window_expectancy_r")
        stress_15_metrics = robustness["cost_stress"][0]["metrics"]
        stress_exp = stress_15_metrics.get("expectancy_r")
        # A modest stability adjustment; all components are validation-only.
        if np.isfinite(val_score):
            val_score += 0.20 * (float(window_median) if window_median is not None else 0.0)
            val_score += 0.20 * (float(stress_exp) if stress_exp is not None else -0.25)
        row = {
            "genome_id": genome.fingerprint(),
            "genome": genome.to_dict(),
            "training_fitness": train_fitness,
            "training_metrics": train_eval["metrics"],
            "validation_metrics": validation_metrics,
            "validation_robustness": robustness,
            "validation_score": val_score if pd.notna(val_score) and val_score != float("-inf") else None,
            "base_validation_pass": bool(base_pass),
            "eligible": passed,
        }
        candidate_rows.append(row)
        if passed:
            eligible.append((val_score, genome, row))

    if not eligible:
        return {
            "timeframe": timeframe,
            "status": "no_robust_candidate",
            "bars": n,
            "gap_diagnostics": gap_diagnostics,
            "train_bars": len(train_frame),
            "validation_bars": len(validation_frame),
            "test_bars": len(test_frame),
            "reason": (
                "No candidate passed base validation, multi-window validation and 1.5x cost stress. "
                "Do not weaken gates to force a winner."
            ),
            "pattern_model": pattern_model.to_dict(),
            "usable_feature_count": len(usable_features),
            "usable_features": list(usable_features),
            "evolution": evolution_report,
            "candidates": candidate_rows,
            "winner": None,
            "final_test_skipped": not run_final_test,
            "final_test_reason": "No eligible candidate was selected; no final test was run." if not run_final_test else None,
            "_training_archive": training_archive,
        }

    eligible.sort(key=lambda item: item[0], reverse=True)
    _, winner, winning_row = eligible[0]

    if run_final_test:
        # Final chronological test: run once only after candidate and robustness gates are frozen.
        test_metrics, test_trades, test_equity = _run_genome(
            test_frame, winner, execution=execution, risk=risk,
            pip_size=pip_size, price_basis=price_basis,
        )
        test_passed = (
            int(test_metrics.get("trades") or 0) >= research.minimum_validation_trades
            and test_metrics.get("expectancy_r") is not None and float(test_metrics["expectancy_r"]) > 0
            and test_metrics.get("profit_factor") is not None and float(test_metrics["profit_factor"]) > 1.0
            and test_metrics.get("max_realized_drawdown_pct") is not None
            and float(test_metrics["max_realized_drawdown_pct"]) <= research.max_drawdown_filter
        )
        status = "test_passed" if test_passed else "test_failed_or_inconclusive"
        test_payload: dict[str, object] = {
            "final_test": test_metrics,
            "test_trades": test_trades.to_dict(orient="records"),
            "test_equity": test_equity.to_dict(orient="records"),
        }
    else:
        status = "candidate_selected_research_only"
        test_payload = {
            "final_test": None,
            "test_trades": [],
            "test_equity": [],
            "final_test_skipped": True,
            "final_test_reason": "Research-only mode: final chronological holdout was not evaluated.",
        }

    return {
        "timeframe": timeframe,
        "status": status,
        "bars": n,
        "gap_diagnostics": gap_diagnostics,
        "train_bars": len(train_frame),
        "validation_bars": len(validation_frame),
        "test_bars": len(test_frame),
        "winner": winner.to_dict(),
        "winner_id": winner.fingerprint(),
        "winner_training_metrics": winning_row["training_metrics"],
        "winner_validation_metrics": winning_row["validation_metrics"],
        "winner_validation_robustness": winning_row["validation_robustness"],
        **test_payload,
        "pattern_model": pattern_model.to_dict(),
        "usable_feature_count": len(usable_features),
        "usable_features": list(usable_features),
        "evolution": evolution_report,
        "candidates": candidate_rows,
        "_training_archive": training_archive,
    }


def run_evolutionary_research(
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
    memory_file: str | Path | None = None,
    run_final_test: bool = True,
) -> dict[str, object]:
    validate_source_timeframe(source_timeframe, timeframes)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_frame = annotate_gaps(source_frame, TIMEFRAME_SECONDS[source_timeframe.upper()])
    source_quality = data_quality_summary(source_frame)
    source_gap_audit = build_gap_audit(source_frame, TIMEFRAME_SECONDS[source_timeframe.upper()])
    source_gap_audit_path = output / "source_gap_audit.csv"
    source_gap_audit.to_csv(source_gap_audit_path, index=False)

    results: list[dict[str, object]] = []
    memory_updates: list[dict[str, object]] = []
    for timeframe in timeframes:
        bars = resample_ohlc(source_frame, timeframe)
        # Read stored genotypes only. The timeframe research function filters
        # them against the actually available training features after fitting
        # its one training-prefix pattern model.
        seeds = load_genome_memory(
            memory_file, symbol=symbol, timeframe=timeframe, allowed_features=None,
        )
        result = research_evolutionary_timeframe(
            bars, timeframe=timeframe, execution=execution, risk=risk,
            research=research, evolution=evolution, pip_size=pip_size,
            price_basis=price_basis, seed_genomes=seeds,
            run_final_test=run_final_test,
        )
        training_archive_payload = result.pop("_training_archive", [])
        memory_result = update_genome_memory(
            memory_file,
            symbol=symbol,
            timeframe=timeframe,
            genomes=[genome_from_dict(item) for item in training_archive_payload],
        ) if memory_file is not None and training_archive_payload else {
            "enabled": bool(memory_file), "stored": 0,
            "reason": "No training archive available for this timeframe." if memory_file else "disabled",
        }
        memory_updates.append({"timeframe": timeframe, **memory_result, "loaded_seed_genomes": len(seeds)})
        result["evolutionary_memory"] = memory_updates[-1]
        results.append(result)

    report = {
        "project": "Project EDGE",
        "version": "0.5.0",
        "engine": "adaptive evolutionary pattern-recognition research engine",
        "symbol": symbol.upper(),
        "source_timeframe": source_timeframe.upper(),
        "timeframes": [tf.upper() for tf in timeframes],
        "research_warning": (
            "Evolutionary search tests many candidate genomes. Validation is used for candidate selection "
            "and is subject to model-selection bias. Moving-block bootstrap intervals are descriptive only. "
            "The previous v0.3.1 holdout has already been inspected; rerunning on the same file does not create "
            "a new independent test. Use --research-only until genuinely new data are available. No live-trading claim is made."
        ),
        "evolution_settings": {
            "population_size": evolution.population_size,
            "generations": evolution.generations,
            "pattern_clusters": evolution.pattern_clusters,
            "validation_pool": evolution.validation_pool,
            "seed": evolution.seed,
            "adaptive_mutation": evolution.adaptive_mutation,
            "initial_mutation_rate": evolution.mutation_rate,
        },
        "pattern_recognition": {
            "model": "training-prefix-only k-means morphology prototypes",
            "causal_dynamics": ["motif_transition_surprise", "motif_run_length", "motif_prior_frequency"],
            "dynamics_policy": "current motif state and prior transitions only; update happens after feature calculation",
        },
        "validation_design": {
            "candidate_selection": "training fitness plus base validation, three ordered windows of the same frozen genome inside the validation partition, and 1.5x execution-cost stress; this is not nested walk-forward retraining",
            "cost_stress_multipliers": [1.5, 2.0],
            "bootstrap": "moving-block bootstrap, 500 repetitions, block length 5 trades, 90% descriptive interval",
            "final_test_run": bool(run_final_test),
            "independent_holdout_warning": "Previously inspected v0.3.1 final-test period is consumed and is not an independent v0.4 test.",
        },
        "evolutionary_memory": {
            "file": str(Path(memory_file).resolve()) if memory_file is not None else None,
            "semantics": "Stores genomes selected by training-only fitness. Historical validation/test metrics are never stored or reused; memory genotypes are re-evaluated on each new training partition.",
            "updates": memory_updates,
        },
        "source_data_quality": source_quality,
        "source_gap_audit": {
            "file": source_gap_audit_path.name,
            "events": int(len(source_gap_audit)),
            "categories": source_gap_audit["category"].value_counts().sort_index().to_dict() if not source_gap_audit.empty else {},
        },
        "gap_handling": {
            "gap_threshold": "source interval * 1.5",
            "no_synthetic_bars": True,
            "raw_source_bars": "observed OHLC path is processed before an open position exits across an edge gap using adverse known endpoints",
            "higher_timeframe_materiality": "source gap is material when at least 25% of target candle duration",
            "feature_state": "resampled indicators reset only after gaps of at least 50 target bars",
            "limitations": "Missing intragap prices are unknown; gap exits are stress assumptions rather than guaranteed execution prices.",
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
            "quote_side_model": "Actual ask OHLC used where present; otherwise bid OHLC ask side is approximated by configured/per-bar spread." if price_basis == "bid" else "Mid OHLC with round-trip spread deduction.",
            "signal_timing": "signal at candle close, entry no earlier than next candle open",
            "same_bar_stop_target": "stop assumed first",
        },
        "results": results,
    }
    (output / "evolution_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    summary_rows = []
    for result in results:
        test = result.get("final_test") or {}
        winner = result.get("winner") or {}
        robustness = result.get("winner_validation_robustness") or {}
        summary_rows.append({
            "timeframe": result["timeframe"],
            "status": result["status"],
            "winner_id": result.get("winner_id"),
            "test_trades": test.get("trades"),
            "test_expectancy_R": test.get("expectancy_r"),
            "test_profit_factor": test.get("profit_factor"),
            "test_max_realized_drawdown": test.get("max_realized_drawdown_pct"),
            "chronological_windows_pass": robustness.get("chronological_windows_pass"),
            "one_and_half_x_cost_pass": robustness.get("one_and_half_x_cost_pass"),
            "contaminated_bars": (result.get("gap_diagnostics") or {}).get("contaminated_bars"),
            "rule_count": len(winner.get("long_conditions", [])) + len(winner.get("short_conditions", [])),
        })
    pd.DataFrame(summary_rows).to_csv(output / "evolution_timeframe_summary.csv", index=False)

    lines = [
        f"# Project EDGE v0.4 evolutionary research: {symbol.upper()}",
        "",
        "> Experimental research only. A candidate selected on validation is not statistically proven or ready for live deployment.",
        "",
        f"Population: {evolution.population_size} | Generations: {evolution.generations} | Pattern clusters: {evolution.pattern_clusters}",
        f"Adaptive mutation: {evolution.adaptive_mutation} | Final-test run: {run_final_test}",
        "",
        "| Timeframe | Result | Winner ID | Test trades | Validation windows | 1.5x costs | Test expectancy (R) |",
        "|---|---|---|---:|---|---|---:|",
    ]
    for row in summary_rows:
        def fmt(value: object) -> str:
            return "n/a" if value is None else str(value)
        lines.append(
            f"| {row['timeframe']} | {row['status']} | {fmt(row['winner_id'])} | "
            f"{fmt(row['test_trades'])} | {fmt(row['chronological_windows_pass'])} | "
            f"{fmt(row['one_and_half_x_cost_pass'])} | {fmt(row['test_expectancy_R'])} |"
        )
    lines.extend([
        "",
        "## v0.4 additions",
        "",
        "- Canonical genomes remove duplicate and contradictory feature conditions.",
        "- Adaptive mutation responds to stagnation and population diversity.",
        "- Lineage records parent fingerprints and genetic operators.",
        "- Genome memory persists genotypes only; old validation/test scores are never reused.",
        "- Causal motif dynamics describe transition surprise, persistence and prior motif frequency.",
        "- Candidate selection requires base validation, ordered validation windows and 1.5x cost stress.",
        "- Moving-block bootstrap intervals are reported as descriptive uncertainty, not as significance proof.",
        "",
        "## Holdout warning",
        "",
        "The v0.3.1 final-test period was already inspected. Running v0.4 against the same file does not create a new independent test. Use research-only mode until a genuinely new, later data period or independently sourced unseen sample is available.",
        "",
        "## Execution limitations",
        "",
        "OHLC simulation is not tick replay. Actual paired bid/ask OHLC is used when supplied; otherwise ask prices are estimated from spread assumptions. Swap is only represented when broker-specific signed pip-equivalent rollover rates are configured; account-currency financing conversion, latency, partial fills and rejected orders are not modelled.",
    ])
    (output / "evolution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
