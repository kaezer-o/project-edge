from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import BacktestSettings, run_backtest
from .config import ExecutionConfig, RiskConfig, TIMEFRAME_SECONDS
from .data import annotate_gaps, data_quality_summary, resample_ohlc, validate_source_interval
from .evolution import StrategyGenome, generate_genome_signal, genome_from_dict
from .features import add_features
from .metrics import calculate_metrics
from .pattern_recognition import (
    add_causal_motif_dynamics,
    add_morphology_features,
    apply_pattern_model,
    pattern_model_from_dict,
)
from .data import load_ohlc_csv


ASK_OHLC = {"ask_open", "ask_high", "ask_low", "ask_close"}


def _utc_timestamp(value: str, *, inclusive_date_end: bool = False) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    if inclusive_date_end and len(value.strip()) == 10:
        stamp = stamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return stamp


def _extract_candidate(report_path: str | Path, timeframe: str) -> tuple[dict[str, object], StrategyGenome, object]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("Evolution report has no results list")
    result = next((row for row in results if str(row.get("timeframe", "")).upper() == timeframe.upper()), None)
    if result is None:
        raise ValueError(f"Report contains no result for timeframe {timeframe}")
    winner = result.get("winner")
    if not isinstance(winner, dict):
        raise ValueError(f"No frozen winner exists for timeframe {timeframe}")
    pattern_payload = result.get("pattern_model")
    if not isinstance(pattern_payload, dict):
        raise ValueError("Selected result has no serialized pattern model")
    genome = genome_from_dict(winner)
    pattern_model = pattern_model_from_dict(pattern_payload)
    if pattern_model.n_clusters < 2:
        raise ValueError("Frozen candidate depends on a fitted pattern model, but report model is disabled")
    used_features = sorted({
        condition.feature
        for condition in (*genome.long_conditions, *genome.short_conditions)
    })
    # These sequence features depend on transition/frequency counters that
    # were accumulated during the original run. v0.5 preserves the saved
    # morphology centroids but does not serialize those counters, so it must
    # refuse candidates that depend on them rather than silently cold-starting.
    stateful_features = {
        "motif_transition_surprise", "motif_run_length", "motif_prior_frequency"
    }
    unsupported_stateful = sorted(stateful_features.intersection(used_features))
    if unsupported_stateful:
        raise ValueError(
            "Frozen replay cannot yet reproduce the original training-seeded motif state for "
            f"these genome features: {unsupported_stateful}. A future report format must persist "
            "the transition/frequency state before this candidate can be replayed exactly."
        )
    report_meta = {
        "project_version": report.get("version"),
        "symbol": report.get("symbol"),
        "timeframe": timeframe.upper(),
        "candidate_id": result.get("winner_id") or genome.fingerprint(),
        "features_used_by_frozen_genome": used_features,
        "stateful_motif_features_used": unsupported_stateful,
        "status_in_original_report": result.get("status"),
        "genome": genome.to_dict(),
        "pattern_model_status": pattern_model.status,
        "pattern_model_clusters": pattern_model.n_clusters,
        "pattern_model_fit_rows": pattern_model.fit_rows,
        "original_validation_end_utc": max(
            (window.get("end_utc", "") for window in (result.get("winner_validation_robustness") or {}).get("chronological_validation_windows", [])),
            default=None,
        ),
    }
    return report_meta, genome, pattern_model


def replay_frozen_candidate(
    input_path: str | Path,
    report_path: str | Path,
    *,
    timeframe: str = "H4",
    source_timeframe: str = "M1",
    provider_name: str,
    start_utc: str,
    end_utc: str,
    pip_size: float,
    execution: ExecutionConfig,
    risk: RiskConfig,
    output_dir: str | Path,
    minimum_warmup_bars: int = 205,
) -> dict[str, object]:
    """Replay an unchanged genome and its frozen training-only pattern model.

    The model is never refitted. The evaluation period must begin strictly
    after the original report's last chronological validation window.
    """
    timeframe = timeframe.upper()
    source_timeframe = source_timeframe.upper()
    if timeframe not in TIMEFRAME_SECONDS or source_timeframe not in TIMEFRAME_SECONDS:
        raise ValueError("Unsupported timeframe")
    if TIMEFRAME_SECONDS[timeframe] < TIMEFRAME_SECONDS[source_timeframe]:
        raise ValueError("Replay timeframe cannot be lower resolution than source data")
    if pip_size <= 0:
        raise ValueError("pip_size must be > 0")

    metadata, genome, pattern_model = _extract_candidate(report_path, timeframe)
    start = _utc_timestamp(start_utc)
    end = _utc_timestamp(end_utc, inclusive_date_end=True)
    if end <= start:
        raise ValueError("end_utc must be later than start_utc")
    original_validation_end = metadata.get("original_validation_end_utc")
    if original_validation_end:
        boundary = _utc_timestamp(str(original_validation_end))
        if start <= boundary:
            raise ValueError(
                f"Replay start {start.isoformat()} must be after consumed validation end "
                f"{boundary.isoformat()}"
            )

    source = load_ohlc_csv(input_path)
    if not ASK_OHLC.issubset(source.columns):
        raise ValueError(
            "Frozen independent-feed replay requires actual ask-side OHLC columns "
            "(ask_open, ask_high, ask_low, ask_close). Create them from paired bid/ask ticks; "
            "estimated spread-only ask quotes are not accepted for this validation."
        )
    validate_source_interval(source, source_timeframe)
    source = annotate_gaps(source, TIMEFRAME_SECONDS[source_timeframe])
    quality = data_quality_summary(source)

    warmup_start = start - pd.Timedelta(
    	TIMEFRAME_SECONDS[timeframe] * minimum_warmup_bars,
    	unit="s",
    )
    bars = resample_ohlc(source, timeframe)
    warmup_bars = int((bars.index < start).sum())
    if warmup_bars < minimum_warmup_bars:
        raise ValueError(
            f"Only {warmup_bars} {timeframe} warm-up bars precede replay start; "
            f"need at least {minimum_warmup_bars}. Forex weekend closures and source gaps reduce "
            f"the number of bars, so download further back than {warmup_start.isoformat()} "
            "(typically 6-8 calendar weeks before the replay start)."
        )

    featured = add_features(bars)
    featured = add_morphology_features(featured)
    featured = apply_pattern_model(featured, pattern_model)
    featured = add_causal_motif_dynamics(featured, pattern_model)

    evaluation_bars = featured.loc[featured.index <= end].copy()
    if evaluation_bars.empty or not bool((evaluation_bars.index >= start).any()):
        raise ValueError("No bars fall inside requested replay interval")
    signal = generate_genome_signal(evaluation_bars, genome)
    signal.loc[evaluation_bars.index < start] = 0

    trades, equity = run_backtest(
        evaluation_bars,
        signal,
        stop_atr=genome.stop_atr,
        reward_risk=genome.reward_risk,
        execution=execution,
        risk=risk,
        settings=BacktestSettings(
            pip_size=pip_size,
            max_holding_bars=genome.max_holding_bars,
            price_basis="bid",
        ),
    )
    if not trades.empty:
        if (trades["entry_time"] < start).any() or (trades["entry_time"] > end).any():
            raise RuntimeError("Replay generated a trade outside the requested evaluation interval")
    metrics = calculate_metrics(trades, equity)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trades_path = output / "frozen_replay_trades.csv"
    equity_path = output / "frozen_replay_equity.csv"
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)
    replay_report = {
        "project": "Project EDGE",
        "version": "0.5.0",
        "run_type": "frozen_candidate_independent_feed_replay",
        "provider_name_user_supplied": provider_name,
        "provider_independence_verified_by_software": False,
        "source_file": str(Path(input_path).resolve()),
        "source_timeframe": source_timeframe,
        "strategy_timeframe": timeframe,
        "requested_test_start_utc": start.isoformat(),
        "requested_test_end_utc": end.isoformat(),
        "warmup_bars_available": warmup_bars,
        "original_candidate": metadata,
        "frozen_genome_refitted": False,
        "pattern_model_refitted": False,
        "pattern_model_application": "Original report means/scales/centroids applied unchanged; causal transition state computed sequentially from replay feed only.",
        "source_data_quality": quality,
        "actual_bid_ask_ohlc": True,
        "execution_assumptions": {
            "price_basis": "actual bid and ask OHLC",
            "spread_from_quote_sides": True,
            "slippage_pips_per_side": execution.slippage_pips_per_side,
            "commission_round_trip_pips": execution.commission_round_trip_pips,
            "swap_long_pips_per_rollover": execution.swap_long_pips_per_rollover,
            "swap_short_pips_per_rollover": execution.swap_short_pips_per_rollover,
            "rollover_timezone": execution.rollover_timezone,
            "rollover_local_time": execution.rollover_local_time,
            "triple_swap_weekday": execution.triple_swap_weekday,
            "triple_swap_multiplier": execution.triple_swap_multiplier,
        },
        "metrics": metrics,
        "trade_count": int(len(trades)),
        "trades_file": trades_path.name,
        "equity_file": equity_path.name,
        "limitations": [
            "Independent provider identity is supplied by the operator and is not cryptographically verified.",
            "M1 quote OHLC replay is not tick-by-tick order-book execution; intrabar bid/ask event order is unavailable.",
            "Swap inputs must be sourced from the intended broker and instrument. Zero defaults mean no swap was charged.",
            "Broker-specific financing conversion, margin, latency, rejected orders and partial fills are not simulated.",
            "A positive replay result is not proof of future profitability or live execution equivalence.",
        ],
    }
    report_path_out = output / "frozen_replay_report.json"
    report_path_out.write_text(json.dumps(replay_report, indent=2, default=str), encoding="utf-8")
    summary_lines = [
        f"# EDGE v0.5 frozen candidate replay: {provider_name}",
        "",
        f"Candidate: {metadata['candidate_id']} ({timeframe})",
        f"Evaluation: {start.isoformat()} to {end.isoformat()}",
        f"Trades: {len(trades)}",
        f"Expectancy (R): {metrics.get('expectancy_r')}",
        f"Profit factor: {metrics.get('profit_factor')}",
        f"Maximum realized drawdown: {metrics.get('max_realized_drawdown_pct')}",
        "",
        "The genome and training-fitted pattern centroids were held fixed. The source provider label was supplied by the operator and is not independently authenticated.",
        "",
        "This OHLC replay is a research diagnostic, not a live-trading performance claim.",
    ]
    (output / "frozen_replay_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    replay_report["report_file"] = str(report_path_out.resolve())
    return replay_report
