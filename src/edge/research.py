from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import BacktestSettings, run_backtest
from .config import ExecutionConfig, ResearchConfig, RiskConfig, SUPPORTED_TIMEFRAMES
from .data import (
    annotate_gaps, build_gap_audit, data_quality_summary,
    gap_annotation_summary, resample_ohlc, validate_source_timeframe,
)
from .config import TIMEFRAME_SECONDS
from .features import add_features
from .metrics import calculate_metrics
from .strategies import StrategyCandidate, generate_candidates, generate_signal


def _slice(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return frame.iloc[start:end].copy()


def _score(metrics: dict[str, object]) -> float:
    expectancy = metrics.get("expectancy_r")
    trades = int(metrics.get("trades") or 0)
    drawdown = metrics.get("max_realized_drawdown_pct")
    if expectancy is None or drawdown is None or trades == 0:
        return float("-inf")
    # A simple ranking statistic, not a forecast or proof of significance.
    return float(expectancy) * (trades ** 0.5) / (1.0 + 5.0 * float(drawdown))


def _run_segment(
    frame: pd.DataFrame,
    signal: pd.Series,
    candidate: StrategyCandidate,
    execution: ExecutionConfig,
    risk: RiskConfig,
    settings: BacktestSettings,
) -> dict[str, object]:
    trades, equity = run_backtest(
        frame,
        signal,
        stop_atr=candidate.stop_atr,
        reward_risk=candidate.reward_risk,
        execution=execution,
        risk=risk,
        settings=settings,
    )
    metrics = calculate_metrics(trades, equity)
    return {"metrics": metrics, "trades_frame": trades, "equity_frame": equity}


def research_timeframe(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    research: ResearchConfig,
    settings: BacktestSettings,
) -> dict[str, object]:
    featured = add_features(frame)
    gap_diagnostics = gap_annotation_summary(featured)
    n = len(featured)
    train_end = int(n * research.train_fraction)
    validation_end = int(n * (research.train_fraction + research.validation_fraction))
    if train_end < 250 or validation_end - train_end < 100 or n - validation_end < 100:
        return {
            "timeframe": timeframe,
            "status": "insufficient_data",
            "bars": n,
            "gap_diagnostics": gap_diagnostics,
            "reason": "Need more bars in train, validation, and final test partitions.",
            "candidates": [],
            "winner": None,
        }

    train_frame = _slice(featured, 0, train_end)
    validation_frame = _slice(featured, train_end, validation_end)
    test_frame = _slice(featured, validation_end, n)
    candidate_rows: list[dict[str, object]] = []
    eligible: list[tuple[float, StrategyCandidate, pd.Series, dict[str, object]]] = []

    for candidate in generate_candidates():
        full_signal = generate_signal(featured, candidate)
        train_signal = full_signal.loc[train_frame.index]
        validation_signal = full_signal.loc[validation_frame.index]
        train_result = _run_segment(
            train_frame, train_signal, candidate, execution, risk, settings
        )
        validation_result = _run_segment(
            validation_frame, validation_signal, candidate, execution, risk, settings
        )
        train_metrics = train_result["metrics"]
        validation_metrics = validation_result["metrics"]

        train_trades = int(train_metrics["trades"] or 0)
        validation_trades = int(validation_metrics["trades"] or 0)
        train_expectancy = train_metrics["expectancy_r"]
        validation_expectancy = validation_metrics["expectancy_r"]
        validation_pf = validation_metrics["profit_factor"]
        validation_dd = validation_metrics["max_realized_drawdown_pct"]

        passed = (
            train_trades >= research.minimum_train_trades
            and validation_trades >= research.minimum_validation_trades
            and train_expectancy is not None
            and float(train_expectancy) > 0
            and validation_expectancy is not None
            and float(validation_expectancy) > 0
            and validation_pf is not None
            and float(validation_pf) >= 1.05
            and validation_dd is not None
            and float(validation_dd) <= research.max_drawdown_filter
        )
        score = _score(validation_metrics)
        row = {
            "name": candidate.name,
            "family": candidate.family,
            "params": candidate.params,
            "stop_atr": candidate.stop_atr,
            "reward_risk": candidate.reward_risk,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_score": score if score != float("-inf") else None,
            "eligible": bool(passed),
        }
        candidate_rows.append(row)
        if passed:
            eligible.append((score, candidate, full_signal, row))

    if not eligible:
        return {
            "timeframe": timeframe,
            "status": "no_validated_candidate",
            "bars": n,
            "gap_diagnostics": gap_diagnostics,
            "train_bars": len(train_frame),
            "validation_bars": len(validation_frame),
            "test_bars": len(test_frame),
            "reason": (
                "No candidate passed the predeclared train/validation filters. "
                "Do not lower thresholds merely to force a winner."
            ),
            "candidates": candidate_rows,
            "winner": None,
        }

    eligible.sort(key=lambda item: item[0], reverse=True)
    _, winner, winner_signal, winner_validation_row = eligible[0]
    test_signal = winner_signal.loc[test_frame.index]
    test_result = _run_segment(
        test_frame, test_signal, winner, execution, risk, settings
    )
    test_metrics = test_result["metrics"]

    # The final test is a report-only holdout. It is not used to choose or tune.
    status = "test_passed" if (
        int(test_metrics["trades"] or 0) >= research.minimum_validation_trades
        and test_metrics["expectancy_r"] is not None
        and float(test_metrics["expectancy_r"]) > 0
        and test_metrics["profit_factor"] is not None
        and float(test_metrics["profit_factor"]) > 1.0
        and test_metrics["max_realized_drawdown_pct"] is not None
        and float(test_metrics["max_realized_drawdown_pct"]) <= research.max_drawdown_filter
    ) else "test_failed_or_inconclusive"

    return {
        "timeframe": timeframe,
        "status": status,
        "bars": n,
        "gap_diagnostics": gap_diagnostics,
        "train_bars": len(train_frame),
        "validation_bars": len(validation_frame),
        "test_bars": len(test_frame),
        "winner": winner.to_dict(),
        "winner_validation": winner_validation_row["validation"],
        "final_test": test_metrics,
        "test_trades": test_result["trades_frame"].to_dict(orient="records"),
        "test_equity": test_result["equity_frame"].to_dict(orient="records"),
        "candidates": candidate_rows,
    }


def run_research(
    source_frame: pd.DataFrame,
    *,
    source_timeframe: str,
    timeframes: list[str],
    symbol: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    research: ResearchConfig,
    settings: BacktestSettings,
    output_dir: str | Path,
) -> dict[str, object]:
    validate_source_timeframe(source_timeframe, timeframes)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_frame = annotate_gaps(
        source_frame, TIMEFRAME_SECONDS[source_timeframe.upper()]
    )
    source_quality = data_quality_summary(source_frame)
    source_gap_audit = build_gap_audit(
        source_frame, TIMEFRAME_SECONDS[source_timeframe.upper()]
    )
    source_gap_audit_path = output / "source_gap_audit.csv"
    source_gap_audit.to_csv(source_gap_audit_path, index=False)

    results: list[dict[str, object]] = []
    for timeframe in timeframes:
        bars = resample_ohlc(source_frame, timeframe)
        result = research_timeframe(
            bars,
            timeframe=timeframe,
            execution=execution,
            risk=risk,
            research=research,
            settings=settings,
        )
        results.append(result)

    report = {
        "project": "Project EDGE",
        "symbol": symbol.upper(),
        "source_timeframe": source_timeframe.upper(),
        "timeframes": [timeframe.upper() for timeframe in timeframes],
        "source_data_quality": source_quality,
        "source_gap_audit": {
            "file": source_gap_audit_path.name,
            "events": int(len(source_gap_audit)),
            "categories": (
                source_gap_audit["category"].value_counts().sort_index().to_dict()
                if not source_gap_audit.empty else {}
            ),
        },
        "gap_handling": {
            "gap_threshold": "source interval * 1.5",
            "no_synthetic_bars": True,
            "raw_source_bars": "valid OHLC path processed, then open positions exit across edge_gap_after using adverse close/open endpoint",
            "resampled_gap_touching_bars": "excluded from features and new entries; open trades use the worst observed contaminated-bar extreme and clean endpoints",
            "indicator_state": "rolling and exponential features restart after non-weekend source gaps; normal weekend closures do not reset state",
            "limitation": "Missing intra-gap prices are unknowable; gap exits are explicitly conservative approximations, not recoverable fills.",
        },
        "assumptions": {
            "price_basis": settings.price_basis,
            "prices": (
                "Bid OHLC; explicit ask OHLC preferred. If absent, ask OHLC is approximated from configured/per-bar spread. Longs enter ask and exit bid; shorts enter bid and exit ask."
                if settings.price_basis == "bid" else "Mid-price OHLC"
            ),
            "signal_timing": "Signal at candle close; entry no earlier than next candle open",
            "same_bar_stop_target": "Stop is assumed to occur first",
            "costs": {
                "spread_model": "Actual ask OHLC used where present; otherwise estimated from configured/per-bar spread for bid basis.",
                "slippage_pips_per_side": execution.slippage_pips_per_side,
                "commission_round_trip_pips": execution.commission_round_trip_pips,
                "swap_long_pips_per_rollover": execution.swap_long_pips_per_rollover,
                "swap_short_pips_per_rollover": execution.swap_short_pips_per_rollover,
                "rollover_timezone": execution.rollover_timezone,
                "rollover_local_time": execution.rollover_local_time,
                "triple_swap_weekday": execution.triple_swap_weekday,
                "triple_swap_multiplier": execution.triple_swap_multiplier,
            },
            "account_currency": (
                "Equity compounding is an R-multiple approximation. Broker-specific "
                "contract size and quote-to-account currency conversion are not yet modeled."
            ),
            "final_test": "One untouched chronological holdout; never used for selection",
        },
        "results": results,
    }

    json_path = output / "research_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    summary_rows = []
    for result in results:
        winner = result.get("winner") or {}
        test = result.get("final_test") or {}
        summary_rows.append(
            {
                "timeframe": result["timeframe"],
                "status": result["status"],
                "winner": winner.get("name"),
                "test_trades": test.get("trades"),
                "test_expectancy_R": test.get("expectancy_r"),
                "test_profit_factor": test.get("profit_factor"),
                "test_max_realized_drawdown": test.get("max_realized_drawdown_pct"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output / "timeframe_summary.csv", index=False)

    lines = [
        f"# Project EDGE research report: {symbol.upper()}",
        "",
        "> Research output, not a promise of future profitability or a live trading recommendation.",
        "",
        f"Source timeframe: `{source_timeframe.upper()}`",
        f"Evaluated timeframes: {', '.join(timeframes)}",
        "",
        "| Timeframe | Result | Candidate | Test trades | Test expectancy (R) | Profit factor | Max realized drawdown |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        def fmt(value: object) -> str:
            return "n/a" if value is None else str(value)
        lines.append(
            "| {timeframe} | {status} | {winner} | {trades} | {expectancy} | {pf} | {dd} |".format(
                timeframe=row["timeframe"],
                status=row["status"],
                winner=fmt(row["winner"]),
                trades=fmt(row["test_trades"]),
                expectancy=fmt(row["test_expectancy_R"]),
                pf=fmt(row["test_profit_factor"]),
                dd=fmt(row["test_max_realized_drawdown"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`test_passed` means the selected candidate met the configured minimum holdout checks. "
            "It does not establish a durable edge. `test_failed_or_inconclusive` means the final "
            "holdout did not meet those checks. `no_validated_candidate` means no candidate passed "
            "the predeclared train/validation filters.",
            "",
            "Do not deploy this output to a live account. Verify it with broker-specific bid/ask "
            "data, instrument contract specifications, realistic execution, and forward paper trading.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
