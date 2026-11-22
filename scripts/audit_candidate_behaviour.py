from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from edge.config import ExecutionConfig, RiskConfig, TIMEFRAME_SECONDS
from edge.data import annotate_gaps, load_ohlc_csv, resample_ohlc
from edge.features import add_features
from edge.program import StrategyProgram, evaluate_program
from edge.program_research import _evaluate


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def signal_similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    """Compare aligned directional signals; report active overlap separately."""
    a = np.asarray(left, dtype=np.int8)
    b = np.asarray(right, dtype=np.int8)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Signal arrays must be one-dimensional and identically sized")
    long_a, long_b = set(np.flatnonzero(a == 1)), set(np.flatnonzero(b == 1))
    short_a, short_b = set(np.flatnonzero(a == -1)), set(np.flatnonzero(b == -1))
    active_a, active_b = set(np.flatnonzero(a != 0)), set(np.flatnonzero(b != 0))
    directional_conflicts = int(np.count_nonzero(((a == 1) & (b == -1)) | ((a == -1) & (b == 1))))
    n = len(a)
    return {
        "bars": n,
        "exact_signal_agreement": float(np.mean(a == b)) if n else 1.0,
        "active_signal_jaccard": _jaccard(active_a, active_b),
        "long_signal_jaccard": _jaccard(long_a, long_b),
        "short_signal_jaccard": _jaccard(short_a, short_b),
        "directional_conflict_rate": directional_conflicts / n if n else 0.0,
        "active_bars_a": len(active_a),
        "active_bars_b": len(active_b),
    }


def trade_entry_events(trades: pd.DataFrame) -> set[str]:
    if trades.empty:
        return set()
    required = {"entry_time", "direction"}
    if not required.issubset(trades.columns):
        raise ValueError("Backtest trade frame is missing entry_time or direction")
    timestamps = pd.to_datetime(trades["entry_time"], utc=True)
    return {
        f"{timestamp.isoformat()}|{str(direction).lower()}"
        for timestamp, direction in zip(timestamps, trades["direction"])
    }


def trade_overlap(left: set[str], right: set[str]) -> dict[str, int | float]:
    shared = left & right
    union = left | right
    return {
        "entries_a": len(left),
        "entries_b": len(right),
        "shared_entries": len(shared),
        "union_entries": len(union),
        "entry_jaccard": len(shared) / len(union) if union else 0.0,
    }


def _rule_fingerprint(program: StrategyProgram) -> str:
    payload = {
        "long_entry": program.long_entry.to_dict(),
        "short_entry": program.short_entry.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prepare_train_validation(
    raw_frame: pd.DataFrame, *, timeframe: str, source_timeframe: str
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Mirror the research partition while never slicing/evaluating the final holdout."""
    target = timeframe.upper()
    source = source_timeframe.upper()
    if target not in TIMEFRAME_SECONDS or source not in TIMEFRAME_SECONDS:
        raise ValueError("Unsupported timeframe")
    if TIMEFRAME_SECONDS[target] < TIMEFRAME_SECONDS[source]:
        raise ValueError("Target timeframe cannot be lower than source timeframe")
    annotated = annotate_gaps(raw_frame, TIMEFRAME_SECONDS[source])
    market = annotated if target == source else resample_ohlc(annotated, target)
    featured = add_features(market)
    count = len(featured)
    train_end = int(count * 0.60)
    validation_end = int(count * 0.80)
    if train_end < 300 or validation_end - train_end < 100 or count - validation_end < 100:
        raise ValueError("Insufficient bars for chronological train/validation/test partitions")
    return featured.iloc[:train_end].copy(), featured.iloc[train_end:validation_end].copy(), count - validation_end


def _candidate_behaviour(
    seed: int,
    rank: int,
    candidate: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
) -> dict[str, Any]:
    program_data = candidate.get("program")
    if not isinstance(program_data, dict):
        raise ValueError(f"Seed {seed} candidate {rank} has no program")
    program = StrategyProgram.from_dict(program_data)

    train_signal = evaluate_program(program, train).to_numpy(dtype=np.int8, copy=False)
    validation_signal = evaluate_program(program, validation).to_numpy(dtype=np.int8, copy=False)
    train_metrics, train_trades = _evaluate(
        train, program, execution=execution, risk=risk, pip_size=pip_size,
        price_basis=price_basis, signal=pd.Series(train_signal, index=train.index),
    )
    validation_metrics, validation_trades = _evaluate(
        validation, program, execution=execution, risk=risk, pip_size=pip_size,
        price_basis=price_basis, signal=pd.Series(validation_signal, index=validation.index),
    )
    return {
        "seed": seed,
        "rank": rank,
        "validation_gate_pass": bool((candidate.get("validation") or {}).get("pass", False)),
        "rule_fingerprint": _rule_fingerprint(program),
        "risk_parameters": {
            "stop_atr": program.stop_atr,
            "reward_risk": program.reward_risk,
            "max_holding_bars": program.max_holding_bars,
        },
        "training": {
            "metrics": train_metrics,
            "signal": train_signal,
            "trade_events": trade_entry_events(train_trades),
        },
        "validation": {
            "metrics": validation_metrics,
            "signal": validation_signal,
            "trade_events": trade_entry_events(validation_trades),
        },
    }


def build_behaviour_audit(
    reports: list[tuple[int, dict[str, Any]]],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("At least two seed reports are required")
    seeds = [seed for seed, _ in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed identifiers must be unique")
    reference = reports[0][1]
    expected_data = reference.get("data", {})
    for seed, report in reports:
        if report.get("data", {}).get("final_test_evaluated") is not False:
            raise ValueError(f"Seed {seed} indicates final-test evaluation; audit stopped")
        if (
            report.get("symbol") != reference.get("symbol")
            or report.get("timeframe") != reference.get("timeframe")
            or report.get("source_timeframe") != reference.get("source_timeframe")
        ):
            raise ValueError(f"Seed {seed} has different symbol/timeframe metadata")
        if report.get("data") != expected_data:
            raise ValueError(f"Seed {seed} has different data partition metadata")
    if int(expected_data.get("train_bars", -1)) != len(train):
        raise ValueError("Input data does not match the reports' training partition size")
    if int(expected_data.get("validation_bars", -1)) != len(validation):
        raise ValueError("Input data does not match the reports' validation partition size")

    candidate_results: list[dict[str, Any]] = []
    internal: list[dict[str, Any]] = []
    for seed, report in reports:
        candidates = report.get("candidates") or []
        if not candidates:
            raise ValueError(f"Seed {seed} contains no finalists")
        for rank, candidate in enumerate(candidates, start=1):
            result = _candidate_behaviour(
                seed, rank, candidate, train, validation,
                execution=execution, risk=risk, pip_size=pip_size,
                price_basis=price_basis,
            )
            internal.append(result)
            candidate_results.append({
                "seed": seed,
                "rank": rank,
                "validation_gate_pass": result["validation_gate_pass"],
                "rule_fingerprint": result["rule_fingerprint"],
                "risk_parameters": result["risk_parameters"],
                "training": {
                    "trades": int(result["training"]["metrics"].get("trades", 0)),
                    "signal_activity_rate": float(np.mean(result["training"]["signal"] != 0)),
                    "long_signal_rate": float(np.mean(result["training"]["signal"] == 1)),
                    "short_signal_rate": float(np.mean(result["training"]["signal"] == -1)),
                },
                "validation": {
                    "trades": int(result["validation"]["metrics"].get("trades", 0)),
                    "signal_activity_rate": float(np.mean(result["validation"]["signal"] != 0)),
                    "long_signal_rate": float(np.mean(result["validation"]["signal"] == 1)),
                    "short_signal_rate": float(np.mean(result["validation"]["signal"] == -1)),
                },
            })

    pairs: list[dict[str, Any]] = []
    for left, right in combinations(internal, 2):
        if left["seed"] == right["seed"]:
            continue
        row: dict[str, Any] = {
            "seed_a": left["seed"], "rank_a": left["rank"],
            "seed_b": right["seed"], "rank_b": right["rank"],
            "same_entry_rules": left["rule_fingerprint"] == right["rule_fingerprint"],
            "training_signal_similarity": signal_similarity(
                left["training"]["signal"], right["training"]["signal"]
            ),
            "validation_signal_similarity": signal_similarity(
                left["validation"]["signal"], right["validation"]["signal"]
            ),
            "training_trade_overlap": trade_overlap(
                left["training"]["trade_events"], right["training"]["trade_events"]
            ),
            "validation_trade_overlap": trade_overlap(
                left["validation"]["trade_events"], right["validation"]["trade_events"]
            ),
        }
        pairs.append(row)

    def ranked(field: str, metric: str, limit: int = 30) -> list[dict[str, Any]]:
        return sorted(
            pairs,
            key=lambda row: (
                -float(row[field][metric]),
                -int(row[field].get("shared_entries", 0)),
                row["seed_a"], row["rank_a"], row["seed_b"], row["rank_b"],
            ),
        )[:limit]

    return {
        "audit_version": 1,
        "scope": {
            "seeds": seeds,
            "symbol": reference.get("symbol"),
            "timeframe": reference.get("timeframe"),
            "source_timeframe": reference.get("source_timeframe"),
            "candidate_count": len(candidate_results),
            "cross_seed_pair_count": len(pairs),
            "training_bars": len(train),
            "validation_bars": len(validation),
            "final_test_evaluated": False,
            "selection_or_validation_gates_modified": False,
            "price_basis": price_basis,
            "execution": execution.__dict__,
            "risk": risk.__dict__,
            "pip_size": pip_size,
        },
        "candidate_results": candidate_results,
        "cross_seed_pairs": pairs,
        "top_training_active_signal_similarity": ranked(
            "training_signal_similarity", "active_signal_jaccard"
        ),
        "top_validation_active_signal_similarity": ranked(
            "validation_signal_similarity", "active_signal_jaccard"
        ),
        "top_training_trade_overlap": ranked("training_trade_overlap", "entry_jaccard"),
        "top_validation_trade_overlap": ranked("validation_trade_overlap", "entry_jaccard"),
        "limitations": [
            "Exact signal agreement can be inflated by bars where both candidates emit no signal; active-signal Jaccard and directional overlaps must be read alongside it.",
            "Trade-entry overlap depends on the supplied execution and risk assumptions. The source data is bid-only, so configured spread is an estimate rather than observed ask-side history.",
            "Validation has been repeatedly examined during development. Validation similarities are descriptive and are not independent performance evidence.",
            "The final chronological holdout is not evaluated by this script.",
            "No similarity result changes fitness, selection, migration, validation gates or final-test access.",
            "Behavioural similarity is historical and does not establish profitability, future stability or live execution quality.",
        ],
    }


def _parse_report(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        return int(seed_text), Path(path_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Report must use SEED=PATH format") from exc


def _write_markdown(audit: dict[str, Any], path: Path) -> None:
    lines = [
        "# Candidate behavioural similarity audit", "",
        f"- Symbol/timeframe: {audit['scope']['symbol']} / {audit['scope']['timeframe']}",
        f"- Seeds: {', '.join(map(str, audit['scope']['seeds']))}",
        f"- Final test evaluated: no",
        f"- Candidates: {audit['scope']['candidate_count']}",
        f"- Cross-seed pairs: {audit['scope']['cross_seed_pair_count']}", "",
        "## Highest training active-signal overlap", "",
        "| Seed/rank A | Seed/rank B | Active Jaccard | Long Jaccard | Short Jaccard | Exact agreement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in audit["top_training_active_signal_similarity"][:15]:
        m = row["training_signal_similarity"]
        lines.append(
            f"| {row['seed_a']}/{row['rank_a']} | {row['seed_b']}/{row['rank_b']} | "
            f"{m['active_signal_jaccard']:.3f} | {m['long_signal_jaccard']:.3f} | "
            f"{m['short_signal_jaccard']:.3f} | {m['exact_signal_agreement']:.3f} |"
        )
    lines.extend([
        "", "## Highest validation active-signal overlap", "",
        "| Seed/rank A | Seed/rank B | Active Jaccard | Long Jaccard | Short Jaccard | Exact agreement |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in audit["top_validation_active_signal_similarity"][:15]:
        m = row["validation_signal_similarity"]
        lines.append(
            f"| {row['seed_a']}/{row['rank_a']} | {row['seed_b']}/{row['rank_b']} | "
            f"{m['active_signal_jaccard']:.3f} | {m['long_signal_jaccard']:.3f} | "
            f"{m['short_signal_jaccard']:.3f} | {m['exact_signal_agreement']:.3f} |"
        )
    for title, key in (
        ("Highest training trade-entry overlap", "top_training_trade_overlap"),
        ("Highest validation trade-entry overlap", "top_validation_trade_overlap"),
    ):
        lines.extend(["", f"## {title}", "",
            "| Seed/rank A | Seed/rank B | Shared | Union | Entry Jaccard |",
            "|---|---|---:|---:|---:|"])
        for row in audit[key][:15]:
            m = row["training_trade_overlap" if "training" in key else "validation_trade_overlap"]
            lines.append(
                f"| {row['seed_a']}/{row['rank_a']} | {row['seed_b']}/{row['rank_b']} | "
                f"{m['shared_entries']} | {m['union_entries']} | {m['entry_jaccard']:.3f} |"
            )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in audit["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay finalist programs and compare historical signal/trade behaviour across seeds."
    )
    parser.add_argument("--input", required=True, help="same source OHLC CSV used for the evolutionary runs")
    parser.add_argument("--report", action="append", type=_parse_report, required=True, help="SEED=REPORT_JSON; repeat per seed")
    parser.add_argument("--output", default="results/candidate-behaviour-audit.json")
    parser.add_argument("--price-basis", choices=("bid", "mid"), default="bid")
    parser.add_argument("--pip-size", type=float, default=0.0001)
    parser.add_argument("--spread-pips", type=float, default=0.8)
    parser.add_argument("--slippage-pips", type=float, default=0.1)
    parser.add_argument("--commission-pips", type=float, default=0.0)
    parser.add_argument("--swap-long-pips", type=float, default=0.0)
    parser.add_argument("--swap-short-pips", type=float, default=0.0)
    parser.add_argument("--rollover-timezone", default="America/New_York")
    parser.add_argument("--rollover-local-time", default="17:00")
    parser.add_argument("--triple-swap-weekday", type=int, choices=range(7), default=None)
    parser.add_argument("--triple-swap-multiplier", type=float, default=3.0)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.0025)
    parser.add_argument("--max-daily-loss", type=float, default=0.01)
    parser.add_argument("--max-trades-per-day", type=int, default=4)
    args = parser.parse_args()

    reports = [(seed, json.loads(path.read_text(encoding="utf-8"))) for seed, path in args.report]
    reference = reports[0][1]
    timeframe = str(reference["timeframe"])
    source_timeframe = str(reference["source_timeframe"])
    train, validation, final_test_bar_count = _prepare_train_validation(
        load_ohlc_csv(args.input),
        timeframe=timeframe,
        source_timeframe=source_timeframe,
    )
    expected_test_bars = int(reference.get("data", {}).get("final_test_bars", -1))
    if final_test_bar_count != expected_test_bars:
        raise ValueError("Input data does not match the reports' final-test partition size")
    execution = ExecutionConfig(
        spread_pips=args.spread_pips,
        slippage_pips_per_side=args.slippage_pips,
        commission_round_trip_pips=args.commission_pips,
        swap_long_pips_per_rollover=args.swap_long_pips,
        swap_short_pips_per_rollover=args.swap_short_pips,
        rollover_timezone=args.rollover_timezone,
        rollover_local_time=args.rollover_local_time,
        triple_swap_weekday=args.triple_swap_weekday,
        triple_swap_multiplier=args.triple_swap_multiplier,
    )
    risk = RiskConfig(
        starting_balance=args.starting_balance,
        risk_per_trade=args.risk_per_trade,
        max_daily_loss=args.max_daily_loss,
        max_trades_per_day=args.max_trades_per_day,
    )
    audit = build_behaviour_audit(
        reports, train, validation, execution=execution, risk=risk,
        pip_size=args.pip_size, price_basis=args.price_basis,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    _write_markdown(audit, markdown_path)
    print(f"Audited {len(reports)} runs and {audit['scope']['candidate_count']} finalists.")
    print(f"Cross-seed candidate pairs: {audit['scope']['cross_seed_pair_count']}")
    print(f"Final test evaluated: {audit['scope']['final_test_evaluated']}")
    print(f"JSON report: {output}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
