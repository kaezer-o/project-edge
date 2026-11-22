from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from edge.config import ExecutionConfig, RiskConfig
from edge.island_program_research import _classify_strategy_families
from edge.program import StrategyProgram

try:
    from audit_candidate_behaviour import (
        _candidate_behaviour,
        _prepare_train_validation,
        signal_similarity,
        trade_overlap,
    )
except ModuleNotFoundError:
    from scripts.audit_candidate_behaviour import (
        _candidate_behaviour,
        _prepare_train_validation,
        signal_similarity,
        trade_overlap,
    )
from edge.data import load_ohlc_csv


def _signal_fingerprint(signal: np.ndarray) -> str:
    values = np.asarray(signal, dtype=np.int8)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _trade_fingerprint(events: set[str]) -> str:
    payload = json.dumps(sorted(events), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _duplicate_groups(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(int(row["rank"]))
    return [
        {"fingerprint": fingerprint, "ranks": ranks, "size": len(ranks)}
        for fingerprint, ranks in sorted(groups.items(), key=lambda item: (-len(item[1]), item[1][0]))
        if len(ranks) > 1
    ]


def _pair_summary(rows: list[dict[str, Any]], *, signal_field: str, trade_field: str) -> dict[str, Any]:
    signal_key = "active_signal_jaccard"
    trade_key = "entry_jaccard"
    signal_values = [float(row[signal_field][signal_key]) for row in rows]
    trade_values = [float(row[trade_field][trade_key]) for row in rows]
    return {
        "pair_count": len(rows),
        "mean_active_signal_jaccard": float(np.mean(signal_values)) if rows else 0.0,
        "median_active_signal_jaccard": float(np.median(signal_values)) if rows else 0.0,
        "pairs_at_or_above_0_50_signal_jaccard": sum(value >= 0.50 for value in signal_values),
        "pairs_at_or_above_0_75_signal_jaccard": sum(value >= 0.75 for value in signal_values),
        "pairs_at_or_above_0_90_signal_jaccard": sum(value >= 0.90 for value in signal_values),
        "pairs_at_or_above_0_50_trade_jaccard": sum(value >= 0.50 for value in trade_values),
        "mean_trade_entry_jaccard": float(np.mean(trade_values)) if rows else 0.0,
        "median_trade_entry_jaccard": float(np.median(trade_values)) if rows else 0.0,
    }


def _exclusive_family_class(families: list[str]) -> str:
    if "hybrid" in families:
        return "hybrid"
    return families[0] if len(families) == 1 else "other"


def _family_pair_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        classes = sorted([row["family_class_a"], row["family_class_b"]])
        key = " | ".join(classes)
        grouped.setdefault(key, []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for key, group in sorted(grouped.items()):
        result[key] = {
            "pair_count": len(group),
            "mean_training_active_signal_jaccard": float(np.mean([
                row["training_signal_similarity"]["active_signal_jaccard"] for row in group
            ])),
            "mean_validation_active_signal_jaccard": float(np.mean([
                row["validation_signal_similarity"]["active_signal_jaccard"] for row in group
            ])),
            "training_pairs_at_or_above_0_75": sum(
                row["training_signal_similarity"]["active_signal_jaccard"] >= 0.75 for row in group
            ),
            "validation_pairs_at_or_above_0_75": sum(
                row["validation_signal_similarity"]["active_signal_jaccard"] >= 0.75 for row in group
            ),
            "exact_training_signal_pairs": sum(row["exact_training_signal_match"] for row in group),
            "exact_validation_signal_pairs": sum(row["exact_validation_signal_match"] for row in group),
            "qualified_pairs": sum(row["both_validation_qualified"] for row in group),
        }
    return result


def build_redundancy_audit(
    reports: list[tuple[int, dict[str, Any]]],
    train,
    validation,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one seed report is required")
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

    per_seed: dict[str, Any] = {}
    all_pairs: list[dict[str, Any]] = []
    for seed, report in reports:
        candidates = report.get("candidates") or []
        if not candidates:
            raise ValueError(f"Seed {seed} contains no finalists")
        evaluated: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            result = _candidate_behaviour(
                seed, rank, candidate, train, validation,
                execution=execution, risk=risk, pip_size=pip_size,
                price_basis=price_basis,
            )
            program = StrategyProgram.from_dict(candidate["program"])
            training_signal = result["training"]["signal"]
            validation_signal = result["validation"]["signal"]
            training_trades = result["training"]["trade_events"]
            validation_trades = result["validation"]["trade_events"]
            evaluated.append({
                "seed": seed,
                "rank": rank,
                "validation_gate_pass": result["validation_gate_pass"],
                "families": _classify_strategy_families(program),
                "training_signal_fingerprint": _signal_fingerprint(training_signal),
                "validation_signal_fingerprint": _signal_fingerprint(validation_signal),
                "training_trade_fingerprint": _trade_fingerprint(training_trades),
                "validation_trade_fingerprint": _trade_fingerprint(validation_trades),
                "training_signal": training_signal,
                "validation_signal": validation_signal,
                "training_trade_events": training_trades,
                "validation_trade_events": validation_trades,
            })

        pairs: list[dict[str, Any]] = []
        for left, right in combinations(evaluated, 2):
            pairs.append({
                "seed": seed,
                "rank_a": left["rank"],
                "rank_b": right["rank"],
                "families_a": left["families"],
                "families_b": right["families"],
                "family_class_a": _exclusive_family_class(left["families"]),
                "family_class_b": _exclusive_family_class(right["families"]),
                "both_validation_qualified": left["validation_gate_pass"] and right["validation_gate_pass"],
                "validation_qualified_a": left["validation_gate_pass"],
                "validation_qualified_b": right["validation_gate_pass"],
                "exact_training_signal_match": left["training_signal_fingerprint"] == right["training_signal_fingerprint"],
                "exact_validation_signal_match": left["validation_signal_fingerprint"] == right["validation_signal_fingerprint"],
                "exact_training_trade_match": left["training_trade_fingerprint"] == right["training_trade_fingerprint"],
                "exact_validation_trade_match": left["validation_trade_fingerprint"] == right["validation_trade_fingerprint"],
                "training_signal_similarity": signal_similarity(left["training_signal"], right["training_signal"]),
                "validation_signal_similarity": signal_similarity(left["validation_signal"], right["validation_signal"]),
                "training_trade_overlap": trade_overlap(left["training_trade_events"], right["training_trade_events"]),
                "validation_trade_overlap": trade_overlap(left["validation_trade_events"], right["validation_trade_events"]),
            })
        all_pairs.extend(pairs)

        candidate_rows = [{
            "rank": row["rank"],
            "validation_gate_pass": row["validation_gate_pass"],
            "families": row["families"],
            "training_signal_fingerprint": row["training_signal_fingerprint"],
            "validation_signal_fingerprint": row["validation_signal_fingerprint"],
            "training_trade_fingerprint": row["training_trade_fingerprint"],
            "validation_trade_fingerprint": row["validation_trade_fingerprint"],
        } for row in evaluated]
        qualified_pairs = [row for row in pairs if row["both_validation_qualified"]]
        per_seed[str(seed)] = {
            "candidate_count": len(evaluated),
            "validation_qualified_count": sum(row["validation_gate_pass"] for row in evaluated),
            "unique_training_signal_streams": len({row["training_signal_fingerprint"] for row in evaluated}),
            "unique_validation_signal_streams": len({row["validation_signal_fingerprint"] for row in evaluated}),
            "unique_training_trade_entry_sets": len({row["training_trade_fingerprint"] for row in evaluated}),
            "unique_validation_trade_entry_sets": len({row["validation_trade_fingerprint"] for row in evaluated}),
            "exact_duplicate_training_signal_groups": _duplicate_groups(candidate_rows, "training_signal_fingerprint"),
            "exact_duplicate_validation_signal_groups": _duplicate_groups(candidate_rows, "validation_signal_fingerprint"),
            "exact_duplicate_training_trade_groups": _duplicate_groups(candidate_rows, "training_trade_fingerprint"),
            "exact_duplicate_validation_trade_groups": _duplicate_groups(candidate_rows, "validation_trade_fingerprint"),
            "all_pair_summary": _pair_summary(
                pairs, signal_field="training_signal_similarity", trade_field="training_trade_overlap"
            ),
            "qualified_pair_summary": _pair_summary(
                qualified_pairs, signal_field="training_signal_similarity", trade_field="training_trade_overlap"
            ),
            "family_pair_redundancy": _family_pair_summary(pairs),
            "candidates": candidate_rows,
            "pairs": pairs,
        }

    def top_pairs(field: str, metric: str, limit: int = 30) -> list[dict[str, Any]]:
        return sorted(
            all_pairs,
            key=lambda row: (
                -float(row[field][metric]),
                -int(row[field].get("shared_entries", 0)),
                row["seed"], row["rank_a"], row["rank_b"],
            ),
        )[:limit]

    return {
        "audit_version": 1,
        "scope": {
            "seeds": seeds,
            "symbol": reference.get("symbol"),
            "timeframe": reference.get("timeframe"),
            "source_timeframe": reference.get("source_timeframe"),
            "candidate_count": sum(item["candidate_count"] for item in per_seed.values()),
            "within_seed_pair_count": len(all_pairs),
            "training_bars": len(train),
            "validation_bars": len(validation),
            "final_test_evaluated": False,
            "selection_or_validation_gates_modified": False,
            "price_basis": price_basis,
            "execution": execution.__dict__,
            "risk": risk.__dict__,
            "pip_size": pip_size,
        },
        "per_seed": per_seed,
        "top_training_signal_pairs": top_pairs("training_signal_similarity", "active_signal_jaccard"),
        "top_validation_signal_pairs": top_pairs("validation_signal_similarity", "active_signal_jaccard"),
        "top_training_trade_pairs": top_pairs("training_trade_overlap", "entry_jaccard"),
        "top_validation_trade_pairs": top_pairs("validation_trade_overlap", "entry_jaccard"),
        "limitations": [
            "This is a finalist-only audit; it does not measure redundancy in the full evolving population.",
            "Validation has been repeatedly examined during development and is descriptive, not independent performance evidence.",
            "Exact signal agreement can be inflated by neutral bars; active-signal Jaccard is reported separately.",
            "Trade-entry overlap depends on execution and risk assumptions. EURUSD source data is bid-only, so spread is estimated.",
            "The final chronological holdout is never evaluated.",
            "No audit result changes fitness, selection, migration, validation gates or final-test access.",
        ],
    }


def _parse_report(value: str) -> tuple[int, Path]:
    try:
        seed, path = value.split("=", 1)
        return int(seed), Path(path)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Report must use SEED=PATH format") from exc


def _write_markdown(audit: dict[str, Any], path: Path) -> None:
    scope = audit["scope"]
    lines = [
        "# Intra-run finalist behavioural redundancy audit", "",
        f"- Symbol/timeframe: {scope['symbol']} / {scope['timeframe']}",
        f"- Seeds: {', '.join(map(str, scope['seeds']))}",
        f"- Final test evaluated: no",
        f"- Finalists: {scope['candidate_count']}",
        f"- Within-seed pairs: {scope['within_seed_pair_count']}", "",
        "## Per-seed summary", "",
        "| Seed | Finalists | Qualified | Unique train signals | Unique validation signals | Unique train trades | Unique validation trades |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, item in audit["per_seed"].items():
        lines.append(
            f"| {seed} | {item['candidate_count']} | {item['validation_qualified_count']} | "
            f"{item['unique_training_signal_streams']} | {item['unique_validation_signal_streams']} | "
            f"{item['unique_training_trade_entry_sets']} | {item['unique_validation_trade_entry_sets']} |"
        )
    for title, key, metric_key in (
        ("Highest training signal overlap", "top_training_signal_pairs", "training_signal_similarity"),
        ("Highest validation signal overlap", "top_validation_signal_pairs", "validation_signal_similarity"),
    ):
        lines.extend(["", f"## {title}", "",
            "| Seed | Candidate A | Candidate B | Active Jaccard | Long | Short | Both qualified |",
            "|---:|---:|---:|---:|---:|---:|---|"])
        for row in audit[key][:15]:
            m = row[metric_key]
            lines.append(
                f"| {row['seed']} | {row['rank_a']} | {row['rank_b']} | "
                f"{m['active_signal_jaccard']:.3f} | {m['long_signal_jaccard']:.3f} | "
                f"{m['short_signal_jaccard']:.3f} | {row['both_validation_qualified']} |"
            )
    lines.extend(["", "## Exact duplicate signal groups", ""])
    for seed, item in audit["per_seed"].items():
        lines.append(f"### Seed {seed}")
        for key, label in (
            ("exact_duplicate_training_signal_groups", "Training"),
            ("exact_duplicate_validation_signal_groups", "Validation"),
        ):
            groups = item[key]
            lines.append(f"- {label}: " + (
                "; ".join(f"ranks {group['ranks']} (n={group['size']})" for group in groups)
                if groups else "none"
            ))
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {text}" for text in audit["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure behavioural redundancy among finalists within each evolutionary run.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", action="append", type=_parse_report, required=True, help="SEED=REPORT_JSON; repeat per seed")
    parser.add_argument("--output", default="results/candidate-redundancy-audit.json")
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
    train, validation, holdout_count = _prepare_train_validation(
        load_ohlc_csv(args.input),
        timeframe=str(reference["timeframe"]),
        source_timeframe=str(reference["source_timeframe"]),
    )
    if holdout_count != int(reference.get("data", {}).get("final_test_bars", -1)):
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
    audit = build_redundancy_audit(
        reports, train, validation, execution=execution, risk=risk,
        pip_size=args.pip_size, price_basis=args.price_basis,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    markdown = output.with_suffix(".md")
    _write_markdown(audit, markdown)
    print(f"Audited {audit['scope']['candidate_count']} finalists across {len(reports)} runs.")
    print(f"Within-seed candidate pairs: {audit['scope']['within_seed_pair_count']}")
    print("Final test evaluated: False")
    print(f"JSON report: {output}")
    print(f"Markdown report: {markdown}")


if __name__ == "__main__":
    main()
