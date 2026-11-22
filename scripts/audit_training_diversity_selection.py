from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from edge.data import load_ohlc_csv
from edge.program import StrategyProgram, evaluate_program

try:
    from audit_candidate_behaviour import _prepare_train_validation, signal_similarity
except ModuleNotFoundError:
    from scripts.audit_candidate_behaviour import _prepare_train_validation, signal_similarity


DEFAULT_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)


def select_training_diverse(
    candidates: list[dict[str, Any]],
    signals: dict[int, np.ndarray],
    *,
    similarity_threshold: float = 0.90,
    limit: int | None = None,
) -> dict[str, Any]:
    """Greedily retain high-fitness candidates with distinct training signals only."""
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between zero and one")
    ranked = sorted(
        candidates,
        key=lambda row: (-float(row["training_fitness"]), int(row["rank"])),
    )
    target = len(ranked) if limit is None else max(0, min(int(limit), len(ranked)))
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in ranked:
        if len(selected) >= target:
            break
        rank = int(candidate["rank"])
        comparisons = [
            (int(row["rank"]), signal_similarity(signals[rank], signals[int(row["rank"])]))
            for row in selected
        ]
        highest = max(
            (float(item["active_signal_jaccard"]) for _, item in comparisons),
            default=0.0,
        )
        if comparisons and highest >= similarity_threshold:
            rejected.append({
                "rank": rank,
                "training_fitness": float(candidate["training_fitness"]),
                "most_similar_selected_rank": max(
                    comparisons,
                    key=lambda pair: float(pair[1]["active_signal_jaccard"]),
                )[0],
                "max_active_signal_jaccard": highest,
            })
            continue
        selected.append({
            "rank": rank,
            "training_fitness": float(candidate["training_fitness"]),
            "max_active_signal_jaccard_to_selected": highest,
        })
    selected_ranks = {int(item["rank"]) for item in selected}
    rejected_ranks = {int(item["rank"]) for item in rejected}
    return {
        "requested_limit": target,
        "similarity_threshold": similarity_threshold,
        "selected_count": len(selected),
        "selected": selected,
        "rejected_as_redundant": rejected,
        "not_considered_after_limit": [
            int(row["rank"]) for row in ranked
            if int(row["rank"]) not in selected_ranks
            and int(row["rank"]) not in rejected_ranks
        ],
    }


def _signal_key(signal: np.ndarray) -> bytes:
    return np.asarray(signal, dtype=np.int8).tobytes()


def build_selection_audit(
    reports: list[tuple[int, dict[str, Any]]],
    train,
    *,
    similarity_threshold: float = 0.90,
    limit: int | None = None,
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
            or report.get("data") != expected_data
        ):
            raise ValueError(f"Seed {seed} has mismatched run or data partition metadata")
    if int(expected_data.get("train_bars", -1)) != len(train):
        raise ValueError("Input data does not match the reports' training partition size")

    per_seed: dict[str, Any] = {}
    for seed, report in reports:
        candidates = report.get("candidates") or []
        if not candidates:
            raise ValueError(f"Seed {seed} contains no finalists")
        signals: dict[int, np.ndarray] = {}
        rows: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            fitness = candidate.get("training_fitness")
            if fitness is None or not np.isfinite(float(fitness)):
                raise ValueError(f"Seed {seed} candidate {rank} has invalid training_fitness")
            program_data = candidate.get("program")
            if not isinstance(program_data, dict):
                raise ValueError(f"Seed {seed} candidate {rank} has no program")
            program = StrategyProgram.from_dict(program_data)
            signals[rank] = np.asarray(evaluate_program(program, train), dtype=np.int8)
            rows.append({"rank": rank, "training_fitness": float(fitness)})
        selection = select_training_diverse(
            rows, signals, similarity_threshold=similarity_threshold, limit=limit,
        )
        selected_count = selection["selected_count"]
        fitness_ranked = sorted(
            rows, key=lambda row: (-row["training_fitness"], row["rank"])
        )
        # Fair comparison: fitness-only receives the same number of slots as
        # the diversity selector actually filled at this threshold.
        baseline = fitness_ranked[:selected_count]
        baseline_ranks = [int(row["rank"]) for row in baseline]
        selected_ranks = [int(row["rank"]) for row in selection["selected"]]
        per_seed[str(seed)] = {
            "candidate_count": len(rows),
            "unique_training_signal_streams": len({
                _signal_key(signal) for signal in signals.values()
            }),
            "fitness_only_top_ranks": baseline_ranks,
            "fitness_only_unique_signals": len({
                _signal_key(signals[rank]) for rank in baseline_ranks
            }),
            "fitness_only_selected_count": len(baseline),
            "diversity_selected_ranks": selected_ranks,
            "diversity_selected_unique_signals": len({
                _signal_key(signals[rank]) for rank in selected_ranks
            }),
            "strongest_training_fitness": fitness_ranked[0]["training_fitness"],
            "strongest_fitness_retained": bool(
                selected_ranks and selected_ranks[0] == int(fitness_ranked[0]["rank"])
            ),
            "lowest_selected_training_fitness": min(
                (float(row["training_fitness"]) for row in selection["selected"]),
                default=None,
            ),
            "selection": selection,
        }
    return {
        "audit_version": 2,
        "scope": {
            "seeds": seeds,
            "symbol": reference.get("symbol"),
            "timeframe": reference.get("timeframe"),
            "source_timeframe": reference.get("source_timeframe"),
            "similarity_threshold": similarity_threshold,
            "final_test_evaluated": False,
            "selection_uses_training_fitness_and_training_signals_only": True,
            "validation_used_for_selection": False,
            "production_selection_changed": False,
            "fitness_only_baseline_matches_diversity_selected_count": True,
        },
        "per_seed": per_seed,
        "limitations": [
            "This is an offline finalist-selection simulation, not a change to production evolution.",
            "It operates only on the existing finalist pool and cannot recover candidates discarded earlier.",
            "Training fitness and training signals are used for selection; validation is not read by the selection algorithm.",
            "The fitness-only baseline is truncated to the number of candidates actually retained by the diversity selector.",
            "The validation partition remains development data and is not independent performance evidence.",
            "The final chronological holdout is never evaluated.",
            "Bid-only source data requires an estimated spread for execution-based audits; this signal-only simulation does not model fills.",
        ],
    }


def build_threshold_sensitivity_audit(
    reports: list[tuple[int, dict[str, Any]]],
    train,
    *,
    thresholds: list[float],
    limit: int | None = 20,
) -> dict[str, Any]:
    if not thresholds:
        raise ValueError("At least one similarity threshold is required")
    normalized = [float(value) for value in thresholds]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Similarity thresholds must be unique")
    if any(not 0.0 <= value <= 1.0 for value in normalized):
        raise ValueError("Similarity thresholds must be between zero and one")
    audits = {
        f"{value:.2f}": build_selection_audit(
            reports, train, similarity_threshold=value, limit=limit,
        )
        for value in normalized
    }
    first = next(iter(audits.values()))
    return {
        "audit_version": 1,
        "scope": {
            **first["scope"],
            "similarity_thresholds": normalized,
            "similarity_threshold": None,
        },
        "threshold_results": audits,
        "limitations": first["limitations"],
    }


def _parse_report(value: str) -> tuple[int, Path]:
    try:
        seed, path = value.split("=", 1)
        return int(seed), Path(path)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Report must use SEED=PATH format") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark training-only diversity selection across similarity thresholds."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", action="append", type=_parse_report, required=True)
    parser.add_argument("--output", default="results/training-diversity-threshold-sensitivity.json")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    reports = [(seed, json.loads(path.read_text(encoding="utf-8"))) for seed, path in args.report]
    reference = reports[0][1]
    train, _, holdout_count = _prepare_train_validation(
        load_ohlc_csv(args.input),
        timeframe=str(reference["timeframe"]),
        source_timeframe=str(reference["source_timeframe"]),
    )
    if holdout_count != int(reference.get("data", {}).get("final_test_bars", -1)):
        raise ValueError("Input data does not match the reports' final-test partition size")
    audit = build_threshold_sensitivity_audit(
        reports, train, thresholds=args.thresholds, limit=args.limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Training-only diversity threshold sensitivity", "",
        f"- Seeds: {', '.join(map(str, audit['scope']['seeds']))}",
        f"- Thresholds: {', '.join(f'{value:.2f}' for value in args.thresholds)}",
        f"- Maximum shortlist size: {args.limit}",
        "- Selection inputs: training fitness and training signals only",
        "- Validation used for selection: no",
        "- Final test evaluated: no", "",
        "| Seed | Threshold | Diversity selected | Fitness-only same-size unique | Diversity unique | Strongest fitness retained |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for threshold, threshold_audit in audit["threshold_results"].items():
        for seed, item in threshold_audit["per_seed"].items():
            lines.append(
                f"| {seed} | {threshold} | {item['selection']['selected_count']} | "
                f"{item['fitness_only_unique_signals']} | "
                f"{item['diversity_selected_unique_signals']} | "
                f"{'yes' if item['strongest_fitness_retained'] else 'no'} |"
            )
    for threshold, threshold_audit in audit["threshold_results"].items():
        lines.extend(["", f"## Threshold {threshold}", ""])
        for seed, item in threshold_audit["per_seed"].items():
            selection = item["selection"]
            lines.append(
                f"- Seed {seed}: selected ranks {item['diversity_selected_ranks']}; "
                f"rejected as redundant {len(selection['rejected_as_redundant'])}; "
                f"lowest selected training fitness "
                f"{item['lowest_selected_training_fitness']:.6f}."
            )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in audit["limitations"])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Audited {audit['scope']['seeds']} runs across thresholds "
        f"{', '.join(f'{value:.2f}' for value in args.thresholds)}. "
        "Final test evaluated: False"
    )
    print(f"JSON report: {output}")
    print(f"Markdown report: {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
