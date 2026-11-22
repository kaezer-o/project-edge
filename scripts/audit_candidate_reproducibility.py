from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


COMMUTATIVE_OPS = {"and", "or", "add", "mul", "min", "max", "eq", "ne"}
MOTIFS = (
    "ema_pair_comparison",
    "price_vs_ema_filter",
    "donchian_high_breakout",
    "donchian_low_breakout",
    "rsi_threshold_cross",
    "atr_vs_rolling_atr",
    "ema_direction_with_atr_regime_gate",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def normalize_expression(node: dict[str, Any]) -> dict[str, Any]:
    """Preserve operators/features while replacing tunable constants with a marker."""
    if not isinstance(node, dict) or not isinstance(node.get("op"), str):
        raise ValueError("Expression node must be an object with a string op")
    args = node.get("args", [])
    if not isinstance(args, list):
        raise ValueError("Expression args must be a list")
    normalized_args = [normalize_expression(child) for child in args]
    op = node["op"]
    if op in COMMUTATIVE_OPS:
        normalized_args.sort(key=_stable_json)
    value = "*" if op == "const" else node.get("value")
    return {"op": op, "value": value, "args": normalized_args}


def _walk(node: dict[str, Any]):
    yield node
    for child in node.get("args", []):
        yield from _walk(child)


def _is_feature(node: dict[str, Any], name: str) -> bool:
    return node.get("op") == "feature" and node.get("value") == name


def _contains(node: dict[str, Any], predicate) -> bool:
    return any(predicate(item) for item in _walk(node))


def _comparison_has_features(node: dict[str, Any], op: set[str], left: str, right: str) -> bool:
    return (
        node.get("op") in op
        and len(node.get("args", [])) == 2
        and _is_feature(node["args"][0], left)
        and _is_feature(node["args"][1], right)
    )


def _has_rsi_threshold_cross(root: dict[str, Any]) -> bool:
    return _contains(
        root,
        lambda n: n.get("op") in {"cross_above", "cross_below"}
        and len(n.get("args", [])) == 2
        and _is_feature(n["args"][0], "rsi_14")
        and n["args"][1].get("op") == "const",
    )


def _has_atr_rolling_comparison(root: dict[str, Any]) -> bool:
    def match(node):
        if node.get("op") not in {"gt", "ge", "lt", "le", "cross_above", "cross_below"}:
            return False
        args = node.get("args", [])
        if len(args) != 2:
            return False
        return any(
            _is_feature(a, "atr_14")
            and b.get("op") == "rolling_mean"
            and len(b.get("args", [])) == 1
            and _is_feature(b["args"][0], "atr_14")
            for a, b in (args, args[::-1])
        )
    return _contains(root, match)


def _has_ema_direction_atr_gate(root: dict[str, Any]) -> bool:
    def is_price_ema(node):
        args = node.get("args", [])
        return node.get("op") in {"gt", "ge", "lt", "le"} and len(args) == 2 and any(
            _is_feature(a, "close") and str(b.get("value", "")).startswith("ema_")
            for a, b in (args, args[::-1])
        )
    return _contains(
        root,
        lambda node: node.get("op") == "and"
        and _contains(node, is_price_ema)
        and _has_atr_rolling_comparison(node),
    )


def classify_motifs(program: dict[str, Any]) -> dict[str, list[str]]:
    sides = {}
    for side in ("long_entry", "short_entry"):
        root = program.get(side)
        if not isinstance(root, dict):
            raise ValueError(f"Candidate program is missing {side}")
        found = set()
        if _contains(root, lambda n: n.get("op") in {"gt", "ge", "lt", "le", "cross_above", "cross_below"}
                     and len(n.get("args", [])) == 2
                     and all(str(a.get("value", "")).startswith("ema_") and a.get("op") == "feature"
                             for a in n["args"])):
            found.add("ema_pair_comparison")
        if _contains(root, lambda n: n.get("op") in {"gt", "ge", "lt", "le"}
                     and len(n.get("args", [])) == 2
                     and any(_is_feature(a, "close") and b.get("op") == "feature"
                             and str(b.get("value", "")).startswith("ema_")
                             for a, b in (n["args"], n["args"][::-1]))):
            found.add("price_vs_ema_filter")
        if _contains(root, lambda n: _comparison_has_features(n, {"gt", "ge"}, "close", "donchian_high_20")
                     or _comparison_has_features(n, {"gt", "ge"}, "close", "donchian_high_55")):
            found.add("donchian_high_breakout")
        if _contains(root, lambda n: _comparison_has_features(n, {"lt", "le"}, "close", "donchian_low_20")
                     or _comparison_has_features(n, {"lt", "le"}, "close", "donchian_low_55")):
            found.add("donchian_low_breakout")
        if _has_rsi_threshold_cross(root):
            found.add("rsi_threshold_cross")
        if _has_atr_rolling_comparison(root):
            found.add("atr_vs_rolling_atr")
        if _has_ema_direction_atr_gate(root):
            found.add("ema_direction_with_atr_regime_gate")
        sides[side] = sorted(found)
    return sides


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _candidate_record(seed: int, rank: int, candidate: dict[str, Any]) -> dict[str, Any]:
    program = candidate.get("program")
    if not isinstance(program, dict):
        raise ValueError(f"Seed {seed} candidate {rank} has no program object")
    long_rule = normalize_expression(program["long_entry"])
    short_rule = normalize_expression(program["short_entry"])
    rule_signature = {"long_entry": long_rule, "short_entry": short_rule}
    risk_signature = {
        "stop_atr": program.get("stop_atr"),
        "reward_risk": program.get("reward_risk"),
        "max_holding_bars": program.get("max_holding_bars"),
    }
    validation = candidate.get("validation") or {}
    metrics = validation.get("metrics") or {}
    stress = validation.get("cost_stress") or []
    stress_by_multiplier = {str(row.get("cost_multiplier")): bool(row.get("pass")) for row in stress}
    motifs_by_side = classify_motifs(program)
    motif_union = sorted(set(motifs_by_side["long_entry"]) | set(motifs_by_side["short_entry"]))
    return {
        "seed": seed,
        "rank": rank,
        "genome_fingerprint": candidate.get("fingerprint"),
        "rule_structure_fingerprint": _digest(rule_signature),
        "full_structure_fingerprint": _digest({"rules": rule_signature, "risk": risk_signature}),
        "risk_parameters": risk_signature,
        "motifs": motif_union,
        "long_motifs": motifs_by_side["long_entry"],
        "short_motifs": motifs_by_side["short_entry"],
        "validation_entry_fingerprint": (validation.get("trade_event_diagnostics") or {}).get("entry_event_fingerprint"),
        "validation_pass": bool(validation.get("pass", False)),
        "passed_windows": validation.get("passed_windows", 0),
        "trades": metrics.get("trades", 0),
        "expectancy_r": metrics.get("expectancy_r"),
        "profit_factor": metrics.get("profit_factor"),
        "stress_1_5x_pass": validation.get("one_and_half_x_cost_pass"),
        "stress_2x_pass": stress_by_multiplier.get("2.0", stress_by_multiplier.get("2", None)),
    }


def _groups(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        value = row.get(field)
        if value:
            grouped[str(value)].append(row)
    return [
        {"fingerprint": fingerprint, "members": [
            {"seed": row["seed"], "rank": row["rank"], "validation_pass": row["validation_pass"]}
            for row in members
        ]}
        for fingerprint, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(members) > 1
    ]


def build_audit(reports: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("At least two seed reports are required")
    seeds = [seed for seed, _ in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed identifiers must be unique")
    reference = reports[0][1]
    for seed, report in reports:
        # Check holdout access before cross-run metadata equality so the
        # safety failure is reported directly, even if this flag differs.
        if report.get("data", {}).get("final_test_evaluated") is not False:
            raise ValueError(f"Seed {seed} indicates final-test evaluation; audit stopped")
        if report.get("symbol") != reference.get("symbol") or report.get("timeframe") != reference.get("timeframe"):
            raise ValueError(f"Seed {seed} has a different symbol or timeframe")
        if report.get("data") != reference.get("data"):
            raise ValueError(f"Seed {seed} has different data partition metadata")

    records = []
    run_summaries = []
    for seed, report in reports:
        candidates = report.get("candidates") or []
        if not candidates:
            raise ValueError(f"Seed {seed} contains no candidates")
        for rank, candidate in enumerate(candidates, start=1):
            records.append(_candidate_record(seed, rank, candidate))
        history = report.get("runtime", {}).get("history", [])
        final_history = history[-1] if history else {}
        run_summaries.append({
            "seed": seed,
            "status": report.get("status"),
            "generation": report.get("runtime", {}).get("generation"),
            "candidate_count": report.get("candidate_count"),
            "validated_candidate_count": report.get("validated_candidate_count"),
            "unique_genomes": final_history.get("unique_genomes"),
            "unique_behaviours": final_history.get("unique_behaviours"),
            "unique_signal_streams": final_history.get("unique_signal_streams"),
            "exclusive_family_counts": final_history.get("strategy_family_class_population_counts"),
            "motif_population_counts": (final_history.get("population_motif_diagnostics") or {}).get("motif_population_counts"),
        })

    cross_seed_pairs = []
    for left, right in combinations(records, 2):
        if left["seed"] == right["seed"]:
            continue
        left_motifs, right_motifs = set(left["motifs"]), set(right["motifs"])
        cross_seed_pairs.append({
            "seed_a": left["seed"], "rank_a": left["rank"],
            "seed_b": right["seed"], "rank_b": right["rank"],
            "same_rule_structure": left["rule_structure_fingerprint"] == right["rule_structure_fingerprint"],
            "same_full_structure": left["full_structure_fingerprint"] == right["full_structure_fingerprint"],
            "same_validation_entry_set": bool(left["validation_entry_fingerprint"])
                and left["validation_entry_fingerprint"] == right["validation_entry_fingerprint"],
            "motif_jaccard": round(_jaccard(left_motifs, right_motifs), 6),
            "shared_motifs": sorted(left_motifs & right_motifs),
        })

    per_seed_motifs = {}
    qualified_motifs = Counter()
    unqualified_motifs = Counter()
    for seed in seeds:
        rows = [row for row in records if row["seed"] == seed]
        counts = Counter(motif for row in rows for motif in row["motifs"])
        per_seed_motifs[str(seed)] = {
            "candidate_count": len(rows),
            "qualified_count": sum(row["validation_pass"] for row in rows),
            "motif_candidate_counts": dict(sorted(counts.items())),
        }
    for row in records:
        (qualified_motifs if row["validation_pass"] else unqualified_motifs).update(row["motifs"])

    exact_rule_pairs = [row for row in cross_seed_pairs if row["same_rule_structure"]]
    exact_event_pairs = [row for row in cross_seed_pairs if row["same_validation_entry_set"]]
    motif_similar_pairs = sorted(
        cross_seed_pairs,
        key=lambda row: (-row["motif_jaccard"], row["seed_a"], row["rank_a"], row["seed_b"], row["rank_b"]),
    )
    return {
        "audit_version": 1,
        "scope": {
            "seeds": seeds,
            "symbol": reference.get("symbol"),
            "timeframe": reference.get("timeframe"),
            "candidates_per_seed": {str(seed): len(report.get("candidates") or []) for seed, report in reports},
            "final_test_evaluated": False,
            "selection_or_validation_gates_modified": False,
        },
        "run_summaries": run_summaries,
        "candidate_records": records,
        "per_seed_candidate_motifs": per_seed_motifs,
        "qualified_candidate_motif_counts": dict(sorted(qualified_motifs.items())),
        "unqualified_candidate_motif_counts": dict(sorted(unqualified_motifs.items())),
        "exact_cross_seed_rule_matches": exact_rule_pairs,
        "exact_cross_seed_validation_entry_matches": exact_event_pairs,
        "top_cross_seed_motif_similarities": motif_similar_pairs[:30],
        "limitations": [
            "Validation entry events are represented by SHA-256 fingerprints in the source reports. Equality detects exact matches only; approximate cross-seed Jaccard overlap cannot be reconstructed.",
            "Motif labels are coarse structural indicators and do not establish equivalent trading behaviour.",
            "The validation partition has been repeatedly examined during development and is not an independent performance estimate.",
            "A structural match across seeds is descriptive evidence only and does not justify changing selection or validation thresholds.",
        ],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Candidate reproducibility audit",
        "",
        f"- Symbol/timeframe: {audit['scope']['symbol']} / {audit['scope']['timeframe']}",
        f"- Seeds: {', '.join(map(str, audit['scope']['seeds']))}",
        "- Final test evaluated: no",
        "",
        "## Run summaries",
        "",
        "| Seed | Status | Qualified | Unique signals |",
        "|---:|---|---:|---:|",
    ]
    for row in audit["run_summaries"]:
        lines.append(
            f"| {row['seed']} | {row['status']} | {row['validated_candidate_count']} | {row['unique_signal_streams']} |"
        )
    lines.extend([
        "",
        "## Cross-seed exact matches",
        "",
        f"- Exact rule-structure pairs: {len(audit['exact_cross_seed_rule_matches'])}",
        f"- Exact validation entry-set pairs: {len(audit['exact_cross_seed_validation_entry_matches'])}",
        "",
        "## Candidate motif counts",
        "",
        "| Seed | Candidates | Qualified | Motif counts |",
        "|---:|---:|---:|---|",
    ])
    for seed, row in audit["per_seed_candidate_motifs"].items():
        counts = ", ".join(f"{key}: {value}" for key, value in row["motif_candidate_counts"].items()) or "none"
        lines.append(f"| {seed} | {row['candidate_count']} | {row['qualified_count']} | {counts} |")
    lines.extend([
        "",
        "## Interpretation constraints",
        "",
    ])
    lines.extend(f"- {item}" for item in audit["limitations"])
    lines.append("")
    return "\n".join(lines)


def _parse_report_spec(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        return int(seed_text), Path(path_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Report must use SEED=PATH format") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare finalist structures and validation fingerprints across independent EDGE runs."
    )
    parser.add_argument(
        "--report", action="append", type=_parse_report_spec, required=True,
        help="Seed report in SEED=PATH format; provide at least two reports.",
    )
    parser.add_argument("--output", default="results/candidate-reproducibility-audit.json")
    parser.add_argument("--markdown-output", default="results/candidate-reproducibility-audit.md")
    args = parser.parse_args()
    if len(args.report) < 2:
        parser.error("provide at least two --report arguments")

    reports = []
    for seed, path in args.report:
        try:
            reports.append((seed, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"could not load report {path}: {exc}")

    try:
        audit = build_audit(reports)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    output = Path(args.output)
    markdown_output = Path(args.markdown_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Audited {len(reports)} runs and {len(audit['candidate_records'])} finalists.")
    print(f"Exact cross-seed rule matches: {len(audit['exact_cross_seed_rule_matches'])}")
    print(f"Exact cross-seed validation entry matches: {len(audit['exact_cross_seed_validation_entry_matches'])}")
    print(f"JSON report: {output}")
    print(f"Markdown report: {markdown_output}")


if __name__ == "__main__":
    main()
