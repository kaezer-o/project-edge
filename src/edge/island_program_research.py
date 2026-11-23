from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExecutionConfig, RiskConfig, TIMEFRAME_SECONDS
from .data import annotate_gaps, resample_ohlc
from .features import FEATURE_COLUMNS, add_features
from .island_evolution import (
    IslandEvolutionConfig,
    run_island_evolution,
)
from .program import StrategyProgram, evaluate_program, canonicalize_program
from .program_evolution import (
    available_strategy_families,
    crossover_programs,
    mutate_program,
    strategy_family_program,
)
from .program_research import (
    _diverse_ranked,
    _evaluate,
    _random_program,
    _training_fitness,
    _validation_check,
)


def _prepare_frames(
    raw_frame: pd.DataFrame,
    *,
    timeframe: str,
    source_timeframe: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    timeframe = timeframe.upper()
    source_timeframe = source_timeframe.upper()
    if timeframe not in TIMEFRAME_SECONDS or source_timeframe not in TIMEFRAME_SECONDS:
        raise ValueError("Unsupported timeframe")
    if TIMEFRAME_SECONDS[timeframe] < TIMEFRAME_SECONDS[source_timeframe]:
        raise ValueError("Target timeframe cannot be lower than source timeframe")
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
    # Include close for causal Donchian breakout templates. Raw price levels
    # remain protected by the expression unit checks against scalar thresholds.
    features = tuple(
        column for column in (*FEATURE_COLUMNS, "close")
        if column in train.columns
        and pd.api.types.is_numeric_dtype(train[column])
        and train[column].notna().any()
    )
    if len(features) < 4:
        raise ValueError("Fewer than four usable numeric features in training data")
    return train, validation, test, features


def _contains_expr(root, predicate) -> bool:
    return predicate(root) or any(
        _contains_expr(child, predicate) for child in root.args
    )


def _feature_name(expr) -> str | None:
    return str(expr.value) if expr.op == "feature" else None


def _direct_comparison(root, op: str, left: str, right: str) -> bool:
    return (
        root.op == op
        and len(root.args) == 2
        and _feature_name(root.args[0]) == left
        and _feature_name(root.args[1]) == right
    )


def _volatility_gate(root, op: str) -> bool:
    if root.op != op or len(root.args) != 2:
        return False
    left, right = root.args
    return (
        _feature_name(left) == "atr_14"
        and right.op == "rolling_mean"
        and len(right.args) == 1
        and _feature_name(right.args[0]) == "atr_14"
        and isinstance(right.value, int)
    )


def _volatility_family_lookbacks(
    root,
    *,
    volatility_op: str,
    direction_op: str,
    ema_name: str,
) -> set[int]:
    """Find lookbacks where one AND subtree contains both family-defining rules."""
    lookbacks: set[int] = set()

    def visit(node) -> None:
        if node.op == "and":
            gates: set[int] = set()
            directions = False

            def inspect(subtree) -> None:
                nonlocal directions
                if _volatility_gate(subtree, volatility_op):
                    gates.add(int(subtree.args[1].value))
                if _direct_comparison(
                    subtree, direction_op, "close", ema_name
                ):
                    directions = True
                for child in subtree.args:
                    inspect(child)

            inspect(node)
            if directions:
                lookbacks.update(gates)
        for child in node.args:
            visit(child)

    visit(root)
    return lookbacks


def _expression_structure(root) -> dict[str, object]:
    """Return a value-normalized AST shape and compact structural metrics."""
    operators: dict[str, int] = {}
    features: set[str] = set()
    lookbacks: set[int] = set()
    node_count = 0
    max_depth = 0

    def visit(node, depth: int):
        nonlocal node_count, max_depth
        node_count += 1
        max_depth = max(max_depth, depth)
        operators[node.op] = operators.get(node.op, 0) + 1
        if node.op == "feature":
            features.add(str(node.value))
        if node.op == "rolling_mean" and isinstance(node.value, int):
            lookbacks.add(int(node.value))
        if node.op == "const":
            value = "*"
        elif node.op == "rolling_mean":
            value = node.value
        elif node.op == "feature":
            value = str(node.value)
        else:
            value = None
        return {
            "op": node.op,
            "value": value,
            "args": [visit(child, depth + 1) for child in node.args],
        }

    tree = visit(root, 1)
    return {
        "tree": tree,
        "nodes": node_count,
        "depth": max_depth,
        "operators": dict(sorted(operators.items())),
        "features": sorted(features),
        "lookbacks": sorted(lookbacks),
    }


def _classify_structural_motifs(root) -> list[str]:
    """Label recurring AST motifs without assigning strategy families."""
    motifs: set[str] = set()

    def visit(node) -> None:
        if node.op in {"gt", "lt", "cross_above", "cross_below"} and len(node.args) == 2:
            left, right = node.args
            left_name = _feature_name(left)
            right_name = _feature_name(right)
            if left_name and right_name and left_name.startswith("ema_") and right_name.startswith("ema_"):
                motifs.add("ema_pair_comparison")
            if left_name == "close" and right_name and right_name.startswith("ema_"):
                motifs.add("price_vs_ema_filter")
            if left_name == "close" and right_name and right_name.startswith("donchian_high_") and node.op == "gt":
                motifs.add("donchian_high_breakout")
            if left_name == "close" and right_name and right_name.startswith("donchian_low_") and node.op == "lt":
                motifs.add("donchian_low_breakout")
            if left_name == "rsi_14" and right.op == "const" and node.op in {"cross_above", "cross_below"}:
                motifs.add("rsi_threshold_cross")
            if (left_name == "atr_14" and right.op == "rolling_mean" and len(right.args) == 1
                    and _feature_name(right.args[0]) == "atr_14" and isinstance(right.value, int)
                    and node.op in {"gt", "lt"}):
                motifs.add("atr_vs_rolling_atr")
        if node.op == "and":
            subtree_nodes: list[object] = []
            def collect(current) -> None:
                subtree_nodes.append(current)
                for child in current.args:
                    collect(child)
            collect(node)
            has_atr_gate = any(
                current.op in {"gt", "lt"} and len(current.args) == 2
                and _feature_name(current.args[0]) == "atr_14"
                and current.args[1].op == "rolling_mean" and len(current.args[1].args) == 1
                and _feature_name(current.args[1].args[0]) == "atr_14"
                for current in subtree_nodes
            )
            has_price_ema = any(
                current.op in {"gt", "lt"} and len(current.args) == 2
                and _feature_name(current.args[0]) == "close"
                and (_feature_name(current.args[1]) or "").startswith("ema_")
                for current in subtree_nodes
            )
            if has_atr_gate and has_price_ema:
                motifs.add("ema_direction_with_atr_regime_gate")
        for child in node.args:
            visit(child)

    visit(root)
    return sorted(motifs)

def _program_structure_diagnostics(program: StrategyProgram) -> dict[str, object]:
    """Describe AST topology without changing evaluation or selection."""
    long = _expression_structure(program.long_entry)
    short = _expression_structure(program.short_entry)
    long_features = set(long["features"])
    short_features = set(short["features"])
    long_motifs = _classify_structural_motifs(program.long_entry)
    short_motifs = _classify_structural_motifs(program.short_entry)
    motifs = sorted(set(long_motifs) | set(short_motifs))
    payload = {
        "long": long["tree"],
        "short": short["tree"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    operator_counts = {
        "long": long["operators"],
        "short": short["operators"],
    }
    return {
        "fingerprint": fingerprint,
        "long_nodes": long["nodes"],
        "short_nodes": short["nodes"],
        "long_depth": long["depth"],
        "short_depth": short["depth"],
        "operator_counts": operator_counts,
        "features": sorted(long_features | short_features),
        "lookbacks": sorted(set(long["lookbacks"]) | set(short["lookbacks"])),
        "long_only_features": sorted(long_features - short_features),
        "short_only_features": sorted(short_features - long_features),
        "long_motifs": long_motifs,
        "short_motifs": short_motifs,
        "motifs": motifs,
    }


def _classify_strategy_families(program: StrategyProgram) -> list[str]:
    """Classify recognizable rule structures, not incidental feature usage."""
    long_rule, short_rule = program.long_entry, program.short_entry
    families: set[str] = set()

    if (
        _contains_expr(long_rule, lambda node: _direct_comparison(
            node, "gt", "ema_10", "ema_40"
        ))
        and _contains_expr(short_rule, lambda node: _direct_comparison(
            node, "lt", "ema_10", "ema_40"
        ))
    ):
        families.add("trend")

    if (
        _contains_expr(long_rule, lambda node: (
            node.op == "cross_above"
            and _feature_name(node.args[0]) == "rsi_14"
            and node.args[1].op == "const"
        ))
        and _contains_expr(short_rule, lambda node: (
            node.op == "cross_below"
            and _feature_name(node.args[0]) == "rsi_14"
            and node.args[1].op == "const"
        ))
    ):
        families.add("mean_reversion")

    if (
        _contains_expr(long_rule, lambda node: _direct_comparison(
            node, "gt", "close", "donchian_high_20"
        ))
        and _contains_expr(short_rule, lambda node: _direct_comparison(
            node, "lt", "close", "donchian_low_20"
        ))
    ):
        families.add("breakout")

    for label, volatility_op, ema_name in (
        ("volatility_expansion", "gt", "ema_40"),
        ("volatility_contraction", "lt", "ema_20"),
    ):
        long_lookbacks = _volatility_family_lookbacks(
            long_rule,
            volatility_op=volatility_op,
            direction_op="gt",
            ema_name=ema_name,
        )
        short_lookbacks = _volatility_family_lookbacks(
            short_rule,
            volatility_op=volatility_op,
            direction_op="lt",
            ema_name=ema_name,
        )
        if long_lookbacks & short_lookbacks:
            families.add(label)

    if not families:
        return ["other"]
    if len(families) > 1:
        families.add("hybrid")
    return sorted(families)


def _behaviour_descriptor(
    frame: pd.DataFrame,
    program: StrategyProgram,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    signal: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Describe signal rates and exact executed entry-time/direction events."""
    if signal is None:
        signal = evaluate_program(program, frame)
    if trades is None:
        _, trades = _evaluate(
            frame,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            signal=signal,
        )
    chunks = np.array_split(np.arange(len(frame)), 3)
    rates: list[float] = []
    for positions in chunks:
        if not len(positions):
            rates.extend((0.0, 0.0, 0.0))
            continue
        segment = signal.iloc[positions]
        start_time = pd.Timestamp(frame.index[int(positions[0])])
        end_time = pd.Timestamp(frame.index[int(positions[-1])])
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize("UTC")
            end_time = end_time.tz_localize("UTC")
        else:
            start_time = start_time.tz_convert("UTC")
            end_time = end_time.tz_convert("UTC")
        if not trades.empty:
            entry_times = pd.to_datetime(trades["entry_time"], utc=True)
            trade_count = int(((entry_times >= start_time) & (entry_times <= end_time)).sum())
        else:
            trade_count = 0
        rates.extend((
            float((segment == 1).mean()),
            float((segment == -1).mean()),
            float(trade_count / len(positions)),
        ))
    trade_events: list[str] = []
    if not trades.empty:
        for trade in trades.itertuples(index=False):
            timestamp = pd.Timestamp(trade.entry_time)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            trade_events.append(f"{timestamp.isoformat()}|{str(trade.direction).lower()}")
    families = _classify_strategy_families(program)
    signal_fingerprint = hashlib.sha256(
        signal.to_numpy(dtype="int8", copy=False).tobytes()
    ).hexdigest()
    unique_trade_events = sorted(set(trade_events))
    behaviour_payload = json.dumps(
        {
            "signals": signal_fingerprint,
            "trades": unique_trade_events,
            "stop_atr": program.stop_atr,
            "reward_risk": program.reward_risk,
            "max_holding_bars": program.max_holding_bars,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    behaviour_fingerprint = hashlib.sha256(behaviour_payload).hexdigest()
    return {
        "rates": rates,
        "trades": unique_trade_events,
        "signal_fingerprint": signal_fingerprint,
        "behaviour_fingerprint": behaviour_fingerprint,
        "families": sorted(families),
        "structure": _program_structure_diagnostics(program),
    }



def _directional_signal_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Jaccard overlap of (bar, direction) training signals."""
    left_values = np.asarray(left, dtype=np.int8)
    right_values = np.asarray(right, dtype=np.int8)
    left_active = {
        (int(index), int(left_values[index]))
        for index in np.flatnonzero(left_values != 0)
    }
    right_active = {
        (int(index), int(right_values[index]))
        for index in np.flatnonzero(right_values != 0)
    }
    union = left_active | right_active
    return len(left_active & right_active) / len(union) if union else 0.0


def _training_trade_path_events(
    trades: pd.DataFrame,
) -> set[tuple[int, str, int, str]]:
    """Fingerprint executed training trades with compact UTC nanosecond events."""
    if trades.empty:
        return set()
    events: set[tuple[int, str, int, str]] = set()
    for trade in trades.itertuples(index=False):
        entry_time = pd.Timestamp(trade.entry_time)
        exit_time = pd.Timestamp(trade.exit_time)
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        else:
            entry_time = entry_time.tz_convert("UTC")
        if exit_time.tzinfo is None:
            exit_time = exit_time.tz_localize("UTC")
        else:
            exit_time = exit_time.tz_convert("UTC")
        events.add((
            int(entry_time.value),
            str(trade.direction).lower(),
            int(exit_time.value),
            str(trade.exit_reason).lower(),
        ))
    return events


def _set_jaccard(left: set, right: set) -> float:
    intersection_size = len(left & right)
    union_size = len(left) + len(right) - intersection_size
    return intersection_size / union_size if union_size else 0.0


def _risk_profile_similarity(left: StrategyProgram, right: StrategyProgram) -> float:
    """Return one minus the largest relative difference in risk parameters."""
    left_values = (
        float(left.stop_atr),
        float(left.reward_risk),
        float(left.max_holding_bars),
    )
    right_values = (
        float(right.stop_atr),
        float(right.reward_risk),
        float(right.max_holding_bars),
    )
    differences = (
        abs(a - b) / max(abs(a), abs(b), 1e-12)
        for a, b in zip(left_values, right_values)
    )
    return max(0.0, 1.0 - max(differences, default=0.0))


def _training_signal_diverse_shortlist(
    ranked: list[tuple[float, StrategyProgram, dict[str, object], dict[str, object]]],
    frame: pd.DataFrame,
    *,
    similarity_threshold: float,
    limit: int,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
) -> tuple[list[tuple[float, StrategyProgram, dict[str, object], dict[str, object]]], dict[str, object]]:
    """Filter finalists using cached training behaviour and exact greedy ranking."""
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("finalist similarity threshold must be between zero and one")

    signals: dict[str, np.ndarray] = {}
    trade_paths: dict[str, set[tuple[int, str, int, str]]] = {}
    programs_by_fingerprint: dict[str, StrategyProgram] = {}
    signal_evaluations = 0
    training_backtests = 0
    for item in ranked:
        program = item[1]
        fingerprint = program.fingerprint()
        if fingerprint in programs_by_fingerprint:
            continue
        programs_by_fingerprint[fingerprint] = program
        signal = np.asarray(evaluate_program(program, frame), dtype=np.int8)
        signal_evaluations += 1
        _, trades = _evaluate(
            frame,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            signal=pd.Series(signal, index=frame.index),
        )
        training_backtests += 1
        signals[fingerprint] = signal
        trade_paths[fingerprint] = _training_trade_path_events(trades)

    selected = []
    selected_ranks: list[int] = []
    selected_fingerprints: list[str] = []
    rejected = []
    pairwise_comparisons = 0
    short_circuited_comparisons = 0
    for rank, item in enumerate(ranked, start=1):
        if len(selected) >= limit:
            break
        fingerprint = item[1].fingerprint()
        best_comparison = None
        for selected_rank, selected_fingerprint in zip(
            selected_ranks, selected_fingerprints
        ):
            pairwise_comparisons += 1
            signal_similarity = _directional_signal_jaccard(
                signals[fingerprint], signals[selected_fingerprint]
            )
            # The combined score is the minimum of the three similarities.
            # A signal score no greater than the current best cannot improve it.
            if (
                best_comparison is not None
                and signal_similarity <= best_comparison["combined_similarity"]
            ):
                short_circuited_comparisons += 1
                continue
            trade_similarity = _set_jaccard(
                trade_paths[fingerprint], trade_paths[selected_fingerprint]
            )
            if (
                best_comparison is not None
                and trade_similarity <= best_comparison["combined_similarity"]
            ):
                short_circuited_comparisons += 1
                continue
            risk_similarity = _risk_profile_similarity(
                item[1], programs_by_fingerprint[selected_fingerprint]
            )
            combined_similarity = min(
                signal_similarity, trade_similarity, risk_similarity
            )
            if (
                best_comparison is None
                or combined_similarity > best_comparison["combined_similarity"]
            ):
                best_comparison = {
                    "fitness_rank": selected_rank,
                    "directional_signal_jaccard": signal_similarity,
                    "training_trade_path_jaccard": trade_similarity,
                    "risk_profile_similarity": risk_similarity,
                    "combined_similarity": combined_similarity,
                }
        if (
            best_comparison is not None
            and best_comparison["combined_similarity"] >= similarity_threshold
        ):
            rejected.append({
                "fitness_rank": rank,
                "training_fitness": float(item[0]),
                "most_similar_selected_fitness_rank": best_comparison["fitness_rank"],
                "max_directional_signal_jaccard": best_comparison["directional_signal_jaccard"],
                "max_training_trade_path_jaccard": best_comparison["training_trade_path_jaccard"],
                "max_risk_profile_similarity": best_comparison["risk_profile_similarity"],
                "max_combined_similarity": best_comparison["combined_similarity"],
            })
            continue
        selected.append(item)
        selected_ranks.append(rank)
        selected_fingerprints.append(fingerprint)
    return selected, {
        "method": "greedy_training_signal_and_trade_path_jaccard",
        "similarity_threshold": float(similarity_threshold),
        "requested_limit": int(limit),
        "fitness_ranked_pool_count": len(ranked),
        "selected_count": len(selected),
        "selected_fitness_ranks": selected_ranks,
        "performance": {
            "ranked_program_count": len(ranked),
            "unique_program_count": len(programs_by_fingerprint),
            "signal_evaluations": signal_evaluations,
            "training_backtests": training_backtests,
            "pairwise_comparisons": pairwise_comparisons,
            "short_circuited_comparisons": short_circuited_comparisons,
        },
        "rejected_as_redundant": rejected,
        "selection_inputs": [
            "training_fitness",
            "directional_training_signals",
            "training_trade_entry_exit_paths",
            "risk_management_profile",
        ],
        "validation_used_for_selection": False,
        "final_test_used_for_selection": False,
    }


def _training_eligible_finalists(
    evaluated: list[tuple[float, StrategyProgram, dict[str, object], dict[str, object]]],
) -> tuple[list[tuple[float, StrategyProgram, dict[str, object], dict[str, object]]], dict[str, object]]:
    """Exclude candidates that fail an existing chronological training-era gate."""
    eligible = []
    rejected = []
    for score, program, metrics, robustness in evaluated:
        all_windows_eligible = robustness.get("all_windows_eligible") is True
        finite_score = bool(np.isfinite(score))
        score_above_ineligible_sentinel = finite_score and float(score) > -1e8
        if all_windows_eligible and score_above_ineligible_sentinel:
            eligible.append((score, program, metrics, robustness))
        else:
            rejected.append({
                "program_fingerprint": program.fingerprint(),
                "training_fitness": float(score) if finite_score else None,
                "all_windows_eligible": all_windows_eligible,
                "reason": (
                    "training_window_ineligible" if not all_windows_eligible
                    else "non_finite_or_ineligible_training_fitness"
                ),
            })
    return eligible, {
        "method": "existing_three_window_training_eligibility",
        "evaluated_population_count": len(evaluated),
        "eligible_population_count": len(eligible),
        "rejected_population_count": len(rejected),
        "rejected": rejected,
        "validation_used": False,
        "final_test_used": False,
    }


def _candidate_validation_overlap(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Measure pairwise overlap of candidate trade entries on validation only."""
    event_sets = [
        set(
            candidate.get("validation", {})
            .get("trade_event_diagnostics", {})
            .get("events", [])
        )
        for candidate in candidates
    ]
    comparisons: list[dict[str, object]] = []
    for first_index, first_events in enumerate(event_sets):
        for second_index in range(first_index + 1, len(event_sets)):
            second_events = event_sets[second_index]
            # Empty trade sets carry no useful overlap signal.
            if not first_events or not second_events:
                continue
            union = first_events | second_events
            shared = first_events & second_events
            comparisons.append({
                "candidate_a": first_index + 1,
                "candidate_b": second_index + 1,
                "shared_entry_events": len(shared),
                "union_entry_events": len(union),
                "jaccard_similarity": len(shared) / len(union),
            })
    return sorted(
        comparisons,
        key=lambda row: (
            float(row["jaccard_similarity"]),
            int(row["shared_entry_events"]),
        ),
        reverse=True,
    )


def _run_signature(
    source_data: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    source_timeframe: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    config: IslandEvolutionConfig,
    seed: int,
    min_train_trades: int,
    min_validation_trades: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(source_data, index=True).values.tobytes())
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "source_timeframe": source_timeframe,
        "data_sha256": digest.hexdigest(),
        "execution": execution.__dict__,
        "risk": risk.__dict__,
        "pip_size": pip_size,
        "price_basis": price_basis,
        "island_config": config.__dict__,
        "seed": seed,
        "min_train_trades": min_train_trades,
        "min_validation_trades": min_validation_trades,
        "fitness_version": 7,
        "strategy_seed_version": 1,
        "strategy_family_classifier_version": 3,
        "structure_diagnostics_version": 2,
        "structural_motif_taxonomy_version": 1,
        "population_motif_telemetry_version": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_island_program_research(
    raw_frame: pd.DataFrame,
    *,
    timeframe: str,
    source_timeframe: str,
    symbol: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    generations: int,
    seed: int,
    min_train_trades: int,
    min_validation_trades: int,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: IslandEvolutionConfig | None = None,
    resume: bool = False,
    run_final_test: bool = False,
    finalist_similarity_threshold: float | None = None,
    finalist_limit: int = 50,
) -> dict[str, object]:
    """Run resumable multi-island program evolution; only training reaches the evaluator."""
    if generations < 1:
        raise ValueError("generations must be at least one")
    if finalist_similarity_threshold is not None and not 0.0 <= finalist_similarity_threshold <= 1.0:
        raise ValueError("finalist_similarity_threshold must be between zero and one")
    if not isinstance(finalist_limit, int) or isinstance(finalist_limit, bool) or finalist_limit < 1:
        raise ValueError("finalist_limit must be a positive integer")
    config = config or IslandEvolutionConfig()
    timeframe = timeframe.upper()
    source_timeframe = source_timeframe.upper()
    train, validation, test, features = _prepare_frames(
        raw_frame, timeframe=timeframe, source_timeframe=source_timeframe
    )
    signature = _run_signature(
        pd.concat((train, validation, test)),
        symbol=symbol,
        timeframe=timeframe,
        source_timeframe=source_timeframe,
        execution=execution,
        risk=risk,
        pip_size=pip_size,
        price_basis=price_basis,
        config=config,
        seed=seed,
        min_train_trades=min_train_trades,
        min_validation_trades=min_validation_trades,
    )

    training_results: OrderedDict[str, tuple[float, dict[str, object], dict[str, object]]] = OrderedDict()
    training_result_cache_limit = max(256, config.islands * config.population_per_island * 4)

    def evaluate(genome: dict[str, object]) -> tuple[float, dict[str, object]]:
        program = StrategyProgram.from_dict(genome)
        signal = evaluate_program(program, train)
        aggregate_result = _evaluate(
            train,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            signal=signal,
        )
        score, metrics, robustness = _training_fitness(
            train,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            min_train_trades=min_train_trades,
            aggregate_result=aggregate_result,
        )
        fingerprint = program.fingerprint()
        training_results[fingerprint] = (score, metrics, robustness)
        training_results.move_to_end(fingerprint)
        if len(training_results) > training_result_cache_limit:
            training_results.popitem(last=False)
        descriptor = _behaviour_descriptor(
            train,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            signal=signal,
            trades=aggregate_result[1],
        )
        return float(score), descriptor

    def reproduce(
        first: dict[str, object],
        second: dict[str, object],
        rng: np.random.Generator,
    ) -> dict[str, object]:
        parent_a = StrategyProgram.from_dict(first)
        parent_b = StrategyProgram.from_dict(second)
        if rng.random() < 0.55:
            child = crossover_programs(parent_a, parent_b, rng)
        else:
            child = mutate_program(parent_a, features, rng)
        return canonicalize_program(child).to_dict()

    initial_populations = None
    initial_family_counts: dict[str, int] | None = None
    if not resume:
        initial_family_counts = {}
        seed_sequence = np.random.SeedSequence(seed)
        initial_populations = []
        family_names = available_strategy_families(features)
        if not family_names:
            raise ValueError("No supported strategy-family templates for available features")
        for island_index, child_seed in enumerate(seed_sequence.spawn(config.islands)):
            rng = np.random.default_rng(child_seed)
            population: list[dict[str, object]] = []
            seen: set[str] = set()
            while len(population) < config.population_per_island:
                slot = len(population)
                family = family_names[
                    (island_index * config.population_per_island + slot)
                    % len(family_names)
                ]
                candidate = canonicalize_program(
                    strategy_family_program(family, features, rng)
                )
                fingerprint = candidate.fingerprint()
                attempts = 0
                while fingerprint in seen and attempts < 16:
                    candidate = canonicalize_program(
                        strategy_family_program(family, features, rng)
                    )
                    fingerprint = candidate.fingerprint()
                    attempts += 1
                assigned_family = family
                if fingerprint in seen:
                    candidate = canonicalize_program(_random_program(features, rng))
                    fingerprint = candidate.fingerprint()
                    assigned_family = "random_fallback"
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    population.append(candidate.to_dict())
                    initial_family_counts[assigned_family] = (
                        initial_family_counts.get(assigned_family, 0) + 1
                    )
            initial_populations.append(population)

    result = run_island_evolution(
        initial_populations,
        evaluate=evaluate,
        reproduce=reproduce,
        config=config,
        seed=seed,
        generations=generations,
        checkpoint_path=checkpoint_path,
        run_signature=signature,
        resume=resume,
    )

    final_evaluated = []
    for population in result.islands:
        for genome in population:
            program = StrategyProgram.from_dict(genome)
            cached = training_results.get(program.fingerprint())
            if cached is None:
                score, metrics, robustness = _training_fitness(
                    train,
                    program,
                    execution=execution,
                    risk=risk,
                    pip_size=pip_size,
                    price_basis=price_basis,
                    min_train_trades=min_train_trades,
                )
            else:
                score, metrics, robustness = cached
            final_evaluated.append((score, program, metrics, robustness))
    training_eligible_evaluated, training_eligibility = _training_eligible_finalists(
        final_evaluated
    )
    fitness_ranked_shortlist = _diverse_ranked(
        training_eligible_evaluated,
        min(finalist_limit, len(training_eligible_evaluated)),
        max_per_structure=1,
    )
    shortlist = fitness_ranked_shortlist
    finalist_selection: dict[str, object] = {
        "training_eligibility": training_eligibility,
        "method": "existing_fitness_and_structure_ranking",
        "similarity_filter_enabled": finalist_similarity_threshold is not None,
        "fitness_ranked_pool_count": len(fitness_ranked_shortlist),
        "selected_count": len(fitness_ranked_shortlist),
        "requested_limit": finalist_limit,
        "validation_used_for_selection": False,
        "final_test_used_for_selection": False,
    }
    if finalist_similarity_threshold is not None:
        shortlist, finalist_selection = _training_signal_diverse_shortlist(
            fitness_ranked_shortlist,
            train,
            similarity_threshold=finalist_similarity_threshold,
            limit=finalist_limit,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
        )
        finalist_selection["similarity_filter_enabled"] = True
        finalist_selection["training_eligibility"] = training_eligibility
    fitness_ranked_shortlist_report = [
        {
            "fitness_rank": rank,
            "training_fitness": float(score),
            "program_fingerprint": program.fingerprint(),
        }
        for rank, (score, program, _metrics, _robustness)
        in enumerate(fitness_ranked_shortlist, start=1)
    ]
    fitness_rank_by_fingerprint = {
        program.fingerprint(): rank
        for rank, (_score, program, _metrics, _robustness)
        in enumerate(fitness_ranked_shortlist, start=1)
    }
    candidates: list[dict[str, object]] = []
    accepted: list[tuple[float, StrategyProgram, dict[str, object]]] = []
    for score, program, train_metrics, robustness in shortlist:
        validation_result = _validation_check(
            validation,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            min_trades=max(5, min_validation_trades // 3),
        )
        row = {
            "program": program.to_dict(),
            "fingerprint": program.fingerprint(),
            "fitness_rank": fitness_rank_by_fingerprint[program.fingerprint()],
            "training_fitness": float(score),
            "training_metrics": train_metrics,
            "training_robustness": robustness,
            "validation": validation_result,
        }
        candidates.append(row)
        if validation_result["pass"]:
            accepted.append((score, program, row))

    candidate_validation_overlap = _candidate_validation_overlap(candidates)
    # Keep the compact overlap summary, not every raw validation timestamp.
    for candidate in candidates:
        validation_diagnostics = candidate["validation"].get("trade_event_diagnostics", {})
        validation_diagnostics.pop("events", None)
        for window in candidate["validation"].get("windows", []):
            window.get("trade_event_diagnostics", {}).pop("events", None)

    winner = max(accepted, key=lambda item: item[0]) if accepted else None
    final_test = None
    if winner is not None and run_final_test:
        test_metrics, _ = _evaluate(
            test,
            winner[1],
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
        )
        final_test = {
            "metrics": test_metrics,
            "warning": "Single historical holdout; not prospective or live-trading evidence.",
        }

    report: dict[str, object] = {
        "project": "Project EDGE",
        "engine": "resumable_multi_island_program_evolution",
        "version": 1,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_timeframe": source_timeframe,
        "data": {
            "train_bars": len(train),
            "validation_bars": len(validation),
            "final_test_bars": len(test),
            "features": list(features),
            "final_test_evaluated": final_test is not None,
        },
        "runtime": {
            "generation": result.generation,
            "checkpoint": str(Path(checkpoint_path)),
            "resumed": resume,
            "islands": config.islands,
            "population_per_island": config.population_per_island,
            "total_population": config.islands * config.population_per_island,
            "migration_interval": config.migration_interval,
            "migrants_per_island": config.migrants_per_island,
            "novelty_weight": config.novelty_weight,
            "minimum_family_elites": config.minimum_family_elites,
            "initial_strategy_family_counts": initial_family_counts,
            "available_strategy_families": list(available_strategy_families(features)),
            "generation_performance": result.runtime_metrics,
            "history": result.history,
        },
        "finalist_selection": {
            **finalist_selection,
            "fitness_ranked_shortlist": fitness_ranked_shortlist_report,
        },
        "candidate_count": len(candidates),
        "validated_candidate_count": len(accepted),
        "status": "validated_candidate" if winner else "no_robust_candidate",
        "candidates": candidates,
        "candidate_validation_overlap": candidate_validation_overlap,
        "winner": winner[2] if winner else None,
        "final_test": final_test,
        "final_test_skipped": final_test is None,
        "validation_warning": (
            "Validation is development data if previously inspected. "
            "The final chronological partition must remain untouched until the methodology is frozen."
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "island_program_evolution_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
