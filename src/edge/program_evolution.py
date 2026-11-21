from __future__ import annotations

from dataclasses import replace
import numpy as np

from .program import (
    Expr,
    StrategyProgram,
    validate_expr,
)


NUMERIC_GENERATORS = ("add", "sub", "div", "min", "max")
WINDOW_GENERATORS = ("lag", "rolling_mean", "rolling_std")
BOOLEAN_GENERATORS = ("gt", "lt", "cross_above", "cross_below")
LOGICAL_GENERATORS = ("and", "or", "not")


def _terminal_number(
    rng: np.random.Generator,
    features: tuple[str, ...],
) -> Expr:
    # Free-floating constants have no reliable scale across forex indicators.
    # Constants are introduced only by the scale-aware comparison templates.
    return Expr("feature", value=str(rng.choice(features)))


def random_expression(
    expected: str,
    features: tuple[str, ...] | list[str],
    rng: np.random.Generator,
    *,
    max_depth: int = 4,
) -> Expr:
    """Create a bounded, type-correct expression without executable code."""
    feature_names = tuple(sorted({
        str(name) for name in features
        if str(name).strip() and not str(name).startswith("edge_")
    }))
    if expected not in {"number", "bool"}:
        raise ValueError("expected must be 'number' or 'bool'")
    if not feature_names:
        raise ValueError("At least one feature name is required")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    def numeric(depth: int) -> Expr:
        if depth >= max_depth or rng.random() < 0.35:
            return _terminal_number(rng, feature_names)
        choice = float(rng.random())
        if choice < 0.55:
            op = str(rng.choice(NUMERIC_GENERATORS))
            return Expr(op, (numeric(depth + 1), numeric(depth + 1)))
        if choice < 0.85:
            op = str(rng.choice(WINDOW_GENERATORS))
            return Expr(
                op,
                (numeric(depth + 1),),
                value=int(rng.integers(1, 31)),
            )
        return Expr("abs", (numeric(depth + 1),))

    rsi_features = tuple(name for name in feature_names if name.startswith("rsi_"))
    ratio_features = tuple(
        name for name in feature_names
        if name.endswith("_atr") or name in {"ema_gap_atr", "price_change_atr"}
    )
    price_features = tuple(
        name for name in feature_names
        if (
            name.startswith("ema_")
            or name.startswith("donchian_")
            or name in {"open", "high", "low", "close"}
        )
    )
    atr_features = tuple(name for name in feature_names if name.startswith("atr_"))

    def comparison_atom(depth: int) -> Expr:
        op = str(rng.choice(BOOLEAN_GENERATORS))

        # RSI thresholds use the indicator's native 0-100 scale.
        if rsi_features and rng.random() < 0.25:
            feature = Expr("feature", value=str(rng.choice(rsi_features)))
            threshold = float(rng.choice((20, 30, 40, 50, 60, 70, 80)))
            return Expr(op, (feature, Expr("const", value=threshold)))

        # Existing ATR-normalized features can use dimensionless thresholds.
        if ratio_features and rng.random() < 0.20:
            feature = Expr("feature", value=str(rng.choice(ratio_features)))
            threshold = float(rng.uniform(-2.0, 2.0))
            return Expr(op, (feature, Expr("const", value=threshold)))

        # Price differences normalized by ATR are dimensionless, so bounded
        # scalar thresholds have a stable interpretation across price levels.
        if len(price_features) >= 2 and atr_features and rng.random() < 0.25:
            left_name = str(rng.choice(price_features))
            right_options = tuple(name for name in price_features if name != left_name)
            if right_options:
                right_name = str(rng.choice(right_options))
                difference = Expr(
                    "sub",
                    (
                        Expr("feature", value=left_name),
                        Expr("feature", value=right_name),
                    ),
                )
                normalized = Expr(
                    "div",
                    (difference, Expr("feature", value=str(rng.choice(atr_features)))),
                )
                threshold = float(rng.uniform(-2.0, 2.0))
                return Expr(op, (normalized, Expr("const", value=threshold)))

        left = numeric(depth + 1)
        right = numeric(depth + 1)
        if left == right:
            right = Expr("lag", (right,), value=1)
        return Expr(op, (left, right))

    def boolean(depth: int) -> Expr:
        if depth >= max_depth or rng.random() < 0.35:
            return comparison_atom(depth)
        op = str(rng.choice(LOGICAL_GENERATORS))
        if op == "not":
            return Expr(op, (boolean(depth + 1),))
        return Expr(op, (boolean(depth + 1), boolean(depth + 1)))

    # Random trees are retried when their feature units are incompatible.
    # This keeps the search space valid without weakening expression checks.
    for _ in range(128):
        result = numeric(1) if expected == "number" else boolean(1)
        try:
            validate_expr(result, expected=expected)
        except ValueError:
            continue
        return result

    # Guaranteed-valid fallback; the normal path should almost always succeed.
    terminal = Expr("feature", value=feature_names[0])
    if expected == "number":
        return terminal
    fallback = Expr("gt", (terminal, Expr("const", value=0.0)))
    validate_expr(fallback, expected="bool")
    return fallback


def _node_type(node: Expr) -> str:
    for expected in ("number", "bool"):
        try:
            validate_expr(node, expected=expected)
            return expected
        except ValueError:
            pass
    raise ValueError("Cannot determine expression node type")


def _nodes_with_paths(
    root: Expr,
    *,
    path: tuple[int, ...] = (),
) -> list[tuple[tuple[int, ...], Expr, str]]:
    found = [(path, root, _node_type(root))]
    for index, child in enumerate(root.args):
        found.extend(
            _nodes_with_paths(child, path=path + (index,))
        )
    return found


def _replace_subtree(
    root: Expr,
    path: tuple[int, ...],
    replacement: Expr,
) -> Expr:
    if not path:
        return replacement
    index = path[0]
    children = list(root.args)
    children[index] = _replace_subtree(
        children[index],
        path[1:],
        replacement,
    )
    return Expr(root.op, tuple(children), root.value)


def _tree_depth(root: Expr) -> int:
    if not root.args:
        return 1
    return 1 + max(_tree_depth(child) for child in root.args)


def _tree_size(root: Expr) -> int:
    return 1 + sum(_tree_size(child) for child in root.args)


def mutate_expression(
    expr: Expr,
    features: tuple[str, ...] | list[str],
    rng: np.random.Generator,
    *,
    max_depth: int = 12,
    max_nodes: int = 128,
) -> Expr:
    """Replace one randomly selected subtree with a same-type expression."""
    validate_expr(expr, expected=_node_type(expr))
    candidates = _nodes_with_paths(expr)
    rng.shuffle(candidates)

    for path, node, kind in candidates:
        depth_before = len(path)
        remaining_depth = max_depth - depth_before
        if remaining_depth < 1:
            continue
        replacement = random_expression(
            kind,
            features,
            rng,
            max_depth=min(5, remaining_depth),
        )
        candidate = _replace_subtree(expr, path, replacement)
        if _tree_size(candidate) > max_nodes:
            continue
        if _tree_depth(candidate) > max_depth:
            continue
        try:
            validate_expr(candidate, expected=_node_type(expr))
        except ValueError:
            continue
        return candidate
    return expr


def crossover_expressions(
    first: Expr,
    second: Expr,
    rng: np.random.Generator,
    *,
    max_depth: int = 12,
    max_nodes: int = 128,
) -> Expr:
    """Exchange a randomly chosen, type-compatible subtree from the second parent."""
    expected = _node_type(first)
    if _node_type(second) != expected:
        raise ValueError("Crossover parents must have the same root type")
    validate_expr(second, expected=expected)
    first_nodes = _nodes_with_paths(first)
    second_nodes = [
        item for item in _nodes_with_paths(second) if item[2] == expected
    ]
    rng.shuffle(first_nodes)
    rng.shuffle(second_nodes)

    for path, _, kind in first_nodes:
        if kind != expected:
            continue
        for _, donor, _ in second_nodes:
            candidate = _replace_subtree(first, path, donor)
            if _tree_size(candidate) > max_nodes:
                continue
            if _tree_depth(candidate) > max_depth:
                continue
            try:
                validate_expr(candidate, expected=expected)
            except ValueError:
                continue
            return candidate
    return first


def mutate_program(
    program: StrategyProgram,
    features: tuple[str, ...] | list[str],
    rng: np.random.Generator,
) -> StrategyProgram:
    """Mutate one strategy side or one bounded risk parameter."""
    choice = int(rng.integers(0, 5))
    if choice == 0:
        return replace(
            program,
            long_entry=mutate_expression(program.long_entry, features, rng),
        )
    if choice == 1:
        return replace(
            program,
            short_entry=mutate_expression(program.short_entry, features, rng),
        )
    if choice == 2:
        return replace(
            program,
            stop_atr=float(np.clip(
                program.stop_atr * rng.choice((0.75, 0.9, 1.1, 1.25)),
                0.25,
                8.0,
            )),
        )
    if choice == 3:
        return replace(
            program,
            reward_risk=float(np.clip(
                program.reward_risk * rng.choice((0.75, 0.9, 1.1, 1.25)),
                0.25,
                10.0,
            )),
        )
    return replace(
        program,
        max_holding_bars=int(np.clip(
            round(program.max_holding_bars * rng.choice((0.75, 0.9, 1.1, 1.25))),
            1,
            5000,
        )),
    )


def crossover_programs(
    first: StrategyProgram,
    second: StrategyProgram,
    rng: np.random.Generator,
) -> StrategyProgram:
    """Build one offspring from compatible long/short trees and risk genes."""
    return StrategyProgram(
        long_entry=crossover_expressions(first.long_entry, second.long_entry, rng),
        short_entry=crossover_expressions(first.short_entry, second.short_entry, rng),
        stop_atr=float(np.clip(
            (first.stop_atr + second.stop_atr) / 2.0,
            0.25,
            8.0,
        )),
        reward_risk=float(np.clip(
            (first.reward_risk + second.reward_risk) / 2.0,
            0.25,
            10.0,
        )),
        max_holding_bars=int(rng.choice((
            first.max_holding_bars,
            second.max_holding_bars,
        ))),
    )
