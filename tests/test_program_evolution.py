import numpy as np
import pytest

from edge.program import Expr, StrategyProgram, validate_expr
from edge.program_evolution import (
    crossover_expressions,
    crossover_programs,
    mutate_expression,
    mutate_program,
    random_expression,
)


FEATURES = ("close", "ema_gap_atr", "rsi_14", "atr_14")


def _assert_typed_and_bounded(expr, expected):
    validate_expr(expr, expected=expected)


@pytest.mark.parametrize("expected", ["number", "bool"])
def test_random_expression_generates_valid_typed_trees(expected):
    rng = np.random.default_rng(41)

    for _ in range(100):
        expr = random_expression(expected, FEATURES, rng, max_depth=5)
        _assert_typed_and_bounded(expr, expected)


def test_generator_excludes_metadata_and_respects_feature_units():
    rng = np.random.default_rng(2026)
    features = (
        "ema_10", "ema_50", "atr_14", "rsi_14",
        "edge_weekend_before", "edge_segment_id",
    )

    for _ in range(100):
        expr = random_expression("bool", features, rng, max_depth=5)
        validate_expr(expr, expected="bool")

        def collect_features(node):
            values = [node.value] if node.op == "feature" else []
            for child in node.args:
                values.extend(collect_features(child))
            return values

        assert all(not str(name).startswith("edge_") for name in collect_features(expr))


def test_random_expression_requires_features_and_valid_depth():
    rng = np.random.default_rng(2)

    with pytest.raises(ValueError, match="feature"):
        random_expression("number", [], rng)
    with pytest.raises(ValueError, match="max_depth"):
        random_expression("bool", FEATURES, rng, max_depth=0)
    with pytest.raises(ValueError, match="expected"):
        random_expression("string", FEATURES, rng)


@pytest.mark.parametrize("expected", ["number", "bool"])
def test_mutation_preserves_expression_type_and_bounds(expected):
    rng = np.random.default_rng(99)
    original = random_expression(expected, FEATURES, rng, max_depth=4)

    for _ in range(30):
        mutated = mutate_expression(original, FEATURES, rng)
        _assert_typed_and_bounded(mutated, expected)


@pytest.mark.parametrize("expected", ["number", "bool"])
def test_crossover_exchanges_only_type_compatible_subtrees(expected):
    rng = np.random.default_rng(123)
    first = random_expression(expected, FEATURES, rng, max_depth=4)
    second = random_expression(expected, FEATURES, rng, max_depth=4)

    for _ in range(30):
        child = crossover_expressions(first, second, rng)
        _assert_typed_and_bounded(child, expected)


def test_program_mutation_keeps_program_valid():
    rng = np.random.default_rng(8)
    program = StrategyProgram(
        long_entry=Expr("gt", (Expr("feature", value="ema_gap_atr"), Expr("const", value=0.5))),
        short_entry=Expr("lt", (Expr("feature", value="rsi_14"), Expr("const", value=30.0))),
    )

    for _ in range(100):
        program = mutate_program(program, FEATURES, rng)
        validate_expr(program.long_entry, expected="bool")
        validate_expr(program.short_entry, expected="bool")
        assert 0.25 <= program.stop_atr <= 8.0
        assert 0.25 <= program.reward_risk <= 10.0
        assert 1 <= program.max_holding_bars <= 5000


def test_program_crossover_combines_valid_trees_and_risk_parameters():
    rng = np.random.default_rng(18)
    first = StrategyProgram(
        long_entry=Expr("gt", (Expr("feature", value="ema_gap_atr"), Expr("const", value=0.5))),
        short_entry=Expr("lt", (Expr("feature", value="rsi_14"), Expr("const", value=30.0))),
        stop_atr=1.0,
        reward_risk=1.5,
        max_holding_bars=40,
    )
    second = StrategyProgram(
        long_entry=Expr("cross_above", (Expr("feature", value="ema_gap_atr"), Expr("const", value=0.25))),
        short_entry=Expr("cross_below", (Expr("feature", value="ema_gap_atr"), Expr("const", value=-0.25))),
        stop_atr=2.0,
        reward_risk=3.0,
        max_holding_bars=80,
    )

    child = crossover_programs(first, second, rng)

    validate_expr(child.long_entry, expected="bool")
    validate_expr(child.short_entry, expected="bool")
    assert child.stop_atr == 1.5
    assert child.reward_risk == 2.25
    assert child.max_holding_bars in {40, 80}


def test_crossover_rejects_different_root_types():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        crossover_expressions(
            Expr("feature", value="close"),
            Expr("gt", (Expr("feature", value="close"), Expr("const", value=0.0))),
            rng,
        )



def _walk(expr):
    yield expr
    for child in expr.args:
        yield from _walk(child)


def test_generated_rules_avoid_constant_only_and_raw_price_thresholds():
    rng = np.random.default_rng(20261121)

    for _ in range(250):
        expr = random_expression("bool", FEATURES, rng, max_depth=5)
        validate_expr(expr, expected="bool")

        for node in _walk(expr):
            if node.op not in {"gt", "lt", "cross_above", "cross_below"}:
                continue
            left, right = node.args
            left_features = {
                child.value for child in _walk(left) if child.op == "feature"
            }
            right_features = {
                child.value for child in _walk(right) if child.op == "feature"
            }
            assert left_features or right_features

            if left.op == "const":
                assert right_features
                assert (
                    right.op == "div"
                    or any(
                        str(name).startswith("rsi_")
                        or str(name).endswith("_atr")
                        or str(name) == "ema_gap_atr"
                        for name in right_features
                    )
                )
                if any(str(name).startswith("rsi_") for name in right_features):
                    assert 20.0 <= float(left.value) <= 80.0
                else:
                    assert -2.0 <= float(left.value) <= 2.0

            if right.op == "const":
                assert left_features
                assert (
                    left.op == "div"
                    or any(
                        str(name).startswith("rsi_")
                        or str(name).endswith("_atr")
                        or str(name) == "ema_gap_atr"
                        for name in left_features
                    )
                )
                if any(str(name).startswith("rsi_") for name in left_features):
                    assert 20.0 <= float(right.value) <= 80.0
                else:
                    assert -2.0 <= float(right.value) <= 2.0


def test_numeric_generator_uses_market_features_not_free_floating_constants():
    rng = np.random.default_rng(11)
    for _ in range(100):
        expr = random_expression("number", FEATURES, rng, max_depth=5)
        nodes = list(_walk(expr))
        assert any(node.op == "feature" for node in nodes)
        assert all(node.op != "const" for node in nodes)
