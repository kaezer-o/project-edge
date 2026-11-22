import numpy as np
import pandas as pd
import pytest

from edge.program import (
    Expr,
    StrategyProgram,
    evaluate_expr,
    evaluate_program,
    validate_expr,
    canonicalize_expr,
    canonicalize_program,
    semantic_fingerprint,
)


def feature(name):
    return Expr("feature", value=name)


def const(value):
    return Expr("const", value=value)


def test_program_generates_long_and_short_signals_from_composed_rules():
    index = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    frame = pd.DataFrame({"momentum": [-2.0, -1.0, 0.0, 1.0, 2.0]}, index=index)
    program = StrategyProgram(
        long_entry=Expr("gt", (feature("momentum"), const(0.5))),
        short_entry=Expr("lt", (feature("momentum"), const(-0.5))),
    )

    signal = evaluate_program(program, frame)

    assert signal.tolist() == [-1, -1, 0, 1, 1]
    assert signal.dtype == np.dtype("int8")


def test_conflicting_long_and_short_conditions_produce_no_trade():
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    always_true = Expr("gt", (feature("x"), const(0.0)))
    program = StrategyProgram(long_entry=always_true, short_entry=always_true)

    assert evaluate_program(program, frame).tolist() == [0, 0]


def test_cross_above_uses_current_and_previous_closed_rows():
    frame = pd.DataFrame({"x": [0.0, 1.0, 0.0, 2.0]})
    expr = Expr("cross_above", (feature("x"), const(0.5)))

    assert evaluate_expr(expr, frame).fillna(False).tolist() == [False, True, False, True]


def test_lag_and_rolling_mean_are_causal():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    lagged = evaluate_expr(Expr("lag", (feature("x"),), value=1), frame)
    mean = evaluate_expr(Expr("rolling_mean", (feature("x"),), value=2), frame)

    assert lagged.iloc[0] != lagged.iloc[0]
    assert lagged.iloc[1:].tolist() == [1.0, 2.0, 3.0]
    assert mean.iloc[:1].isna().all()
    assert mean.iloc[1:].tolist() == [1.5, 2.5, 3.5]


def test_division_by_zero_returns_missing_values_not_infinity():
    frame = pd.DataFrame({"x": [1.0, 2.0], "zero": [0.0, 0.0]})
    expr = Expr("div", (feature("x"), feature("zero")))

    result = evaluate_expr(expr, frame)

    assert result.isna().all()


def test_program_round_trip_preserves_fingerprint():
    program = StrategyProgram(
        long_entry=Expr(
            "and",
            (
                Expr("gt", (feature("x"), const(1.0))),
                Expr("lt", (feature("y"), const(4.0))),
            ),
        ),
        short_entry=Expr("lt", (feature("x"), const(-1.0))),
        stop_atr=1.25,
        reward_risk=2.5,
        max_holding_bars=80,
    )

    restored = StrategyProgram.from_dict(program.to_dict())

    assert restored == program
    assert restored.fingerprint() == program.fingerprint()


@pytest.mark.parametrize(
    "expr",
    [
        Expr("python", value="__import__('os')"),
        Expr("add", (feature("x"),)),
        Expr("const", value=float("nan")),
        Expr("lag", (feature("x"),), value=251),
        Expr("gt", (feature("x"), const(1.0)), value="ignored"),
    ],
)
def test_invalid_or_unsafe_expressions_are_rejected(expr):
    with pytest.raises(ValueError):
        validate_expr(expr)


@pytest.mark.parametrize(
    "expr",
    [
        Expr("gt", (feature("ema_50"), feature("rsi_14"))),
        Expr("cross_above", (feature("ema_20"), feature("edge_weekend_before"))),
        Expr("add", (feature("ema_20"), feature("rsi_14"))),
        Expr("mul", (feature("ema_20"), feature("atr_14"))),
    ],
)
def test_incompatible_feature_units_are_rejected(expr):
    with pytest.raises(ValueError):
        validate_expr(expr, expected="bool" if expr.op in {"gt", "cross_above"} else "number")


def test_price_indicators_can_be_compared_and_combined():
    valid = Expr("gt", (feature("ema_20"), feature("donchian_high_20")))
    validate_expr(valid, expected="bool")


def test_boolean_expression_cannot_be_used_as_numeric_child():
    invalid = Expr("add", (Expr("gt", (feature("x"), const(0))), const(1)))

    with pytest.raises(ValueError, match="numeric arguments"):
        validate_expr(invalid)


def test_expression_depth_and_size_are_bounded():
    expr = feature("x")
    for _ in range(13):
        expr = Expr("abs", (expr,))

    with pytest.raises(ValueError, match="maximum depth"):
        validate_expr(expr)


def test_program_requires_chronological_index():
    frame = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.to_datetime(["2025-01-02", "2025-01-01"], utc=True),
    )
    program = StrategyProgram(
        long_entry=Expr("gt", (feature("x"), const(0))),
        short_entry=Expr("lt", (feature("x"), const(0))),
    )

    with pytest.raises(ValueError, match="chronological"):
        evaluate_program(program, frame)


def test_logical_not_does_not_turn_missing_features_into_entries():
    frame = pd.DataFrame({"x": [np.nan, -1.0, 1.0]})
    not_positive = Expr(
        "not",
        (Expr("gt", (feature("x"), const(0.0))),),
    )
    program = StrategyProgram(
        long_entry=not_positive,
        short_entry=Expr("lt", (feature("x"), const(-2.0))),
    )

    assert evaluate_program(program, frame).tolist() == [0, 1, 0]



def test_comparisons_preserve_missing_values_for_boolean_composition():
    frame = pd.DataFrame({"x": [np.nan, -1.0, 1.0]})
    comparison = evaluate_expr(
        Expr("gt", (feature("x"), const(0.0))),
        frame,
    )

    assert comparison.isna().tolist() == [True, False, False]


def test_crossovers_require_current_and_previous_observations():
    frame = pd.DataFrame({"x": [np.nan, -1.0, 1.0, np.nan, 2.0]})
    crossing = evaluate_expr(
        Expr("cross_above", (feature("x"), const(0.0))),
        frame,
    )

    assert crossing.isna().tolist() == [True, True, False, True, True]



@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("min", [1.0, np.nan, 2.0]),
        ("max", [3.0, np.nan, 4.0]),
    ],
)
def test_min_and_max_propagate_missing_operands(operation, expected):
    frame = pd.DataFrame({
        "left": [1.0, np.nan, 4.0],
        "right": [3.0, 2.0, 2.0],
    })
    result = evaluate_expr(
        Expr(operation, (feature("left"), feature("right"))),
        frame,
    )

    assert result.iloc[[0, 2]].tolist() == [expected[0], expected[2]]
    assert pd.isna(result.iloc[1])



def test_raw_price_levels_and_volatility_distances_are_distinct():
    with pytest.raises(ValueError, match="compatible feature units"):
        validate_expr(
            Expr("gt", (feature("ema_20"), feature("atr_14"))),
            expected="bool",
        )

    valid = Expr(
        "gt",
        (
            Expr(
                "sub",
                (feature("ema_10"), feature("ema_50")),
            ),
            feature("atr_14"),
        ),
    )
    validate_expr(valid, expected="bool")


def test_raw_price_and_atr_features_reject_unscaled_scalar_thresholds():
    for name in ("ema_20", "atr_14", "donchian_high_20"):
        with pytest.raises(ValueError, match="normalize by ATR"):
            validate_expr(
                Expr("gt", (feature(name), const(1.0))),
                expected="bool",
            )


def test_oscillator_thresholds_and_atr_normalized_price_thresholds_are_valid():
    validate_expr(
        Expr("lt", (feature("rsi_14"), const(30.0))),
        expected="bool",
    )
    normalized_gap = Expr(
        "div",
        (
            Expr("sub", (feature("ema_10"), feature("ema_50"))),
            feature("atr_14"),
        ),
    )
    validate_expr(
        Expr("gt", (normalized_gap, const(0.5))),
        expected="bool",
    )


def test_constant_only_comparisons_are_rejected():
    with pytest.raises(ValueError, match="at least one market feature"):
        validate_expr(
            Expr("cross_below", (const(-1.0), const(0.5))),
            expected="bool",
        )



def test_ema_gap_atr_is_classified_as_normalized_ratio():
    validate_expr(
        Expr(
            "gt",
            (
                feature("ema_gap_atr"),
                const(0.5),
            ),
        ),
        expected="bool",
    )



def test_rolling_std_of_price_level_has_price_distance_units():
    dispersion = Expr(
        "rolling_std",
        (feature("ema_50"),),
        value=20,
    )

    valid = Expr("lt", (dispersion, feature("atr_14")))
    validate_expr(valid, expected="bool")

    invalid = Expr("lt", (dispersion, feature("ema_50")))
    with pytest.raises(ValueError, match="compatible feature units"):
        validate_expr(invalid, expected="bool")


def test_rolling_mean_of_price_level_remains_price_level():
    average = Expr(
        "rolling_mean",
        (feature("ema_50"),),
        value=20,
    )
    validate_expr(
        Expr("gt", (average, feature("ema_200"))),
        expected="bool",
    )


def test_canonicalization_removes_double_negation_and_duplicate_terms():
    positive = Expr("gt", (feature("x"), const(0.0)))
    redundant = Expr("not", (Expr("not", (Expr("and", (positive, positive)),)),))
    assert canonicalize_expr(redundant) == positive


def test_canonicalization_removes_identical_min_and_positive_price_abs():
    assert canonicalize_expr(Expr("min", (feature("ema_10"), feature("ema_10")))) == feature("ema_10")
    assert canonicalize_expr(Expr("abs", (feature("close"),))) == feature("close")


def test_canonical_program_and_semantic_fingerprint():
    frame = pd.DataFrame({"x": [1.0, -1.0, 2.0]})
    rule = Expr("gt", (feature("x"), const(0.0)))
    first = StrategyProgram(long_entry=rule, short_entry=Expr("lt", (feature("x"), const(0.0))))
    second = StrategyProgram(long_entry=Expr("not", (Expr("not", (rule,)),)), short_entry=first.short_entry)
    assert canonicalize_program(second) == first
    assert semantic_fingerprint(first, frame) == semantic_fingerprint(second, frame)
