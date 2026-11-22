from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import pandas as pd


NUMERIC_OPS = {
    "const", "feature", "add", "sub", "mul", "div", "abs", "min", "max",
    "lag", "rolling_mean", "rolling_std",
}
BOOLEAN_OPS = {
    "gt", "lt", "cross_above", "cross_below", "and", "or", "not",
}
ARITY = {
    "const": 0, "feature": 0, "add": 2, "sub": 2, "mul": 2, "div": 2,
    "abs": 1, "min": 2, "max": 2, "lag": 1, "rolling_mean": 1,
    "rolling_std": 1, "gt": 2, "lt": 2, "cross_above": 2,
    "cross_below": 2, "and": 2, "or": 2, "not": 1,
}
WINDOW_OPS = {"lag", "rolling_mean", "rolling_std"}


@dataclass(frozen=True)
class Expr:
    """A typed, serializable expression node. Never executes arbitrary code."""

    op: str
    args: tuple[Expr, ...] = ()
    value: str | float | int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "op": self.op,
            "value": self.value,
            "args": [arg.to_dict() for arg in self.args],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Expr:
        if not isinstance(payload, dict):
            raise ValueError("Expression must be an object")
        args = payload.get("args", [])
        if not isinstance(args, list):
            raise ValueError("Expression args must be a list")
        return cls(
            op=str(payload.get("op", "")),
            value=payload.get("value"),
            args=tuple(cls.from_dict(arg) for arg in args),
        )


@dataclass(frozen=True)
class StrategyProgram:
    long_entry: Expr
    short_entry: Expr
    stop_atr: float = 1.5
    reward_risk: float = 2.0
    max_holding_bars: int = 100

    def __post_init__(self) -> None:
        if not math.isfinite(self.stop_atr) or not 0.25 <= self.stop_atr <= 8.0:
            raise ValueError("stop_atr must be finite and between 0.25 and 8")
        if not math.isfinite(self.reward_risk) or not 0.25 <= self.reward_risk <= 10.0:
            raise ValueError("reward_risk must be finite and between 0.25 and 10")
        if not 1 <= self.max_holding_bars <= 5000:
            raise ValueError("max_holding_bars must be between 1 and 5000")
        validate_expr(self.long_entry, expected="bool")
        validate_expr(self.short_entry, expected="bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "long_entry": self.long_entry.to_dict(),
            "short_entry": self.short_entry.to_dict(),
            "stop_atr": self.stop_atr,
            "reward_risk": self.reward_risk,
            "max_holding_bars": self.max_holding_bars,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StrategyProgram:
        if payload.get("version") != 1:
            raise ValueError("Unsupported strategy-program version")
        return cls(
            long_entry=Expr.from_dict(payload["long_entry"]),
            short_entry=Expr.from_dict(payload["short_entry"]),
            stop_atr=float(payload["stop_atr"]),
            reward_risk=float(payload["reward_risk"]),
            max_holding_bars=int(payload["max_holding_bars"]),
        )

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]


def _feature_dimension(name: str) -> str:
    """Return a semantic scale category for a strategy feature."""
    if name.startswith("edge_"):
        raise ValueError(
            f"Operational metadata is not a strategy feature: {name}"
        )
    # Check normalized-feature suffixes before broad EMA/price prefixes:
    # e.g. ema_gap_atr is a ratio, not an EMA price level.
    if name.endswith("_atr") or name in {"ema_gap_atr", "price_change_atr"}:
        return "ratio"
    if (
        name.startswith("ema_")
        or name.startswith("donchian_")
        or name in {"open", "high", "low", "close"}
    ):
        return "price_level"
    if name.startswith("atr_"):
        return "price_distance"
    if name.startswith("rsi_"):
        return "oscillator"
    return "generic"


def _expr_dimension(expr: Expr) -> str | None:
    """Infer semantic scale; None represents a scalar constant."""
    if expr.op == "const":
        return None
    if expr.op == "feature":
        return _feature_dimension(str(expr.value))
    dimensions = [_expr_dimension(child) for child in expr.args]
    if expr.op in {"abs", "lag", "rolling_mean"}:
        return dimensions[0]
    if expr.op == "rolling_std":
        # Dispersion of a price level is a price distance, not another
        # absolute price level. Other indicator scales retain their units.
        return (
            "price_distance"
            if dimensions[0] == "price_level"
            else dimensions[0]
        )
    if expr.op in {"add", "sub", "min", "max"}:
        left, right = dimensions
        if left is None:
            return right
        if right is None:
            return left
        if left == right:
            if expr.op == "sub" and left == "price_level":
                return "price_distance"
            return left
        if expr.op == "add" and {left, right} == {
            "price_level", "price_distance"
        }:
            return "price_level"
        if (
            expr.op == "sub"
            and left == "price_level"
            and right == "price_distance"
        ):
            return "price_level"
        raise ValueError(
            f"{expr.op} requires compatible feature units: {left} vs {right}"
        )
    if expr.op == "mul":
        left, right = dimensions
        if left is not None and right is not None:
            raise ValueError(
                "mul requires at least one dimensionless constant operand"
            )
        return left if left is not None else right
    if expr.op == "div":
        left, right = dimensions
        if right is None:
            return left
        if left is not None and left == right:
            return "ratio"
        raise ValueError(
            f"div requires a constant denominator or matching units: {left} / {right}"
        )
    return None


def _expr_has_feature(expr: Expr) -> bool:
    if expr.op == "feature":
        return True
    return any(_expr_has_feature(child) for child in expr.args)


def _require_compatible_comparison(expr: Expr) -> None:
    left, right = (_expr_dimension(child) for child in expr.args)
    if not any(_expr_has_feature(child) for child in expr.args):
        raise ValueError(
            f"{expr.op} requires at least one market feature; constant-only rules never vary"
        )
    raw_scale = {"price_level", "price_distance"}
    if left is None and right in raw_scale:
        raise ValueError(
            "Scalar thresholds cannot be compared with raw price values; normalize by ATR first"
        )
    if right is None and left in raw_scale:
        raise ValueError(
            "Scalar thresholds cannot be compared with raw price values; normalize by ATR first"
        )
    if left is not None and right is not None and left != right:
        raise ValueError(
            f"{expr.op} requires compatible feature units: {left} vs {right}"
        )


def _expr_type(expr: Expr, *, depth: int, count: list[int]) -> str:
    if depth > 12:
        raise ValueError("Expression exceeds maximum depth of 12")
    count[0] += 1
    if count[0] > 128:
        raise ValueError("Expression exceeds maximum size of 128 nodes")
    if expr.op not in ARITY:
        raise ValueError(f"Unsupported expression operation: {expr.op}")
    if expr.op not in ({"const", "feature"} | WINDOW_OPS) and expr.value is not None:
        raise ValueError(f"Operation {expr.op} does not accept a value")
    if len(expr.args) != ARITY[expr.op]:
        raise ValueError(f"Operation {expr.op} requires {ARITY[expr.op]} arguments")

    if expr.op == "const":
        if isinstance(expr.value, bool) or not isinstance(expr.value, (int, float)):
            raise ValueError("const requires a numeric value")
        if not math.isfinite(float(expr.value)):
            raise ValueError("const must be finite")
        return "number"

    if expr.op == "feature":
        if not isinstance(expr.value, str) or not expr.value.strip():
            raise ValueError("feature requires a non-empty column name")
        _feature_dimension(expr.value)
        return "number"

    if expr.op in WINDOW_OPS:
        if isinstance(expr.value, bool) or not isinstance(expr.value, int):
            raise ValueError(f"{expr.op} requires an integer lookback")
        if not 1 <= expr.value <= 250:
            raise ValueError("Lookback must be between 1 and 250")

    child_types = [
        _expr_type(child, depth=depth + 1, count=count)
        for child in expr.args
    ]
    if expr.op in {"add", "sub", "mul", "div", "abs", "min", "max",
                   "lag", "rolling_mean", "rolling_std"}:
        if any(kind != "number" for kind in child_types):
            raise ValueError(f"{expr.op} accepts numeric arguments only")
        _expr_dimension(expr)
        return "number"
    if expr.op in {"gt", "lt", "cross_above", "cross_below"}:
        if child_types != ["number", "number"]:
            raise ValueError(f"{expr.op} requires two numeric arguments")
        _require_compatible_comparison(expr)
        return "bool"
    if expr.op in {"and", "or"}:
        if child_types != ["bool", "bool"]:
            raise ValueError(f"{expr.op} requires two boolean arguments")
        return "bool"
    if expr.op == "not":
        if child_types != ["bool"]:
            raise ValueError("not requires one boolean argument")
        return "bool"
    raise ValueError(f"Unsupported expression operation: {expr.op}")


def validate_expr(expr: Expr, *, expected: str = "number") -> None:
    if expected not in {"number", "bool"}:
        raise ValueError("expected must be 'number' or 'bool'")
    actual = _expr_type(expr, depth=1, count=[0])
    if actual != expected:
        raise ValueError(f"Expected {expected} expression, received {actual}")


def canonicalize_expr(expr: Expr) -> Expr:
    """Return a semantics-preserving, deterministic simplified expression tree."""
    children = tuple(canonicalize_expr(child) for child in expr.args)
    node = Expr(expr.op, children, expr.value)

    if node.op == "not" and node.args[0].op == "not":
        return canonicalize_expr(node.args[0].args[0])
    if node.op in {"and", "or"}:
        left, right = node.args
        if left == right:
            return left
        # These operators are commutative; stable ordering makes equivalent
        # operand orderings share a structural representation.
        ordered = tuple(sorted((left, right), key=_expr_sort_key))
        return Expr(node.op, ordered)
    if node.op in {"min", "max"} and node.args[0] == node.args[1]:
        return node.args[0]
    if node.op == "abs" and node.args[0].op == "feature":
        if _feature_dimension(str(node.args[0].value)) == "price_level":
            return node.args[0]
    return node


def _expr_sort_key(expr: Expr) -> str:
    return json.dumps(expr.to_dict(), sort_keys=True, separators=(",", ":"))


def canonicalize_program(program: StrategyProgram) -> StrategyProgram:
    """Canonicalize entry expressions while preserving execution parameters."""
    return StrategyProgram(
        long_entry=canonicalize_expr(program.long_entry),
        short_entry=canonicalize_expr(program.short_entry),
        stop_atr=program.stop_atr,
        reward_risk=program.reward_risk,
        max_holding_bars=program.max_holding_bars,
    )


def semantic_fingerprint(program: StrategyProgram, frame: pd.DataFrame) -> str:
    """Hash the exact per-bar long/short decision stream on supplied data."""
    signal = evaluate_program(program, frame)
    payload = signal.to_numpy(dtype="int8", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()


def evaluate_expr(expr: Expr, frame: pd.DataFrame) -> pd.Series:
    """Evaluate an expression causally over a chronological feature frame."""
    expected = "bool" if expr.op in BOOLEAN_OPS else "number"
    validate_expr(expr, expected=expected)
    cache: dict[int, pd.Series] = {}

    def evaluate(node: Expr) -> pd.Series:
        key = id(node)
        if key in cache:
            return cache[key]
        op = node.op
        if op == "const":
            result = pd.Series(float(node.value), index=frame.index, dtype=float)
        elif op == "feature":
            if node.value not in frame.columns:
                raise ValueError(f"Missing strategy feature: {node.value}")
            result = pd.to_numeric(frame[node.value], errors="coerce").astype(float)
        else:
            values = [evaluate(child) for child in node.args]
            if op == "add":
                result = values[0] + values[1]
            elif op == "sub":
                result = values[0] - values[1]
            elif op == "mul":
                result = values[0] * values[1]
            elif op == "div":
                denominator = values[1].where(values[1].abs() > 1e-12)
                result = values[0] / denominator
            elif op == "abs":
                result = values[0].abs()
            elif op == "min":
                result = pd.concat(values, axis=1).min(axis=1, skipna=False)
            elif op == "max":
                result = pd.concat(values, axis=1).max(axis=1, skipna=False)
            elif op == "lag":
                result = values[0].shift(int(node.value))
            elif op == "rolling_mean":
                result = values[0].rolling(int(node.value), min_periods=int(node.value)).mean()
            elif op == "rolling_std":
                result = values[0].rolling(int(node.value), min_periods=int(node.value)).std(ddof=0)
            elif op == "gt":
                result = (values[0] > values[1]).astype("boolean")
                result = result.mask(values[0].isna() | values[1].isna(), pd.NA)
            elif op == "lt":
                result = (values[0] < values[1]).astype("boolean")
                result = result.mask(values[0].isna() | values[1].isna(), pd.NA)
            elif op == "cross_above":
                current = (values[0] > values[1]).astype("boolean")
                previous = (values[0].shift(1) <= values[1].shift(1)).astype("boolean")
                result = current & previous
                invalid = (
                    values[0].isna() | values[1].isna()
                    | values[0].shift(1).isna() | values[1].shift(1).isna()
                )
                result = result.mask(invalid, pd.NA)
            elif op == "cross_below":
                current = (values[0] < values[1]).astype("boolean")
                previous = (values[0].shift(1) >= values[1].shift(1)).astype("boolean")
                result = current & previous
                invalid = (
                    values[0].isna() | values[1].isna()
                    | values[0].shift(1).isna() | values[1].shift(1).isna()
                )
                result = result.mask(invalid, pd.NA)
            elif op == "and":
                result = values[0].astype("boolean") & values[1].astype("boolean")
            elif op == "or":
                result = values[0].astype("boolean") | values[1].astype("boolean")
            elif op == "not":
                result = ~values[0].astype("boolean")
            else:
                raise ValueError(f"Unsupported expression operation: {op}")
        cache[key] = result
        return result

    return evaluate(expr)


def evaluate_program(program: StrategyProgram, frame: pd.DataFrame) -> pd.Series:
    """Return close-of-bar intent; the existing backtester fills next bar."""
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Strategy evaluation requires chronological data")
    long_signal = evaluate_expr(program.long_entry, frame).fillna(False).astype(bool)
    short_signal = evaluate_expr(program.short_entry, frame).fillna(False).astype(bool)
    signal = pd.Series(0, index=frame.index, dtype="int8", name="signal")
    signal.loc[long_signal & ~short_signal] = 1
    signal.loc[short_signal & ~long_signal] = -1
    return signal
