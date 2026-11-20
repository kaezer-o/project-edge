from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PANDAS_RULES, SUPPORTED_TIMEFRAMES
from .data import DataQualityError


TICK_TIME_COLUMNS = ("time", "timestamp", "datetime", "date")


def load_tick_csv(path: str | Path) -> pd.DataFrame:
    """Load raw quote ticks with UTC timestamp, bid and ask columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    time_col = next((name for name in TICK_TIME_COLUMNS if name in frame.columns), None)
    if time_col is None:
        raise DataQualityError("Tick CSV needs time, timestamp, datetime, or date")
    required = {"bid", "ask"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataQualityError(f"Tick CSV is missing columns: {sorted(missing)}")
    frame = frame.rename(columns={time_col: "time"})
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame["bid"] = pd.to_numeric(frame["bid"], errors="coerce")
    frame["ask"] = pd.to_numeric(frame["ask"], errors="coerce")
    if frame[["time", "bid", "ask"]].isna().any().any():
        raise DataQualityError("Tick time/bid/ask contains invalid or missing values")
    if (frame[["bid", "ask"]] <= 0).any().any():
        raise DataQualityError("Tick prices must be positive")
    if (frame["ask"] < frame["bid"]).any():
        raise DataQualityError("Found crossed quote ticks where ask < bid")
    frame = frame.sort_values("time", kind="stable").set_index("time")
    frame.index.name = "time"
    return frame


def aggregate_quote_ticks(
    ticks: pd.DataFrame,
    *,
    timeframe: str = "M1",
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """Aggregate observed bid/ask ticks into paired quote-side OHLC bars.

    No empty bars are forward-filled or synthesized. OHLC prices for bid and
    ask are independently aggregated from the observed quote stream. This
    preserves real observed quote-side extrema, although it does not preserve
    the exact intrabar sequence between OHLC points.
    """
    timeframe = timeframe.upper()
    if timeframe not in PANDAS_RULES:
        raise ValueError(f"Unsupported tick aggregation timeframe: {timeframe}")
    if pip_size <= 0:
        raise ValueError("pip_size must be > 0")
    required = {"bid", "ask"}
    if required.difference(ticks.columns):
        raise DataQualityError("Tick frame must contain bid and ask columns")
    if not isinstance(ticks.index, pd.DatetimeIndex) or ticks.index.tz is None:
        raise DataQualityError("Tick index must be timezone-aware")
    if not ticks.index.is_monotonic_increasing:
        ticks = ticks.sort_index(kind="stable")
    if (ticks["ask"] < ticks["bid"]).any():
        raise DataQualityError("Found crossed quotes where ask < bid")

    working = ticks.copy()
    working["spread_pips"] = (working["ask"] - working["bid"]) / pip_size
    rule = PANDAS_RULES[timeframe]
    aggregation = {
        "open": ("bid", "first"),
        "high": ("bid", "max"),
        "low": ("bid", "min"),
        "close": ("bid", "last"),
        "ask_open": ("ask", "first"),
        "ask_high": ("ask", "max"),
        "ask_low": ("ask", "min"),
        "ask_close": ("ask", "last"),
        "spread_pips": ("spread_pips", "mean"),
        "tick_count": ("bid", "size"),
    }
    bars = working.resample(rule, label="left", closed="left").agg(**aggregation)
    bars = bars.dropna(subset=["open", "high", "low", "close", "ask_open", "ask_high", "ask_low", "ask_close"])
    if bars.empty:
        raise DataQualityError("No quote bars could be aggregated from the tick file")
    bars.index.name = "time"
    return bars


def aggregate_tick_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    timeframe: str = "M1",
    pip_size: float = 0.0001,
) -> dict[str, object]:
    ticks = load_tick_csv(input_path)
    bars = aggregate_quote_ticks(ticks, timeframe=timeframe, pip_size=pip_size)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(target, index_label="time")
    return {
        "input": str(Path(input_path).resolve()),
        "output": str(target.resolve()),
        "tick_rows": int(len(ticks)),
        "bar_rows": int(len(bars)),
        "timeframe": timeframe.upper(),
        "start_utc": bars.index.min().isoformat(),
        "end_utc": bars.index.max().isoformat(),
        "mean_spread_pips": float(bars["spread_pips"].mean()),
        "median_spread_pips": float(bars["spread_pips"].median()),
        "actual_bid_ask_ohlc": True,
        "empty_bars_synthesized": False,
    }
