from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PANDAS_RULES, SUPPORTED_TIMEFRAMES, TIMEFRAME_SECONDS

REQUIRED_COLUMNS = ("open", "high", "low", "close")
TIME_COLUMNS = ("time", "timestamp", "datetime", "date")


class DataQualityError(ValueError):
    """Raised when source market data is not safe to research."""


def load_ohlc_csv(path: str | Path) -> pd.DataFrame:
    """Load timestamped OHLC candles and normalize timestamps to UTC.

    Accepted timestamp headers: time, timestamp, datetime, or date.
    Optional spread_pips is retained if present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    time_col = next((name for name in TIME_COLUMNS if name in frame.columns), None)
    if time_col is None:
        raise DataQualityError(
            "CSV needs a timestamp column named time, timestamp, datetime, or date"
        )

    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise DataQualityError(f"CSV is missing required columns: {', '.join(missing)}")

    frame = frame.rename(columns={time_col: "time"})
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if frame["time"].isna().any():
        bad = int(frame["time"].isna().sum())
        raise DataQualityError(f"Found {bad} invalid timestamps")

    numeric_columns = list(REQUIRED_COLUMNS)
    optional_quote_columns = (
        "ask_open", "ask_high", "ask_low", "ask_close", "spread_pips", "tick_count"
    )
    present_quote_columns = [column for column in optional_quote_columns if column in frame.columns]
    ask_columns = {"ask_open", "ask_high", "ask_low", "ask_close"}
    if present_ask := (ask_columns.intersection(frame.columns)):
        if present_ask != ask_columns:
            raise DataQualityError(
                "Actual ask-side OHLC requires all four columns: ask_open, ask_high, ask_low, ask_close"
            )
    numeric_columns.extend(present_quote_columns)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric_columns].isna().any().any():
        raise DataQualityError("OHLC/spread columns contain missing or non-numeric values")

    if (frame[list(REQUIRED_COLUMNS)] <= 0).any().any():
        raise DataQualityError("OHLC prices must be positive")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise DataQualityError("Found candles where high is below open, close, or low")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise DataQualityError("Found candles where low is above open, close, or high")
    if "spread_pips" in frame.columns and (frame["spread_pips"] < 0).any():
        raise DataQualityError("spread_pips cannot be negative")
    if "tick_count" in frame.columns and (frame["tick_count"] < 0).any():
        raise DataQualityError("tick_count cannot be negative")
    if present_ask:
        ask = frame[["ask_open", "ask_high", "ask_low", "ask_close"]]
        if (ask <= 0).any().any():
            raise DataQualityError("Ask OHLC prices must be positive")
        if (ask["ask_high"] < ask[["ask_open", "ask_close", "ask_low"]].max(axis=1)).any():
            raise DataQualityError("Found ask candles where ask_high is invalid")
        if (ask["ask_low"] > ask[["ask_open", "ask_close", "ask_high"]].min(axis=1)).any():
            raise DataQualityError("Found ask candles where ask_low is invalid")
        # When bid and ask bars are aggregated from the same quote stream,
        # ask must never be below bid at the corresponding OHLC statistic.
        for side_column in ("open", "high", "low", "close"):
            if (frame[f"ask_{side_column}"] < frame[side_column]).any():
                raise DataQualityError(
                    f"Found ask_{side_column} below bid {side_column}; quote sides are crossed"
                )

    duplicate_count = int(frame.duplicated(subset="time").sum())
    frame = frame.sort_values("time").drop_duplicates(subset="time", keep="last")
    frame = frame.set_index("time")
    frame.attrs["duplicate_timestamps_removed"] = duplicate_count
    frame.index.name = "time"
    return frame


def validate_source_timeframe(source_timeframe: str, targets: Iterable[str]) -> None:
    source = source_timeframe.upper()
    if source not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported source timeframe {source!r}")
    for target in targets:
        target = target.upper()
        if target not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported target timeframe {target!r}")
        if TIMEFRAME_SECONDS[target] < TIMEFRAME_SECONDS[source]:
            raise ValueError(
                f"Cannot derive {target} from {source}: target is lower resolution"
            )


def _is_weekend_closure(previous: pd.Timestamp, current: pd.Timestamp, seconds: float) -> bool:
    """Recognize the normal FX Friday-to-Sunday/Monday market closure."""
    return (
        seconds >= 24 * 3600
        and previous.weekday() == 4
        and current.weekday() in (6, 0)
    )


def annotate_gaps(
    frame: pd.DataFrame,
    expected_seconds: float,
    *,
    gap_factor: float = 1.5,
) -> pd.DataFrame:
    """Add explicit discontinuity metadata without creating synthetic bars.

    A gap is a timestamp interval larger than ``expected_seconds * gap_factor``.
    ``edge_gap_before`` marks the first observed candle after a gap, while
    ``edge_gap_after`` marks the last observed candle before it. Segment IDs
    increase at the first candle after each discontinuity.
    """
    if expected_seconds <= 0 or gap_factor <= 1.0:
        raise ValueError("expected_seconds must be > 0 and gap_factor must be > 1")
    if not frame.index.is_monotonic_increasing:
        raise DataQualityError("Timestamps must be sorted before gap annotation")
    if frame.index.has_duplicates:
        raise DataQualityError("Timestamps must be unique before gap annotation")

    out = frame.copy()
    deltas = out.index.to_series().diff().dt.total_seconds()
    raw_gap = (deltas > expected_seconds * gap_factor).fillna(False)
    weekend_before = pd.Series(False, index=out.index, dtype=bool)
    for position in range(1, len(out)):
        if bool(raw_gap.iloc[position]):
            weekend_before.iloc[position] = _is_weekend_closure(
                out.index[position - 1], out.index[position], float(deltas.iloc[position])
            )
    # A normal weekend closure is a market-session boundary, not an archive
    # defect. Preserve indicator state across it; abnormal/source gaps reset it.
    gap_before = raw_gap & ~weekend_before
    gap_after = gap_before.shift(-1, fill_value=False).astype(bool)
    gap_seconds_before = deltas.where(gap_before, 0.0).fillna(0.0)
    gap_seconds_after = gap_seconds_before.shift(-1, fill_value=0.0)
    out["edge_gap_before"] = gap_before.to_numpy(dtype=bool)
    out["edge_gap_after"] = gap_after.to_numpy(dtype=bool)
    out["edge_weekend_before"] = weekend_before.to_numpy(dtype=bool)
    out["edge_gap_seconds_before"] = gap_seconds_before.to_numpy(dtype=float)
    out["edge_gap_seconds_after"] = gap_seconds_after.to_numpy(dtype=float)
    out["edge_gap_duration_seconds"] = np.maximum(
        out["edge_gap_seconds_before"].to_numpy(dtype=float),
        out["edge_gap_seconds_after"].to_numpy(dtype=float),
    )
    out["edge_segment_id"] = gap_before.astype("int64").cumsum().to_numpy()
    # On raw candles, the first post-gap bar starts a fresh feature segment.
    out["edge_feature_segment_id"] = out["edge_segment_id"].to_numpy()
    # The raw candle after a gap is still an observed, valid OHLC candle.
    # Its indicators reset via segment_id; higher-timeframe aggregation marks
    # boundary-touching partial candles as contaminated separately.
    out["edge_gap_contaminated"] = False
    return out


def _finalize_resampled_metadata(
    result: pd.DataFrame,
    *,
    source_has_gap_metadata: bool,
    timeframe: str,
) -> pd.DataFrame:
    """Apply timeframe-relative gap policy to resampled candles.

    Source discontinuities remain recorded exactly. A higher-timeframe bar is
    only treated as contaminated when the gap is at least 25% of that bar's
    duration. This prevents a single missing M1 observation from invalidating
    an entire H1/D1 candle.

    Feature state is reset only when a gap spans at least 25% of the longest
    configured 200-bar feature memory (50 target bars). Shorter gaps still
    contaminate boundary bars when material, but do not repeatedly erase all
    indicator history.
    """
    if not source_has_gap_metadata:
        return result

    timeframe = timeframe.upper()
    target_seconds = float(TIMEFRAME_SECONDS[timeframe])
    contamination_threshold = target_seconds * 0.25
    feature_reset_threshold = target_seconds * 50.0

    result["edge_gap_before"] = result.pop("_edge_gap_before").fillna(False).astype(bool)
    result["edge_gap_after"] = result.pop("_edge_gap_after").fillna(False).astype(bool)
    for output_column, source_column in (
        ("edge_gap_seconds_before", "_edge_gap_seconds_before"),
        ("edge_gap_seconds_after", "_edge_gap_seconds_after"),
    ):
        if source_column in result.columns:
            result[output_column] = result.pop(source_column).fillna(0.0).astype(float)
        else:
            result[output_column] = 0.0
    result["edge_gap_duration_seconds"] = result[
        ["edge_gap_seconds_before", "edge_gap_seconds_after"]
    ].max(axis=1)
    if "_edge_weekend_before" in result.columns:
        result["edge_weekend_before"] = result.pop("_edge_weekend_before").fillna(False).astype(bool)

    segment_min = result.pop("_edge_segment_min")
    segment_max = result.pop("_edge_segment_max")
    mixed_segments = segment_min.notna() & segment_max.notna() & segment_min.ne(segment_max)
    result["edge_segment_id"] = segment_max.fillna(0).astype("int64")

    # Preserve exact source events in edge_gap_before/after and their duration
    # columns. Derived flags decide whether the event is material at this
    # timeframe and should affect bars, entries, or open positions.
    material_before = (
        result["edge_gap_before"]
        & result["edge_gap_seconds_before"].ge(contamination_threshold)
    )
    material_after = (
        result["edge_gap_after"]
        & result["edge_gap_seconds_after"].ge(contamination_threshold)
    )
    material_event = material_before | material_after
    material_mixed_gap = mixed_segments & result["edge_gap_duration_seconds"].ge(contamination_threshold)

    result["edge_gap_execution_before"] = material_before.astype(bool)
    result["edge_gap_execution_after"] = material_after.astype(bool)
    result["edge_gap_contaminated"] = (
        material_event | material_mixed_gap
    ).astype(bool)
    result["edge_gap_material_threshold_seconds"] = contamination_threshold
    result["edge_feature_reset_threshold_seconds"] = feature_reset_threshold

    # Reset the feature state only at the first observed candle after a long
    # discontinuity. The boundary candle itself is then excluded from signals.
    feature_reset_before = (
        result["edge_gap_before"]
        & result["edge_gap_seconds_before"].ge(feature_reset_threshold)
    )
    result["edge_feature_reset_before"] = feature_reset_before.astype(bool)
    result["edge_feature_segment_id"] = feature_reset_before.astype("int64").cumsum()
    return result


def _metadata_aggregation() -> dict[str, tuple[str, str]]:
    return {
        "_edge_gap_before": ("edge_gap_before", "max"),
        "_edge_gap_after": ("edge_gap_after", "max"),
        "_edge_weekend_before": ("edge_weekend_before", "max"),
        "_edge_gap_seconds_before": ("edge_gap_seconds_before", "max"),
        "_edge_gap_seconds_after": ("edge_gap_seconds_after", "max"),
        "_edge_segment_min": ("edge_segment_id", "min"),
        "_edge_segment_max": ("edge_segment_id", "max"),
    }


def resample_ohlc(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate source candles without hiding timestamp discontinuities.

    Candle timestamps represent candle-open times. Weekly bars are grouped
    into UTC Monday-to-Monday weeks. Exact source gaps remain in the metadata;
    a target bar is marked contaminated only when its associated gap reaches
    25% of that target timeframe. Feature state resets only after a gap of at
    least 50 target bars.
    """
    timeframe = timeframe.upper()
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")

    has_gap_metadata = {"edge_gap_before", "edge_gap_after", "edge_segment_id"}.issubset(frame.columns)
    if has_gap_metadata:
        frame = frame.copy()
        for column in ("edge_gap_seconds_before", "edge_gap_seconds_after"):
            if column not in frame.columns:
                frame[column] = 0.0
        if "edge_weekend_before" not in frame.columns:
            frame["edge_weekend_before"] = False
    aggregation: dict[str, tuple[str, str]] = {
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
    }
    if "spread_pips" in frame.columns:
        aggregation["spread_pips"] = ("spread_pips", "mean")
    if "tick_count" in frame.columns:
        aggregation["tick_count"] = ("tick_count", "sum")
    for quote_column, method in (
        ("ask_open", "first"), ("ask_high", "max"),
        ("ask_low", "min"), ("ask_close", "last"),
    ):
        if quote_column in frame.columns:
            aggregation[quote_column] = (quote_column, method)
    if has_gap_metadata:
        aggregation.update(_metadata_aggregation())

    if timeframe == "W1":
        iso_week = frame.index.tz_convert("UTC").tz_localize(None).to_period("W-SUN")
        grouped = frame.groupby(iso_week, sort=True)
        result = grouped.agg(**aggregation)
        result.index = pd.DatetimeIndex(
            [period.start_time.tz_localize("UTC") for period in result.index],
            name="time",
        )
    else:
        result = frame.resample(
            PANDAS_RULES[timeframe], label="left", closed="left"
        ).agg(**aggregation)

    result = result.dropna(subset=list(REQUIRED_COLUMNS))
    result = _finalize_resampled_metadata(
        result,
        source_has_gap_metadata=has_gap_metadata,
        timeframe=timeframe,
    )
    return result

def validate_source_interval(frame: pd.DataFrame, timeframe: str) -> None:
    """Check that the median input spacing matches its declared timeframe."""
    timeframe = timeframe.upper()
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported source timeframe {timeframe!r}")
    if len(frame.index) < 3:
        raise DataQualityError("At least three candles are required to verify source timeframe")
    deltas = frame.index.to_series().diff().dropna().dt.total_seconds()
    median_seconds = float(deltas.median())
    expected_seconds = float(TIMEFRAME_SECONDS[timeframe])
    tolerance = max(1.0, expected_seconds * 0.10)
    if abs(median_seconds - expected_seconds) > tolerance:
        raise DataQualityError(
            f"Declared source timeframe {timeframe} does not match median timestamp spacing "
            f"of {median_seconds:g} seconds (expected about {expected_seconds:g})"
        )


def data_quality_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        raise DataQualityError("No usable candles were loaded")
    deltas = frame.index.to_series().diff().dropna()
    median_delta = deltas.median() if not deltas.empty else pd.NaT
    expected_delta = deltas.mode().min() if not deltas.empty else pd.NaT
    threshold = expected_delta * 1.5 if pd.notna(expected_delta) else pd.NaT
    gaps = deltas[deltas > threshold] if pd.notna(threshold) else deltas.iloc[0:0]
    gap_seconds = gaps.dt.total_seconds()
    gap_buckets = {
        "90_seconds_to_5_minutes": int(((gap_seconds > 90) & (gap_seconds <= 300)).sum()),
        "5_to_60_minutes": int(((gap_seconds > 300) & (gap_seconds <= 3600)).sum()),
        "1_to_24_hours": int(((gap_seconds > 3600) & (gap_seconds <= 86400)).sum()),
        "over_24_hours": int((gap_seconds > 86400).sum()),
    }
    weekend_like = 0
    for gap_index, duration in gaps.items():
        previous = gap_index - pd.Timedelta(
    		float(duration.total_seconds()),
    		unit="s",
	)
        if _is_weekend_closure(previous, gap_index, float(duration.total_seconds())):
            weekend_like += 1
    return {
        "rows": int(len(frame)),
        "start_utc": frame.index.min().isoformat(),
        "end_utc": frame.index.max().isoformat(),
        "duplicate_timestamps_removed": int(frame.attrs.get("duplicate_timestamps_removed", 0)),
        "median_bar_interval": str(median_delta) if pd.notna(median_delta) else None,
        "modal_bar_interval": str(expected_delta) if pd.notna(expected_delta) else None,
        "large_gaps": int(len(gaps)),
        "gap_duration_buckets": gap_buckets,
        "weekend_closure_like_gaps": int(weekend_like),
        "non_weekend_gaps": int(len(gaps) - weekend_like),
        "has_spread_column": "spread_pips" in frame.columns,
        "has_actual_ask_ohlc": {"ask_open", "ask_high", "ask_low", "ask_close"}.issubset(frame.columns),
    }


def build_gap_audit(
    frame: pd.DataFrame,
    expected_seconds: float,
    *,
    gap_factor: float = 1.5,
) -> pd.DataFrame:
    """Return one row per timestamp discontinuity, without synthesizing bars.

    Weekend-like closures are distinguished from non-weekend source gaps.
    ``estimated_missing_bars`` is only a timestamp-based estimate; it does not
    establish how many quotes the provider should have published.
    """
    if expected_seconds <= 0 or gap_factor <= 1.0:
        raise ValueError("expected_seconds must be > 0 and gap_factor must be > 1")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise DataQualityError("Gap audit requires sorted, unique timestamps")

    deltas = frame.index.to_series().diff().dt.total_seconds()
    positions = [i for i, value in enumerate(deltas.to_numpy())
                 if i > 0 and pd.notna(value) and value > expected_seconds * gap_factor]
    rows: list[dict[str, object]] = []
    for position in positions:
        previous = frame.index[position - 1]
        current = frame.index[position]
        seconds = float(deltas.iloc[position])
        weekend = _is_weekend_closure(previous, current, seconds)
        estimated_missing = max(0, int(round(seconds / expected_seconds)) - 1)
        if weekend:
            category = "weekend_market_closure_like"
        elif seconds <= 5 * 60:
            category = "short_non_weekend_gap"
        elif seconds <= 60 * 60:
            category = "medium_non_weekend_gap"
        elif seconds <= 24 * 60 * 60:
            category = "long_non_weekend_gap"
        else:
            category = "over_24h_non_weekend_gap"
        rows.append(
            {
                "previous_bar_utc": previous.isoformat(),
                "next_bar_utc": current.isoformat(),
                "gap_seconds": seconds,
                "gap_minutes": seconds / 60.0,
                "estimated_missing_bars": estimated_missing,
                "category": category,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "previous_bar_utc", "next_bar_utc", "gap_seconds", "gap_minutes",
            "estimated_missing_bars", "category",
        ],
    )


def gap_annotation_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize discontinuity metadata already attached to a timeframe."""
    def boolean_count(column: str) -> int:
        if column not in frame.columns:
            return 0
        return int(frame[column].fillna(False).astype(bool).sum())

    duration = (
        pd.to_numeric(frame["edge_gap_duration_seconds"], errors="coerce")
        if "edge_gap_duration_seconds" in frame.columns
        else pd.Series(dtype=float)
    )
    feature_segments = (
        int(frame["edge_feature_segment_id"].nunique())
        if "edge_feature_segment_id" in frame.columns and not frame.empty
        else 1 if not frame.empty else 0
    )
    threshold = (
        float(frame["edge_gap_material_threshold_seconds"].iloc[0])
        if "edge_gap_material_threshold_seconds" in frame.columns and not frame.empty
        else None
    )
    return {
        "gap_events": boolean_count("edge_gap_before"),
        "gap_boundaries_before": boolean_count("edge_gap_before"),
        "gap_boundaries_after": boolean_count("edge_gap_after"),
        "execution_gap_events_before": boolean_count("edge_gap_execution_before"),
        "execution_gap_events_after": boolean_count("edge_gap_execution_after"),
        "weekend_closures": boolean_count("edge_weekend_before"),
        "gap_contaminated_bars": boolean_count("edge_gap_contaminated"),
        "feature_resets": boolean_count("edge_feature_reset_before"),
        "feature_segments": feature_segments,
        "material_gap_threshold_seconds": threshold,
        "feature_reset_threshold_seconds": (
            float(frame["edge_feature_reset_threshold_seconds"].iloc[0])
            if "edge_feature_reset_threshold_seconds" in frame.columns and not frame.empty
            else None
        ),
        "max_gap_duration_seconds": (
            float(duration.max()) if not duration.empty and duration.notna().any() else 0.0
        ),
    }

