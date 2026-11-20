from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_ohlc_csv


def compare_ohlc_sources(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    pip_size: float,
    reference_name: str = "reference",
    comparison_name: str = "comparison",
) -> dict[str, object]:
    """Compare two normalized OHLC feeds on exact overlapping timestamps.

    Differences are descriptive. Two brokers/providers may construct candles
    from different quote streams, timezone/session boundaries or bar-close
    conventions; mismatch is not automatically evidence that either feed is bad.
    """
    if pip_size <= 0:
        raise ValueError("pip_size must be > 0")
    required = {"open", "high", "low", "close"}
    for label, frame in ((reference_name, reference), (comparison_name, comparison)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing OHLC columns: {sorted(missing)}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{label} must use a DatetimeIndex")
        if frame.index.tz is None:
            raise ValueError(f"{label} timestamps must be timezone-aware")
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError(f"{label} timestamps must be sorted and unique")

    ref_index = reference.index
    cmp_index = comparison.index
    overlap = ref_index.intersection(cmp_index)
    only_ref = ref_index.difference(cmp_index)
    only_cmp = cmp_index.difference(ref_index)
    if len(overlap) == 0:
        return {
            "reference": reference_name,
            "comparison": comparison_name,
            "reference_rows": int(len(reference)),
            "comparison_rows": int(len(comparison)),
            "overlap_rows": 0,
            "reference_only_timestamps": int(len(only_ref)),
            "comparison_only_timestamps": int(len(only_cmp)),
            "timestamp_overlap_fraction_reference": 0.0 if len(reference) else None,
            "timestamp_overlap_fraction_comparison": 0.0 if len(comparison) else None,
            "price_comparison": None,
            "warning": "No exact timestamp overlap; check timezone and candle-open/close conventions.",
        }

    left = reference.loc[overlap, list(required)].astype(float)
    right = comparison.loc[overlap, list(required)].astype(float)
    price_report: dict[str, object] = {}
    for column in ("open", "high", "low", "close"):
        difference = (left[column] - right[column]).to_numpy(dtype=float) / pip_size
        absolute = np.abs(difference)
        price_report[column] = {
            "mean_signed_difference_pips": float(np.mean(difference)),
            "mean_absolute_difference_pips": float(np.mean(absolute)),
            "median_absolute_difference_pips": float(np.median(absolute)),
            "p95_absolute_difference_pips": float(np.quantile(absolute, 0.95)),
            "p99_absolute_difference_pips": float(np.quantile(absolute, 0.99)),
            "maximum_absolute_difference_pips": float(np.max(absolute)),
            "exact_match_fraction": float(np.isclose(difference, 0.0, atol=1e-9).mean()),
        }

    ref_returns = left["close"].pct_change()
    cmp_returns = right["close"].pct_change()
    valid_returns = ref_returns.notna() & cmp_returns.notna()
    return_left = ref_returns[valid_returns]
    return_right = cmp_returns[valid_returns]
    return_correlation = None
    if (
        len(return_left) >= 3
        and float(return_left.std()) > 0
        and float(return_right.std()) > 0
    ):
        return_correlation = float(return_left.corr(return_right))
    ref_range = (left["high"] - left["low"]) / pip_size
    cmp_range = (right["high"] - right["low"]) / pip_size
    range_correlation = None
    if len(overlap) >= 3 and float(ref_range.std()) > 0 and float(cmp_range.std()) > 0:
        range_correlation = float(ref_range.corr(cmp_range))

    quote_side_comparison: dict[str, object] | None = None
    ask_columns = {"ask_open", "ask_high", "ask_low", "ask_close"}
    if ask_columns.issubset(reference.columns) and ask_columns.issubset(comparison.columns):
        quote_side_comparison = {}
        for column in ("ask_open", "ask_high", "ask_low", "ask_close"):
            difference = (
                reference.loc[overlap, column].astype(float)
                - comparison.loc[overlap, column].astype(float)
            ).to_numpy(dtype=float) / pip_size
            quote_side_comparison[column] = {
                "mean_absolute_difference_pips": float(np.abs(difference).mean()),
                "p95_absolute_difference_pips": float(np.quantile(np.abs(difference), 0.95)),
                "maximum_absolute_difference_pips": float(np.abs(difference).max()),
            }
        ref_spread = (
            reference.loc[overlap, "ask_open"].astype(float)
            - reference.loc[overlap, "open"].astype(float)
        ) / pip_size
        cmp_spread = (
            comparison.loc[overlap, "ask_open"].astype(float)
            - comparison.loc[overlap, "open"].astype(float)
        ) / pip_size
        quote_side_comparison["open_spread"] = {
            "reference_median_pips": float(ref_spread.median()),
            "comparison_median_pips": float(cmp_spread.median()),
            "reference_p95_pips": float(ref_spread.quantile(0.95)),
            "comparison_p95_pips": float(cmp_spread.quantile(0.95)),
        }

    return {
        "reference": reference_name,
        "comparison": comparison_name,
        "reference_rows": int(len(reference)),
        "comparison_rows": int(len(comparison)),
        "overlap_rows": int(len(overlap)),
        "reference_only_timestamps": int(len(only_ref)),
        "comparison_only_timestamps": int(len(only_cmp)),
        "timestamp_overlap_fraction_reference": float(len(overlap) / max(len(reference), 1)),
        "timestamp_overlap_fraction_comparison": float(len(overlap) / max(len(comparison), 1)),
        "overlap_start_utc": overlap.min().isoformat(),
        "overlap_end_utc": overlap.max().isoformat(),
        "close_return_correlation": return_correlation,
        "candle_range_correlation": range_correlation,
        "price_comparison": price_report,
        "quote_side_comparison": quote_side_comparison,
        "interpretation_warning": (
            "Provider prices are not expected to match exactly. Compare timestamp coverage, "
            "return agreement and price-difference distributions; do not treat this report "
            "as proof that either provider is authoritative."
        ),
    }


def compare_ohlc_csv_files(
    reference_path: str | Path,
    comparison_path: str | Path,
    *,
    pip_size: float,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    reference = load_ohlc_csv(reference_path)
    comparison = load_ohlc_csv(comparison_path)
    report = compare_ohlc_sources(
        reference,
        comparison,
        pip_size=pip_size,
        reference_name=Path(reference_path).name,
        comparison_name=Path(comparison_path).name,
    )
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        report["output"] = str(target.resolve())
        target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
