from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = tuple(
    [f"ema_{period}" for period in (10, 20, 40, 50, 100, 200)]
    + ["rsi_14", "atr_14", "donchian_high_20", "donchian_low_20", "donchian_high_55", "donchian_low_55"]
)


def _add_features_contiguous(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate causal indicators within one uninterrupted data segment."""
    out = frame.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    previous_close = close.shift(1)

    for period in (10, 20, 40, 50, 100, 200):
        out[f"ema_{period}"] = close.ewm(
            span=period, adjust=False, min_periods=period
        ).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + relative_strength))
    out.loc[(loss == 0) & (gain > 0), "rsi_14"] = 100
    out.loc[(gain == 0) & (loss > 0), "rsi_14"] = 0

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    for lookback in (20, 55):
        out[f"donchian_high_{lookback}"] = high.rolling(lookback).max().shift(1)
        out[f"donchian_low_{lookback}"] = low.rolling(lookback).min().shift(1)
    return out


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate causal indicators, resetting every time a data gap occurs.

    Resampled frames can provide ``edge_feature_segment_id``. Raw annotated
    frames can provide ``edge_segment_id``. No rolling/EMA state is carried
    across segment boundaries. Gap-contaminated bars are explicitly invalidated.
    """
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    segment_column = (
        "edge_feature_segment_id"
        if "edge_feature_segment_id" in out.columns
        else "edge_segment_id" if "edge_segment_id" in out.columns else None
    )

    if segment_column is None:
        out = _add_features_contiguous(out)
    else:
        for _, positions in out.groupby(segment_column, sort=False, dropna=False).indices.items():
            group = out.iloc[positions].copy()
            calculated = _add_features_contiguous(group)
            for column in FEATURE_COLUMNS:
                out.loc[group.index, column] = calculated[column].to_numpy()

    if "edge_gap_contaminated" in out.columns:
        invalid = out["edge_gap_contaminated"].fillna(False).astype(bool)
        existing = [column for column in FEATURE_COLUMNS if column in out.columns]
        out.loc[invalid, existing] = np.nan
    return out
