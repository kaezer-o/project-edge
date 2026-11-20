from __future__ import annotations

import math

import numpy as np
import pandas as pd


def split_chronological_windows(frame: pd.DataFrame, windows: int = 3) -> list[pd.DataFrame]:
    """Split one validation partition into ordered, non-overlapping windows.

    The strategy genome is frozen across these windows; this is not a nested
    walk-forward retraining procedure.
    """
    if windows < 2:
        raise ValueError("windows must be >= 2")
    if len(frame) < windows * 20:
        return []
    chunks = np.array_split(np.arange(len(frame)), windows)
    return [frame.iloc[chunk].copy() for chunk in chunks if len(chunk)]


def block_bootstrap_expectancy(
    trades: pd.DataFrame,
    *,
    repetitions: int = 500,
    block_length: int = 5,
    seed: int = 2026,
) -> dict[str, float | int | None]:
    """Moving-block bootstrap of chronological per-trade net R observations.

    This is a descriptive uncertainty interval, not a proof of significance.
    It does not correct for candidate selection or non-stationarity.
    """
    if trades.empty or "net_r" not in trades.columns:
        return {"observations": 0, "repetitions": 0, "block_length": block_length,
                "mean": None, "ci_90_low": None, "ci_90_high": None}
    values = trades["net_r"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(values) < 8:
        return {"observations": int(len(values)), "repetitions": 0, "block_length": block_length,
                "mean": float(values.mean()) if len(values) else None,
                "ci_90_low": None, "ci_90_high": None}
    if repetitions < 100 or block_length < 1:
        raise ValueError("repetitions must be >= 100 and block_length must be >= 1")

    rng = np.random.default_rng(seed)
    n = len(values)
    block = min(block_length, n)
    n_blocks = int(math.ceil(n / block))
    estimates = np.empty(repetitions, dtype=float)
    starts = np.arange(n)
    for rep in range(repetitions):
        sampled: list[np.ndarray] = []
        for _ in range(n_blocks):
            start = int(rng.choice(starts))
            indices = (start + np.arange(block)) % n
            sampled.append(values[indices])
        estimates[rep] = np.concatenate(sampled)[:n].mean()
    return {
        "observations": int(n),
        "repetitions": int(repetitions),
        "block_length": int(block),
        "mean": float(values.mean()),
        "ci_90_low": float(np.quantile(estimates, 0.05)),
        "ci_90_high": float(np.quantile(estimates, 0.95)),
    }
