from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


MORPHOLOGY_COLUMNS = (
    "ret_1_atr",
    "ret_3_atr",
    "ret_8_atr",
    "body_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "range_atr",
    "trend_efficiency_10",
    "volatility_ratio_10_50",
    "compression_ratio_8_32",
    "ema_gap_atr",
    "rsi_centered",
    "prior_high_distance_atr",
    "prior_low_distance_atr",
) + tuple(f"shape_ret_{lag:02d}" for lag in range(12, 0, -1))


@dataclass(frozen=True)
class PatternModel:
    columns: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    centroids: tuple[tuple[float, ...], ...]
    n_clusters: int
    fit_rows: int
    status: str = "fitted"
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "means": list(self.means),
            "scales": list(self.scales),
            "centroids": [list(row) for row in self.centroids],
            "n_clusters": self.n_clusters,
            "fit_rows": self.fit_rows,
            "status": self.status,
            "reason": self.reason,
        }


def _add_morphology_features_contiguous(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal, scale-normalized descriptions inside one segment."""
    out = frame.copy()
    close, open_, high, low = (out[c].astype(float) for c in ("close", "open", "high", "low"))
    atr = out["atr_14"].replace(0, np.nan)
    previous_close = close.shift(1)

    out["ret_1_atr"] = (close - previous_close) / atr
    out["ret_3_atr"] = (close - close.shift(3)) / atr
    out["ret_8_atr"] = (close - close.shift(8)) / atr
    out["body_atr"] = (close - open_) / atr
    out["upper_wick_atr"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / atr
    out["lower_wick_atr"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / atr
    out["range_atr"] = (high - low) / atr

    directional = close.diff().abs()
    net_move = (close - close.shift(10)).abs()
    path_length = directional.rolling(10, min_periods=10).sum().replace(0, np.nan)
    out["trend_efficiency_10"] = net_move / path_length

    returns = close.pct_change()
    short_vol = returns.rolling(10, min_periods=10).std()
    long_vol = returns.rolling(50, min_periods=50).std().replace(0, np.nan)
    out["volatility_ratio_10_50"] = short_vol / long_vol

    range_8 = high.rolling(8).max() - low.rolling(8).min()
    range_32 = (high.rolling(32).max() - low.rolling(32).min()).replace(0, np.nan)
    out["compression_ratio_8_32"] = range_8 / range_32
    out["ema_gap_atr"] = (out["ema_20"] - out["ema_50"]) / atr
    out["rsi_centered"] = (out["rsi_14"] - 50.0) / 50.0

    prior_high = high.rolling(20, min_periods=20).max().shift(1)
    prior_low = low.rolling(20, min_periods=20).min().shift(1)
    out["prior_high_distance_atr"] = (close - prior_high) / atr
    out["prior_low_distance_atr"] = (close - prior_low) / atr

    for lag in range(12, 0, -1):
        out[f"shape_ret_{lag:02d}"] = (close.shift(lag - 1) - close.shift(lag)) / atr
    return out


def add_morphology_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build price-shape features without allowing lookbacks across data gaps."""
    required = {"open", "high", "low", "close", "atr_14", "ema_20", "ema_50", "rsi_14"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Frame missing base features: {sorted(missing)}")
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    segment_column = (
        "edge_feature_segment_id"
        if "edge_feature_segment_id" in out.columns
        else "edge_segment_id" if "edge_segment_id" in out.columns else None
    )
    morphology_columns = list(MORPHOLOGY_COLUMNS)
    if segment_column is None:
        out = _add_morphology_features_contiguous(out)
    else:
        for _, positions in out.groupby(segment_column, sort=False, dropna=False).indices.items():
            group = out.iloc[positions].copy()
            calculated = _add_morphology_features_contiguous(group)
            for column in morphology_columns:
                out.loc[group.index, column] = calculated[column].to_numpy()

    if "edge_gap_contaminated" in out.columns:
        invalid = out["edge_gap_contaminated"].fillna(False).astype(bool)
        out.loc[invalid, morphology_columns] = np.nan
    return out


def _kmeans_fit(values: np.ndarray, clusters: int, seed: int, max_iter: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_rows = len(values)
    centroids = np.empty((clusters, values.shape[1]), dtype=float)

    # K-means++ initialization reduces the chance of nearly identical prototypes.
    first = int(rng.integers(n_rows))
    centroids[0] = values[first]
    closest_sq = np.sum((values - centroids[0]) ** 2, axis=1)
    for k in range(1, clusters):
        total = float(closest_sq.sum())
        if total <= 1e-12:
            centroids[k] = values[int(rng.integers(n_rows))]
        else:
            idx = int(rng.choice(n_rows, p=closest_sq / total))
            centroids[k] = values[idx]
        distance_sq = np.sum((values - centroids[k]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, distance_sq)

    for _ in range(max_iter):
        distances = np.sum((values[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = distances.argmin(axis=1)
        updated = centroids.copy()
        for k in range(clusters):
            members = values[labels == k]
            if len(members):
                updated[k] = members.mean(axis=0)
            else:
                updated[k] = values[int(rng.integers(n_rows))]
        if np.allclose(updated, centroids, rtol=1e-5, atol=1e-6):
            centroids = updated
            break
        centroids = updated
    return centroids


def fit_pattern_model(
    frame: pd.DataFrame,
    *,
    fit_end: int,
    n_clusters: int = 6,
    random_state: int = 17,
) -> tuple[pd.DataFrame, PatternModel]:
    """Fit recurring candle/market-state prototypes on training rows only.

    The resulting model is then applied to all timestamps. The fit never sees
    validation or final-test rows, preventing unsupervised look-ahead leakage.
    """
    if not 2 <= n_clusters <= 16:
        raise ValueError("n_clusters must be between 2 and 16")
    if not 0 < fit_end <= len(frame):
        raise ValueError("fit_end must identify a non-empty training prefix")

    missing = set(MORPHOLOGY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Frame missing morphology features: {sorted(missing)}")

    fit_values = frame.iloc[:fit_end].loc[:, MORPHOLOGY_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    # Pattern recognition is an optional research layer. Sparse/gap-heavy
    # partitions should disable clustering explicitly rather than abort the
    # entire evolutionary run. Rule evolution can still use any valid numeric
    # features that remain in the training sample.
    if len(fit_values) < 30:
        out = frame.copy()
        stale_motif_columns = [
            column for column in out.columns
            if column.startswith("motif_")
        ]
        if stale_motif_columns:
            out = out.drop(columns=stale_motif_columns)
        out["motif_distance"] = np.nan
        model = PatternModel(
            columns=tuple(MORPHOLOGY_COLUMNS),
            means=(),
            scales=(),
            centroids=(),
            n_clusters=0,
            fit_rows=len(fit_values),
            status="disabled_insufficient_training_patterns",
            reason=(
                f"Only {len(fit_values)} valid training patterns were available; "
                "at least 30 are required to fit motif clusters. Evolution may "
                "continue using other valid features."
            ),
        )
        return out, model

    means = fit_values.mean(axis=0).to_numpy(dtype=float)
    scales = fit_values.std(axis=0, ddof=0).to_numpy(dtype=float)
    scales[~np.isfinite(scales) | (scales < 1e-9)] = 1.0
    standardized = (fit_values.to_numpy(dtype=float) - means) / scales
    effective_clusters = min(n_clusters, max(2, len(standardized) // 10))
    centroids = _kmeans_fit(standardized, effective_clusters, random_state)

    all_values = frame.loc[:, MORPHOLOGY_COLUMNS].replace([np.inf, -np.inf], np.nan)
    valid_mask = all_values.notna().all(axis=1).to_numpy()
    standardized_all = np.zeros((len(frame), len(MORPHOLOGY_COLUMNS)), dtype=float)
    if valid_mask.any():
        standardized_all[valid_mask] = (all_values.loc[valid_mask].to_numpy(dtype=float) - means) / scales
        distances = np.sum(
            (standardized_all[valid_mask, None, :] - centroids[None, :, :]) ** 2,
            axis=2,
        )
        labels = distances.argmin(axis=1)
        nearest_distance = np.sqrt(distances.min(axis=1))
    else:
        labels = np.array([], dtype=int)
        nearest_distance = np.array([], dtype=float)

    out = frame.copy()
    for cluster_id in range(effective_clusters):
        column = f"motif_{cluster_id:02d}"
        out[column] = np.nan
        out.loc[valid_mask, column] = (labels == cluster_id).astype(float)
    out["motif_distance"] = np.nan
    out.loc[valid_mask, "motif_distance"] = nearest_distance

    model = PatternModel(
        columns=tuple(MORPHOLOGY_COLUMNS),
        means=tuple(float(v) for v in means),
        scales=tuple(float(v) for v in scales),
        centroids=tuple(tuple(float(v) for v in row) for row in centroids),
        n_clusters=effective_clusters,
        fit_rows=len(fit_values),
    )
    return out, model


def add_causal_motif_dynamics(frame: pd.DataFrame, model: PatternModel) -> pd.DataFrame:
    """Add causal motif persistence/transition features using prior observations.

    The current motif is emitted from a frozen training-only centroid model.
    Transition counts and occurrence frequencies are updated only after each
    row's features are calculated, so row t never sees motif labels from t+1.
    A long feature-segment reset restarts run length but retains long-run state
    transition memory; this avoids throwing away all learned market history at
    every ordinary data discontinuity.
    """
    out = frame.copy()
    motif_columns = [
        f"motif_{cluster_id:02d}"
        for cluster_id in range(model.n_clusters)
        if f"motif_{cluster_id:02d}" in out.columns
    ]
    out["motif_transition_surprise"] = np.nan
    out["motif_run_length"] = np.nan
    out["motif_prior_frequency"] = np.nan
    if not motif_columns or model.n_clusters < 2:
        return out

    valid = out[motif_columns].notna().all(axis=1).to_numpy()
    labels = np.full(len(out), -1, dtype=int)
    if valid.any():
        labels[valid] = out.loc[valid, motif_columns].to_numpy(dtype=float).argmax(axis=1)

    segment_column = "edge_feature_segment_id" if "edge_feature_segment_id" in out.columns else None
    segments = out[segment_column].to_numpy() if segment_column else np.zeros(len(out), dtype=int)
    transition_counts = np.ones((model.n_clusters, model.n_clusters), dtype=float)
    state_counts = np.ones(model.n_clusters, dtype=float)
    total_states = float(model.n_clusters)
    previous_label: int | None = None
    previous_segment: object | None = None
    run_length = 0
    surprise = np.full(len(out), np.nan, dtype=float)
    run_values = np.full(len(out), np.nan, dtype=float)
    frequency = np.full(len(out), np.nan, dtype=float)

    for position, current in enumerate(labels):
        segment = segments[position]
        if previous_segment is not None and segment != previous_segment:
            previous_label = None
            run_length = 0
        previous_segment = segment
        if current < 0:
            previous_label = None
            run_length = 0
            continue

        frequency[position] = state_counts[current] / total_states
        if previous_label is None:
            # With no prior transition in the current segment, use the neutral
            # prior surprisal of a uniform transition as a cold-start value.
            surprise[position] = float(np.log(model.n_clusters))
            run_length = 1
        else:
            row = transition_counts[previous_label]
            probability = row[current] / row.sum()
            surprise[position] = float(-np.log(max(probability, 1e-12)))
            if current == previous_label:
                run_length += 1
            else:
                run_length = 1
            transition_counts[previous_label, current] += 1.0
        run_values[position] = float(run_length)
        state_counts[current] += 1.0
        total_states += 1.0
        previous_label = int(current)

    out["motif_transition_surprise"] = surprise
    out["motif_run_length"] = run_values
    out["motif_prior_frequency"] = frequency
    return out


def pattern_model_from_dict(payload: dict[str, object]) -> PatternModel:
    """Restore a frozen model exported in an EDGE evolution report."""
    columns = tuple(str(value) for value in payload.get("columns", []))
    means = tuple(float(value) for value in payload.get("means", []))
    scales = tuple(float(value) for value in payload.get("scales", []))
    centroids = tuple(
        tuple(float(value) for value in row)
        for row in payload.get("centroids", [])
    )
    n_clusters = int(payload.get("n_clusters", len(centroids)))
    fit_rows = int(payload.get("fit_rows", 0))
    status = str(payload.get("status", "fitted"))
    reason = payload.get("reason")
    if n_clusters > 0:
        if not columns or len(means) != len(columns) or len(scales) != len(columns):
            raise ValueError("Frozen pattern model has inconsistent columns/means/scales")
        if len(centroids) != n_clusters or any(len(row) != len(columns) for row in centroids):
            raise ValueError("Frozen pattern model has inconsistent centroid dimensions")
        if any(value <= 0 for value in scales):
            raise ValueError("Frozen pattern model scales must be positive")
    return PatternModel(
        columns=columns,
        means=means,
        scales=scales,
        centroids=centroids,
        n_clusters=n_clusters,
        fit_rows=fit_rows,
        status=status,
        reason=str(reason) if reason is not None else None,
    )


def apply_pattern_model(frame: pd.DataFrame, model: PatternModel) -> pd.DataFrame:
    """Apply stored scaler/centroids without refitting on the replay feed."""
    out = frame.copy()
    stale = [column for column in out.columns if column.startswith("motif_")]
    if stale:
        out = out.drop(columns=stale)
    out["motif_distance"] = np.nan
    if model.n_clusters < 2:
        return out

    missing = set(model.columns).difference(out.columns)
    if missing:
        raise ValueError(f"Replay frame missing frozen model columns: {sorted(missing)}")
    values = out.loc[:, list(model.columns)].replace([np.inf, -np.inf], np.nan)
    valid_mask = values.notna().all(axis=1).to_numpy()
    means = np.asarray(model.means, dtype=float)
    scales = np.asarray(model.scales, dtype=float)
    centroids = np.asarray(model.centroids, dtype=float)
    for cluster_id in range(model.n_clusters):
        out[f"motif_{cluster_id:02d}"] = np.nan
    if valid_mask.any():
        standardized = (values.loc[valid_mask].to_numpy(dtype=float) - means) / scales
        distances = np.sum((standardized[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = distances.argmin(axis=1)
        nearest = np.sqrt(distances.min(axis=1))
        valid_index = values.index[valid_mask]
        for cluster_id in range(model.n_clusters):
            out.loc[valid_index, f"motif_{cluster_id:02d}"] = (labels == cluster_id).astype(float)
        out.loc[valid_index, "motif_distance"] = nearest
    return out
