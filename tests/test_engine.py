from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from edge.backtest import BacktestSettings, run_backtest
from edge.config import ExecutionConfig, RiskConfig
from edge.data import DataQualityError, load_ohlc_csv, resample_ohlc, validate_source_interval
from edge.features import add_features
from edge.strategies import StrategyCandidate, generate_signal


def sample_frame(n: int = 600) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    rng = np.random.default_rng(7)
    changes = rng.normal(0, 0.00008, size=n)
    close = 1.10 + np.cumsum(changes)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.00005
    low = np.minimum(open_, close) - 0.00005
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=index
    )


def test_add_features_creates_atr_and_rsi() -> None:
    featured = add_features(sample_frame())
    assert "atr_14" in featured
    assert "rsi_14" in featured
    assert featured["atr_14"].dropna().gt(0).all()


def test_resample_five_minute_ohlc() -> None:
    source = sample_frame(10)
    result = resample_ohlc(source, "M5")
    assert len(result) == 2
    assert result.iloc[0]["open"] == source.iloc[0]["open"]
    assert result.iloc[0]["close"] == source.iloc[4]["close"]
    assert result.iloc[0]["high"] == source.iloc[:5]["high"].max()
    assert result.iloc[0]["low"] == source.iloc[:5]["low"].min()


def test_bad_ohlc_csv_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {"time": ["2024-01-01T00:00:00Z"], "open": [1.1], "high": [1.0], "low": [1.09], "close": [1.1]}
    ).to_csv(path, index=False)
    with pytest.raises(DataQualityError):
        load_ohlc_csv(path)


def test_backtest_enters_after_signal_and_applies_costs() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.1000, 1.1000, 1.1010, 1.1020, 1.1030],
            "high": [1.1005, 1.1015, 1.1025, 1.1035, 1.1040],
            "low": [1.0995, 1.0998, 1.1008, 1.1018, 1.1025],
            "close": [1.1000, 1.1010, 1.1020, 1.1030, 1.1035],
            "atr_14": [0.001] * 5,
        },
        index=index,
    )
    signal = pd.Series([1, 0, 0, 0, 0], index=index, dtype="int8")
    trades, _ = run_backtest(
        frame,
        signal,
        stop_atr=1.0,
        reward_risk=1.0,
        execution=ExecutionConfig(spread_pips=0.8, slippage_pips_per_side=0.1),
        risk=RiskConfig(risk_per_trade=0.001, max_daily_loss=0.01),
        settings=BacktestSettings(pip_size=0.0001),
    )
    assert len(trades) == 1
    assert trades.iloc[0]["entry_time"] == index[1]
    assert trades.iloc[0]["cost_r"] > 0
    assert trades.iloc[0]["exit_reason"] in {"target", "end_of_data", "time_exit"}


def test_rsi_strategy_signals_only_on_threshold_crossing() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="min", tz="UTC")
    frame = pd.DataFrame({"rsi_14": [35, 24, 22, 34, 76, 80]}, index=index)
    candidate = StrategyCandidate(
        name="rsi_test", family="rsi_mean_reversion",
        params={"lower": 25, "upper": 75}, stop_atr=1.5, reward_risk=1.5
    )
    signal = generate_signal(frame, candidate)
    assert signal.tolist() == [0, 1, 0, 0, -1, 0]


def test_source_timeframe_mismatch_is_rejected() -> None:
    with pytest.raises(DataQualityError):
        validate_source_interval(sample_frame(10), "M5")


def test_loader_reports_removed_duplicate_timestamps(tmp_path) -> None:
    path = tmp_path / "duplicate.csv"
    pd.DataFrame(
        {
            "time": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"],
            "open": [1.1, 1.1, 1.1],
            "high": [1.11, 1.11, 1.11],
            "low": [1.09, 1.09, 1.09],
            "close": [1.1, 1.1, 1.1],
        }
    ).to_csv(path, index=False)
    loaded = load_ohlc_csv(path)
    assert len(loaded) == 2
    assert loaded.attrs["duplicate_timestamps_removed"] == 1


def test_pattern_recognition_fits_training_prefix_only() -> None:
    from edge.pattern_recognition import add_morphology_features, fit_pattern_model

    featured = add_morphology_features(add_features(sample_frame(700)))
    clustered, model = fit_pattern_model(featured, fit_end=420, n_clusters=4, random_state=11)
    assert model.fit_rows < 420
    assert model.n_clusters >= 2
    motif_columns = [c for c in clustered.columns if c.startswith("motif_") and c != "motif_distance"]
    assert len(motif_columns) == model.n_clusters
    valid = clustered[motif_columns].dropna()
    assert (valid.sum(axis=1) == 1).all()
    assert clustered["motif_distance"].notna().sum() > 0


def test_evolved_genome_generates_unambiguous_long_short_signals() -> None:
    from edge.evolution import Condition, StrategyGenome, generate_genome_signal

    index = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    frame = pd.DataFrame({"body_atr": [0.3, -0.4, 0.1, -0.1]}, index=index)
    genome = StrategyGenome(
        long_conditions=(Condition("body_atr", ">", 0.2),),
        short_conditions=(Condition("body_atr", "<", -0.2),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=20,
    )
    assert generate_genome_signal(frame, genome).tolist() == [1, -1, 0, 0]


def test_evolution_runs_multiple_generations_with_novelty() -> None:
    from edge.evolution import EvolutionSettings, evolve_population

    rng = np.random.default_rng(19)
    frame = pd.DataFrame(
        {
            "ret_1_atr": rng.normal(size=200),
            "body_atr": rng.normal(size=200),
            "range_atr": rng.uniform(0.1, 2.0, size=200),
            "motif_00": rng.integers(0, 2, size=200).astype(float),
        }
    )

    def evaluate(genome):
        score = (len(genome.long_conditions) + len(genome.short_conditions)) * 0.1
        return {"fitness": score, "behavior": np.array([score, 0.5, 0.5, 0.1]), "metrics": {}}

    candidates, report = evolve_population(
        frame,
        evaluate=evaluate,
        settings=EvolutionSettings(population_size=8, generations=3, elite_count=2, seed=4),
    )
    assert report["total_evaluated"] == 24
    assert report["generations"] == 3
    assert candidates
    assert len({genome.fingerprint() for _, genome, _ in candidates}) == len(candidates)


def test_bid_price_basis_models_long_entry_at_ask_without_double_charging_spread() -> None:
    from edge.backtest import BacktestSettings

    index = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.1000, 1.1000, 1.1000, 1.1000],
            "high": [1.1002, 1.1013, 1.1002, 1.1002],
            "low": [1.0998, 1.0998, 1.0998, 1.0998],
            "close": [1.1000, 1.1010, 1.1000, 1.1000],
            "atr_14": [0.001] * 4,
        },
        index=index,
    )
    signal = pd.Series([1, 0, 0, 0], index=index, dtype="int8")
    trades, _ = run_backtest(
        frame, signal, stop_atr=1.0, reward_risk=1.0,
        execution=ExecutionConfig(spread_pips=1.0, slippage_pips_per_side=0.0),
        risk=RiskConfig(risk_per_trade=0.001, max_daily_loss=0.01),
        settings=BacktestSettings(pip_size=0.0001, price_basis="bid"),
    )
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(1.1001)
    assert trade["exit_price"] == pytest.approx(1.1011)
    assert trade["spread_cost_pips"] == pytest.approx(0.0)
    assert trade["net_r"] == pytest.approx(1.0)


def test_bid_price_basis_models_short_exit_at_ask() -> None:
    from edge.backtest import BacktestSettings

    index = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.1000, 1.1000, 1.1000, 1.1000],
            "high": [1.1002, 1.1002, 1.1002, 1.1002],
            "low": [1.0998, 1.0987, 1.0998, 1.0998],
            "close": [1.1000, 1.0990, 1.1000, 1.1000],
            "atr_14": [0.001] * 4,
        },
        index=index,
    )
    signal = pd.Series([-1, 0, 0, 0], index=index, dtype="int8")
    trades, _ = run_backtest(
        frame, signal, stop_atr=1.0, reward_risk=1.0,
        execution=ExecutionConfig(spread_pips=1.0, slippage_pips_per_side=0.0),
        risk=RiskConfig(risk_per_trade=0.001, max_daily_loss=0.01),
        settings=BacktestSettings(pip_size=0.0001, price_basis="bid"),
    )
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(1.1000)
    assert trade["exit_price"] == pytest.approx(1.0990)
    assert trade["spread_cost_pips"] == pytest.approx(0.0)
    assert trade["net_r"] == pytest.approx(1.0)


def test_data_quality_summary_buckets_gap_durations() -> None:
    from edge.data import data_quality_summary

    index = pd.DatetimeIndex(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:01:00Z",
            "2024-01-01T00:04:00Z",
            "2024-01-01T01:04:00Z",
            "2024-01-02T01:04:00Z",
        ],
        name="time",
    )
    frame = pd.DataFrame(
        {"open": [1.1] * 5, "high": [1.2] * 5, "low": [1.0] * 5, "close": [1.1] * 5},
        index=index,
    )
    summary = data_quality_summary(frame)
    assert summary["large_gaps"] == 3
    assert summary["gap_duration_buckets"]["5_to_60_minutes"] == 1
    assert summary["gap_duration_buckets"]["1_to_24_hours"] == 1


def test_gap_annotation_and_resampling_marks_touching_bars() -> None:
    from edge.data import annotate_gaps

    index = pd.DatetimeIndex(
        list(pd.date_range("2024-01-01T00:00:00Z", periods=5, freq="min"))
        + list(pd.date_range("2024-01-01T00:10:00Z", periods=10, freq="min")),
        name="time",
    )
    base = sample_frame(15).copy()
    base.index = index
    annotated = annotate_gaps(base, expected_seconds=60)
    assert annotated["edge_gap_before"].sum() == 1
    assert annotated["edge_gap_after"].sum() == 1

    bars = resample_ohlc(annotated, "M5")
    contaminated = bars.index[bars["edge_gap_contaminated"]].tolist()
    assert contaminated == [
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:10:00Z"),
    ]
    assert not bool(bars.loc[pd.Timestamp("2024-01-01T00:15:00Z"), "edge_gap_contaminated"])


def test_indicators_reset_after_a_data_gap() -> None:
    from edge.data import annotate_gaps

    first_index = pd.date_range("2024-01-01", periods=40, freq="min", tz="UTC")
    second_index = pd.date_range(first_index[-1] + pd.Timedelta(6, unit="min"), periods=30, freq="min")
    index = first_index.append(second_index)
    first_close = np.full(len(first_index), 1.10)
    second_close = np.full(len(second_index), 1.20)
    close = np.r_[first_close, second_close]
    frame = pd.DataFrame(
        {"open": close, "high": close + 0.0001, "low": close - 0.0001, "close": close},
        index=index,
    )
    annotated = annotate_gaps(frame, expected_seconds=60)
    featured = add_features(annotated)

    new_segment = featured.iloc[len(first_index):]
    # The first post-gap candle begins a fresh segment. EMA10 becomes valid
    # only after ten candles from that segment, with no old-state inheritance.
    assert new_segment["edge_gap_before"].iloc[0]
    assert new_segment["ema_10"].iloc[:9].isna().all()
    assert new_segment["ema_10"].iloc[9] == pytest.approx(1.20)


def test_backtest_closes_open_position_conservatively_at_data_gap() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.0980, 1.0980, 1.0980],
            "high": [1.1002] * 8,
            "low": [1.0998] * 8,
            "close": [1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.0980, 1.0980, 1.0980],
            "atr_14": [0.01] * 8,
            "edge_gap_before": [False, False, False, False, True, False, False, False],
            "edge_gap_after": [False, False, False, True, False, False, False, False],
            "edge_gap_contaminated": [False, False, False, True, True, False, False, False],
        },
        index=index,
    )
    signal = pd.Series([1, 0, 0, 0, 0, 0, 0, 0], index=index, dtype="int8")
    trades, _ = run_backtest(
        frame, signal, stop_atr=5.0, reward_risk=5.0,
        execution=ExecutionConfig(spread_pips=0.8, slippage_pips_per_side=0.0),
        risk=RiskConfig(risk_per_trade=0.001, max_daily_loss=0.01),
        settings=BacktestSettings(pip_size=0.0001),
    )
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "data_gap_exit"
    assert bool(trade["gap_exit"])
    assert trade["exit_time"] == index[5]
    assert trade["exit_price"] == pytest.approx(1.0980)


def test_backtest_does_not_enter_on_or_immediately_before_gap() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.1] * 5, "high": [1.101] * 5,
            "low": [1.099] * 5, "close": [1.1] * 5,
            "atr_14": [0.001] * 5,
            "edge_gap_before": [False, False, False, True, False],
            "edge_gap_after": [False, True, False, False, False],
            "edge_gap_contaminated": [False, True, False, True, False],
        },
        index=index,
    )
    signal = pd.Series([1, 0, 1, 0, 0], index=index, dtype="int8")
    trades, _ = run_backtest(
        frame, signal, stop_atr=2.0, reward_risk=2.0,
        execution=ExecutionConfig(), risk=RiskConfig(),
        settings=BacktestSettings(),
    )
    # Signal at row 0 would enter on row 1 (gap-adjacent), and signal at row 2
    # would enter on row 3 (gap-contaminated): both must be rejected.
    assert trades.empty


def test_raw_source_gap_closes_open_position_and_records_gap_duration() -> None:
    from edge.data import annotate_gaps

    index = pd.DatetimeIndex(
        [
            "2024-01-02T10:00:00Z",
            "2024-01-02T10:01:00Z",
            "2024-01-02T10:02:00Z",
            "2024-01-02T10:03:00Z",
            "2024-01-02T10:04:00Z",
            "2024-01-02T11:05:00Z",
            "2024-01-02T11:06:00Z",
        ],
        name="time",
    )
    close = np.array([1.1000, 1.1000, 1.1000, 1.1000, 1.0990, 1.0950, 1.0950])
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "atr_14": [0.01] * len(index),
        },
        index=index,
    )
    annotated = annotate_gaps(frame, expected_seconds=60)
    signal = pd.Series([1, 0, 0, 0, 0, 0, 0], index=index, dtype="int8")
    trades, _ = run_backtest(
        annotated,
        signal,
        stop_atr=5.0,
        reward_risk=5.0,
        execution=ExecutionConfig(spread_pips=0.0, slippage_pips_per_side=0.0),
        risk=RiskConfig(risk_per_trade=0.001, max_daily_loss=0.01),
        settings=BacktestSettings(pip_size=0.0001),
    )

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "data_gap_exit"
    assert trade["exit_time"] == index[5]
    assert trade["exit_price"] == pytest.approx(1.0950)
    assert trade["gap_duration_seconds"] == pytest.approx(3660.0)
    assert trade["gap_exit_assumption"] == "adverse_of_pre_gap_close_and_post_gap_open"


def test_gap_audit_classifies_source_gap_and_estimates_missing_bars() -> None:
    from edge.data import build_gap_audit

    index = pd.DatetimeIndex(
        [
            "2024-01-02T10:00:00Z",
            "2024-01-02T10:01:00Z",
            "2024-01-02T11:02:00Z",
            "2024-01-05T21:58:00Z",
            "2024-01-08T22:00:00Z",
        ],
        name="time",
    )
    frame = pd.DataFrame(
        {"open": [1.1] * 5, "high": [1.2] * 5, "low": [1.0] * 5, "close": [1.1] * 5},
        index=index,
    )
    audit = build_gap_audit(frame, expected_seconds=60)

    assert len(audit) == 3
    assert audit.iloc[0]["category"] == "long_non_weekend_gap"
    assert audit.iloc[0]["estimated_missing_bars"] == 60
    assert audit.iloc[1]["category"] == "over_24h_non_weekend_gap"
    assert audit.iloc[2]["category"] == "weekend_market_closure_like"


def test_gap_duration_metadata_survives_resampling() -> None:
    from edge.data import annotate_gaps

    index = pd.DatetimeIndex(
        list(pd.date_range("2024-01-02T10:00:00Z", periods=5, freq="min"))
        + list(pd.date_range("2024-01-02T11:05:00Z", periods=10, freq="min")),
        name="time",
    )
    base = sample_frame(15).copy()
    base.index = index
    annotated = annotate_gaps(base, expected_seconds=60)
    bars = resample_ohlc(annotated, "M5")

    assert bars["edge_gap_duration_seconds"].max() == pytest.approx(3660.0)
    assert bars["edge_gap_contaminated"].sum() >= 2


def test_resampled_gap_materiality_scales_with_target_timeframe() -> None:
    from edge.data import annotate_gaps

    index = pd.DatetimeIndex(
        list(pd.date_range("2024-01-02T10:00:00Z", periods=5, freq="min"))
        + list(pd.date_range("2024-01-02T10:15:00Z", periods=15, freq="min")),
        name="time",
    )
    base = sample_frame(len(index)).copy()
    base.index = index
    annotated = annotate_gaps(base, expected_seconds=60)

    m15 = resample_ohlc(annotated, "M15")
    h1 = resample_ohlc(annotated, "H1")

    assert m15["edge_gap_contaminated"].sum() >= 2
    assert m15["edge_gap_execution_before"].any()
    assert m15["edge_gap_execution_after"].any()
    # The same 11-minute source discontinuity is less than 25% of an H1 bar.
    assert not h1["edge_gap_contaminated"].any()
    assert not h1["edge_gap_execution_before"].any()
    assert not h1["edge_gap_execution_after"].any()


def test_short_source_gap_does_not_contaminate_m15_bar() -> None:
    from edge.data import annotate_gaps

    index = pd.DatetimeIndex(
        list(pd.date_range("2024-01-02T10:00:00Z", periods=5, freq="min"))
        + list(pd.date_range("2024-01-02T10:07:00Z", periods=13, freq="min")),
        name="time",
    )
    base = sample_frame(len(index)).copy()
    base.index = index
    annotated = annotate_gaps(base, expected_seconds=60)
    m15 = resample_ohlc(annotated, "M15")

    # 180 seconds is below 25% of a 15-minute candle (225 seconds).
    assert m15["edge_gap_duration_seconds"].max() == pytest.approx(180.0)
    assert not m15["edge_gap_contaminated"].any()
    assert not m15["edge_gap_execution_before"].any()
    assert not m15["edge_gap_execution_after"].any()


def test_feature_state_resets_only_after_very_long_resampled_gap() -> None:
    from edge.data import annotate_gaps

    first = pd.date_range("2024-01-02T00:00:00Z", periods=120, freq="min")
    moderate = pd.date_range(first[-1] + pd.Timedelta(2, unit="h"), periods=120, freq="min")
    long = pd.date_range(moderate[-1] + pd.Timedelta(13, unit="h"), periods=120, freq="min")
    index = first.append(moderate).append(long)
    base = sample_frame(len(index)).copy()
    base.index = index
    annotated = annotate_gaps(base, expected_seconds=60)
    m15 = resample_ohlc(annotated, "M15")

    # A two-hour interruption is significant for M15 execution but is shorter
    # than 50 M15 bars, so it does not erase the long feature-state history.
    moderate_boundary = moderate[0].floor("15min")
    assert bool(m15.loc[moderate_boundary, "edge_gap_contaminated"])
    assert not bool(m15.loc[moderate_boundary, "edge_feature_reset_before"])

    # A 13-hour interruption exceeds 50 M15 bars and begins a new feature state.
    long_boundary = long[0].floor("15min")
    assert bool(m15.loc[long_boundary, "edge_feature_reset_before"])
    assert m15["edge_feature_segment_id"].nunique() >= 2


def test_pattern_model_disables_clustering_when_training_patterns_are_sparse() -> None:
    from edge.pattern_recognition import add_morphology_features, fit_pattern_model

    featured = add_morphology_features(add_features(sample_frame(250)))
    clustered, model = fit_pattern_model(
        featured, fit_end=20, n_clusters=6, random_state=3
    )

    assert model.n_clusters == 0
    assert model.status == "disabled_insufficient_training_patterns"
    assert model.fit_rows < 30
    assert clustered["motif_distance"].isna().all()
    assert not any(
        column.startswith("motif_") and column != "motif_distance"
        for column in clustered.columns
    )


def test_backtest_uses_timeframe_relative_execution_gap_flags() -> None:
    index = pd.date_range("2024-01-02T10:00:00Z", periods=5, freq="15min")
    frame = pd.DataFrame(
        {
            "open": [1.1] * 5,
            "high": [1.1002] * 5,
            "low": [1.0998] * 5,
            "close": [1.1] * 5,
            "atr_14": [0.01] * 5,
            # Exact source events are retained for diagnostics.
            "edge_gap_before": [False, False, True, False, False],
            "edge_gap_after": [False, True, False, False, False],
            # This gap is below the M15 materiality threshold.
            "edge_gap_execution_before": [False] * 5,
            "edge_gap_execution_after": [False] * 5,
            "edge_gap_contaminated": [False] * 5,
        },
        index=index,
    )
    signal = pd.Series([1, 0, 0, 0, 0], index=index, dtype="int8")
    trades, _ = run_backtest(
        frame,
        signal,
        stop_atr=5.0,
        reward_risk=5.0,
        execution=ExecutionConfig(spread_pips=0.0, slippage_pips_per_side=0.0),
        risk=RiskConfig(risk_per_trade=0.001, max_daily_loss=0.01),
        settings=BacktestSettings(pip_size=0.0001, max_holding_bars=2),
    )

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] != "data_gap_exit"


def test_gap_heavy_multitimeframe_evolution_does_not_abort_pattern_fit() -> None:
    from edge.data import annotate_gaps
    from edge.config import ResearchConfig
    from edge.evolution import EvolutionSettings
    from edge.evolutionary_research import research_evolutionary_timeframe

    rng = np.random.default_rng(10)
    source_count = 18_000
    source_index = pd.date_range(
        "2024-01-02T00:00:00Z", periods=source_count, freq="min"
    )
    keep = np.ones(source_count, dtype=bool)
    for start in range(1_000, source_count, 700):
        keep[start:start + 6] = False
    for start in range(5_000, source_count, 5_000):
        keep[start:start + 61] = False
    index = source_index[keep]

    changes = rng.normal(0.0, 0.00005, size=len(index))
    close = 1.10 + np.cumsum(changes)
    open_ = np.r_[close[0], close[:-1]]
    source = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.00003,
            "low": np.minimum(open_, close) - 0.00003,
            "close": close,
        },
        index=index,
    )
    annotated = annotate_gaps(source, expected_seconds=60)
    bars = resample_ohlc(annotated, "M15")

    result = research_evolutionary_timeframe(
        bars,
        timeframe="M15",
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        research=ResearchConfig(minimum_train_trades=1, minimum_validation_trades=1),
        evolution=EvolutionSettings(
            population_size=8,
            generations=2,
            elite_count=2,
            tournament_size=2,
            validation_pool=4,
            seed=4,
        ),
        pip_size=0.0001,
        price_basis="bid",
    )

    assert result["status"] in {
        "no_robust_candidate",
        "test_passed",
        "test_failed_or_inconclusive",
    }
    assert result["pattern_model"]["status"] == "fitted"
    assert result["pattern_model"]["fit_rows"] >= 30
    assert result["gap_diagnostics"]["feature_segments"] == 1


def test_genome_canonicalization_removes_duplicate_and_contradictory_rules() -> None:
    from edge.evolution import Condition, StrategyGenome, canonicalize_genome

    genome = StrategyGenome(
        long_conditions=(
            Condition("rsi_centered", "<", -0.1),
            Condition("rsi_centered", "<", -0.2),
            Condition("body_atr", ">", 0.5),
            Condition("body_atr", "<", 0.2),  # contradiction: remove body_atr
            Condition("rsi_centered", "<", -0.2),
        ),
        short_conditions=(
            Condition("ema_gap_atr", ">", 0.1),
            Condition("ema_gap_atr", ">", 0.2),
        ),
        stop_atr=8.0,
        reward_risk=0.1,
        max_holding_bars=5000,
    )
    cleaned = canonicalize_genome(genome)
    assert cleaned.long_conditions == (Condition("rsi_centered", "<", -0.2),)
    assert cleaned.short_conditions == (Condition("ema_gap_atr", ">", 0.2),)
    assert cleaned.stop_atr == 4.0
    assert cleaned.reward_risk == 0.5
    assert cleaned.max_holding_bars == 1000


def test_evolution_adapts_mutation_and_records_lineage() -> None:
    from edge.evolution import EvolutionSettings, evolve_population

    rng = np.random.default_rng(91)
    frame = pd.DataFrame({
        "ret_1_atr": rng.normal(size=150),
        "ret_3_atr": rng.normal(size=150),
        "body_atr": rng.normal(size=150),
        "range_atr": rng.uniform(0.1, 2.0, size=150),
        "rsi_centered": rng.normal(size=150),
    })

    def evaluate(genome):
        quality = -1.0  # Force stagnation, exercising the adaptive mutation increase.
        return {"fitness": quality, "behavior": np.array([0.1, 0.4, 0.5, 0.01]), "metrics": {}}

    _, report = evolve_population(
        frame,
        evaluate=evaluate,
        settings=EvolutionSettings(
            population_size=8, generations=4, elite_count=2,
            tournament_size=2, mutation_rate=0.20, seed=12,
        ),
    )
    history = report["generation_history"]
    assert len(history) == 4
    assert history[-1]["mutation_rate_used"] > history[0]["mutation_rate_used"]
    assert report["lineage"]
    assert any(event["operator"].startswith("crossover") for event in report["lineage"])


def test_causal_motif_dynamics_do_not_use_future_motif_labels() -> None:
    from edge.pattern_recognition import PatternModel, add_causal_motif_dynamics

    index = pd.date_range("2024-01-01", periods=6, freq="min", tz="UTC")
    frame = pd.DataFrame({
        "motif_00": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        "motif_01": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        "edge_feature_segment_id": [0, 0, 0, 0, 0, 0],
    }, index=index)
    model = PatternModel(
        columns=("x",), means=(0.0,), scales=(1.0,),
        centroids=((0.0,), (1.0,)), n_clusters=2, fit_rows=10,
    )
    first = add_causal_motif_dynamics(frame, model)
    altered = frame.copy()
    altered.loc[index[4]:, ["motif_00", "motif_01"]] = [0.0, 1.0]
    second = add_causal_motif_dynamics(altered, model)
    pd.testing.assert_series_equal(
        first.loc[index[:4], "motif_transition_surprise"],
        second.loc[index[:4], "motif_transition_surprise"],
    )
    pd.testing.assert_series_equal(
        first.loc[index[:4], "motif_run_length"],
        second.loc[index[:4], "motif_run_length"],
    )
    assert first["motif_run_length"].tolist() == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]


def test_validation_windows_are_chronological_and_non_overlapping() -> None:
    from edge.validation import split_chronological_windows

    index = pd.date_range("2024-01-01", periods=90, freq="min", tz="UTC")
    frame = pd.DataFrame({"x": np.arange(90)}, index=index)
    windows = split_chronological_windows(frame, windows=3)
    assert [len(window) for window in windows] == [30, 30, 30]
    assert windows[0].index.max() < windows[1].index.min()
    assert windows[1].index.max() < windows[2].index.min()


def test_moving_block_bootstrap_is_reproducible() -> None:
    from edge.validation import block_bootstrap_expectancy

    trades = pd.DataFrame({"net_r": [0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.0, 0.1]})
    first = block_bootstrap_expectancy(trades, repetitions=200, block_length=3, seed=9)
    second = block_bootstrap_expectancy(trades, repetitions=200, block_length=3, seed=9)
    assert first == second
    assert first["observations"] == 10
    assert first["ci_90_low"] <= first["mean"] <= first["ci_90_high"]


def test_genome_memory_stores_genotypes_without_performance_metrics(tmp_path) -> None:
    from edge.evolution import Condition, StrategyGenome
    from edge.memory import load_genome_memory, update_genome_memory

    genome = StrategyGenome(
        long_conditions=(Condition("body_atr", ">", 0.1),),
        short_conditions=(Condition("rsi_centered", "<", -0.2),),
        stop_atr=1.5, reward_risk=2.0, max_holding_bars=20,
    )
    path = tmp_path / "memory.json"
    result = update_genome_memory(
        path, symbol="EURUSD", timeframe="M15", genomes=[genome],
    )
    assert result["stored"] == 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "fitness" not in json.dumps(payload).lower()
    seeds = load_genome_memory(
        path, symbol="EURUSD", timeframe="M15", allowed_features={"body_atr", "rsi_centered"},
    )
    assert len(seeds) == 1
    assert seeds[0].fingerprint() == genome.fingerprint()
    assert load_genome_memory(
        path, symbol="GBPUSD", timeframe="M15", allowed_features=None,
    ) == []


def test_compare_independent_ohlc_feeds_reports_timestamp_and_price_differences() -> None:
    from edge.source_comparison import compare_ohlc_sources

    index_a = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    index_b = index_a.delete(2).append(pd.DatetimeIndex([index_a[-1] + pd.Timedelta(1, unit="min")]))
    a = pd.DataFrame({
        "open": [1.1] * 5, "high": [1.1002] * 5,
        "low": [1.0998] * 5, "close": [1.1000, 1.1001, 1.1002, 1.1001, 1.1000],
    }, index=index_a)
    b = pd.DataFrame({
        "open": [1.10001] * 5, "high": [1.10021] * 5,
        "low": [1.09981] * 5, "close": [1.10001, 1.10011, 1.10011, 1.10001, 1.10001],
    }, index=index_b)
    report = compare_ohlc_sources(a, b, pip_size=0.0001)
    assert report["overlap_rows"] == 4
    assert report["reference_only_timestamps"] == 1
    assert report["comparison_only_timestamps"] == 1
    assert report["price_comparison"]["close"]["mean_absolute_difference_pips"] > 0


def test_research_only_pipeline_writes_reports_and_genome_memory(tmp_path) -> None:
    from edge.config import ExecutionConfig, ResearchConfig, RiskConfig
    from edge.evolution import EvolutionSettings
    from edge.evolutionary_research import run_evolutionary_research

    rng = np.random.default_rng(72)
    n = 1500
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 1.1 + np.cumsum(rng.normal(0.0, 0.00012, n))
    open_ = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + rng.uniform(0.00002, 0.00010, n),
        "low": np.minimum(open_, close) - rng.uniform(0.00002, 0.00010, n),
        "close": close,
    }, index=index)
    output = tmp_path / "research"
    memory = tmp_path / "evolution-memory.json"
    report = run_evolutionary_research(
        frame,
        source_timeframe="M15",
        timeframes=["M15"],
        symbol="EURUSD",
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        research=ResearchConfig(minimum_train_trades=1, minimum_validation_trades=1),
        evolution=EvolutionSettings(
            population_size=8, generations=2, elite_count=2,
            tournament_size=2, validation_pool=4, seed=15,
        ),
        pip_size=0.0001,
        price_basis="bid",
        output_dir=output,
        memory_file=memory,
        run_final_test=False,
    )
    assert report["version"] == "0.5.0"
    assert report["validation_design"]["final_test_run"] is False
    assert memory.exists()
    assert (output / "evolution_report.json").exists()
    stored = json.loads(memory.read_text(encoding="utf-8"))
    assert stored["organisms"]
    assert all("fitness" not in item for item in stored["organisms"])
    assert report["results"][0]["final_test_skipped"] or report["results"][0]["status"] == "no_robust_candidate"


def test_cost_stress_scales_observed_spread_column(monkeypatch) -> None:
    import edge.evolutionary_research as evolutionary_research
    from edge.evolution import Condition, StrategyGenome

    index = pd.date_range("2024-01-01", periods=90, freq="15min", tz="UTC")
    frame = pd.DataFrame({"spread_pips": np.ones(90)}, index=index)
    genome = StrategyGenome(
        long_conditions=(Condition("body_atr", ">", 0.0),),
        short_conditions=(Condition("body_atr", "<", 0.0),),
        stop_atr=1.5, reward_risk=2.0, max_holding_bars=20,
    )
    observed: list[tuple[float, float]] = []

    def fake_run(candidate_frame, candidate_genome, *, execution, risk, pip_size, price_basis):
        observed.append((float(candidate_frame["spread_pips"].iloc[0]), float(execution.spread_pips)))
        metrics = {"trades": 0, "expectancy_r": None, "profit_factor": None,
                   "max_realized_drawdown_pct": None}
        return metrics, pd.DataFrame(columns=["net_r"]), pd.DataFrame()

    monkeypatch.setattr(evolutionary_research, "_run_genome", fake_run)
    evolutionary_research._validation_robustness(
        frame, genome, execution=ExecutionConfig(spread_pips=0.8),
        risk=RiskConfig(), pip_size=0.0001, price_basis="bid",
        min_window_trades=8, max_drawdown_filter=0.2,
    )
    assert any(abs(spread - 1.5) < 1e-9 and abs(configured - 1.2) < 1e-9 for spread, configured in observed)
    assert any(abs(spread - 2.0) < 1e-9 and abs(configured - 1.6) < 1e-9 for spread, configured in observed)


def test_continuous_motif_features_use_quantile_thresholds() -> None:
    from edge.evolution import _thresholds

    frame = pd.DataFrame({
        "motif_00": [0.0, 1.0] * 50,
        "motif_prior_frequency": np.linspace(0.05, 0.45, 100),
        "motif_run_length": np.arange(1, 101, dtype=float),
        "motif_transition_surprise": np.linspace(0.1, 2.0, 100),
    })

    thresholds = _thresholds(frame, frame.columns)

    assert thresholds["motif_00"] == (0.5,)
    assert len(thresholds["motif_prior_frequency"]) > 1
    assert max(thresholds["motif_prior_frequency"]) < 0.5
    assert len(thresholds["motif_run_length"]) > 1
    assert len(thresholds["motif_transition_surprise"]) > 1


def test_available_features_does_not_duplicate_continuous_motif_features() -> None:
    from edge.evolution import available_features

    frame = pd.DataFrame({
        "motif_prior_frequency": np.linspace(0.05, 0.45, 20),
        "motif_00": [0.0, 1.0] * 10,
    })

    features = available_features(frame)

    assert features.count("motif_prior_frequency") == 1
    assert features.count("motif_00") == 1


def test_evolution_archives_only_genomes_with_both_directions() -> None:
    from edge.evolution import (
        EvolutionSettings,
        evolve_population,
        has_both_directional_signals,
    )

    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "ret_1_atr": rng.normal(size=300),
        "body_atr": rng.normal(size=300),
        "range_atr": rng.uniform(0.1, 2.0, size=300),
        "rsi_centered": rng.normal(size=300),
    })

    def evaluate(genome):
        return {
            "fitness": 1.0,
            "behavior": np.array([0.1, 0.5, 0.5, 0.1]),
            "metrics": {},
        }

    candidates, _ = evolve_population(
        frame,
        evaluate=evaluate,
        settings=EvolutionSettings(
            population_size=8,
            generations=3,
            elite_count=2,
            seed=42,
        ),
    )

    assert candidates
    assert all(
        has_both_directional_signals(frame, genome)
        for _, genome, _ in candidates
    )
