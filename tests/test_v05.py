from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from edge.backtest import BacktestSettings, _swap_charge_pips, run_backtest
from edge.config import ExecutionConfig, RiskConfig
from edge.data import load_ohlc_csv, resample_ohlc
from edge.evolution import Condition, StrategyGenome
from edge.features import add_features
from edge.pattern_recognition import (
    add_causal_motif_dynamics,
    add_morphology_features,
    apply_pattern_model,
    fit_pattern_model,
    pattern_model_from_dict,
)
from edge.quote_data import aggregate_quote_ticks, load_tick_csv


def test_tick_aggregation_preserves_bid_ask_and_does_not_fill_empty_minutes() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:20Z",
            "2024-01-01T00:02:00Z",
        ],
        name="time",
    )
    ticks = pd.DataFrame(
        {"bid": [1.1000, 1.1002, 1.1005], "ask": [1.1001, 1.1004, 1.1007]},
        index=index,
    )
    bars = aggregate_quote_ticks(ticks, timeframe="M1", pip_size=0.0001)
    assert len(bars) == 2
    assert bars.index.tolist() == [index[0], index[2]]
    assert bars.iloc[0]["open"] == pytest.approx(1.1000)
    assert bars.iloc[0]["high"] == pytest.approx(1.1002)
    assert bars.iloc[0]["ask_open"] == pytest.approx(1.1001)
    assert bars.iloc[0]["ask_high"] == pytest.approx(1.1004)
    assert bars.iloc[0]["tick_count"] == 2
    assert bars.iloc[0]["spread_pips"] == pytest.approx(1.5)


def test_tick_loader_rejects_crossed_quotes(tmp_path) -> None:
    path = tmp_path / "ticks.csv"
    pd.DataFrame(
        {"time": ["2024-01-01T00:00:00Z"], "bid": [1.1002], "ask": [1.1001]}
    ).to_csv(path, index=False)
    from edge.data import DataQualityError

    with pytest.raises(DataQualityError, match="crossed"):
        load_tick_csv(path)


def test_load_ohlc_csv_accepts_and_validates_actual_ask_ohlc(tmp_path) -> None:
    path = tmp_path / "quote_bars.csv"
    pd.DataFrame(
        {
            "time": ["2024-01-01T00:00:00Z"],
            "open": [1.1000], "high": [1.1010], "low": [1.0990], "close": [1.1005],
            "ask_open": [1.1001], "ask_high": [1.1012], "ask_low": [1.0991], "ask_close": [1.1006],
            "spread_pips": [1.0], "tick_count": [10],
        }
    ).to_csv(path, index=False)
    loaded = load_ohlc_csv(path)
    assert loaded.iloc[0]["ask_open"] == pytest.approx(1.1001)
    assert loaded.iloc[0]["tick_count"] == 10


def test_load_ohlc_csv_rejects_crossed_bid_ask_ohlc(tmp_path) -> None:
    from edge.data import DataQualityError

    path = tmp_path / "crossed_quote_bars.csv"
    pd.DataFrame({
        "time": ["2024-01-01T00:00:00Z"],
        "open": [1.1000], "high": [1.1010], "low": [1.0990], "close": [1.1005],
        "ask_open": [1.0999], "ask_high": [1.1012], "ask_low": [1.0991], "ask_close": [1.1006],
    }).to_csv(path, index=False)
    with pytest.raises(DataQualityError, match="ask_open below bid"):
        load_ohlc_csv(path)


def test_resample_keeps_actual_ask_side_ohlc() -> None:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=8, freq="1h")
    frame = pd.DataFrame(
        {
            "open": np.linspace(1.1, 1.1007, 8),
            "high": np.linspace(1.1005, 1.1012, 8),
            "low": np.linspace(1.0995, 1.1002, 8),
            "close": np.linspace(1.1001, 1.1008, 8),
            "ask_open": np.linspace(1.1001, 1.1008, 8),
            "ask_high": np.linspace(1.1006, 1.1013, 8),
            "ask_low": np.linspace(1.0996, 1.1003, 8),
            "ask_close": np.linspace(1.1002, 1.1009, 8),
            "spread_pips": [1.0] * 8,
            "tick_count": [5] * 8,
        }, index=index,
    )
    result = resample_ohlc(frame, "H4")
    assert len(result) == 2
    assert result.iloc[0]["ask_open"] == pytest.approx(frame.iloc[0]["ask_open"])
    assert result.iloc[0]["ask_high"] == pytest.approx(frame.iloc[:4]["ask_high"].max())
    assert result.iloc[0]["ask_low"] == pytest.approx(frame.iloc[:4]["ask_low"].min())
    assert result.iloc[0]["tick_count"] == 20


def test_bid_backtest_uses_actual_ask_ohlc_without_estimated_spread() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.1000] * 4, "high": [1.1002, 1.1015, 1.1002, 1.1002],
            "low": [1.0998] * 4, "close": [1.1000, 1.1010, 1.1000, 1.1000],
            "ask_open": [1.1002] * 4, "ask_high": [1.1004, 1.1017, 1.1004, 1.1004],
            "ask_low": [1.1000, 1.1000, 1.1000, 1.1000], "ask_close": [1.1002, 1.1012, 1.1002, 1.1002],
            "atr_14": [0.001] * 4,
        }, index=index,
    )
    signal = pd.Series([1, 0, 0, 0], index=index, dtype="int8")
    trades, _ = run_backtest(
        frame, signal, stop_atr=1.0, reward_risk=1.0,
        execution=ExecutionConfig(spread_pips=2.0, slippage_pips_per_side=0.0),
        risk=RiskConfig(risk_per_trade=0.001),
        settings=BacktestSettings(pip_size=0.0001, price_basis="bid"),
    )
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(1.1002)
    assert trade["spread_cost_pips"] == 0.0
    assert trade["exit_price"] == pytest.approx(1.1012)


def test_swap_calculation_counts_rollovers_and_triple_day_with_dst_aware_timezone() -> None:
    zone = ZoneInfo("America/New_York")
    entry = pd.Timestamp(datetime(2024, 1, 3, 16, 0, tzinfo=zone))
    exit_ = pd.Timestamp(datetime(2024, 1, 4, 18, 0, tzinfo=zone))
    execution = ExecutionConfig(
        swap_long_pips_per_rollover=-0.4,
        triple_swap_weekday=2,
        triple_swap_multiplier=3.0,
    )
    cost, count = _swap_charge_pips(entry, exit_, 1, execution)
    assert count == 2
    assert cost == pytest.approx(-1.6)

def test_swap_calculation_skips_weekend_rollovers() -> None:
    zone = ZoneInfo("America/New_York")
    entry = pd.Timestamp(datetime(2024, 1, 5, 16, 0, tzinfo=zone))
    exit_ = pd.Timestamp(datetime(2024, 1, 8, 18, 0, tzinfo=zone))

    execution = ExecutionConfig(
        swap_long_pips_per_rollover=-0.5,
    )

    cost, count = _swap_charge_pips(entry, exit_, 1, execution)

    assert count == 2
    assert cost == pytest.approx(-1.0)


def test_swap_is_not_charged_when_trade_does_not_cross_rollover() -> None:
    zone = ZoneInfo("America/New_York")
    entry = pd.Timestamp(datetime(2024, 1, 3, 15, 0, tzinfo=zone))
    exit_ = pd.Timestamp(datetime(2024, 1, 3, 16, 0, tzinfo=zone))
    execution = ExecutionConfig(swap_long_pips_per_rollover=-0.5)
    cost, count = _swap_charge_pips(entry, exit_, 1, execution)
    assert count == 0
    assert cost == 0.0


def test_frozen_pattern_model_apply_reuses_saved_centroids() -> None:
    rng = np.random.default_rng(901)
    index = pd.date_range("2024-01-01", periods=800, freq="4h", tz="UTC")
    changes = rng.normal(0, 0.0008, len(index))
    close = 1.1 + np.cumsum(changes)
    open_ = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + 0.0002,
         "low": np.minimum(open_, close) - 0.0002, "close": close}, index=index,
    )
    prepared = add_morphology_features(add_features(frame))
    fitted, model = fit_pattern_model(prepared, fit_end=500, n_clusters=6, random_state=12)
    restored = pattern_model_from_dict(model.to_dict())
    replayed = apply_pattern_model(prepared, restored)
    cols = [f"motif_{i:02d}" for i in range(model.n_clusters)]
    pd.testing.assert_frame_equal(fitted[cols], replayed[cols])


def test_frozen_replay_reuses_candidate_and_centroids_without_refitting(tmp_path) -> None:
    from edge.frozen_replay import replay_frozen_candidate

    index = pd.date_range("2022-01-01", periods=900, freq="4h", tz="UTC")
    rng = np.random.default_rng(4)
    changes = rng.normal(0, 0.0003, len(index))
    close = 1.10 + np.cumsum(changes)
    open_ = np.r_[close[0], close[:-1]]
    bid = pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + 0.0001,
         "low": np.minimum(open_, close) - 0.0001, "close": close}, index=index,
    )
    quote_bars = bid.copy()
    for col in ("open", "high", "low", "close"):
        quote_bars[f"ask_{col}"] = quote_bars[col] + 0.0001
    quote_bars["spread_pips"] = 1.0
    quote_path = tmp_path / "independent_quotes.csv"
    quote_bars.to_csv(quote_path, index_label="time")

    model_frame = add_morphology_features(add_features(bid))
    _clustered, model = fit_pattern_model(model_frame, fit_end=500, n_clusters=6, random_state=8)
    genome = StrategyGenome(
        long_conditions=(Condition("shape_ret_01", ">", -1000.0),),
        short_conditions=(Condition("shape_ret_01", ">", 1000.0),),
        stop_atr=1.5, reward_risk=2.5, max_holding_bars=10,
    )
    report = {
        "version": "0.4.0", "symbol": "EURUSD",
        "results": [{
            "timeframe": "H4", "status": "candidate_selected_research_only",
            "winner_id": "frozen-test-candidate", "winner": genome.to_dict(),
            "pattern_model": model.to_dict(),
            "winner_validation_robustness": {
                "chronological_validation_windows": [{"end_utc": index[700].isoformat()}]
            },
        }],
    }
    report_path = tmp_path / "evolution_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = replay_frozen_candidate(
        quote_path, report_path, timeframe="H4", source_timeframe="H4",
        provider_name="Independent test source", start_utc=index[710].isoformat(),
        end_utc=index[-1].isoformat(), pip_size=0.0001,
        execution=ExecutionConfig(), risk=RiskConfig(), output_dir=tmp_path / "out",
        minimum_warmup_bars=200,
    )
    assert result["frozen_genome_refitted"] is False
    assert result["pattern_model_refitted"] is False
    assert result["original_candidate"]["candidate_id"] == "frozen-test-candidate"
    trades = pd.read_csv(tmp_path / "out" / "frozen_replay_trades.csv")
    if not trades.empty:
        assert pd.to_datetime(trades["entry_time"], utc=True).min() >= index[710]


def test_frozen_replay_rejects_genome_depending_on_unrestored_motif_state(tmp_path) -> None:
    from edge.frozen_replay import _extract_candidate

    model = {
        "columns": ["x"], "means": [0.0], "scales": [1.0],
        "centroids": [[0.0], [1.0]], "n_clusters": 2, "fit_rows": 100,
    }
    report_path = tmp_path / "stateful_report.json"
    report_path.write_text(json.dumps({"results": [{
        "timeframe": "H4", "winner_id": "stateful-candidate",
        "winner": {"long_conditions": [{"feature": "motif_run_length", "operator": ">", "threshold": 2}],
                   "short_conditions": [], "stop_atr": 1.5, "reward_risk": 2.0,
                   "max_holding_bars": 10},
        "pattern_model": model,
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot yet reproduce"):
        _extract_candidate(report_path, "H4")


def test_frozen_replay_rejects_start_inside_consumed_validation(tmp_path) -> None:
    from edge.frozen_replay import replay_frozen_candidate
    from edge.pattern_recognition import PatternModel

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"results": [{
        "timeframe": "H4", "winner_id": "x",
        "winner": {"long_conditions": [], "short_conditions": [], "stop_atr": 1.5,
                   "reward_risk": 2.0, "max_holding_bars": 10},
        "pattern_model": {"columns": ["x"], "means": [0.0], "scales": [1.0],
                          "centroids": [[0.0], [1.0]], "n_clusters": 2, "fit_rows": 10},
        "winner_validation_robustness": {"chronological_validation_windows": [{
            "end_utc": "2025-11-24T12:00:00+00:00"}]},
    }]}), encoding="utf-8")
    source_path = tmp_path / "unused.csv"
    with pytest.raises(ValueError, match="must be after consumed validation end"):
        replay_frozen_candidate(
            source_path, report_path, timeframe="H4", source_timeframe="H4",
            provider_name="test", start_utc="2025-11-24T12:00:00Z",
            end_utc="2025-12-01T00:00:00Z", pip_size=0.0001,
            execution=ExecutionConfig(), risk=RiskConfig(), output_dir=tmp_path / "out",
        )
