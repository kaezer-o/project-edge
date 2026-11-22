import numpy as np
import pandas as pd

from edge.backtest import BacktestSettings, run_backtest
from edge.config import ExecutionConfig, RiskConfig


def _fixed_fill_frame():
    index = pd.date_range("2025-01-06 10:00", periods=3, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.1000, 1.1000, 1.1001],
            "high": [1.1001, 1.1002, 1.1002],
            "low": [1.0999, 1.0998, 1.1000],
            "close": [1.1000, 1.1001, 1.1001],
            "atr_14": [0.0010, 0.0010, 0.0010],
        },
        index=index,
    )


def _run_fixed_fill(execution):
    frame = _fixed_fill_frame()
    signal = pd.Series([1, 0, 0], index=frame.index, dtype=np.int8)
    trades, _ = run_backtest(
        frame,
        signal,
        stop_atr=1.0,
        reward_risk=100.0,
        execution=execution,
        risk=RiskConfig(),
        settings=BacktestSettings(
            pip_size=0.0001,
            max_holding_bars=1,
            price_basis="mid",
        ),
    )
    assert len(trades) == 1
    return trades.iloc[0]


def test_higher_spread_reduces_net_return_for_identical_mid_price_fills():
    low = _run_fixed_fill(
        ExecutionConfig(spread_pips=0.5, slippage_pips_per_side=0.1,
                        commission_round_trip_pips=0.1)
    )
    high = _run_fixed_fill(
        ExecutionConfig(spread_pips=1.0, slippage_pips_per_side=0.1,
                        commission_round_trip_pips=0.1)
    )

    assert low["entry_price"] == high["entry_price"]
    assert low["exit_price"] == high["exit_price"]
    assert low["gross_r"] == high["gross_r"]
    assert high["spread_cost_pips"] > low["spread_cost_pips"]
    assert high["net_r"] < low["net_r"]


def test_higher_slippage_reduces_net_return_for_identical_mid_price_fills():
    low = _run_fixed_fill(
        ExecutionConfig(spread_pips=0.8, slippage_pips_per_side=0.05,
                        commission_round_trip_pips=0.1)
    )
    high = _run_fixed_fill(
        ExecutionConfig(spread_pips=0.8, slippage_pips_per_side=0.25,
                        commission_round_trip_pips=0.1)
    )

    assert low["entry_price"] == high["entry_price"]
    assert low["exit_price"] == high["exit_price"]
    assert low["gross_r"] == high["gross_r"]
    assert high["slippage_cost_pips"] > low["slippage_cost_pips"]
    assert high["net_r"] < low["net_r"]


def test_higher_commission_reduces_net_return_for_identical_mid_price_fills():
    low = _run_fixed_fill(
        ExecutionConfig(spread_pips=0.8, slippage_pips_per_side=0.1,
                        commission_round_trip_pips=0.1)
    )
    high = _run_fixed_fill(
        ExecutionConfig(spread_pips=0.8, slippage_pips_per_side=0.1,
                        commission_round_trip_pips=0.5)
    )

    assert low["entry_price"] == high["entry_price"]
    assert low["exit_price"] == high["exit_price"]
    assert low["gross_r"] == high["gross_r"]
    assert high["commission_cost_pips"] > low["commission_cost_pips"]
    assert high["net_r"] < low["net_r"]
