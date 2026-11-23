from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _finite_or_none(value: float | None) -> float | None:
    """Keep undefined and non-finite metrics JSON-safe."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def calculate_metrics(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> dict[str, float | int | None]:
    if trades.empty:
        return {
            "trades": 0,
            "gap_exits": 0,
            "win_rate": None,
            "profit_factor": None,
            "expectancy_r": None,
            "average_win_r": None,
            "average_loss_r": None,
            "max_realized_drawdown_pct": None,
            "total_return_pct": 0.0,
            "trade_sharpe_unannualized": None,
        }

    returns = trades["net_r"].astype(float)
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    gross_wins = float(winners.sum())
    gross_losses = float(abs(losers.sum()))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else None

    if equity_curve.empty:
        max_drawdown = None
        total_return = None
    else:
        equity = equity_curve["equity"].astype(float)
        peaks = equity.cummax()
        drawdowns = (peaks - equity) / peaks.replace(0, np.nan)
        max_drawdown = float(drawdowns.max()) if not drawdowns.empty else 0.0
        total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1) if len(equity) > 1 else 0.0

    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    trade_sharpe = (
        float(returns.mean() / std * math.sqrt(len(returns)))
        if std > 0
        else None
    )
    return {
        "trades": int(len(trades)),
        "gap_exits": int(trades.get("gap_exit", pd.Series(False, index=trades.index)).fillna(False).sum()),
        "win_rate": _finite_or_none(float((returns > 0).mean())),
        "profit_factor": _finite_or_none(profit_factor),
        "expectancy_r": _finite_or_none(float(returns.mean())),
        "average_win_r": _finite_or_none(float(winners.mean())) if not winners.empty else None,
        "average_loss_r": _finite_or_none(float(losers.mean())) if not losers.empty else None,
        "max_realized_drawdown_pct": _finite_or_none(max_drawdown),
        "total_return_pct": _finite_or_none(total_return),
        "trade_sharpe_unannualized": _finite_or_none(trade_sharpe),
    }
