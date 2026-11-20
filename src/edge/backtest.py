from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import ExecutionConfig, RiskConfig


@dataclass(frozen=True)
class BacktestSettings:
    pip_size: float = 0.0001
    max_holding_bars: int = 500
    price_basis: str = "mid"

    def __post_init__(self) -> None:
        if self.pip_size <= 0:
            raise ValueError("pip_size must be > 0")
        if self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1")
        if self.price_basis not in {"mid", "bid"}:
            raise ValueError("price_basis must be either 'mid' or 'bid'")


def _swap_charge_pips(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    direction: int,
    execution: ExecutionConfig,
    *,
    exit_bar_seconds: float = 0.0,
) -> tuple[float, int]:
    """Return signed swap cost (pips) and number of charged rollovers.

    Rollover instants are generated in the configured broker timezone, so the
    UTC instant follows daylight-saving transitions. The weekday/triple rule
    is deliberately configurable because brokers and instruments differ.
    """
    rate = (
        execution.swap_long_pips_per_rollover
        if direction == 1 else execution.swap_short_pips_per_rollover
    )
    if rate == 0.0 or exit_time <= entry_time:
        return 0.0, 0

    zone = ZoneInfo(execution.rollover_timezone)
    local_entry = pd.Timestamp(entry_time).to_pydatetime().astimezone(zone)
    conservative_exit_end = pd.Timestamp(exit_time) + pd.Timedelta(
    	max(0.0, exit_bar_seconds),
    	unit="s",
    )
    local_exit = conservative_exit_end.to_pydatetime().astimezone(zone)
    hour, minute, *seconds = [int(part) for part in execution.rollover_local_time.split(":")]
    second = seconds[0] if seconds else 0
    local_clock = time(hour, minute, second)

    day = local_entry.date()
    last_day = local_exit.date()
    total = 0.0
    count = 0
    entry_utc = pd.Timestamp(entry_time).to_pydatetime().astimezone(timezone.utc)
    conservative_exit_utc = conservative_exit_end.to_pydatetime().astimezone(timezone.utc)
    while day <= last_day:
        # Forex rollover is not charged on Saturday or Sunday.
        if day.weekday() < 5:
            rollover_local = datetime.combine(day, local_clock, tzinfo=zone)
            rollover_utc = rollover_local.astimezone(timezone.utc)

            # Count rollover if it occurs during the position's lifetime.
            if entry_utc < rollover_utc <= conservative_exit_utc:
                multiplier = (
                    execution.triple_swap_multiplier
                    if execution.triple_swap_weekday is not None
                    and day.weekday() == execution.triple_swap_weekday
                    else 1.0
                )
                total += rate * multiplier
                count += 1

        day += timedelta(days=1)
    return float(total), count


def run_backtest(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    stop_atr: float,
    reward_risk: float,
    execution: ExecutionConfig,
    risk: RiskConfig,
    settings: BacktestSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-position OHLC simulation with conservative same-bar stop handling.

    Signals are calculated at candle close and filled no earlier than the next
    candle open. If stop and target are both touched in one candle, stop wins.

    For mid-price OHLC, spread is deducted as a round-trip cost. For bid-price
    OHLC, an ask-side OHLC series is approximated by adding the configured or
    per-bar spread to bid OHLC. Long entries use ask and long exits use bid;
    short entries use bid and short exits use ask. Slippage and commission are
    deducted separately. This is still an approximation when historical
    intrabar spread or quote-side tick data is unavailable.
    """
    required = {"open", "high", "low", "close", "atr_14"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Backtest frame missing columns: {sorted(missing)}")
    if not frame.index.equals(signal.index):
        raise ValueError("signal index must exactly match backtest frame index")
    if stop_atr <= 0 or reward_risk <= 0:
        raise ValueError("stop_atr and reward_risk must be > 0")
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["time", "equity"])

    prices = frame[["open", "high", "low", "close", "atr_14"]].to_numpy(dtype=float)
    signals = signal.to_numpy(dtype=np.int8)
    # Resampled research frames provide timeframe-relative execution flags.
    # Raw annotated frames retain the legacy all-gap behaviour.
    gap_before_column = (
        "edge_gap_execution_before"
        if "edge_gap_execution_before" in frame.columns else "edge_gap_before"
    )
    gap_after_column = (
        "edge_gap_execution_after"
        if "edge_gap_execution_after" in frame.columns else "edge_gap_after"
    )
    gap_before = (
        frame[gap_before_column].fillna(False).to_numpy(dtype=bool)
        if gap_before_column in frame.columns else np.zeros(len(frame), dtype=bool)
    )
    gap_after = (
        frame[gap_after_column].fillna(False).to_numpy(dtype=bool)
        if gap_after_column in frame.columns else np.zeros(len(frame), dtype=bool)
    )
    gap_contaminated = (
        frame["edge_gap_contaminated"].fillna(False).to_numpy(dtype=bool)
        if "edge_gap_contaminated" in frame.columns else np.zeros(len(frame), dtype=bool)
    )
    gap_seconds_before = (
        pd.to_numeric(frame["edge_gap_seconds_before"], errors="coerce")
        .fillna(0.0).to_numpy(dtype=float)
        if "edge_gap_seconds_before" in frame.columns else np.zeros(len(frame), dtype=float)
    )
    gap_seconds_after = (
        pd.to_numeric(frame["edge_gap_seconds_after"], errors="coerce")
        .fillna(0.0).to_numpy(dtype=float)
        if "edge_gap_seconds_after" in frame.columns else np.zeros(len(frame), dtype=float)
    )
    if "edge_gap_duration_seconds" in frame.columns:
        gap_duration = (
            pd.to_numeric(frame["edge_gap_duration_seconds"], errors="coerce")
            .fillna(0.0).to_numpy(dtype=float)
        )
    else:
        gap_duration = np.maximum(gap_seconds_before, gap_seconds_after)
    spread_values = (
        frame["spread_pips"].to_numpy(dtype=float)
        if "spread_pips" in frame.columns
        else np.full(len(frame), execution.spread_pips, dtype=float)
    )
    timestamps = frame.index
    timestamp_deltas = timestamps.to_series().diff().dt.total_seconds().dropna()
    nominal_bar_seconds = (
        float(timestamp_deltas[timestamp_deltas <= timestamp_deltas.quantile(0.90)].median())
        if not timestamp_deltas.empty else 0.0
    )

    # Bid-price frames may carry actual ask OHLC. When present, use those
    # quote-side bars rather than reconstructing ask from a spread estimate.
    # available spread estimate. A constant per-bar spread cannot reproduce
    # tick-level spread changes inside a candle.
    has_actual_ask_ohlc = {
        "ask_open", "ask_high", "ask_low", "ask_close"
    }.issubset(frame.columns)
    if settings.price_basis == "bid":
        ask_prices = prices.copy()
        if has_actual_ask_ohlc:
            ask_prices[:, :4] = frame[[
                "ask_open", "ask_high", "ask_low", "ask_close"
            ]].to_numpy(dtype=float)
        else:
            spread_price = spread_values * settings.pip_size
            ask_prices[:, :4] += spread_price[:, None]
    else:
        ask_prices = prices

    equity = float(risk.starting_balance)
    trades: list[dict[str, object]] = []
    equity_points: list[dict[str, object]] = [
        {"time": timestamps[0], "equity": equity}
    ]
    daily_start_equity: dict[date, float] = {}
    daily_realized_pnl: dict[date, float] = {}
    daily_trade_count: dict[date, int] = {}

    i = 0
    n = len(frame)
    while i < n - 1:
        direction = int(signals[i])
        if direction == 0 or gap_contaminated[i]:
            i += 1
            continue

        entry_idx = i + 1
        # Do not open a new position on a partial/contaminated candle or the
        # final observed candle immediately before a known data discontinuity.
        if gap_contaminated[entry_idx] or gap_before[entry_idx] or gap_after[entry_idx]:
            i += 1
            continue
        entry_day = timestamps[entry_idx].date()
        daily_start_equity.setdefault(entry_day, equity)
        daily_realized_pnl.setdefault(entry_day, 0.0)
        daily_trade_count.setdefault(entry_day, 0)

        daily_loss_hit = (
            daily_realized_pnl[entry_day]
            <= -risk.max_daily_loss * daily_start_equity[entry_day]
        )
        if daily_loss_hit or daily_trade_count[entry_day] >= risk.max_trades_per_day:
            i += 1
            continue

        atr_at_signal = float(prices[i, 4])
        if not np.isfinite(atr_at_signal) or atr_at_signal <= 0:
            i += 1
            continue
        stop_distance = atr_at_signal * stop_atr
        if not np.isfinite(stop_distance) or stop_distance <= 0:
            i += 1
            continue

        if settings.price_basis == "bid":
            # A long opens at ask; a short opens at bid.
            entry_price = float(
                ask_prices[entry_idx, 0] if direction == 1 else prices[entry_idx, 0]
            )
            # Long exits trigger on bid; short exits trigger on ask.
            exit_quote_prices = prices if direction == 1 else ask_prices
        else:
            entry_price = float(prices[entry_idx, 0])
            exit_quote_prices = prices

        if direction == 1:
            stop_price = entry_price - stop_distance
            target_price = entry_price + stop_distance * reward_risk
        else:
            stop_price = entry_price + stop_distance
            target_price = entry_price - stop_distance * reward_risk

        exit_idx = n - 1
        exit_price = float(exit_quote_prices[-1, 3])
        exit_reason = "end_of_data"
        gap_exit_duration_seconds = 0.0
        gap_exit_assumption: str | None = None
        final_k = min(n - 1, entry_idx + settings.max_holding_bars - 1)

        for k in range(entry_idx, final_k + 1):
            # A raw source candle can be complete and executable while still
            # ending immediately before an unobserved interval. If a position
            # reaches a post-gap candle without seeing the paired gap_after
            # flag (for example, a truncated evaluation slice), close it using
            # the adverse of the last known close and first known open.
            if gap_before[k] and (k == 0 or not gap_after[k - 1]):
                previous_close = (
                    float(exit_quote_prices[k - 1, 3]) if k > entry_idx
                    else float(entry_price)
                )
                current_open = float(exit_quote_prices[k, 0])
                exit_price = (
                    min(previous_close, current_open)
                    if direction == 1 else max(previous_close, current_open)
                )
                exit_idx = k
                exit_reason = "data_gap_exit"
                gap_exit_duration_seconds = float(gap_seconds_before[k])
                gap_exit_assumption = "adverse_of_pre_gap_close_and_post_gap_open"
                break

            if gap_contaminated[k]:
                # A resampled candle can combine observations from both sides
                # of a source gap. Its intrabar path is unknown. Use the worst
                # observed extreme in the contaminated block and compare it
                # with the nearest clean endpoints. This is an intentionally
                # adverse stress assumption, not a claim about the actual fill.
                block_end = k
                while block_end + 1 < n and gap_contaminated[block_end + 1]:
                    block_end += 1
                previous_clean = k - 1
                while previous_clean >= entry_idx and gap_contaminated[previous_clean]:
                    previous_clean -= 1
                next_clean = block_end + 1
                while next_clean < n and gap_contaminated[next_clean]:
                    next_clean += 1

                if previous_clean >= entry_idx:
                    adverse_values = [float(exit_quote_prices[previous_clean, 3])]
                else:
                    adverse_values = [float(entry_price)]

                if direction == 1:
                    adverse_values.extend(
                        float(exit_quote_prices[j, 2]) for j in range(k, block_end + 1)
                    )
                else:
                    adverse_values.extend(
                        float(exit_quote_prices[j, 1]) for j in range(k, block_end + 1)
                    )

                if next_clean < n:
                    next_open = float(exit_quote_prices[next_clean, 0])
                    adverse_values.append(next_open)
                    exit_idx = next_clean
                else:
                    exit_idx = block_end
                exit_price = min(adverse_values) if direction == 1 else max(adverse_values)
                exit_reason = "data_gap_exit"
                gap_exit_duration_seconds = float(np.max(gap_duration[k:block_end + 1]))
                gap_exit_assumption = "worst_observed_extreme_and_clean_endpoints"
                break

            bar_open, bar_high, bar_low, bar_close = exit_quote_prices[k, :4]
            hit_reason: str | None = None
            hit_price: float | None = None

            if direction == 1:
                if bar_open <= stop_price:
                    hit_reason, hit_price = "stop_gap", float(bar_open)
                elif bar_open >= target_price:
                    hit_reason, hit_price = "target", float(target_price)
                elif bar_low <= stop_price and bar_high >= target_price:
                    hit_reason, hit_price = "stop_same_bar", float(stop_price)
                elif bar_low <= stop_price:
                    hit_reason, hit_price = "stop", float(stop_price)
                elif bar_high >= target_price:
                    hit_reason, hit_price = "target", float(target_price)
            else:
                if bar_open >= stop_price:
                    hit_reason, hit_price = "stop_gap", float(bar_open)
                elif bar_open <= target_price:
                    hit_reason, hit_price = "target", float(target_price)
                elif bar_high >= stop_price and bar_low <= target_price:
                    hit_reason, hit_price = "stop_same_bar", float(stop_price)
                elif bar_high >= stop_price:
                    hit_reason, hit_price = "stop", float(stop_price)
                elif bar_low <= target_price:
                    hit_reason, hit_price = "target", float(target_price)

            if hit_reason is not None:
                exit_idx = k
                exit_price = hit_price if hit_price is not None else float(bar_close)
                exit_reason = hit_reason
                break

            # For raw OHLC rows, the bar itself is not contaminated: its
            # observed path is processed normally above. The position is then
            # closed across the discontinuity using the adverse of the close
            # before it and the first open after it. Missing prices remain
            # unknowable and are explicitly recorded as an approximation.
            if gap_after[k]:
                adverse_values = [float(bar_close)]
                if k + 1 < n:
                    adverse_values.append(float(exit_quote_prices[k + 1, 0]))
                    exit_idx = k + 1
                else:
                    exit_idx = k
                exit_price = min(adverse_values) if direction == 1 else max(adverse_values)
                exit_reason = "data_gap_exit"
                gap_exit_duration_seconds = float(gap_seconds_after[k])
                gap_exit_assumption = "adverse_of_pre_gap_close_and_post_gap_open"
                break

            # An opposite close signal exits at the following candle open.
            if signals[k] == -direction and k + 1 < n:
                exit_idx = k + 1
                exit_price = float(exit_quote_prices[exit_idx, 0])
                exit_reason = "opposite_signal"
                break

            if k == final_k:
                if k + 1 < n and k < entry_idx + settings.max_holding_bars - 1:
                    exit_idx = k + 1
                    exit_price = float(exit_quote_prices[exit_idx, 0])
                    exit_reason = "time_exit"
                else:
                    exit_idx = k
                    exit_price = float(bar_close)
                    exit_reason = "time_exit" if final_k < n - 1 else "end_of_data"

        exit_day = timestamps[exit_idx].date()
        daily_start_equity.setdefault(exit_day, equity)
        daily_realized_pnl.setdefault(exit_day, 0.0)

        if settings.price_basis == "bid":
            spread_cost_pips = 0.0  # Spread is embedded in the quote-side prices.
        else:
            entry_spread = float(spread_values[entry_idx])
            exit_spread = float(spread_values[exit_idx])
            if not np.isfinite(entry_spread):
                entry_spread = execution.spread_pips
            if not np.isfinite(exit_spread):
                exit_spread = execution.spread_pips
            spread_cost_pips = 0.5 * (entry_spread + exit_spread)

        swap_cost_pips, rollover_count = _swap_charge_pips(
            timestamps[entry_idx], timestamps[exit_idx], direction, execution,
            exit_bar_seconds=nominal_bar_seconds,
        )
        total_cost_pips = (
            spread_cost_pips
            + 2 * execution.slippage_pips_per_side
            + execution.commission_round_trip_pips
            + swap_cost_pips
        )
        gross_r = direction * (exit_price - entry_price) / stop_distance
        cost_r = (total_cost_pips * settings.pip_size) / stop_distance
        net_r = gross_r - cost_r

        risk_amount = equity * risk.risk_per_trade
        pnl = risk_amount * net_r
        equity_before = equity
        equity += pnl
        daily_realized_pnl[exit_day] += pnl
        daily_trade_count[entry_day] += 1

        trade = {
            "entry_time": timestamps[entry_idx],
            "exit_time": timestamps[exit_idx],
            "direction": "long" if direction == 1 else "short",
            "price_basis": settings.price_basis,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "stop_distance": stop_distance,
            "gross_r": gross_r,
            "cost_r": cost_r,
            "spread_cost_pips": spread_cost_pips,
            "commission_cost_pips": execution.commission_round_trip_pips,
            "slippage_cost_pips": 2 * execution.slippage_pips_per_side,
            "swap_cost_pips": swap_cost_pips,
            "rollover_count": rollover_count,
            "net_r": net_r,
            "pnl": pnl,
            "equity_before": equity_before,
            "equity_after": equity,
            "exit_reason": exit_reason,
            "gap_exit": exit_reason == "data_gap_exit",
            "gap_duration_seconds": gap_exit_duration_seconds,
            "gap_exit_assumption": gap_exit_assumption,
            "bars_held": int(exit_idx - entry_idx + 1),
        }
        trades.append(trade)
        equity_points.append({"time": timestamps[exit_idx], "equity": equity})

        # Continue at the exit candle. A signal on its close can act next bar.
        i = max(exit_idx, i + 1)

    trades_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_points)
    if not equity_frame.empty:
        equity_frame = equity_frame.drop_duplicates(subset="time", keep="last")
        equity_frame = equity_frame.sort_values("time").reset_index(drop=True)
    return trades_frame, equity_frame
