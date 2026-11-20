from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class ExecutionConfig:
    spread_pips: float = 0.8
    slippage_pips_per_side: float = 0.1
    commission_round_trip_pips: float = 0.0
    # Signed broker swap in pip-equivalent units per rollover. Negative values
    # are financing debits; positive values are credits. Defaults are neutral.
    swap_long_pips_per_rollover: float = 0.0
    swap_short_pips_per_rollover: float = 0.0
    rollover_timezone: str = "America/New_York"
    rollover_local_time: str = "17:00"
    triple_swap_weekday: int | None = None  # Monday=0 ... Sunday=6; broker-specific
    triple_swap_multiplier: float = 3.0

    def __post_init__(self) -> None:
        if self.spread_pips < 0:
            raise ValueError("spread_pips must be >= 0")
        if self.slippage_pips_per_side < 0:
            raise ValueError("slippage_pips_per_side must be >= 0")
        if self.commission_round_trip_pips < 0:
            raise ValueError("commission_round_trip_pips must be >= 0")
        if self.triple_swap_weekday is not None and not 0 <= self.triple_swap_weekday <= 6:
            raise ValueError("triple_swap_weekday must be between 0 (Monday) and 6 (Sunday)")
        if self.triple_swap_multiplier < 1:
            raise ValueError("triple_swap_multiplier must be >= 1")
        try:
            ZoneInfo(self.rollover_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown rollover timezone: {self.rollover_timezone}") from exc
        try:
            time.fromisoformat(self.rollover_local_time)
        except ValueError as exc:
            raise ValueError("rollover_local_time must be HH:MM or HH:MM:SS") from exc


@dataclass(frozen=True)
class RiskConfig:
    starting_balance: float = 10_000.0
    risk_per_trade: float = 0.0025
    max_daily_loss: float = 0.01
    max_trades_per_day: int = 4

    def __post_init__(self) -> None:
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be > 0")
        if not 0 < self.risk_per_trade <= 0.02:
            raise ValueError("risk_per_trade must be between 0 and 0.02")
        if not 0 < self.max_daily_loss <= 0.10:
            raise ValueError("max_daily_loss must be between 0 and 0.10")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day must be >= 1")


@dataclass(frozen=True)
class ResearchConfig:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    minimum_train_trades: int = 30
    minimum_validation_trades: int = 30
    max_drawdown_filter: float = 0.20

    def __post_init__(self) -> None:
        if not 0.40 <= self.train_fraction <= 0.80:
            raise ValueError("train_fraction must be between 0.40 and 0.80")
        if not 0.10 <= self.validation_fraction <= 0.40:
            raise ValueError("validation_fraction must be between 0.10 and 0.40")
        if self.train_fraction + self.validation_fraction >= 0.95:
            raise ValueError("leave at least 5% of observations for final test")
        if self.minimum_train_trades < 1 or self.minimum_validation_trades < 1:
            raise ValueError("minimum trade counts must be >= 1")
        if not 0 < self.max_drawdown_filter <= 1:
            raise ValueError("max_drawdown_filter must be in (0, 1]")


SUPPORTED_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1")
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D1": 24 * 60 * 60,
    "W1": 7 * 24 * 60 * 60,
}
PANDAS_RULES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}
