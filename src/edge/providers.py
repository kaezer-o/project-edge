from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from .data import load_ohlc_csv


class OHLCProvider(Protocol):
    """Contract implemented by every historical OHLC data adapter."""

    name: str

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Return normalized UTC OHLC bars indexed by candle-open timestamp."""
        ...


class CSVOHLCProvider:
    """Local-file provider used by v0.1 and as the adapter contract reference."""

    name = "csv"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # Symbol/timeframe are accepted to preserve the same interface as API adapters.
        if not symbol.strip() or not timeframe.strip():
            raise ValueError("symbol and timeframe must be non-empty")
        return load_ohlc_csv(self.path)
