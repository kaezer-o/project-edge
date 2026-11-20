from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    family: str
    params: dict[str, float | int]
    stop_atr: float
    reward_risk: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate_candidates() -> list[StrategyCandidate]:
    """Small, predeclared hypothesis set; this is not unconstrained brute force."""
    candidates: list[StrategyCandidate] = []
    exit_models = ((1.5, 1.5), (2.0, 2.0))

    for fast, slow in ((10, 40), (20, 50)):
        for stop_atr, reward_risk in exit_models:
            candidates.append(
                StrategyCandidate(
                    name=f"ema_cross_{fast}_{slow}_s{stop_atr}_rr{reward_risk}",
                    family="ema_cross",
                    params={"fast": fast, "slow": slow},
                    stop_atr=stop_atr,
                    reward_risk=reward_risk,
                )
            )

    for lookback in (20, 55):
        for stop_atr, reward_risk in exit_models:
            candidates.append(
                StrategyCandidate(
                    name=f"donchian_{lookback}_s{stop_atr}_rr{reward_risk}",
                    family="donchian_breakout",
                    params={"lookback": lookback},
                    stop_atr=stop_atr,
                    reward_risk=reward_risk,
                )
            )

    for lower, upper in ((25, 75), (30, 70)):
        for stop_atr, reward_risk in exit_models:
            candidates.append(
                StrategyCandidate(
                    name=f"rsi_reversion_{lower}_{upper}_s{stop_atr}_rr{reward_risk}",
                    family="rsi_mean_reversion",
                    params={"lower": lower, "upper": upper},
                    stop_atr=stop_atr,
                    reward_risk=reward_risk,
                )
            )
    return candidates


def generate_signal(frame: pd.DataFrame, candidate: StrategyCandidate) -> pd.Series:
    """Return -1/0/+1 intent at candle close; execution is on the next candle."""
    signal = pd.Series(0, index=frame.index, dtype="int8", name="signal")

    if candidate.family == "ema_cross":
        fast = frame[f"ema_{int(candidate.params['fast'])}"]
        slow = frame[f"ema_{int(candidate.params['slow'])}"]
        long_cross = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        short_cross = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        signal.loc[long_cross.fillna(False)] = 1
        signal.loc[short_cross.fillna(False)] = -1

    elif candidate.family == "donchian_breakout":
        lookback = int(candidate.params["lookback"])
        upper = frame[f"donchian_high_{lookback}"]
        lower = frame[f"donchian_low_{lookback}"]
        long_break = (frame["close"] > upper) & (frame["close"].shift(1) <= upper.shift(1))
        short_break = (frame["close"] < lower) & (frame["close"].shift(1) >= lower.shift(1))
        signal.loc[long_break.fillna(False)] = 1
        signal.loc[short_break.fillna(False)] = -1

    elif candidate.family == "rsi_mean_reversion":
        lower = float(candidate.params["lower"])
        upper = float(candidate.params["upper"])
        rsi = frame["rsi_14"]
        long_reversion = (rsi <= lower) & (rsi.shift(1) > lower)
        short_reversion = (rsi >= upper) & (rsi.shift(1) < upper)
        signal.loc[long_reversion.fillna(False)] = 1
        signal.loc[short_reversion.fillna(False)] = -1

    else:
        raise ValueError(f"Unknown strategy family: {candidate.family}")

    return signal
