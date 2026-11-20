import json
from pathlib import Path

import pandas as pd

from edge.backtest import BacktestSettings, run_backtest
from edge.config import ExecutionConfig, RiskConfig, TIMEFRAME_SECONDS
from edge.data import annotate_gaps, load_ohlc_csv, resample_ohlc
from edge.ecology import HabitatModel, classify_habitats
from edge.evolution import genome_from_dict, generate_genome_signal
from edge.features import add_features
from edge.metrics import calculate_metrics
from edge.pattern_recognition import (
    add_causal_motif_dynamics,
    add_morphology_features,
    apply_pattern_model,
    pattern_model_from_dict,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "EURUSD_M1.csv"
REPORT = ROOT / "results" / "ecological-walk-forward-habitat-v3" / (
    "ecological_walk_forward_report.json"
)

OUTPUT = ROOT / "results" / "frozen-h1-habitat-experiment"
CANDIDATE_ID = "478ff857ca9043f8"
TIMEFRAME = "H1"
PIP_SIZE = 0.0001
PRICE_BASIS = "bid"


def load_frozen_candidate():
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    timeframe_result = next(
        item for item in report["results"]
        if item["timeframe"] == TIMEFRAME
    )

    fold = next(
        item for item in timeframe_result["folds"]
        if item["walk_forward"]["fold"] == 1
    )

    candidate = next(
        item for item in fold["candidates"]
        if item["organism_id"] == CANDIDATE_ID
    )

    if not candidate["eligible"]:
        raise RuntimeError("Saved candidate is not marked eligible.")

    genome = genome_from_dict(candidate["genome"])
    pattern_model = pattern_model_from_dict(fold["pattern_model"])
    habitat_model = HabitatModel.from_dict(fold["habitat_model"])

    cutoff = pd.Timestamp(fold["walk_forward"]["test_end_utc"])

    return genome, pattern_model, habitat_model, cutoff


def run_variant(frame, signal, execution, risk, genome):
    trades, equity = run_backtest(
        frame,
        signal,
        stop_atr=genome.stop_atr,
        reward_risk=genome.reward_risk,
        execution=execution,
        risk=risk,
        settings=BacktestSettings(
            pip_size=PIP_SIZE,
            max_holding_bars=genome.max_holding_bars,
            price_basis=PRICE_BASIS,
        ),
    )

    metrics = calculate_metrics(trades, equity)

    return metrics, trades, equity


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    genome, pattern_model, habitat_model, cutoff = load_frozen_candidate()

    print("Loading source data...")
    source = load_ohlc_csv(SOURCE)
    source = annotate_gaps(source, TIMEFRAME_SECONDS["M1"])

    print("Building H1 features...")
    bars = resample_ohlc(source, TIMEFRAME)

    featured = add_morphology_features(add_features(bars))

    print("Applying frozen pattern model...")
    featured = apply_pattern_model(featured, pattern_model)
    featured = add_causal_motif_dynamics(featured, pattern_model)

    print("Applying frozen habitat model...")
    habitats = classify_habitats(featured, habitat_model)

    # Only bars strictly after the previously inspected test period.
    replay_mask = featured.index > cutoff
    replay_frame = featured.loc[replay_mask].copy()
    replay_habitats = habitats.loc[replay_mask].copy()

    if len(replay_frame) < 100:
        raise RuntimeError("Insufficient later unseen H1 data.")

    if not replay_frame.index.equals(replay_habitats.index):
        raise RuntimeError("Replay and habitat indexes do not match.")

    print(f"Replay starts: {replay_frame.index[0]}")
    print(f"Replay ends:   {replay_frame.index[-1]}")
    print(f"Replay bars:   {len(replay_frame)}")

    # Original execution assumptions from the saved report.
    execution = ExecutionConfig(
        spread_pips=0.8,
        slippage_pips_per_side=0.1,
        commission_round_trip_pips=0.0,
        swap_long_pips_per_rollover=0.0,
        swap_short_pips_per_rollover=0.0,
    )

    # Original evolve-ecology CLI defaults.
    risk = RiskConfig(
        starting_balance=10_000.0,
        risk_per_trade=0.0025,
        max_daily_loss=0.01,
        max_trades_per_day=4,
    )

    # Generate the candidate's unchanged signals.
    original_signal = generate_genome_signal(replay_frame, genome)

    # Habitat is determined from the signal candle, not the entry candle.
    volatility = (
        replay_habitats["habitat_id"]
        .astype("string")
        .str.split("|")
        .str[1]
    )

    high_volatility_mask = (
        replay_habitats["habitat_valid"].fillna(False).astype(bool)
        & volatility.eq("high_volatility").fillna(False)
    )

    filtered_signal = original_signal.where(high_volatility_mask, 0)

    print("\nRunning unfiltered candidate...")
    unfiltered_metrics, unfiltered_trades, unfiltered_equity = run_variant(
        replay_frame,
        original_signal,
        execution,
        risk,
        genome,
    )

    print("Running high-volatility-only candidate...")
    filtered_metrics, filtered_trades, filtered_equity = run_variant(
        replay_frame,
        filtered_signal,
        execution,
        risk,
        genome,
    )

    unfiltered_trades.to_csv(
        OUTPUT / "unfiltered_trades.csv", index=False
    )
    filtered_trades.to_csv(
        OUTPUT / "high_volatility_trades.csv", index=False
    )

    unfiltered_equity.to_csv(
        OUTPUT / "unfiltered_equity.csv", index=False
    )
    filtered_equity.to_csv(
        OUTPUT / "high_volatility_equity.csv", index=False
    )

    summary = {
        "candidate_id": CANDIDATE_ID,
        "timeframe": TIMEFRAME,
        "replay_start": str(replay_frame.index[0]),
        "replay_end": str(replay_frame.index[-1]),
        "replay_bars": len(replay_frame),
        "high_volatility_signal_bars": int(high_volatility_mask.sum()),
        "total_signal_bars": int(original_signal.ne(0).sum()),
        "unfiltered": unfiltered_metrics,
        "high_volatility_only": filtered_metrics,
        "execution_assumptions": {
            "price_basis": PRICE_BASIS,
            "spread_pips": 0.8,
            "slippage_pips_per_side": 0.1,
            "commission_round_trip_pips": 0.0,
            "swap_long_pips_per_rollover": 0.0,
            "swap_short_pips_per_rollover": 0.0,
        },
        "risk_assumptions": {
            "starting_balance": 10000.0,
            "risk_per_trade": 0.0025,
            "max_daily_loss": 0.01,
            "max_trades_per_day": 4,
        },
    }

    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    comparison = pd.DataFrame([
        {"variant": "unfiltered", **unfiltered_metrics},
        {"variant": "high_volatility_only", **filtered_metrics},
    ])

    print("\n=== REPLAY COMPARISON ===")
    print(comparison.to_string(index=False))

    print(f"\nHigh-volatility signal bars: {int(high_volatility_mask.sum())}")
    print(f"All candidate signal bars:   {int(original_signal.ne(0).sum())}")
    print(f"\nSaved results to: {OUTPUT}")


if __name__ == "__main__":
    main()
