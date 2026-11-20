from pathlib import Path

import pandas as pd

from edge.config import ExecutionConfig, RiskConfig, TIMEFRAME_SECONDS
from edge.data import annotate_gaps, load_ohlc_csv, resample_ohlc
from edge.evolution import generate_genome_signal, genome_from_dict
from edge.evolutionary_research import _run_genome
from edge.features import add_features
from edge.pattern_recognition import (
    add_causal_motif_dynamics,
    add_morphology_features,
    fit_pattern_model,
)

INPUT = "data/EURUSD_M1.csv"

TRAIN_END = 4703
OOS_START = 6271

GENOME_DATA = {
    "long_conditions": [
        {
            "feature": "prior_high_distance_atr",
            "operator": "<",
            "threshold": -0.658441,
        },
        {
            "feature": "prior_low_distance_atr",
            "operator": ">",
            "threshold": 2.040495,
        },
    ],
    "short_conditions": [
        {
            "feature": "shape_ret_10",
            "operator": ">",
            "threshold": 0.565111,
        },
    ],
    "stop_atr": 3.3,
    "reward_risk": 3.3,
    "max_holding_bars": 40,
}


def main():
    output = Path("results/h4_candidate_oos")
    output.mkdir(parents=True, exist_ok=True)

    print("Loading EURUSD data...")
    source = load_ohlc_csv(INPUT)
    source = annotate_gaps(source, TIMEFRAME_SECONDS["M1"])
    bars = resample_ohlc(source, "H4")

    if len(bars) <= OOS_START:
        raise ValueError(
            f"Not enough H4 bars: {len(bars)}. "
            f"Need more than {OOS_START}."
        )

    print(f"Total H4 bars: {len(bars)}")
    print(f"Training cutoff: {bars.index[TRAIN_END - 1]}")
    print(f"OOS start: {bars.index[OOS_START]}")
    print(f"OOS end: {bars.index[-1]}")

    print("\nBuilding causal features...")
    featured = add_morphology_features(add_features(bars))

    print("Fitting frozen pattern model on original training data...")
    featured, pattern_model = fit_pattern_model(
        featured,
        fit_end=TRAIN_END,
        n_clusters=6,
        random_state=2026,
    )

    print(f"Pattern model status: {pattern_model.status}")
    print(f"Pattern model training rows: {pattern_model.fit_rows}")

    print("Calculating causal motif dynamics...")
    featured = add_causal_motif_dynamics(featured, pattern_model)

    genome = genome_from_dict(GENOME_DATA)

    execution = ExecutionConfig(
        spread_pips=0.8,
        slippage_pips_per_side=0.1,
    )

    risk = RiskConfig(
        starting_balance=10000.0,
        risk_per_trade=0.0025,
        max_daily_loss=0.01,
        max_trades_per_day=4,
    )

    test_frame = featured.iloc[OOS_START:].copy()
    signals = generate_genome_signal(test_frame, genome)

    signal_output = test_frame[
        [
            "prior_high_distance_atr",
            "prior_low_distance_atr",
            "shape_ret_10",
        ]
    ].copy()
    signal_output["signal"] = signals
    signal_output.to_csv(output / "signals.csv")

    print("\nRunning frozen candidate...")
    metrics, trades, equity = _run_genome(
        test_frame,
        genome,
        execution=execution,
        risk=risk,
        pip_size=0.0001,
        price_basis="bid",
    )

    print("\nOUT-OF-SAMPLE RESULTS")
    print("=" * 55)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSignals:")
    print(f"Long: {(signals == 1).sum()}")
    print(f"Short: {(signals == -1).sum()}")
    print(f"Neutral: {(signals == 0).sum()}")

    if not trades.empty:
        trades.to_csv(output / "trades.csv", index=False)
        equity.to_csv(output / "equity.csv", index=False)

        print("\nTrades by direction:")
        print(
            trades.groupby("direction")["net_r"]
            .agg(["count", "mean", "sum"])
            .to_string()
        )

        print("\nTrades by exit reason:")
        print(
            trades.groupby("exit_reason")["net_r"]
            .agg(["count", "mean", "sum"])
            .to_string()
        )

        print(f"\nSaved trades and equity to: {output}")
    else:
        print("\nNo trades generated.")

    print(f"Saved signals to: {output}")


if __name__ == "__main__":
    main()
