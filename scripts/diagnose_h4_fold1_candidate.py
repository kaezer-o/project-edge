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
VALIDATION_END = 5487
FOLD_END = 6271

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


def diagnose(name, frame, genome, execution, risk):
    long_1 = frame["prior_high_distance_atr"] < -0.658441
    long_2 = frame["prior_low_distance_atr"] > 2.040495
    short_1 = frame["shape_ret_10"] > 0.565111

    signal = generate_genome_signal(frame, genome)

    print(f"\n{'=' * 65}")
    print(name)
    print(f"{'=' * 65}")
    print(f"Bars:                    {len(frame)}")
    print(f"Start:                   {frame.index[0]}")
    print(f"End:                     {frame.index[-1]}")

    print("\nIndividual condition matches:")
    print(f"Long condition 1:        {long_1.sum()}")
    print(f"Long condition 2:        {long_2.sum()}")
    print(f"Both long conditions:    {(long_1 & long_2).sum()}")
    print(f"Short condition:         {short_1.sum()}")

    print("\nFinal signals:")
    print(f"Long signals:            {(signal == 1).sum()}")
    print(f"Short signals:           {(signal == -1).sum()}")
    print(f"Neutral bars:            {(signal == 0).sum()}")
    print(f"Conflicting matches:     {(long_1 & long_2 & short_1).sum()}")

    print("\nFeature distributions:")
    print(
        frame[
            [
                "prior_high_distance_atr",
                "prior_low_distance_atr",
                "shape_ret_10",
            ]
        ].describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
        ).to_string()
    )

    output = Path("results")
    output.mkdir(exist_ok=True)

    signal_output = frame[
        [
            "prior_high_distance_atr",
            "prior_low_distance_atr",
            "shape_ret_10",
        ]
    ].copy()
    signal_output["signal"] = signal
    signal_output.to_csv(
        output / f"h4_fold1_{name.lower()}_signals.csv"
    )

    metrics, trades, equity = _run_genome(
        frame,
        genome,
        execution=execution,
        risk=risk,
        pip_size=0.0001,
        price_basis="bid",
    )

    print("\nBacktest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    if not trades.empty:
        print("\nTrades by exit reason:")
        print(
            trades.groupby("exit_reason")["net_r"]
            .agg(["count", "mean", "sum"])
            .to_string()
        )

        print("\nTrades by direction:")
        print(
            trades.groupby("direction")["net_r"]
            .agg(["count", "mean", "sum"])
            .to_string()
        )

        trades.to_csv(
            output / f"h4_fold1_{name.lower()}_trades.csv",
            index=False,
        )

    print(f"\nSaved signal data for {name}.")


def main():
    source = load_ohlc_csv(INPUT)
    source = annotate_gaps(source, TIMEFRAME_SECONDS["M1"])
    bars = resample_ohlc(source, "H4")

    print(f"Full H4 bars: {len(bars)}")
    print(f"Expected fold bars: {FOLD_END}")

    if len(bars) < FOLD_END:
        raise ValueError("Source data has fewer bars than the recorded fold.")

    fold_frame = bars.iloc[:FOLD_END].copy()
    featured = add_morphology_features(add_features(fold_frame))

    featured, pattern_model = fit_pattern_model(
        featured,
        fit_end=TRAIN_END,
        n_clusters=6,
        random_state=2026,
    )
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

    partitions = [
        ("TRAINING", featured.iloc[:TRAIN_END].copy()),
        (
            "VALIDATION",
            featured.iloc[TRAIN_END:VALIDATION_END].copy(),
        ),
        ("TEST", featured.iloc[VALIDATION_END:FOLD_END].copy()),
    ]

    for name, frame in partitions:
        diagnose(name, frame, genome, execution, risk)


if __name__ == "__main__":
    main()
