from __future__ import annotations

import pandas as pd

from edge.backtest import BacktestSettings
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
from edge.ecology import fit_habitat_model, classify_habitats

INPUT = "data/EURUSD_M1.csv"

TRAIN_END = 20671
VALIDATION_END = 23624
FOLD_END = 26577

GENOME_DATA = {
    "long_conditions": [
        {"feature": "ret_1_atr", "operator": ">", "threshold": 0.258963},
        {"feature": "rsi_centered", "operator": ">", "threshold": 0.259044},
    ],
    "short_conditions": [
        {
            "feature": "compression_ratio_8_32",
            "operator": "<",
            "threshold": 0.260031,
        },
    ],
    "stop_atr": 3.515625,
    "reward_risk": 1.9602,
    "max_holding_bars": 160,
}


def diagnose(name, frame, habitats, genome, execution, risk):
    signal = generate_genome_signal(frame, genome)

    print(f"\n{'=' * 65}")
    print(name)
    print(f"{'=' * 65}")
    print(f"Bars:          {len(frame)}")
    print(f"Start:         {frame.index[0]}")
    print(f"End:           {frame.index[-1]}")
    print(f"Long signals:  {(signal == 1).sum()}")
    print(f"Short signals: {(signal == -1).sum()}")
    print(f"Neutral bars:  {(signal == 0).sum()}")

    metrics, trades, equity = _run_genome(
        frame,
        genome,
        execution=execution,
        risk=risk,
        pip_size=0.0001,
        price_basis="bid",
    )

    print("\nOverall metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    if trades.empty:
        print("\nNo trades.")
        return

    # Attribute each trade to the habitat of its signal candle.
    entry_positions = frame.index.get_indexer(
        pd.DatetimeIndex(trades["entry_time"])
    )

    if (entry_positions <= 0).any():
        raise ValueError(
            "Could not locate every trade's entry candle and preceding signal candle."
        )

    signal_positions = entry_positions - 1
    signal_habitats = habitats.iloc[signal_positions].reset_index(drop=True)

    for column in (
        "habitat_id",
        "habitat_regime",
        "habitat_volatility",
        "habitat_structure",
    ):
        trades[column] = signal_habitats[column].to_numpy()

    print("\nPerformance by habitat:")
    habitat_groups = trades.groupby(
        [
            "direction",
            "habitat_regime",
            "habitat_volatility",
            "habitat_structure",
        ],
        dropna=False,
    )

    habitat_stats = habitat_groups["net_r"].agg(
        trades="size",
        expectancy_r="mean",
        total_r="sum",
        gross_profit=lambda s: s[s > 0].sum(),
        gross_loss=lambda s: s[s < 0].sum(),
    )

    habitat_stats["profit_factor"] = (
        habitat_stats["gross_profit"]
        / habitat_stats["gross_loss"].abs().replace(0, float("nan"))
    )

    print(habitat_stats.to_string())

    print("\nTrade columns:")
    print(list(trades.columns))

    if "exit_reason" not in trades.columns:
        print("\nERROR: exit_reason column missing.")
        return

    print("\nTrades by exit reason:")
    numeric_columns = [
        col for col in trades.select_dtypes(include="number").columns
        if any(term in col.lower() for term in ("r", "pnl", "profit", "return"))
    ]

    aggregation = {"exit_reason": "count"}
    for col in numeric_columns:
        aggregation[col] = ["mean", "sum"]

    grouped = trades.groupby("exit_reason").agg(aggregation)
    print(grouped.to_string())

    print("\nPerformance by direction:")

    for direction, group in trades.groupby("direction", sort=False):
        wins = group.loc[group["net_r"] > 0, "net_r"].sum()
        losses = group.loc[group["net_r"] < 0, "net_r"].sum()
        profit_factor = wins / abs(losses) if losses < 0 else float("inf")

        print(f"\n{direction.upper()}")
        print(f"Trades:          {len(group)}")
        print(f"Win rate:        {(group['net_r'] > 0).mean():.4f}")
        print(f"Profit factor:   {profit_factor:.4f}")
        print(f"Expectancy R:    {group['net_r'].mean():.5f}")
        print(f"Total net R:     {group['net_r'].sum():.5f}")
        print(f"Median bars:     {group['bars_held'].median():.1f}")
        print(f"Mean bars:       {group['bars_held'].mean():.1f}")

    print("\nDirection × exit reason:")

    breakdown = trades.groupby(
        ["direction", "exit_reason"]
    ).agg(
        trades=("net_r", "size"),
        expectancy_r=("net_r", "mean"),
        total_r=("net_r", "sum"),
        median_bars=("bars_held", "median"),
    )

    print(breakdown.to_string())

    opposite = trades[trades["exit_reason"] == "opposite_signal"]

    print(f"\nOpposite-signal exits: {len(opposite)}")
    if not opposite.empty:
        print(opposite.to_string(index=False, max_rows=100))

    trades.to_csv(f"results/diagnostic_{name.lower()}_trades.csv", index=False)
    print(f"\nSaved trades: results/diagnostic_{name.lower()}_trades.csv")


def main():
    source = load_ohlc_csv(INPUT)
    source = annotate_gaps(source, TIMEFRAME_SECONDS["M1"])

    bars = resample_ohlc(source, "H1")

    print(f"Full H1 bars: {len(bars)}")
    print(f"Expected fold bars: {FOLD_END}")

    if len(bars) < FOLD_END:
        raise ValueError("H1 source has fewer bars than the recorded fold.")

    # Reconstruct only the data available to H1 fold 2.
    fold_frame = bars.iloc[:FOLD_END].copy()

    if len(fold_frame) != FOLD_END:
        raise ValueError("Fold length mismatch.")

    featured = add_morphology_features(add_features(fold_frame))

    featured, pattern_model = fit_pattern_model(
        featured,
        fit_end=TRAIN_END,
        n_clusters=6,
        random_state=2026,
    )
    featured = add_causal_motif_dynamics(featured, pattern_model)

    # Fit habitat thresholds on training data only.
    habitat_model = fit_habitat_model(
        featured,
        fit_end=TRAIN_END,
    )
    all_habitats = classify_habitats(featured, habitat_model)

    train_frame = featured.iloc[:TRAIN_END].copy()
    validation_frame = featured.iloc[TRAIN_END:VALIDATION_END].copy()

    train_habitats = all_habitats.iloc[:TRAIN_END].copy()
    validation_habitats = all_habitats.iloc[TRAIN_END:VALIDATION_END].copy()

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

    print("\nReconstructed partitions:")
    print(f"Training:   {len(train_frame)} bars")
    print(f"Validation: {len(validation_frame)} bars")

    diagnose(
        "TRAINING",
        train_frame,
        train_habitats,
        genome,
        execution,
        risk,
    )
    diagnose(
        "VALIDATION",
        validation_frame,
        validation_habitats,
        genome,
        execution,
        risk,
    )


if __name__ == "__main__":
    main()
