
import json
from pathlib import Path

import pandas as pd

from edge.config import TIMEFRAME_SECONDS
from edge.data import annotate_gaps, load_ohlc_csv, resample_ohlc
from edge.ecology import HabitatModel, classify_habitats
from edge.features import add_features
from edge.pattern_recognition import add_morphology_features


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ecological-walk-forward-habitat-v3"
SOURCE = ROOT / "data" / "EURUSD_M1.csv"

REPORT_PATH = RESULTS / "ecological_walk_forward_report.json"
OUTPUT_PATH = RESULTS / "test_habitat_attribution.csv"


def analyze_fold(source_frame, timeframe, fold):
    number = fold["walk_forward"]["fold"]
    validation_end = fold["walk_forward"]["validation_end_position"]
    test_end = fold["walk_forward"]["test_end_position"]

    trades_path = RESULTS / (
        f"{timeframe}_fold_{number:02d}_test_trades.csv"
    )

    if not trades_path.exists():
        return []

    trades = pd.read_csv(trades_path)

    if trades.empty:
        return []

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"], utc=True
    )

    # Reproduce the original resampling and feature pipeline.
    bars = resample_ohlc(source_frame, timeframe)
    fold_bars = bars.iloc[:test_end].copy()

    featured = add_morphology_features(add_features(fold_bars))

    model = HabitatModel.from_dict(fold["habitat_model"])
    habitats = classify_habitats(featured, model)

    # Include the final validation candle. It is the signal candle
    # for a trade entering on the first test candle.
    test_habitats = habitats.iloc[validation_end - 1:test_end].copy()

    # Attribute each trade to the candle immediately before entry.
    entry_positions = test_habitats.index.get_indexer(
        pd.DatetimeIndex(trades["entry_time"])
    )

    rows = []

    for trade_number, (_, trade) in enumerate(trades.iterrows()):
        entry_position = entry_positions[trade_number]
        signal_position = entry_position - 1

        if entry_position < 0 or signal_position < 0:
            habitat_id = "unclassified"
        else:
            signal = test_habitats.iloc[signal_position]

            if not bool(signal["habitat_valid"]):
                habitat_id = "unclassified"
            else:
                habitat_id = signal["habitat_id"]

        rows.append({
            "timeframe": timeframe,
            "fold": number,
            "entry_time": trade["entry_time"],
            "habitat_id": habitat_id,
            "net_r": float(trade["net_r"]),
        })

    return rows


def main():
    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    source = load_ohlc_csv(SOURCE)
    source = annotate_gaps(
        source,
        TIMEFRAME_SECONDS["M1"],
    )

    attributed = []

    for timeframe_result in report["results"]:
        timeframe = timeframe_result["timeframe"]

        for fold in timeframe_result["folds"]:
            if not fold.get("winner_id"):
                continue

            attributed.extend(
                analyze_fold(source, timeframe, fold)
            )

    if not attributed:
        raise RuntimeError("No saved test trades could be attributed.")

    trades = pd.DataFrame(attributed)

    summary = (
        trades[trades["habitat_id"] != "unclassified"]
        .groupby(["timeframe", "fold", "habitat_id"])
        .agg(
            trade_count=("net_r", "size"),
            expectancy_r=("net_r", "mean"),
            gross_profit_r=(
                "net_r",
                lambda values: values[values > 0].sum(),
            ),
            gross_loss_r=(
                "net_r",
                lambda values: -values[values < 0].sum(),
            ),
        )
        .reset_index()
    )

    summary["profit_factor"] = (
        summary["gross_profit_r"] / summary["gross_loss_r"]
    )

    summary.loc[
        summary["gross_loss_r"] == 0, "profit_factor"
    ] = float("inf")

    summary = summary.drop(
        columns=["gross_profit_r", "gross_loss_r"]
    )

    summary.to_csv(OUTPUT_PATH, index=False)

    print(summary.to_string(index=False))
    print()
    print(f"Attributed trades: {len(trades)}")
    print(
        "Unclassified trades:",
        int((trades["habitat_id"] == "unclassified").sum()),
    )
    print(f"Saved: {OUTPUT_PATH}")

    print("\n=== HABITAT DIMENSIONS BY TIMEFRAME ===")

    classified_trades = trades[
        trades["habitat_id"] != "unclassified"
    ].copy()

    for dimension, position in [
        ("regime", 0),
        ("volatility", 1),
        ("structure", 2),
    ]:
        dimension_trades = classified_trades.copy()

        dimension_trades[dimension] = (
            dimension_trades["habitat_id"]
            .str.split("|")
            .str[position]
        )

        grouped = dimension_trades.groupby(
            ["timeframe", dimension]
        )["net_r"]

        result = grouped.agg(
            trade_count="size",
            expectancy_r="mean",
            gross_profit_r=lambda x: x[x > 0].sum(),
            gross_loss_r=lambda x: -x[x < 0].sum(),
        )

        result["profit_factor"] = (
            result["gross_profit_r"] / result["gross_loss_r"]
        )

        result.loc[
            result["gross_loss_r"] == 0,
            "profit_factor",
        ] = float("inf")

        print(f"\n{dimension.upper()}")
        print(
            result.drop(
                columns=["gross_profit_r", "gross_loss_r"]
            ).to_string()
        )


if __name__ == "__main__":
    main()
