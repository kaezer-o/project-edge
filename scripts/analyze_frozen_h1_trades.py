import pandas as pd

from edge.config import TIMEFRAME_SECONDS
from edge.data import annotate_gaps, load_ohlc_csv, resample_ohlc
from edge.ecology import HabitatModel, classify_habitats
from edge.features import add_features
from edge.pattern_recognition import (
    add_causal_motif_dynamics,
    add_morphology_features,
    apply_pattern_model,
    pattern_model_from_dict,
)

from replay_frozen_h1_habitat import (
    OUTPUT,
    REPORT,
    SOURCE,
    TIMEFRAME,
    load_frozen_candidate,
)


def summarize(group):
    values = group["net_r"]
    gains = values[values > 0].sum()
    losses = abs(values[values < 0].sum())

    return pd.Series({
        "trades": len(values),
        "win_rate": (values > 0).mean(),
        "expectancy_r": values.mean(),
        "profit_factor": gains / losses if losses else float("inf"),
        "total_r": values.sum(),
    })


def main():
    print("Rebuilding frozen H1 habitat classifications...")

    _, pattern_model, habitat_model, cutoff = load_frozen_candidate()

    source = annotate_gaps(
        load_ohlc_csv(SOURCE),
        TIMEFRAME_SECONDS["M1"],
    )
    bars = resample_ohlc(source, TIMEFRAME)

    featured = add_morphology_features(add_features(bars))
    featured = apply_pattern_model(featured, pattern_model)
    featured = add_causal_motif_dynamics(featured, pattern_model)

    habitats = classify_habitats(featured, habitat_model)

    replay_bars = featured.loc[featured.index > cutoff]
    replay_habitats = habitats.loc[replay_bars.index]

    trades = pd.read_csv(OUTPUT / "high_volatility_trades.csv")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)

    # Entry occurs after the signal candle. Attribute using the preceding bar.
    positions = replay_bars.index.get_indexer(trades["entry_time"])

    if (positions <= 0).any():
        raise RuntimeError(
            "Could not locate a preceding signal candle for every trade."
        )

    signal_positions = positions - 1

    trades["signal_time"] = replay_bars.index.take(signal_positions)
    trades["habitat_id"] = (
        replay_habitats["habitat_id"].iloc[signal_positions].to_numpy()
    )

    trades["month"] = trades["entry_time"].dt.to_period("M").astype(str)

    trades.to_csv(OUTPUT / "high_volatility_trades_attributed.csv", index=False)

    reports = {
        "habitat": trades.groupby("habitat_id").apply(
            summarize, include_groups=False
        ),
        "direction": trades.groupby("direction").apply(
            summarize, include_groups=False
        ),
        "habitat_direction": trades.groupby(
            ["habitat_id", "direction"]
        ).apply(summarize, include_groups=False),
        "month": trades.groupby("month").apply(
            summarize, include_groups=False
        ),
    }

    for name, result in reports.items():
        print(f"\n=== {name.upper()} ===")
        print(result.round(4).to_string())

        result.to_csv(OUTPUT / f"breakdown_{name}.csv")

    print("\nSaved attributed trades and breakdowns to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
