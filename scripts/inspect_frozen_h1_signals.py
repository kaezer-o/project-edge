import pandas as pd

from edge.evolution import generate_genome_signal
from analyze_frozen_h1_trades import (
    SOURCE, TIMEFRAME, load_frozen_candidate,
)
from edge.config import TIMEFRAME_SECONDS
from edge.data import annotate_gaps, load_ohlc_csv, resample_ohlc
from edge.ecology import classify_habitats
from edge.features import add_features
from edge.pattern_recognition import (
    add_causal_motif_dynamics,
    add_morphology_features,
    apply_pattern_model,
)

genome, pattern_model, habitat_model, cutoff = load_frozen_candidate()

source = annotate_gaps(load_ohlc_csv(SOURCE), TIMEFRAME_SECONDS["M1"])
bars = resample_ohlc(source, TIMEFRAME)
frame = add_morphology_features(add_features(bars))
frame = apply_pattern_model(frame, pattern_model)
frame = add_causal_motif_dynamics(frame, pattern_model)

habitats = classify_habitats(frame, habitat_model)

mask = frame.index > cutoff
frame = frame.loc[mask]
habitats = habitats.loc[mask]

signals = generate_genome_signal(frame, genome)

data = pd.DataFrame({
    "habitat": habitats["habitat_id"],
    "signal": signals,
})

data = data[data["signal"] != 0].copy()
data["direction"] = data["signal"].map({1: "long", -1: "short"})

print("\n=== RAW SIGNALS BY HABITAT AND DIRECTION ===")
print(
    data.groupby(["habitat", "direction"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\n=== TOTAL SIGNALS ===")
print(data["direction"].value_counts().to_string())
