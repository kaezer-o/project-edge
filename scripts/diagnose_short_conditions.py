import pandas as pd

from edge.config import TIMEFRAME_SECONDS
from edge.data import annotate_gaps, load_ohlc_csv, resample_ohlc
from edge.evolution import generate_genome_signal
from edge.ecology import classify_habitats
from edge.features import add_features
from edge.pattern_recognition import (
    add_causal_motif_dynamics,
    add_morphology_features,
    apply_pattern_model,
)
from replay_frozen_h1_habitat import (
    SOURCE, TIMEFRAME, load_frozen_candidate,
)

genome, pattern_model, habitat_model, cutoff = load_frozen_candidate()

source = annotate_gaps(load_ohlc_csv(SOURCE), TIMEFRAME_SECONDS["M1"])
bars = resample_ohlc(source, TIMEFRAME)

frame = add_morphology_features(add_features(bars))
frame = apply_pattern_model(frame, pattern_model)
frame = add_causal_motif_dynamics(frame, pattern_model)

habitats = classify_habitats(frame, habitat_model)

periods = {
    "TRAINING": (0, 17718),
    "VALIDATION": (17718, 20671),
    "ORIGINAL TEST": (20671, 23624),
    "LATER REPLAY": (23624, len(frame)),
}

short_conditions = genome.short_conditions
long_conditions = genome.long_conditions

def evaluate(conditions, data):
    result = pd.Series(True, index=data.index)
    for condition in conditions:
        values = data[condition.feature]
        if condition.operator == ">":
            result &= values > condition.threshold
        else:
            result &= values < condition.threshold
    return result.fillna(False)

print("=== CONDITION ACTIVATION BY PERIOD ===")

for name, (start, end) in periods.items():
    data = frame.iloc[start:end]

    print(f"\n{name}: {len(data)} bars")

    for condition in short_conditions:
        values = data[condition.feature]
        passed = (
            values > condition.threshold
            if condition.operator == ">"
            else values < condition.threshold
        )
        print(
            f"{condition.feature} {condition.operator} "
            f"{condition.threshold}: {int(passed.fillna(False).sum())}"
        )

    short_match = evaluate(short_conditions, data)
    long_match = evaluate(long_conditions, data)

    print("Short conjunction:", int(short_match.sum()))
    print("Long conjunction:", int(long_match.sum()))
    print("Final short signals:", int((short_match & ~long_match).sum()))

short_conditions = genome.short_conditions
long_conditions = genome.long_conditions

def evaluate(conditions):
    result = pd.Series(True, index=frame.index)
    for condition in conditions:
        values = frame[condition.feature]
        if condition.operator == ">":
            result &= values > condition.threshold
        else:
            result &= values < condition.threshold
    return result.fillna(False)

print("=== SHORT CONDITION DIAGNOSTICS ===")

short_masks = []

for condition in short_conditions:
    values = frame[condition.feature]

    if condition.operator == ">":
        passed = values > condition.threshold
    else:
        passed = values < condition.threshold

    passed = passed.fillna(False)
    short_masks.append(passed)

    print(f"\n{condition.feature} {condition.operator} {condition.threshold}")
    print(f"Passing bars: {int(passed.sum())} / {len(frame)}")
    print(f"Passing percentage: {passed.mean() * 100:.2f}%")
    print(f"Non-null values: {int(values.notna().sum())}")
    print("Value distribution:")
    print(values.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

short_match = evaluate(short_conditions)
long_match = evaluate(long_conditions)

print("\n=== COMBINED CONDITIONS ===")
print("Short conjunction:", int(short_match.sum()))
print("Long conjunction:", int(long_match.sum()))
print("Both sides match:", int((short_match & long_match).sum()))
print("Final short signals:", int((short_match & ~long_match).sum()))
print("Final long signals:", int((long_match & ~short_match).sum()))

print("\n=== SHORT CONDITION CROSS-TAB ===")
print(pd.crosstab(
    short_masks[0],
    short_masks[1],
    rownames=["motif_04 condition"],
    colnames=["prior_frequency condition"],
).to_string())

print("\n=== SHORT MATCHES BY HABITAT ===")
print(
    habitats.loc[short_match, "habitat_id"]
    .value_counts()
    .to_string()
)
