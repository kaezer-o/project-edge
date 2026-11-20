from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


BASE_FEATURES = (
    "ret_1_atr", "ret_3_atr", "ret_8_atr", "body_atr", "upper_wick_atr",
    "lower_wick_atr", "range_atr", "trend_efficiency_10",
    "volatility_ratio_10_50", "compression_ratio_8_32", "ema_gap_atr",
    "rsi_centered", "prior_high_distance_atr", "prior_low_distance_atr",
    "motif_distance", "motif_transition_surprise", "motif_run_length",
    "motif_prior_frequency",
)


@dataclass(frozen=True)
class Condition:
    feature: str
    operator: str
    threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyGenome:
    long_conditions: tuple[Condition, ...]
    short_conditions: tuple[Condition, ...]
    stop_atr: float
    reward_risk: float
    max_holding_bars: int

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {
            "long_conditions": [c.to_dict() for c in self.long_conditions],
            "short_conditions": [c.to_dict() for c in self.short_conditions],
            "stop_atr": self.stop_atr,
            "reward_risk": self.reward_risk,
            "max_holding_bars": self.max_holding_bars,
        }


@dataclass(frozen=True)
class EvolutionSettings:
    population_size: int = 32
    generations: int = 12
    elite_count: int = 3
    tournament_size: int = 3
    mutation_rate: float = 0.28
    random_injection_rate: float = 0.10
    novelty_weight: float = 0.20
    seed: int = 2026
    pattern_clusters: int = 6
    validation_pool: int = 24
    adaptive_mutation: bool = True
    min_mutation_rate: float = 0.10
    max_mutation_rate: float = 0.65
    stagnation_generations: int = 2

    def __post_init__(self) -> None:
        if self.population_size < 8:
            raise ValueError("population_size must be >= 8")
        if self.generations < 2:
            raise ValueError("generations must be >= 2")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be between 1 and population_size-1")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("invalid tournament_size")
        if not 0 <= self.mutation_rate <= 1 or not 0 <= self.random_injection_rate < 1:
            raise ValueError("mutation and injection rates must be in [0, 1]")
        if not 0 <= self.novelty_weight <= 1:
            raise ValueError("novelty_weight must be in [0, 1]")
        if not 2 <= self.pattern_clusters <= 16:
            raise ValueError("pattern_clusters must be between 2 and 16")
        if self.validation_pool < 1:
            raise ValueError("validation_pool must be >= 1")
        if not 0 <= self.min_mutation_rate <= self.max_mutation_rate <= 1:
            raise ValueError("mutation-rate bounds must satisfy 0 <= min <= max <= 1")
        if not self.min_mutation_rate <= self.mutation_rate <= self.max_mutation_rate:
            raise ValueError("mutation_rate must be within adaptive mutation bounds")
        if self.stagnation_generations < 1:
            raise ValueError("stagnation_generations must be >= 1")


def canonicalize_conditions(
    conditions: Sequence[Condition],
    *,
    max_conditions: int = 4,
) -> tuple[Condition, ...]:
    """Remove duplicate/redundant rules and discard contradictory feature bounds.

    For a feature, ``x > a`` keeps the strongest lower bound (largest a), while
    ``x < b`` keeps the strongest upper bound (smallest b). If the two bounds
    cannot both be true (a >= b), that feature is removed from the side rather
    than leaving an organism with an impossible conjunction.
    """
    if max_conditions < 1:
        raise ValueError("max_conditions must be >= 1")
    bounds: dict[str, dict[str, Condition]] = {}
    for condition in conditions:
        if condition.operator not in (">", "<"):
            raise ValueError(f"Unsupported rule operator: {condition.operator}")
        if not condition.feature or not np.isfinite(condition.threshold):
            continue
        feature_bounds = bounds.setdefault(condition.feature, {})
        current = feature_bounds.get(condition.operator)
        if current is None:
            feature_bounds[condition.operator] = condition
        elif condition.operator == ">" and condition.threshold > current.threshold:
            feature_bounds[condition.operator] = condition
        elif condition.operator == "<" and condition.threshold < current.threshold:
            feature_bounds[condition.operator] = condition

    cleaned: list[Condition] = []
    for feature in sorted(bounds):
        feature_bounds = bounds[feature]
        lower = feature_bounds.get(">")
        upper = feature_bounds.get("<")
        if lower is not None and upper is not None and lower.threshold >= upper.threshold:
            continue
        if lower is not None:
            cleaned.append(lower)
        if upper is not None:
            cleaned.append(upper)
    return tuple(cleaned[:max_conditions])


def canonicalize_genome(genome: StrategyGenome) -> StrategyGenome:
    """Return a bounded, deduplicated genome with valid rule intervals."""
    long_conditions = canonicalize_conditions(genome.long_conditions)
    short_conditions = canonicalize_conditions(genome.short_conditions)
    return StrategyGenome(
        long_conditions=long_conditions,
        short_conditions=short_conditions,
        stop_atr=float(np.clip(genome.stop_atr, 0.5, 4.0)),
        reward_risk=float(np.clip(genome.reward_risk, 0.5, 5.0)),
        max_holding_bars=int(np.clip(genome.max_holding_bars, 1, 1000)),
    )


def genome_from_dict(payload: dict[str, object]) -> StrategyGenome:
    """Decode a stored genome without importing any historical fitness score."""
    def decode_side(key: str) -> tuple[Condition, ...]:
        rows = payload.get(key, [])
        if not isinstance(rows, list):
            raise ValueError(f"Genome field {key} must be a list")
        conditions: list[Condition] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Genome field {key} contains an invalid condition")
            conditions.append(Condition(
                feature=str(row["feature"]),
                operator=str(row["operator"]),
                threshold=float(row["threshold"]),
            ))
        return tuple(conditions)

    genome = StrategyGenome(
        long_conditions=decode_side("long_conditions"),
        short_conditions=decode_side("short_conditions"),
        stop_atr=float(payload["stop_atr"]),
        reward_risk=float(payload["reward_risk"]),
        max_holding_bars=int(payload["max_holding_bars"]),
    )
    return canonicalize_genome(genome)


def available_features(frame: pd.DataFrame) -> tuple[str, ...]:
    motif_features = tuple(sorted(
        c for c in frame.columns
        if c.startswith("motif_")
        and c != "motif_distance"
    ))
    shape_features = tuple(sorted(
        c for c in frame.columns
        if c.startswith("shape_ret_")
    ))

    features = dict.fromkeys(
        (*BASE_FEATURES, *shape_features, *motif_features)
    )

    return tuple(
        feature
        for feature in features
        if feature in frame.columns and frame[feature].notna().any()
    )


def _thresholds(
    frame: pd.DataFrame,
    features: Iterable[str],
) -> dict[str, tuple[float, ...]]:
    result: dict[str, tuple[float, ...]] = {}

    for feature in features:
        values = (
            frame[feature]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if values.empty:
            continue

        # One-hot cluster indicators are categorical.
        suffix = feature.removeprefix("motif_")
        if suffix.isdigit():
            result[feature] = (0.5,)
            continue

        # Continuous motif statistics use their actual training distribution.
        quantiles = values.quantile(
            [0.15, 0.30, 0.50, 0.70, 0.85]
        ).to_numpy(dtype=float)

        unique = tuple(sorted({
            round(float(value), 6)
            for value in quantiles
            if np.isfinite(value)
        }))

        result[feature] = unique or (float(values.median()),)

    return result


def _random_condition(rng: np.random.Generator, ranges: dict[str, tuple[float, ...]]) -> Condition:
    feature = str(rng.choice(tuple(ranges.keys())))
    operator = ">" if rng.random() < 0.5 else "<"
    threshold = float(rng.choice(ranges[feature]))
    return Condition(feature, operator, threshold)


def random_genome(rng: np.random.Generator, ranges: dict[str, tuple[float, ...]]) -> StrategyGenome:
    def random_side() -> tuple[Condition, ...]:
        for _ in range(12):
            conditions = tuple(_random_condition(rng, ranges) for _ in range(int(rng.integers(1, 4))))
            cleaned = canonicalize_conditions(conditions)
            if cleaned:
                return cleaned
        return (_random_condition(rng, ranges),)

    return canonicalize_genome(StrategyGenome(
        long_conditions=random_side(),
        short_conditions=random_side(),
        stop_atr=float(rng.choice([0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])),
        reward_risk=float(rng.choice([0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])),
        max_holding_bars=int(rng.choice([5, 10, 20, 40, 80, 160])),
    ))


def generate_genome_signal(frame: pd.DataFrame, genome: StrategyGenome) -> pd.Series:
    """Evaluate a genome as conjunctions; conflicting long/short matches abstain."""
    genome = canonicalize_genome(genome)

    def side_match(conditions: tuple[Condition, ...]) -> pd.Series:
        matches = pd.Series(bool(conditions), index=frame.index)
        for condition in conditions:
            if condition.feature not in frame.columns:
                raise ValueError(f"Genome requires missing feature: {condition.feature}")
            values = frame[condition.feature]
            if condition.operator == ">":
                matches &= values > condition.threshold
            elif condition.operator == "<":
                matches &= values < condition.threshold
            else:
                raise ValueError(f"Unsupported rule operator: {condition.operator}")
        return matches.fillna(False)

    long_match = side_match(genome.long_conditions)
    short_match = side_match(genome.short_conditions)
    signal = pd.Series(0, index=frame.index, dtype="int8", name="signal")
    signal.loc[long_match & ~short_match] = 1
    signal.loc[short_match & ~long_match] = -1
    return signal

def has_both_directional_signals(
    frame: pd.DataFrame,
    genome: StrategyGenome,
) -> bool:
    """Require at least one exclusive long and short signal."""
    signal = generate_genome_signal(frame, genome)
    return bool((signal == 1).any() and (signal == -1).any())


def _mutate_condition(rng: np.random.Generator, condition: Condition, ranges: dict[str, tuple[float, ...]]) -> Condition:
    if rng.random() < 0.35:
        return _random_condition(rng, ranges)
    values = ranges[condition.feature]
    return Condition(
        feature=condition.feature,
        operator=condition.operator if rng.random() < 0.5 else ("<" if condition.operator == ">" else ">"),
        threshold=float(rng.choice(values)),
    )


def mutate_genome(rng: np.random.Generator, genome: StrategyGenome, ranges: dict[str, tuple[float, ...]]) -> StrategyGenome:
    genome = canonicalize_genome(genome)

    def mutate_side(conditions: tuple[Condition, ...]) -> tuple[Condition, ...]:
        items = list(conditions) or [_random_condition(rng, ranges)]
        action = rng.random()
        if action < 0.20 and len(items) > 1:
            items.pop(int(rng.integers(len(items))))
        elif action < 0.40 and len(items) < 4:
            items.append(_random_condition(rng, ranges))
        else:
            idx = int(rng.integers(len(items)))
            items[idx] = _mutate_condition(rng, items[idx], ranges)
        cleaned = canonicalize_conditions(items)
        return cleaned or (_random_condition(rng, ranges),)

    return canonicalize_genome(StrategyGenome(
        long_conditions=mutate_side(genome.long_conditions),
        short_conditions=mutate_side(genome.short_conditions),
        stop_atr=float(np.clip(genome.stop_atr * rng.choice([0.75, 0.9, 1.1, 1.25]), 0.5, 4.0)),
        reward_risk=float(np.clip(genome.reward_risk * rng.choice([0.75, 0.9, 1.1, 1.25]), 0.5, 5.0)),
        max_holding_bars=int(rng.choice([5, 10, 20, 40, 80, 160])) if rng.random() < 0.25 else genome.max_holding_bars,
    ))


def crossover(rng: np.random.Generator, first: StrategyGenome, second: StrategyGenome) -> StrategyGenome:
    first = canonicalize_genome(first)
    second = canonicalize_genome(second)
    long_conditions = first.long_conditions if rng.random() < 0.5 else second.long_conditions
    short_conditions = first.short_conditions if rng.random() < 0.5 else second.short_conditions
    if rng.random() < 0.5:
        long_conditions = tuple((first.long_conditions + second.long_conditions)[:4]) or first.long_conditions
    if rng.random() < 0.5:
        short_conditions = tuple((first.short_conditions + second.short_conditions)[:4]) or second.short_conditions
    long_clean = canonicalize_conditions(long_conditions) or first.long_conditions or second.long_conditions
    short_clean = canonicalize_conditions(short_conditions) or first.short_conditions or second.short_conditions
    return canonicalize_genome(StrategyGenome(
        long_conditions=long_clean,
        short_conditions=short_clean,
        stop_atr=float(np.clip((first.stop_atr + second.stop_atr) / 2, 0.5, 4.0)),
        reward_risk=float(np.clip((first.reward_risk + second.reward_risk) / 2, 0.5, 5.0)),
        max_holding_bars=int(rng.choice([first.max_holding_bars, second.max_holding_bars])),
    ))


def behavior_descriptor(trades: pd.DataFrame, bars: int) -> np.ndarray:
    if trades.empty:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    direction = trades["direction"]
    long_fraction = float((direction == "long").mean())
    return np.array(
        [
            len(trades) / max(bars, 1) * 1000.0,
            float((trades["net_r"] > 0).mean()),
            long_fraction,
            float(trades["bars_held"].mean()) / max(bars, 1),
        ],
        dtype=float,
    )


def novelty_scores(descriptors: list[np.ndarray]) -> np.ndarray:
    if len(descriptors) <= 1:
        return np.zeros(len(descriptors), dtype=float)
    matrix = np.vstack(descriptors)
    scale = matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (matrix - matrix.mean(axis=0)) / scale
    distances = np.sqrt(((normalized[:, None, :] - normalized[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    k = min(5, len(descriptors) - 1)
    return np.partition(distances, kth=k - 1, axis=1)[:, :k].mean(axis=1)


def combined_selection_scores(quality: np.ndarray, novelty: np.ndarray, novelty_weight: float) -> np.ndarray:
    def percentile_rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
        return order / max(len(values) - 1, 1)
    return (1.0 - novelty_weight) * percentile_rank(quality) + novelty_weight * percentile_rank(novelty)


def _feature_counts(genomes: Sequence[StrategyGenome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for genome in genomes:
        for condition in genome.long_conditions + genome.short_conditions:
            counts[condition.feature] = counts.get(condition.feature, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))



def evolve_population(
    train_frame: pd.DataFrame,
    *,
    evaluate,
    settings: EvolutionSettings,
    seed_genomes: Iterable[StrategyGenome] = (),
) -> tuple[list[tuple[float, StrategyGenome, dict[str, object]]], dict[str, object]]:
    """Evolve genomes on training rows only, preserving lineage and diversity.

    Historical memory contributes genotypes only. Every memory seed is evaluated
    again on the current training partition; no old validation/test metrics are
    imported into fitness or selection.
    """
    rng = np.random.default_rng(settings.seed)
    features = available_features(train_frame)
    ranges = _thresholds(train_frame, features)

    if len(ranges) < 4:
        raise ValueError("Not enough usable market-pattern features for evolution")

    population: list[StrategyGenome] = []
    lineage: list[dict[str, object]] = []
    seen: set[str] = set()

    def random_valid_genome() -> StrategyGenome:
        for _ in range(500):
            genome = random_genome(rng, ranges)
            fingerprint = genome.fingerprint()

            if (
                fingerprint not in seen
                and has_both_directional_signals(train_frame, genome)
            ):
                seen.add(fingerprint)
                return genome

        raise ValueError(
            "Could not generate a genome with both long and short "
            "signals on the training data after 500 attempts"
        )

    memory_pool = list(seed_genomes)
    rng.shuffle(memory_pool)

    memory_seed_limit = max(1, int(settings.population_size * 0.15))

    for seed_genome in memory_pool[:memory_seed_limit]:
        genome = canonicalize_genome(seed_genome)

        required = {
            c.feature
            for c in genome.long_conditions + genome.short_conditions
        }

        if (
            not required.issubset(ranges)
            or not genome.long_conditions
            or not genome.short_conditions
            or not has_both_directional_signals(train_frame, genome)
        ):
            continue

        fingerprint = genome.fingerprint()

        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        population.append(genome)

        lineage.append({
            "generation": 0,
            "genome_id": fingerprint,
            "operator": "memory_seed",
            "parents": [],
        })

    while len(population) < settings.population_size:
        genome = random_valid_genome()
        population.append(genome)

        lineage.append({
            "generation": 0,
            "genome_id": genome.fingerprint(),
            "operator": "random_initialization",
            "parents": [],
        })

    archive: dict[str, tuple[float, StrategyGenome, dict[str, object]]] = {}
    archive_lineage: dict[str, dict[str, object]] = {}
    generation_log: list[dict[str, object]] = []
    total_evaluated = 0
    current_mutation_rate = settings.mutation_rate
    best_so_far = float("-inf")
    stagnation = 0
    lineage_events: list[dict[str, object]] = list(lineage)

    for generation in range(settings.generations):
        results: list[tuple[float, StrategyGenome, dict[str, object], np.ndarray]] = []

        for genome, origin in zip(population, lineage):
            fingerprint = genome.fingerprint()

            if not has_both_directional_signals(train_frame, genome):
                raise RuntimeError(
                    "Inactive directional genome entered the evaluation population"
                )

            evaluated = evaluate(genome)
            quality = float(evaluated["fitness"])
            descriptor = np.asarray(evaluated["behavior"], dtype=float)
            results.append((quality, genome, evaluated, descriptor))

            existing = archive.get(fingerprint)
            if existing is None or quality > existing[0]:
                archive[fingerprint] = (quality, genome, evaluated)
                archive_lineage[fingerprint] = dict(origin)

            total_evaluated += 1

        qualities = np.array([row[0] for row in results], dtype=float)
        novelties = novelty_scores([row[3] for row in results])
        selection = combined_selection_scores(
            qualities, novelties, settings.novelty_weight
        )

        finite_quality = qualities[np.isfinite(qualities)]
        generation_best = (
            float(finite_quality.max())
            if len(finite_quality)
            else float("-inf")
        )

        diversity_ratio = (
            len({row[1].fingerprint() for row in results})
            / max(len(results), 1)
        )

        improved = generation_best > best_so_far + 1e-10

        if improved:
            best_so_far = generation_best
            stagnation = 0
        else:
            stagnation += 1

        generation_log.append({
            "generation": generation + 1,
            "best_training_fitness": (
                generation_best if np.isfinite(generation_best) else None
            ),
            "mean_training_fitness": (
                float(finite_quality.mean()) if len(finite_quality) else None
            ),
            "mean_novelty": float(novelties.mean()),
            "unique_genomes": len({r[1].fingerprint() for r in results}),
            "population_diversity_ratio": float(diversity_ratio),
            "mutation_rate_used": float(current_mutation_rate),
            "stagnation_generations": stagnation,
            "top_feature_frequency": _feature_counts([
                row[1]
                for row in sorted(
                    results,
                    key=lambda x: x[0],
                    reverse=True,
                )[:max(1, settings.elite_count * 2)]
            ]),
        })

        if generation == settings.generations - 1:
            break

        if settings.adaptive_mutation:
            if (
                diversity_ratio < 0.60
                or stagnation >= settings.stagnation_generations
            ):
                current_mutation_rate = min(
                    settings.max_mutation_rate,
                    current_mutation_rate + 0.05,
                )
            elif improved and diversity_ratio > 0.80:
                current_mutation_rate = max(
                    settings.min_mutation_rate,
                    current_mutation_rate - 0.02,
                )

        elite_order = np.argsort(selection)[::-1]
        elite_indices = [
            int(idx) for idx in elite_order[:settings.elite_count]
        ]

        next_population = [results[i][1] for i in elite_indices]

        next_lineage = [
            {
                "generation": generation + 2,
                "genome_id": results[i][1].fingerprint(),
                "operator": "elite_survival",
                "parents": [results[i][1].fingerprint()],
            }
            for i in elite_indices
        ]

        lineage_events.extend(dict(event) for event in next_lineage)

        while len(next_population) < settings.population_size:
            accepted = False

            for _ in range(500):
                if rng.random() < settings.random_injection_rate:
                    child = random_genome(rng, ranges)
                    operator = "random_immigrant"
                    parents: list[str] = []
                else:
                    def tournament() -> int:
                        indices = rng.choice(
                            len(population),
                            size=settings.tournament_size,
                            replace=False,
                        )
                        return int(max(
                            indices,
                            key=lambda idx: selection[int(idx)],
                        ))

                    parent_a_idx = tournament()
                    parent_b_idx = tournament()
                    parent_a = population[parent_a_idx]
                    parent_b = population[parent_b_idx]

                    child = crossover(rng, parent_a, parent_b)
                    parents = [
                        parent_a.fingerprint(),
                        parent_b.fingerprint(),
                    ]
                    operator = "crossover"

                    if rng.random() < current_mutation_rate:
                        child = mutate_genome(rng, child, ranges)
                        operator = "crossover_plus_mutation"

                child = canonicalize_genome(child)

                if not has_both_directional_signals(train_frame, child):
                    continue

                child_id = child.fingerprint()

                if child_id in {
                    genome.fingerprint() for genome in next_population
                }:
                    continue

                accepted = True
                break

            if not accepted:
                raise ValueError(
                    "Could not produce a unique genome with both directional "
                    "signals after 500 attempts"
                )

            next_population.append(child)

            event = {
                "generation": generation + 2,
                "genome_id": child_id,
                "operator": operator,
                "parents": parents,
            }

            next_lineage.append(event)
            lineage_events.append(event)

        population = next_population
        lineage = next_lineage

    candidates = sorted(
        archive.values(),
        key=lambda row: row[0],
        reverse=True,
    )

    top_genomes = [
        row[1] for row in candidates[:min(50, len(candidates))]
    ]

    return candidates, {
        "algorithm": "adaptive genetic programming with canonical rules, crossover, mutation, elitism, novelty selection and random immigrants",
        "population_size": settings.population_size,
        "generations": settings.generations,
        "total_evaluated": total_evaluated,
        "unique_genomes": len(archive),
        "novelty_weight": settings.novelty_weight,
        "seed": settings.seed,
        "adaptive_mutation": settings.adaptive_mutation,
        "initial_mutation_rate": settings.mutation_rate,
        "final_mutation_rate": float(current_mutation_rate),
        "generation_history": generation_log,
        "lineage": lineage_events,
        "archive_top_feature_frequency": _feature_counts(top_genomes),
        "memory_seed_count": sum(
            1 for item in lineage_events
            if item.get("operator") == "memory_seed"
        ),
    }



def fitness_from_metrics(metrics: dict[str, object], trades: pd.DataFrame, bars: int, genome: StrategyGenome, min_trades: int) -> tuple[float, np.ndarray]:
    count = int(metrics.get("trades") or 0)
    expectancy = metrics.get("expectancy_r")
    drawdown = metrics.get("max_realized_drawdown_pct")
    profit_factor = metrics.get("profit_factor")
    descriptor = behavior_descriptor(trades, bars)
    complexity = len(genome.long_conditions) + len(genome.short_conditions)
    if count < min_trades or expectancy is None or drawdown is None:
        deficit = max(0.0, min_trades - count) / max(min_trades, 1)
        return -2.0 - deficit - 0.01 * complexity, descriptor
    pf = min(float(profit_factor), 5.0) if profit_factor is not None else 0.0
    score = (
        float(expectancy) * math.sqrt(count)
        / (1.0 + 4.0 * float(drawdown))
        + 0.03 * math.log1p(max(pf, 0.0))
        - 0.015 * complexity
    )
    # Penalize highly concentrated gross profits: a genome should not survive
    # solely because of one or two unusually large wins.
    positive = trades.loc[trades["net_r"] > 0, "net_r"].astype(float).sort_values(ascending=False)
    if not positive.empty:
        top_n = max(1, int(math.ceil(len(positive) * 0.05)))
        concentration = float(positive.iloc[:top_n].sum() / positive.sum()) if positive.sum() > 0 else 1.0
        score -= 0.10 * max(0.0, concentration - 0.20)
    return float(score), descriptor
