from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import numpy as np
import hashlib
import json
from typing import Mapping

from .evolution import (
    StrategyGenome,
    canonicalize_genome,
    crossover,
    has_both_directional_signals,
    mutate_genome,
    random_genome,
)
import pandas as pd


HABITAT_FEATURES = (
    "trend_efficiency_10",
    "ema_gap_atr",
    "volatility_ratio_10_50",
    "compression_ratio_8_32",
)


@dataclass(frozen=True, order=True)
class Habitat:
    """One combination of observable market conditions."""

    regime: str
    volatility: str
    structure: str

    def __post_init__(self) -> None:
        if self.regime not in {
            "trend_up", "trend_down", "range", "transitional"
        }:
            raise ValueError("Invalid habitat regime")

        if self.volatility not in {
            "low_volatility", "normal_volatility", "high_volatility"
        }:
            raise ValueError("Invalid habitat volatility")

        if self.structure not in {
            "compression", "balanced", "expansion"
        }:
            raise ValueError("Invalid habitat structure")

    @property
    def tags(self) -> tuple[str, ...]:
        return (self.regime, self.volatility, self.structure)

    @property
    def habitat_id(self) -> str:
        return "|".join(self.tags)


@dataclass(frozen=True)
class HabitatModel:
    """Thresholds fitted exclusively on a historical training prefix."""

    trend_efficiency_low: float
    trend_efficiency_high: float
    ema_gap_abs_min: float

    volatility_low: float
    volatility_high: float

    compression_low: float
    expansion_high: float

    fit_rows: int
    fit_start_utc: str | None
    fit_end_utc: str | None

    version: int = 1

    def __post_init__(self) -> None:
        values = (
            self.trend_efficiency_low,
            self.trend_efficiency_high,
            self.ema_gap_abs_min,
            self.volatility_low,
            self.volatility_high,
            self.compression_low,
            self.expansion_high,
        )

        if not all(np.isfinite(value) for value in values):
            raise ValueError("Habitat thresholds must be finite")

        if self.fit_rows < 30:
            raise ValueError("At least 30 valid training rows are required")

        if self.trend_efficiency_low > self.trend_efficiency_high:
            raise ValueError("Invalid trend thresholds")

        if self.volatility_low > self.volatility_high:
            raise ValueError("Invalid volatility thresholds")

        if self.compression_low > self.expansion_high:
            raise ValueError("Invalid structure thresholds")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> HabitatModel:
        return cls(**payload)


def fit_habitat_model(
    frame: pd.DataFrame,
    *,
    fit_end: int,
) -> HabitatModel:
    """Learn environmental thresholds from rows [0, fit_end) only."""

    if not 0 < fit_end <= len(frame):
        raise ValueError("fit_end must identify a non-empty training prefix")

    missing = set(HABITAT_FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing habitat features: {sorted(missing)}")

    training = (
        frame.iloc[:fit_end]
        .loc[:, HABITAT_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(training) < 30:
        raise ValueError(
            f"Only {len(training)} valid habitat rows; at least 30 required"
        )

    trend = training["trend_efficiency_10"].to_numpy(dtype=float)
    ema_gap = training["ema_gap_atr"].abs().to_numpy(dtype=float)
    volatility = training["volatility_ratio_10_50"].to_numpy(dtype=float)
    compression = training["compression_ratio_8_32"].to_numpy(dtype=float)

    valid_index = training.index

    return HabitatModel(
        trend_efficiency_low=float(np.quantile(trend, 0.40)),
        trend_efficiency_high=float(np.quantile(trend, 0.65)),
        ema_gap_abs_min=float(np.quantile(ema_gap, 0.50)),
        volatility_low=float(np.quantile(volatility, 0.25)),
        volatility_high=float(np.quantile(volatility, 0.75)),
        compression_low=float(np.quantile(compression, 0.25)),
        expansion_high=float(np.quantile(compression, 0.75)),
        fit_rows=len(training),
        fit_start_utc=valid_index[0].isoformat(),
        fit_end_utc=valid_index[-1].isoformat(),
    )


def classify_habitats(
    frame: pd.DataFrame,
    model: HabitatModel,
) -> pd.DataFrame:
    """Classify each row using a previously fitted, frozen habitat model.

    Each valid row receives one regime, one volatility state and one
    market-structure state. The three dimensions can overlap.
    """

    missing = set(HABITAT_FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing habitat features: {sorted(missing)}")

    values = (
        frame.loc[:, HABITAT_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=float)
    )

    valid = np.isfinite(values).all(axis=1)

    trend = values[:, 0]
    ema_gap = values[:, 1]
    volatility_ratio = values[:, 2]
    compression_ratio = values[:, 3]

    regime = np.full(len(frame), "unknown", dtype=object)

    strong_trend = (
        valid
        & (trend >= model.trend_efficiency_high)
        & (np.abs(ema_gap) >= model.ema_gap_abs_min)
        & (np.abs(ema_gap) > 1e-12)
    )

    regime[strong_trend & (ema_gap > 0)] = "trend_up"
    regime[strong_trend & (ema_gap < 0)] = "trend_down"

    ranging = (
        valid
        & (trend <= model.trend_efficiency_low)
        & (np.abs(ema_gap) <= model.ema_gap_abs_min)
    )
    regime[ranging] = "range"

    transitional = valid & ~strong_trend & ~ranging
    regime[transitional] = "transitional"

    volatility = np.full(len(frame), "unknown", dtype=object)
    volatility[valid] = "normal_volatility"

    low_vol = valid & (volatility_ratio <= model.volatility_low)
    high_vol = valid & ~low_vol & (
        volatility_ratio >= model.volatility_high
    )

    volatility[low_vol] = "low_volatility"
    volatility[high_vol] = "high_volatility"

    structure = np.full(len(frame), "unknown", dtype=object)
    structure[valid] = "balanced"

    compressed = valid & (
        compression_ratio <= model.compression_low
    )
    expanding = valid & ~compressed & (
        compression_ratio >= model.expansion_high
    )

    structure[compressed] = "compression"
    structure[expanding] = "expansion"

    habitats: list[Habitat | None] = []
    for i in range(len(frame)):
        if not valid[i]:
            habitats.append(None)
        else:
            habitats.append(
                Habitat(
                    regime=str(regime[i]),
                    volatility=str(volatility[i]),
                    structure=str(structure[i]),
                )
            )

    return pd.DataFrame(
        {
            "habitat": habitats,
            "habitat_id": [
                item.habitat_id if item is not None else None
                for item in habitats
            ],
            "habitat_regime": regime,
            "habitat_volatility": volatility,
            "habitat_structure": structure,
            "habitat_valid": valid,
        },
        index=frame.index,
    )


# ---------------------------------------------------------------------------
# Organisms, genetic similarity, populations and species
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Organism:
    """An individual strategy living inside the ecosystem."""

    organism_id: str
    genome: StrategyGenome
    generation: int
    parent_ids: tuple[str, ...] = ()
    age_generations: int = 0
    species_id: str | None = None

    def __post_init__(self) -> None:
        if not self.organism_id:
            raise ValueError("organism_id cannot be empty")
        if not isinstance(self.genome, StrategyGenome):
            raise TypeError("genome must be a StrategyGenome")
        if self.generation < 0 or self.age_generations < 0:
            raise ValueError("generation and age cannot be negative")
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("parent_ids must be unique")
        if any(not parent for parent in self.parent_ids):
            raise ValueError("parent IDs cannot be empty")

    @classmethod
    def born(
        cls,
        genome: StrategyGenome,
        generation: int,
        *,
        parent_ids: tuple[str, ...] = (),
        birth_index: int = 0,
    ) -> Organism:
        """Create a reproducibly identified organism from a genome."""
        if generation < 0 or birth_index < 0:
            raise ValueError("generation and birth_index cannot be negative")

        canonical = canonicalize_genome(genome)
        parents = tuple(parent_ids)

        payload = {
            "genome": canonical.to_dict(),
            "generation": generation,
            "parents": parents,
            "birth_index": birth_index,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        organism_id = hashlib.sha256(encoded).hexdigest()[:16]

        return cls(
            organism_id=organism_id,
            genome=canonical,
            generation=generation,
            parent_ids=parents,
        )


@dataclass(frozen=True)
class Species:
    """A genetic family represented by its founding organism."""

    species_id: str
    founder_organism_id: str
    founder_genome: StrategyGenome
    member_ids: tuple[str, ...]
    origin_generation: int
    last_active_generation: int

    def __post_init__(self) -> None:
        if not self.species_id or not self.founder_organism_id:
            raise ValueError("Species identifiers cannot be empty")
        if not self.member_ids:
            raise ValueError("A species must contain at least one organism")
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("Species member IDs must be unique")
        # The founder is a historical ancestor and need not remain alive.
        if self.origin_generation < 0:
            raise ValueError("origin_generation cannot be negative")
        if self.last_active_generation < self.origin_generation:
            raise ValueError("last_active_generation precedes species origin")

    @property
    def size(self) -> int:
        return len(self.member_ids)

@dataclass(frozen=True)
class SpeciesHabitatProfile:
    species_id: str
    habitat_expectancy: tuple[tuple[str, float], ...]
    preferred_habitat: str | None
    specialization_score: float
    sampled_habitats: int

    def __post_init__(self) -> None:
        if not self.species_id:
            raise ValueError("species_id cannot be empty")

        if not 0.0 <= self.specialization_score <= 1.0:
            raise ValueError("specialization_score must be between 0 and 1")

        if self.sampled_habitats < 0:
            raise ValueError("sampled_habitats cannot be negative")

@dataclass(frozen=True)
class Population:
    """All organisms currently participating in one generation."""

    population_id: str
    generation: int
    organisms: tuple[Organism, ...]

    def __post_init__(self) -> None:
        if not self.population_id:
            raise ValueError("population_id cannot be empty")
        if self.generation < 0:
            raise ValueError("population generation cannot be negative")

        ids = [organism.organism_id for organism in self.organisms]
        if len(ids) != len(set(ids)):
            raise ValueError("Population contains duplicate organism IDs")

        if any(org.generation > self.generation for org in self.organisms):
            raise ValueError("Population contains an organism from the future")

    @property
    def size(self) -> int:
        return len(self.organisms)


def _condition_side_distance(
    first: tuple,
    second: tuple,
    feature_scales: Mapping[str, float],
) -> float:
    """Return a normalized genetic distance between two rule sets."""
    left = {
        (condition.feature, condition.operator): condition.threshold
        for condition in first
    }
    right = {
        (condition.feature, condition.operator): condition.threshold
        for condition in second
    }

    keys = set(left) | set(right)
    if not keys:
        return 0.0

    distances = []

    for feature, operator in keys:
        key = (feature, operator)

        if key not in left or key not in right:
            distances.append(1.0)
            continue

        scale = float(
            feature_scales.get(
                feature,
                max(1.0, abs(left[key]), abs(right[key])),
            )
        )

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid genetic scale for feature {feature}")

        difference = abs(left[key] - right[key]) / scale
        distances.append(min(float(difference), 1.0))

    return float(np.mean(distances))


def genetic_distance(
    first: StrategyGenome,
    second: StrategyGenome,
    *,
    feature_scales: Mapping[str, float] | None = None,
) -> float:
    """Measure structural and numerical similarity between two genomes.

    Returns a normalized distance:
        0.0 = genetically identical
        1.0 = maximally different

    Feature scales can be supplied from training data to normalize
    numerical threshold differences across differently scaled features.
    """
    first = canonicalize_genome(first)
    second = canonicalize_genome(second)
    scales = feature_scales or {}

    long_distance = _condition_side_distance(
        first.long_conditions, second.long_conditions, scales
    )
    short_distance = _condition_side_distance(
        first.short_conditions, second.short_conditions, scales
    )

    stop_distance = min(abs(first.stop_atr - second.stop_atr) / 3.5, 1.0)
    reward_distance = min(
        abs(first.reward_risk - second.reward_risk) / 4.5, 1.0
    )
    holding_distance = min(
        abs(np.log(first.max_holding_bars / second.max_holding_bars))
        / np.log(1000.0),
        1.0,
    )

    distance = (
        0.30 * long_distance
        + 0.30 * short_distance
        + 0.15 * stop_distance
        + 0.15 * reward_distance
        + 0.10 * holding_distance
    )

    return float(np.clip(distance, 0.0, 1.0))


def speciate_population(
    population: Population,
    *,
    distance_threshold: float = 0.25,
    feature_scales: Mapping[str, float] | None = None,
    previous_species: tuple[Species, ...] | list[Species] = (),
) -> tuple[Population, tuple[Species, ...]]:
    """Group organisms while preserving surviving species identities.

    Existing species retain their identity while at least one living
    organism remains genetically close to their historical founder.

    New organisms are assigned to the closest qualifying species.
    Genetically distant organisms establish new species.
    """
    if not 0.0 <= distance_threshold <= 1.0:
        raise ValueError("distance_threshold must be between 0 and 1")

    organisms = sorted(
    	population.organisms,
    	key=lambda organism: (
        	organism.genome.fingerprint(),
        	organism.organism_id,
    	),
    )

    previous_by_id = {item.species_id: item for item in previous_species}

    if len(previous_by_id) != len(previous_species):
        raise ValueError("Duplicate previous species IDs")

    clusters: list[dict[str, object]] = []

    for item in sorted(previous_species, key=lambda value: value.species_id):
        clusters.append({
            "species_id": item.species_id,
            "founder_id": item.founder_organism_id,
            "founder_genome": item.founder_genome,
            "origin_generation": item.origin_generation,
            "members": [],
        })

    used_species_ids = {
    	item.species_id for item in previous_species
    } 

    for organism in organisms:
        preferred = next(
            (
                cluster for cluster in clusters
                if cluster["species_id"] == organism.species_id
            ),
            None,
        )

        if preferred is not None:
            distance = genetic_distance(
                organism.genome,
                preferred["founder_genome"],
                feature_scales=feature_scales,
            )
            if distance <= distance_threshold:
                preferred["members"].append(organism)
                continue

        matches = [
            (
                genetic_distance(
                    organism.genome,
                    cluster["founder_genome"],
                    feature_scales=feature_scales,
                ),
                index,
            )
            for index, cluster in enumerate(clusters)
        ]

        if matches:
            distance, closest_index = min(matches)
        else:
            distance, closest_index = float("inf"), -1

        if distance <= distance_threshold:
            clusters[closest_index]["members"].append(organism)
        else:
            species_id = "sp-" + hashlib.sha256(
                organism.organism_id.encode("utf-8")
            ).hexdigest()[:12]

            if species_id in used_species_ids:
              raise RuntimeError(
        	f"Species identity collision: {species_id}"
              )

            used_species_ids.add(species_id)

            clusters.append({
                "species_id": species_id,
                "founder_id": organism.organism_id,
                "founder_genome": organism.genome,
                "origin_generation": organism.generation,
                "members": [organism],
            })

    assigned: dict[str, Organism] = {}
    assigned_species: dict[str, str] = {}
    species_list: list[Species] = []

    for cluster in clusters:
        members = cluster["members"]

        if not members:
            continue

        species_id = str(cluster["species_id"])
        member_ids = tuple(member.organism_id for member in members)

        species_list.append(
            Species(
                species_id=species_id,
                founder_organism_id=str(cluster["founder_id"]),
                founder_genome=cluster["founder_genome"],
                member_ids=member_ids,
                origin_generation=int(cluster["origin_generation"]),
                last_active_generation=population.generation,
            )
        )

        for member in members:
            if member.organism_id in assigned_species:
                raise RuntimeError(
                    f"Organism assigned to multiple species: "
                    f"{member.organism_id}"
                )

            assigned_species[member.organism_id] = species_id
            assigned[member.organism_id] = replace(
                member,
                species_id=species_id,
            )

    if set(assigned) != {
        organism.organism_id for organism in population.organisms
    }:
        raise RuntimeError(
           "Speciation failed to assign every organism exactly once"
        )

    updated_population = replace(
        population,
        organisms=tuple(
            assigned[organism.organism_id]
            for organism in population.organisms
        ),
    )

    return (
        updated_population,
        tuple(sorted(species_list, key=lambda item: item.species_id)),
    )


# ---------------------------------------------------------------------------
# Multi-objective ecological fitness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HabitatOutcome:
    """An organism's observed performance in one market habitat."""

    habitat_id: str
    expectancy_r: float | None
    trade_count: int

    def __post_init__(self) -> None:
        if not self.habitat_id:
            raise ValueError("habitat_id cannot be empty")
        if self.trade_count < 0:
            raise ValueError("trade_count cannot be negative")
        if self.trade_count == 0:
            if self.expectancy_r is not None:
                raise ValueError("expectancy must be None when there are no trades")
        elif self.expectancy_r is None or not np.isfinite(self.expectancy_r):
            raise ValueError("Expectancy must be finite when trades exist")


@dataclass(frozen=True)
class OrganismFitness:
    """Independent objectives used to evaluate an organism's survival."""

    organism_id: str
    expectancy_r: float
    max_drawdown_pct: float
    complexity: int
    novelty_score: float
    habitat_outcomes: tuple[HabitatOutcome, ...] = ()
    temporal_persistence_score: float = 0.0
    temporal_habitat_persistence: tuple[TemporalHabitatPersistence, ...] = ()
    trade_count: int = 0
    confidence_adjusted_expectancy_r: float | None = None

    def __post_init__(self) -> None:
        if not self.organism_id:
            raise ValueError("organism_id cannot be empty")

        numeric = (
            self.expectancy_r,
            self.max_drawdown_pct,
            self.novelty_score,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("Fitness measurements must be finite")

        if (
            self.confidence_adjusted_expectancy_r is not None
            and not np.isfinite(self.confidence_adjusted_expectancy_r)
        ):
            raise ValueError(
                "Confidence-adjusted expectancy must be finite"
            )

        if self.max_drawdown_pct < 0:
            raise ValueError("Drawdown cannot be negative")
        if self.complexity < 0:
            raise ValueError("Complexity cannot be negative")
        if self.novelty_score < 0:
            raise ValueError("Novelty cannot be negative")
        if (
            not isinstance(self.trade_count, (int, np.integer))
            or self.trade_count < 0
        ):
            raise ValueError("trade_count must be a non-negative integer")

        habitat_ids = [item.habitat_id for item in self.habitat_outcomes]
        if len(habitat_ids) != len(set(habitat_ids)):
            raise ValueError("Habitat outcomes must have unique habitat IDs")
        persistence_ids = [
            item.habitat_id for item in self.temporal_habitat_persistence
        ]
        if len(persistence_ids) != len(set(persistence_ids)):
            raise ValueError(
                "Temporal habitat persistence IDs must be unique"
            )

        if (
            not np.isfinite(self.temporal_persistence_score)
            or not 0.0 <= self.temporal_persistence_score <= 1.0
        ):
            raise ValueError(
                "Temporal persistence score must be between 0 and 1"
            )

    @property
    def selection_expectancy_r(self) -> float:
        if self.confidence_adjusted_expectancy_r is not None:
            return float(self.confidence_adjusted_expectancy_r)
        return float(self.expectancy_r)

    def adaptability_score(self, minimum_trades: int = 5) -> float:
        """Fraction of sufficiently sampled habitats with positive expectancy."""
        if minimum_trades < 1:
            raise ValueError("minimum_trades must be >= 1")

        qualified = [
            item for item in self.habitat_outcomes
            if item.trade_count >= minimum_trades
        ]

        if not qualified:
            return 0.0

        return float(
            sum(item.expectancy_r > 0 for item in qualified) / len(qualified)
        )

    def niche_performance_score(self, minimum_trades: int = 5) -> float:
        """Best expectancy in any sufficiently sampled habitat."""
        if minimum_trades < 1:
            raise ValueError("minimum_trades must be >= 1")

        qualified = [
            item.expectancy_r
            for item in self.habitat_outcomes
            if item.trade_count >= minimum_trades
        ]
        return float(max(qualified, default=0.0))

    def objectives(self, minimum_habitat_trades: int = 5) -> tuple[float, ...]:
        """Return objectives with larger values always representing improvement.
        Objective order:
          1. Overall expectancy: maximize
          2. Drawdown: minimize
          3. Habitat adaptability: maximize
          4. Niche performance: maximize
          5. Complexity: minimize
          6. Novelty: maximize
          7. Temporal persistence: maximize
        """
        return (
            self.selection_expectancy_r,
            -float(self.max_drawdown_pct),
            self.adaptability_score(minimum_habitat_trades),
            self.niche_performance_score(minimum_habitat_trades),
            -float(self.complexity),
            float(self.novelty_score),
            float(self.temporal_persistence_score),
        )


def pareto_dominates(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> bool:
    """Return whether first is at least as good in every objective
    and strictly better in at least one. All objectives are maximized.
    """
    if len(first) != len(second) or not first:
        raise ValueError("Objective vectors must have equal non-zero dimensions")

    if not all(np.isfinite(value) for value in (*first, *second)):
        raise ValueError("Objective vectors must contain finite values")

    return all(a >= b for a, b in zip(first, second)) and any(
        a > b for a, b in zip(first, second)
    )


def pareto_fronts(
    objective_map: Mapping[str, tuple[float, ...]],
) -> list[list[str]]:
    """Return deterministic non-dominated fronts using Pareto sorting."""
    identifiers = sorted(objective_map)

    if not identifiers:
        return []

    vectors = [objective_map[key] for key in identifiers]
    dimensions = len(vectors[0])

    if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("All organisms must have equally sized objectives")

    if any(
        not all(np.isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise ValueError("Objective vectors must contain finite values")

    dominated_by: dict[str, int] = {key: 0 for key in identifiers}
    dominates: dict[str, list[str]] = {key: [] for key in identifiers}

    for i, first_id in enumerate(identifiers):
        for second_id in identifiers[i + 1:]:
            if pareto_dominates(
                objective_map[first_id], objective_map[second_id]
            ):
                dominates[first_id].append(second_id)
                dominated_by[second_id] += 1
            elif pareto_dominates(
                objective_map[second_id], objective_map[first_id]
            ):
                dominates[second_id].append(first_id)
                dominated_by[first_id] += 1

    fronts: list[list[str]] = []
    current = [
        key for key in identifiers if dominated_by[key] == 0
    ]

    while current:
        current.sort()
        fronts.append(current)
        following: list[str] = []

        for identifier in current:
            for weaker_id in dominates[identifier]:
                dominated_by[weaker_id] -= 1
                if dominated_by[weaker_id] == 0:
                    following.append(weaker_id)

        current = following

    if sum(len(front) for front in fronts) != len(identifiers):
        raise RuntimeError("Pareto sorting failed to rank every organism")

    return fronts

# ---------------------------------------------------------------------------
# Ecological competition and natural selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EcologicalSelection:
    """Outcome of one population's ecological selection cycle."""

    population: Population
    species: tuple[Species, ...]
    extinct_species_ids: tuple[str, ...]
    pareto_fronts: tuple[tuple[str, ...], ...]
    crowding_distance: tuple[tuple[str, float], ...]
    species_champion_ids: tuple[str, ...] = ()
    habitat_champion_ids: tuple[str, ...] = ()
    global_competition_ids: tuple[str, ...] = ()

    @property
    def survivor_ids(self) -> tuple[str, ...]:
        return tuple(org.organism_id for org in self.population.organisms)


def _crowding_distances(
    fronts: list[list[str]],
    objectives: Mapping[str, tuple[float, ...]],
) -> dict[str, float]:
    """Reward organisms that occupy less crowded objective-space regions."""
    distances: dict[str, float] = {}

    for front in fronts:
        for identifier in front:
            distances[identifier] = 0.0

        if len(front) <= 2:
            for identifier in front:
                distances[identifier] = float("inf")
            continue

        dimensions = len(objectives[front[0]])

        for dimension in range(dimensions):
            ordered = sorted(
                front,
                key=lambda identifier: (
                    objectives[identifier][dimension], identifier
                ),
            )

            low = objectives[ordered[0]][dimension]
            high = objectives[ordered[-1]][dimension]
            span = high - low

            if span <= 1e-12:
                continue

            distances[ordered[0]] = float("inf")
            distances[ordered[-1]] = float("inf")

            for position in range(1, len(ordered) - 1):
                identifier = ordered[position]

                if np.isinf(distances[identifier]):
                    continue

                previous_value = objectives[ordered[position - 1]][dimension]
                next_value = objectives[ordered[position + 1]][dimension]

                distances[identifier] += (
                    next_value - previous_value
                ) / span

    return distances


def select_ecological_survivors(
    population: Population,
    species: tuple[Species, ...] | list[Species],
    fitness: Mapping[str, OrganismFitness],
    *,
    capacity: int,
    minimum_habitat_trades: int = 5,
    minimum_total_trades: int = 0,
) -> EcologicalSelection:
    """Select survivors using Pareto rank, crowding and species protection.

    The strongest representative of each species receives protection
    when population capacity permits. Remaining places are filled by
    Pareto rank and crowding distance.

    If there are more species than available places, weaker species
    can become extinct.
    """
    if capacity < 1:
        raise ValueError("capacity must be >= 1")
    if minimum_habitat_trades < 1:
        raise ValueError("minimum_habitat_trades must be >= 1")
    if minimum_total_trades < 0:
        raise ValueError("minimum_total_trades cannot be negative")
    if not population.organisms:
        raise ValueError("Cannot select survivors from an empty population")

    organisms = {
        organism.organism_id: organism
        for organism in population.organisms
    }

    species_by_id = {item.species_id: item for item in species}

    if len(organisms) != population.size:
        raise ValueError("Population contains duplicate organism IDs")
    if len(species_by_id) != len(species):
        raise ValueError("Duplicate species IDs")

    membership: dict[str, str] = {}

    for item in species:
        for member_id in item.member_ids:
            if member_id not in organisms:
                raise ValueError(
                    f"Species references unknown organism: {member_id}"
                )
            if member_id in membership:
                raise ValueError(
                    f"Organism belongs to multiple species: {member_id}"
                )
            membership[member_id] = item.species_id

    if set(membership) != set(organisms):
        raise ValueError("Every organism must belong to exactly one species")

    for identifier, organism in organisms.items():
        if organism.species_id != membership[identifier]:
            raise ValueError(
                f"Organism species assignment mismatch: {identifier}"
            )

    if set(fitness) != set(organisms):
        raise ValueError("Fitness records must match population organisms")

    for identifier, record in fitness.items():
        if record.organism_id != identifier:
            raise ValueError(f"Fitness identity mismatch: {identifier}")

    objective_map = {
        identifier: record.objectives(minimum_habitat_trades)
        for identifier, record in fitness.items()
    }

    fronts = pareto_fronts(objective_map)
    distances = _crowding_distances(fronts, objective_map)

    front_rank = {
        identifier: rank
        for rank, front in enumerate(fronts)
        for identifier in front
    }

    def selection_key(identifier: str) -> tuple[int, float, str]:
        return (
            front_rank[identifier],
            -distances[identifier],
            identifier,
        )

    def champion_key(
        identifier: str,
    ) -> tuple[bool, bool, tuple[int, float, str]]:
        record = fitness[identifier]
        return (
            record.trade_count < minimum_total_trades,
            record.selection_expectancy_r <= 0,
            selection_key(identifier),
        )

    target_capacity = min(capacity, len(organisms))
    # Preserve one representative per species when capacity allows.
    # Profitable species receive priority over unprofitable species.
    champions = []

    for item in species:
        champion = min(
            item.member_ids,
            key=champion_key,
        )
        champions.append(champion)

    champions.sort(
        key=champion_key
    )

    # Preserve every species when capacity allows. When species
    # outnumber capacity, reserve half the slots for species champions.
    if len(species) <= target_capacity:
        species_quota = len(species)
    else:
        species_quota = max(1, target_capacity // 2)
    species_quota = min(target_capacity, species_quota)
    species_champion_ids = tuple(champions[:species_quota])

    selected = list(species_champion_ids)
    selected_set = set(selected)

    # Identify robust representatives of profitable habitats.
    habitat_champions: dict[str, tuple[str, float]] = {}

    for identifier, record in fitness.items():
        if (
            record.trade_count < minimum_total_trades
            or record.selection_expectancy_r <= 0
        ):
            continue

        persistence_by_habitat = {
            item.habitat_id: item
            for item in record.temporal_habitat_persistence
        }

        for outcome in record.habitat_outcomes:
            if (
                outcome.trade_count < minimum_habitat_trades
                or outcome.expectancy_r is None
                or outcome.expectancy_r <= 0
            ):
                continue

            persistence = persistence_by_habitat.get(outcome.habitat_id)

            if (
                persistence is None
                or persistence.qualified_periods < 2
                or persistence.profitable_periods < 2
                or persistence.weighted_expectancy_r is None
                or persistence.weighted_expectancy_r <= 0
            ):
                continue

            score = persistence.weighted_expectancy_r
            current = habitat_champions.get(outcome.habitat_id)

            if (
                current is None
                or score > current[1]
                or (
                    score == current[1]
                    and selection_key(identifier) < selection_key(current[0])
                )
            ):
                habitat_champions[outcome.habitat_id] = (
                    identifier,
                    score,
                )

    ordered_habitat_champions = sorted(
        habitat_champions.values(),
        key=lambda item: (-item[1], selection_key(item[0])),
    )

    # Reserve habitat slots only when every species can already
    # be represented and survivor capacity remains.
    habitat_quota = (
        max(1, target_capacity // 4)
        if len(species) < target_capacity
        else 0
    )
    habitat_champion_ids = []

    for identifier, _ in ordered_habitat_champions:
        if (
            len(habitat_champion_ids) >= habitat_quota
            or len(selected) >= target_capacity
        ):
            break

        if identifier in selected_set:
            continue

        selected.append(identifier)
        selected_set.add(identifier)
        habitat_champion_ids.append(identifier)

    # Remaining positions go to the strongest organisms globally.
    # Prefer an organism from a species not yet represented.
    global_competition_ids = []
    remaining = set(organisms) - selected_set

    while len(selected) < target_capacity and remaining:
        represented_species = {
            organisms[identifier].species_id
            for identifier in selected
        }

        unrepresented_candidates = [
            identifier
            for identifier in remaining
            if organisms[identifier].species_id not in represented_species
        ]

        candidates = (
            unrepresented_candidates
            if unrepresented_candidates
            else list(remaining)
        )
        identifier = min(candidates, key=champion_key)

        selected.append(identifier)
        selected_set.add(identifier)
        remaining.remove(identifier)
        global_competition_ids.append(identifier)

    surviving_organisms = tuple(
        org for org in population.organisms
        if org.organism_id in selected_set
    )

    surviving_species = []
    extinct_species = []

    for item in species:
        members = tuple(
            identifier for identifier in item.member_ids
            if identifier in selected_set
        )

        if not members:
            extinct_species.append(item.species_id)
            continue

        surviving_species.append(
            replace(
                item,
                member_ids=members,
                last_active_generation=population.generation,
            )
        )

    return EcologicalSelection(
        population=replace(
            population,
            organisms=surviving_organisms,
        ),
        species=tuple(sorted(
            surviving_species,
            key=lambda item: item.species_id,
        )),
        extinct_species_ids=tuple(sorted(extinct_species)),
        species_champion_ids=species_champion_ids,
        habitat_champion_ids=tuple(habitat_champion_ids),
        global_competition_ids=tuple(global_competition_ids),
        pareto_fronts=tuple(tuple(front) for front in fronts),
        crowding_distance=tuple(sorted(distances.items())),
    )

def _parent_selection_probabilities(
    members: list[Organism],
    fitness: Mapping[str, OrganismFitness],
    pareto_ranks: Mapping[str, int],
    crowding: Mapping[str, float],
) -> np.ndarray:
    """Calculate bounded, fitness-weighted parent probabilities."""

    if not members:
        raise ValueError("Parent selection requires at least one member")

    weights = []

    for organism in members:
        identifier = organism.organism_id

        if identifier not in fitness:
            raise ValueError(f"Missing fitness record: {identifier}")

        if identifier not in pareto_ranks:
            raise ValueError(f"Missing Pareto rank: {identifier}")

        rank = pareto_ranks[identifier]
        if rank < 0:
            raise ValueError("Pareto ranks cannot be negative")

        record = fitness[identifier]

        rank_weight = 1.0 / (1.0 + rank)
        distance = crowding.get(identifier, 0.0)

        if np.isnan(distance) or distance < 0:
            raise ValueError(
                f"Invalid crowding distance for {identifier}: {distance}"
            )

        if np.isposinf(distance):
            crowding_weight = 1.25
        else:
            crowding_weight = 0.75 + 0.5 * (
                distance / (1.0 + distance)
            )

        expectancy_weight = 1.0 + 0.2 * np.tanh(
            record.expectancy_r
        )

        persistence_weight = (
            0.8 + 0.4 * record.temporal_persistence_score
        )

        weight = (
            0.10
            + rank_weight
            * crowding_weight
            * expectancy_weight
            * persistence_weight
        )

        weights.append(weight)

    probabilities = np.asarray(weights, dtype=float)
    total = probabilities.sum()

    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid parent-selection weights")

    return probabilities / total

# ---------------------------------------------------------------------------
# Ecological reproduction, mutation and migration
# ---------------------------------------------------------------------------


def reproduce_population(
    selection: EcologicalSelection,
    *,
    target_size: int,
    feature_ranges: Mapping[str, tuple[float, ...]],
    rng: np.random.Generator,
    fitness: Mapping[str, OrganismFitness] | None = None,
    mutation_rate: float = 0.30,
    immigrant_rate: float = 0.05,
    migration_rate: float = 0.05,
    species_profiles: tuple[SpeciesHabitatProfile, ...] | None = None,
    signal_frame: pd.DataFrame | None = None,
) -> Population:
    """Produce the next generation from ecological survivors.

    Species are selected uniformly for reproduction to reduce domination
    by large species. Parents normally reproduce within their species.

    Migration permits occasional cross-species reproduction.
    Immigrants introduce fresh random genetic material.
    """
    from .evolution import crossover, mutate_genome, random_genome

    if target_size < 1:
        raise ValueError("target_size must be >= 1")

    if target_size < selection.population.size:
        raise ValueError("target_size cannot be smaller than survivor count")

    for name, value in (
        ("mutation_rate", mutation_rate),
        ("immigrant_rate", immigrant_rate),
        ("migration_rate", migration_rate),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")

    if not selection.population.organisms:
        raise ValueError("Cannot reproduce an empty population")

    if not feature_ranges:
        raise ValueError("feature_ranges cannot be empty")

    if signal_frame is not None and not isinstance(
        signal_frame, pd.DataFrame
    ):
        raise TypeError("signal_frame must be a pandas DataFrame")

    survivors = tuple(
        replace(organism, age_generations=organism.age_generations + 1)
        for organism in selection.population.organisms
    )

    if signal_frame is not None:
        inactive_survivors = [
            organism.organism_id
            for organism in survivors
            if not has_both_directional_signals(
                signal_frame, organism.genome
            )
        ]
        if inactive_survivors:
            raise ValueError(
                "Survivors must produce at least one exclusive long and "
                "short signal; inactive organisms: "
                f"{inactive_survivors[:5]}"
            )

    species_members: dict[str, list[Organism]] = {
        item.species_id: []
        for item in selection.species
    }

    for organism in selection.population.organisms:
        if organism.species_id not in species_members:
            raise ValueError(
                f"Survivor has no living species: {organism.organism_id}"
            )
        species_members[organism.species_id].append(organism)

    active_species = sorted(
        species_members,
        key=lambda identifier: identifier,
    )
    if not active_species:
        raise ValueError("No surviving species available for reproduction")

    pareto_ranks = {
        organism_id: rank
        for rank, front in enumerate(selection.pareto_fronts)
        for organism_id in front
    }

    crowding = dict(selection.crowding_distance)

    if fitness is not None:
        all_survivor_ids = {
            organism.organism_id
            for members in species_members.values()
            for organism in members
        }

        missing_fitness = all_survivor_ids - set(fitness)
        if missing_fitness:
            raise ValueError(
                f"Missing fitness records: {sorted(missing_fitness)}"
            )

        missing_ranks = all_survivor_ids - set(pareto_ranks)
        if missing_ranks:
            raise ValueError(
                f"Missing Pareto ranks: {sorted(missing_ranks)}"
            )

    def choose_parent(
        members: list[Organism],
        *,
        exclude_id: str | None = None,
    ) -> Organism:
        candidates = [
            organism
            for organism in members
            if organism.organism_id != exclude_id
        ]

        if not candidates:
            candidates = members

        if fitness is None:
            index = int(rng.integers(len(candidates)))
        else:
            probabilities = _parent_selection_probabilities(
                candidates,
                fitness,
                pareto_ranks,
                crowding,
            )
            index = int(rng.choice(len(candidates), p=probabilities))

        return candidates[index]
    
    profile_by_species = {}

    if species_profiles is not None:
        for profile in species_profiles:
            if profile.species_id in profile_by_species:
                raise ValueError("Duplicate species habitat profile")

            profile_by_species[profile.species_id] = profile

        missing = set(active_species) - set(profile_by_species)

        if missing:
            raise ValueError(
                f"Missing habitat profiles for surviving species: {sorted(missing)}"
            )    

    for members in species_members.values():
        for organism in members:
            required = {
                condition.feature
                for condition in (
                    organism.genome.long_conditions
                    + organism.genome.short_conditions
                )
            }
            if not required.issubset(feature_ranges):
                raise ValueError(
                    "feature_ranges omit features used by a survivor"
                )

    next_generation = selection.population.generation + 1
    next_organisms = list(survivors)
    used_ids = {item.organism_id for item in next_organisms}
    used_genome_fingerprints = {
         item.genome.fingerprint() for item in next_organisms
    }
    birth_index = 0

    while len(next_organisms) < target_size:
        if rng.random() < immigrant_rate:
            child_genome = random_genome(rng, dict(feature_ranges))
            parent_ids: tuple[str, ...] = ()
            operator = "immigration"
        else:
            species_a = str(rng.choice(active_species))
            members_a = species_members[species_a]
            parent_a = choose_parent(members_a)

            parent_b = parent_a

            if len(active_species) > 1 and rng.random() < migration_rate:
                other_species = [
                    identifier
                    for identifier in active_species
                    if identifier != species_a
                ]

                partner_candidates = []
                log_weights = []

                if species_profiles is not None:
                    profile_a = profile_by_species[species_a]
                    expectancy_a = (
                        dict(profile_a.habitat_expectancy).get(
                            profile_a.preferred_habitat
                        )
                        if profile_a.preferred_habitat is not None
                        else None
                    )

                    if expectancy_a is not None and expectancy_a > 0:
                        for identifier in other_species:
                            profile_b = profile_by_species[identifier]

                            if (
                                profile_b.preferred_habitat is None
                                or profile_b.preferred_habitat
                                == profile_a.preferred_habitat
                            ):
                                continue

                            expectancy_b = dict(
                                profile_b.habitat_expectancy
                            ).get(profile_b.preferred_habitat)

                            if expectancy_b is None or expectancy_b <= 0:
                                continue

                            partner_candidates.append(identifier)
                            log_weights.append(
                                float(np.log(expectancy_a) + np.log(expectancy_b))
                            )

                if partner_candidates:
                    log_weights_array = np.asarray(log_weights)
                    weights = np.exp(
                        log_weights_array - log_weights_array.max()
                    )
                    probabilities = weights / weights.sum()

                    species_b = str(rng.choice(
                        partner_candidates,
                        p=probabilities,
                    ))
                else:
                    species_b = str(rng.choice(other_species))

                members_b = species_members[species_b]
                parent_b = choose_parent(members_b)
                operator = "inter_species_crossover"

            else:
                parent_b = choose_parent(
                    members_a,
                    exclude_id=parent_a.organism_id,
                )
                operator = (
                    "self_reproduction"
                    if parent_a.organism_id == parent_b.organism_id
                    else "intra_species_crossover"
                )

            child_genome = crossover(
                rng, parent_a.genome, parent_b.genome
            )
            parent_ids = (
                (parent_a.organism_id,)
                if parent_a.organism_id == parent_b.organism_id
                else (parent_a.organism_id, parent_b.organism_id)
            )

            if rng.random() < mutation_rate:
                child_genome = mutate_genome(
                    rng, child_genome, dict(feature_ranges)
                )
                operator += "+mutation"
        child_genome = canonicalize_genome(child_genome)
        fingerprint = child_genome.fingerprint()
        dedup_attempts = 0

        while True:
            unique = fingerprint not in used_genome_fingerprints
            directionally_active = (
                signal_frame is None
                or has_both_directional_signals(signal_frame, child_genome)
            )

            # Preserve the bounded duplicate fallback, but never admit an
            # inactive genome when the caller supplies a signal frame.
            if directionally_active and (unique or dedup_attempts >= 5):
                break

            if dedup_attempts >= 500:
                raise ValueError(
                    "Could not produce an offspring with at least one "
                    "exclusive long and short signal after "
                    "500 mutation attempts"
                )

            child_genome = canonicalize_genome(
                mutate_genome(
                    rng,
                    child_genome,
                    dict(feature_ranges),
                )
            )
            fingerprint = child_genome.fingerprint()
            dedup_attempts += 1

            if not unique:
                operator += "+dedup_mutation"
            if not directionally_active:
                operator += "+signal_repair_mutation"

        child = Organism.born(
            child_genome,
            generation=next_generation,
            parent_ids=parent_ids,
            birth_index=birth_index,
        )

        birth_index += 1

        if child.organism_id in used_ids:
            raise RuntimeError(
                "Organism identity collision during reproduction"
            )

        used_ids.add(child.organism_id)
        used_genome_fingerprints.add(fingerprint)
        next_organisms.append(child)

    return Population(
        population_id=(
            f"{selection.population.population_id}-g{next_generation}"
        ),
        generation=next_generation,
        organisms=tuple(next_organisms),
    )

# ---------------------------------------------------------------------------
# Habitat specialization and organism performance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HabitatPerformanceSummary:
    """Performance grouped by the market habitat at signal time."""

    outcomes: tuple[HabitatOutcome, ...]
    classified_trades: int
    unclassified_trades: int

    def __post_init__(self) -> None:
        if self.classified_trades < 0 or self.unclassified_trades < 0:
            raise ValueError("Trade counts cannot be negative")

        if sum(item.trade_count for item in self.outcomes) != self.classified_trades:
            raise ValueError("Habitat trade counts do not match classified total")

@dataclass(frozen=True)
class TemporalHabitatPeriod:
    period_index: int
    start_time_utc: str
    end_time_utc: str
    outcomes: tuple[HabitatOutcome, ...]
    classified_trades: int
    unclassified_trades: int

    def __post_init__(self) -> None:
        if self.period_index < 0:
            raise ValueError("period_index cannot be negative")

        if not self.start_time_utc or not self.end_time_utc:
            raise ValueError("Period timestamps cannot be empty")

        if self.classified_trades < 0 or self.unclassified_trades < 0:
            raise ValueError("Trade counts cannot be negative")

        if sum(item.trade_count for item in self.outcomes) != self.classified_trades:
            raise ValueError("Habitat counts do not match classified trades")

        identifiers = [item.habitat_id for item in self.outcomes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Period habitat IDs must be unique")


@dataclass(frozen=True)
class TemporalHabitatPerformanceSummary:
    periods: tuple[TemporalHabitatPeriod, ...]
    total_trades: int
    unassigned_trades: int

    def __post_init__(self) -> None:
        if len(self.periods) < 2:
            raise ValueError("At least two periods are required")

        if self.total_trades < 0 or self.unassigned_trades < 0:
            raise ValueError("Trade counts cannot be negative")

        indices = [period.period_index for period in self.periods]
        if indices != list(range(len(self.periods))):
            raise ValueError("Period indices must be sequential")

        accounted = sum(
            period.classified_trades + period.unclassified_trades
            for period in self.periods
        ) + self.unassigned_trades

        if accounted != self.total_trades:
            raise ValueError("Temporal trade counts do not match total")

@dataclass(frozen=True)
class TemporalHabitatPersistence:
    habitat_id: str
    period_expectancies: tuple[tuple[int, float | None, int], ...]
    qualified_periods: int
    profitable_periods: int
    persistence_ratio: float
    weighted_expectancy_r: float | None
    worst_period_expectancy_r: float | None

    def __post_init__(self) -> None:
        if not self.habitat_id:
            raise ValueError("habitat_id cannot be empty")

        indices = [item[0] for item in self.period_expectancies]

        if indices != sorted(set(indices)) or any(i < 0 for i in indices):
            raise ValueError("Period indices must be unique and ordered")

        for _, expectancy, count in self.period_expectancies:
            if count < 0:
                raise ValueError("Trade counts cannot be negative")
            if count == 0 and expectancy is not None:
                raise ValueError("Expectancy must be None without trades")
            if count > 0 and (
                expectancy is None or not np.isfinite(expectancy)
            ):
                raise ValueError("Expectancy must be finite when trades exist")

        if self.qualified_periods < 0:
            raise ValueError("qualified_periods cannot be negative")

        if not 0 <= self.profitable_periods <= self.qualified_periods:
            raise ValueError("Invalid profitable period count")

        if not np.isfinite(self.persistence_ratio) or not 0 <= self.persistence_ratio <= 1:
            raise ValueError("persistence_ratio must be between 0 and 1")

        expected_ratio = (
            self.profitable_periods / self.qualified_periods
            if self.qualified_periods
            else 0.0
        )

        if not np.isclose(self.persistence_ratio, expected_ratio):
            raise ValueError("Persistence ratio does not match period counts")

        if self.qualified_periods == 0:
            if self.weighted_expectancy_r is not None:
                raise ValueError("Weighted expectancy must be None without qualified periods")
            if self.worst_period_expectancy_r is not None:
                raise ValueError("Worst expectancy must be None without qualified periods")
        else:
            if (
                self.weighted_expectancy_r is None
                or self.worst_period_expectancy_r is None
                or not np.isfinite(self.weighted_expectancy_r)
                or not np.isfinite(self.worst_period_expectancy_r)
            ):
                raise ValueError("Qualified periods require finite expectancy values")

def measure_habitat_performance(
    trades: pd.DataFrame,
    habitat_frame: pd.DataFrame,
    *,
    timeframe_seconds: int,
) -> HabitatPerformanceSummary:
    """Measure an organism's performance in each observed market habitat.

    Trades are assigned using the completed signal candle immediately
    before their entry candle. Entry-candle habitat labels are deliberately
    not used because they contain information unavailable at signal time.

    Habitats with no trades are retained with trade_count=0.
    Unclassifiable trades are counted separately.
    """
    if timeframe_seconds <= 0:
        raise ValueError("timeframe_seconds must be positive")

    if not trades.empty:
        required_trades = {"entry_time", "net_r"}
        missing_trades = required_trades.difference(trades.columns)
        if missing_trades:
            raise ValueError(
                f"Trades missing columns: {sorted(missing_trades)}"
            )

    required_habitats = {"habitat_id", "habitat_valid"}
    missing_habitats = required_habitats.difference(habitat_frame.columns)
    if missing_habitats:
        raise ValueError(
            f"Habitat frame missing columns: {sorted(missing_habitats)}"
        )

    index = habitat_frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("Habitat frame needs a timezone-aware DatetimeIndex")

    index = index.tz_convert("UTC")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("Habitat index must be sorted and unique")

    if trades.empty:
        habitat_ids = sorted({
            str(value)
            for value, valid in zip(
                habitat_frame["habitat_id"],
                habitat_frame["habitat_valid"].fillna(False).astype(bool),
            )
            if valid and pd.notna(value) and str(value)
        })
        return HabitatPerformanceSummary(
            outcomes=tuple(
                HabitatOutcome(identifier, None, 0)
                for identifier in habitat_ids
            ),
            classified_trades=0,
            unclassified_trades=0,
        )

    entry_times = pd.to_datetime(
        trades["entry_time"], utc=True, errors="coerce"
    )
    results = pd.to_numeric(trades["net_r"], errors="coerce").to_numpy(
        dtype=float
    )

    if entry_times.isna().any():
        raise ValueError("Trade entry timestamps contain invalid values")
    if not np.isfinite(results).all():
        raise ValueError("Trade net_r values must be finite")

    # Backtest enters on the candle immediately after the signal candle.
    # Use row adjacency rather than subtracting a fixed duration: weekends
    # and missing bars make adjacent H4 candle timestamps more than 4h apart.
    entry_positions = index.get_indexer(pd.DatetimeIndex(entry_times))
    positions = np.where(
        entry_positions > 0,
        entry_positions - 1,
        -1,
    )

    valid_rows = habitat_frame["habitat_valid"].fillna(False).astype(
        bool
    ).to_numpy()
    habitat_ids_at_time = habitat_frame["habitat_id"].to_numpy(dtype=object)

    grouped: dict[str, list[float]] = {}

    # Retain every valid habitat observed in the supplied evaluation frame,
    # even when this organism generated no trades in that habitat.
    known_habitats = {
        str(identifier)
        for identifier, is_valid in zip(habitat_ids_at_time, valid_rows)
        if is_valid and pd.notna(identifier) and str(identifier)
    }

    classified = 0

    for position, result in zip(positions, results):
        if position < 0 or not valid_rows[position]:
            continue

        identifier = habitat_ids_at_time[position]
        if pd.isna(identifier) or not str(identifier):
            continue

        key = str(identifier)
        grouped.setdefault(key, []).append(float(result))
        classified += 1

    outcomes = []

    for identifier in sorted(known_habitats):
        values = grouped.get(identifier, [])

        outcomes.append(
            HabitatOutcome(
                habitat_id=identifier,
                expectancy_r=float(np.mean(values)) if values else None,
                trade_count=len(values),
            )
        )

    return HabitatPerformanceSummary(
        outcomes=tuple(outcomes),
        classified_trades=classified,
        unclassified_trades=len(trades) - classified,
    )

def measure_temporal_habitat_performance(
    trades: pd.DataFrame,
    habitat_frame: pd.DataFrame,
    *,
    timeframe_seconds: int,
    windows: int = 4,
) -> TemporalHabitatPerformanceSummary:
    """Measure habitat performance across chronological evaluation windows.

    Windows contain approximately equal numbers of observed candles.
    Trades are assigned by entry candle. Habitat classification still
    uses the preceding signal candle.
    """

    if windows < 2:
        raise ValueError("windows must be >= 2")

    index = habitat_frame.index

    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("Habitat frame needs a timezone-aware DatetimeIndex")

    index = index.tz_convert("UTC")

    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("Habitat index must be sorted and unique")

    if len(index) < windows:
        raise ValueError("Not enough habitat rows for requested windows")

    if trades.empty:
        trades = trades.copy()

        if "entry_time" not in trades.columns:
            trades["entry_time"] = pd.Series(
                index=trades.index,
                dtype="datetime64[ns, UTC]",
            )

        if "net_r" not in trades.columns:
            trades["net_r"] = pd.Series(
                index=trades.index,
                dtype=float,
            )

    if "entry_time" not in trades.columns:
        raise ValueError("Trades missing columns: ['entry_time']")

    entry_times = pd.to_datetime(
        trades["entry_time"],
        utc=True,
        errors="coerce",
    )

    if entry_times.isna().any():
        raise ValueError("Trade entry timestamps contain invalid values")

    entry_positions = index.get_indexer(
        pd.DatetimeIndex(entry_times)
    )

    chunks = np.array_split(np.arange(len(index)), windows)
    periods = []

    for period_index, chunk in enumerate(chunks):
        start = int(chunk[0])
        end = int(chunk[-1])

        trade_mask = (
            (entry_positions >= start)
            & (entry_positions <= end)
        )

        period_trades = trades.iloc[
            np.flatnonzero(trade_mask)
        ]

        # Include the previous candle as a signal-time lookup boundary.
        frame_start = max(0, start - 1)

        local_habitats = habitat_frame.iloc[
            frame_start:end + 1
        ]

        measured = measure_habitat_performance(
            period_trades,
            local_habitats,
            timeframe_seconds=timeframe_seconds,
        )

        period_frame = habitat_frame.iloc[start:end + 1]

        valid_rows = period_frame["habitat_valid"].fillna(False).astype(bool)

        known_habitats = {
            str(identifier)
            for identifier, valid in zip(
                period_frame["habitat_id"],
                valid_rows,
            )
            if valid and pd.notna(identifier) and str(identifier)
        }

        # Retain habitats that actually received trades, including a
        # signal immediately before the period boundary.
        outcome_map = {
            item.habitat_id: item
            for item in measured.outcomes
        }

        traded_habitats = {
            item.habitat_id
            for item in measured.outcomes
            if item.trade_count > 0
        }

        relevant_habitats = known_habitats | traded_habitats

        outcomes = tuple(
            outcome_map[identifier]
            for identifier in sorted(relevant_habitats)
            if identifier in outcome_map
        )

        periods.append(TemporalHabitatPeriod(
            period_index=period_index,
            start_time_utc=index[start].isoformat(),
            end_time_utc=index[end].isoformat(),
            outcomes=outcomes,
            classified_trades=measured.classified_trades,
            unclassified_trades=measured.unclassified_trades,
        ))

    return TemporalHabitatPerformanceSummary(
        periods=tuple(periods),
        total_trades=len(trades),
        unassigned_trades=int((entry_positions < 0).sum()),
    )

def profile_species_habitats(
    species: tuple[Species, ...],
    fitness: Mapping[str, OrganismFitness],
    *,
    minimum_trades: int = 5,
) -> tuple[SpeciesHabitatProfile, ...]:
    """Measure each species' trade-count-weighted habitat performance."""

    if minimum_trades < 1:
        raise ValueError("minimum_trades must be >= 1")

    profiles = []

    for group in species:
        missing = sorted(
            set(group.member_ids) - set(fitness)
        )
        if missing:
            raise ValueError(
                f"Missing fitness records for species members: {missing}"
            )

        habitat_values: dict[str, list[tuple[float, int]]] = {}

        for organism_id in group.member_ids:
            member = fitness[organism_id]

            for outcome in member.habitat_outcomes:
                if outcome.trade_count < minimum_trades:
                    continue

                if outcome.expectancy_r is None:
                    continue

                habitat_values.setdefault(
                    outcome.habitat_id, []
                ).append((
                    outcome.expectancy_r,
                    outcome.trade_count,
                ))

        averages = tuple(sorted(
            (
                habitat_id,
                float(
                    sum(value * count for value, count in observations)
                    / sum(count for _, count in observations)
                ),
            )
            for habitat_id, observations in habitat_values.items()
        ))

        if averages:
            preferred = max(averages, key=lambda item: item[1])[0]
            values = [item[1] for item in averages]

            spread = max(values) - min(values)
            scale = max(abs(value) for value in values)

            specialization = (
                spread / (spread + scale)
                if spread + scale > 0
                else 0.0
            )
        else:
            preferred = None
            specialization = 0.0

        profiles.append(SpeciesHabitatProfile(
            species_id=group.species_id,
            habitat_expectancy=averages,
            preferred_habitat=preferred,
            specialization_score=float(specialization),
            sampled_habitats=len(averages),
        ))

    return tuple(profiles)


def profile_temporal_habitats(
    summary: TemporalHabitatPerformanceSummary,
    *,
    minimum_period_trades: int = 5,
) -> tuple[TemporalHabitatPersistence, ...]:
    """Measure how consistently an organism performs in each habitat."""

    if minimum_period_trades < 1:
        raise ValueError("minimum_period_trades must be >= 1")

    habitat_ids = sorted({
        outcome.habitat_id
        for period in summary.periods
        for outcome in period.outcomes
    })

    profiles = []

    for habitat_id in habitat_ids:
        observations = []
        qualified = []

        for period in summary.periods:
            outcome = next(
                (
                    item for item in period.outcomes
                    if item.habitat_id == habitat_id
                ),
                None,
            )

            # An unobserved habitat is not counted as a failure.
            if outcome is None:
                continue

            observations.append((
                period.period_index,
                outcome.expectancy_r,
                outcome.trade_count,
            ))

            if (
                outcome.trade_count >= minimum_period_trades
                and outcome.expectancy_r is not None
            ):
                qualified.append(outcome)

        profitable = sum(
            item.expectancy_r > 0
            for item in qualified
        )

        qualified_count = len(qualified)

        total_qualified_trades = sum(
            item.trade_count for item in qualified
        )

        weighted_expectancy = (
            sum(
                item.expectancy_r * item.trade_count
                for item in qualified
            ) / total_qualified_trades
            if total_qualified_trades
            else None
        )

        worst_expectancy = (
            min(item.expectancy_r for item in qualified)
            if qualified
            else None
        )

        profiles.append(TemporalHabitatPersistence(
            habitat_id=habitat_id,
            period_expectancies=tuple(observations),
            qualified_periods=qualified_count,
            profitable_periods=profitable,
            persistence_ratio=(
                profitable / qualified_count
                if qualified_count
                else 0.0
            ),
            weighted_expectancy_r=(
                float(weighted_expectancy)
                if weighted_expectancy is not None
                else None
            ),
            worst_period_expectancy_r=(
                float(worst_expectancy)
                if worst_expectancy is not None
                else None
            ),
        ))

    return tuple(profiles)