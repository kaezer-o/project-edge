from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .backtest import BacktestSettings
from .config import ExecutionConfig, RiskConfig
from .ecology import (
    Organism,
    OrganismFitness,
    Population,
    Species,
    SpeciesHabitatProfile,
    TemporalHabitatPersistence,
    TemporalHabitatPerformanceSummary,
    measure_habitat_performance,
    measure_temporal_habitat_performance,
    profile_temporal_habitats,
    profile_species_habitats,
)
from .evolution import (
    available_features,
    behavior_descriptor,
    has_both_directional_signals,
    novelty_scores,
    random_genome,
)
from .evolutionary_research import _run_genome


@dataclass(frozen=True)
class OrganismEvaluation:
    organism_id: str
    metrics: dict[str, object]
    trades: pd.DataFrame
    equity: pd.DataFrame
    descriptor: tuple[float, ...]
    temporal_habitat_performance: TemporalHabitatPerformanceSummary
    temporal_habitat_persistence: tuple[TemporalHabitatPersistence, ...]


@dataclass(frozen=True)
class PopulationEvaluation:
    fitness: Mapping[str, OrganismFitness]
    evaluations: Mapping[str, OrganismEvaluation]


def initialize_ecological_population(
    frame: pd.DataFrame,
    *,
    population_id: str,
    population_size: int,
    feature_ranges: Mapping[str, tuple[float, ...]],
    seed: int,
    max_attempts_per_organism: int = 500,
) -> Population:
    """Create a unique initial population with both directions active."""
    if not population_id:
        raise ValueError("population_id cannot be empty")
    if population_size < 1:
        raise ValueError("population_size must be >= 1")
    if max_attempts_per_organism < 1:
        raise ValueError("max_attempts_per_organism must be >= 1")
    if frame.empty:
        raise ValueError("Cannot initialize a population from an empty frame")
    if not feature_ranges:
        raise ValueError("feature_ranges cannot be empty")

    rng = np.random.default_rng(seed)
    seen_fingerprints: set[str] = set()
    organisms: list[Organism] = []

    for birth_index in range(population_size):
        accepted_genome = None

        for _ in range(max_attempts_per_organism):
            genome = random_genome(rng, dict(feature_ranges))
            fingerprint = genome.fingerprint()

            if fingerprint in seen_fingerprints:
                continue
            if not has_both_directional_signals(frame, genome):
                continue

            accepted_genome = genome
            seen_fingerprints.add(fingerprint)
            break

        if accepted_genome is None:
            raise ValueError(
                "Could not initialize a unique genome with both long and "
                "short signals after "
                f"{max_attempts_per_organism} attempts for organism "
                f"{birth_index}"
            )

        organisms.append(
            Organism.born(
                accepted_genome,
                generation=0,
                birth_index=birth_index,
            )
        )

    return Population(
        population_id=population_id,
        generation=0,
        organisms=tuple(organisms),
    )


def evaluate_ecological_population(
    population: Population,
    frame: pd.DataFrame,
    habitat_frame: pd.DataFrame,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    timeframe_seconds: int,
    minimum_habitat_trades: int = 5,
    temporal_windows: int = 4,
    minimum_period_trades: int = 5,
) -> PopulationEvaluation:
    """Evaluate every organism on identical training data."""

    if not population.organisms:
        raise ValueError("Cannot evaluate an empty population")

    if len(frame) != len(habitat_frame):
        raise ValueError("Market and habitat frames must have equal lengths")

    if not frame.index.equals(habitat_frame.index):
        raise ValueError("Market and habitat indexes must match")

    if minimum_habitat_trades < 1:
        raise ValueError("minimum_habitat_trades must be >= 1")

    if temporal_windows < 2:
        raise ValueError("temporal_windows must be >= 2")

    if temporal_windows > len(frame):
        raise ValueError("temporal_windows cannot exceed available rows")
    
    if minimum_period_trades < 1:
        raise ValueError("minimum_period_trades must be >= 1")

    evaluations: dict[str, OrganismEvaluation] = {}
    descriptors: list[np.ndarray] = []

    persistence_scores: dict[str, float] = {}

    for organism in population.organisms:
        metrics, trades, equity = _run_genome(
            frame,
            organism.genome,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
        )

        descriptor = behavior_descriptor(trades, len(frame))


        temporal_performance = measure_temporal_habitat_performance(
            trades,
            habitat_frame,
            timeframe_seconds=timeframe_seconds,
            windows=temporal_windows,
        )

        temporal_persistence = profile_temporal_habitats(
            temporal_performance,
            minimum_period_trades=minimum_period_trades,
        )

        qualified_periods = sum(
            profile.qualified_periods
            for profile in temporal_persistence
        )

        profitable_periods = sum(
            profile.profitable_periods
            for profile in temporal_persistence
        )

        persistence_scores[organism.organism_id] = (
            profitable_periods / qualified_periods
            if qualified_periods > 0
            else 0.0
         )

        evaluations[organism.organism_id] = OrganismEvaluation(
            organism_id=organism.organism_id,
            metrics=metrics,
            trades=trades,
            equity=equity,
            descriptor=tuple(float(x) for x in descriptor),
            temporal_habitat_performance=temporal_performance,
            temporal_habitat_persistence=temporal_persistence,
        )

        descriptors.append(descriptor)

    novelty = novelty_scores(descriptors)

    fitness: dict[str, OrganismFitness] = {}

    for index, organism in enumerate(population.organisms):
        evaluation = evaluations[organism.organism_id]
        metrics = evaluation.metrics

        expectancy = metrics.get("expectancy_r")
        drawdown = metrics.get("max_realized_drawdown_pct")

        expectancy_value = (
            float(expectancy)
            if expectancy is not None and np.isfinite(expectancy)
            else -1.0
        )

        drawdown_value = (
            float(drawdown)
            if drawdown is not None and np.isfinite(drawdown)
            else 1.0
        )

        habitat_performance = measure_habitat_performance(
            evaluation.trades,
            habitat_frame,
            timeframe_seconds=timeframe_seconds,
        )

        complexity = (
            len(organism.genome.long_conditions)
            + len(organism.genome.short_conditions)
        )

        fitness[organism.organism_id] = OrganismFitness(
            organism_id=organism.organism_id,
            expectancy_r=expectancy_value,
            max_drawdown_pct=drawdown_value,
            complexity=complexity,
            novelty_score=float(novelty[index]),
            habitat_outcomes=habitat_performance.outcomes,
            temporal_persistence_score=persistence_scores[organism.organism_id],
            temporal_habitat_persistence=evaluation.temporal_habitat_persistence,
            trade_count=len(evaluation.trades),
        )

    return PopulationEvaluation(
        fitness=fitness,
        evaluations=evaluations,
    )

from .ecology import (
    EcologicalSelection,
    Species,
    reproduce_population,
    select_ecological_survivors,
    speciate_population,
)
from .evolution import EvolutionSettings, _thresholds


@dataclass(frozen=True)
class EcologicalRun:
    final_population: Population
    final_species: tuple[Species, ...]
    final_species_profiles: tuple[SpeciesHabitatProfile, ...]
    final_evaluation: PopulationEvaluation
    final_selection: EcologicalSelection
    history: tuple[dict[str, object], ...]

def _adaptive_mutation_rate(
    base_rate: float,
    *,
    stagnation_count: int,
    species_count: int,
    population_size: int,
    settings: EvolutionSettings,
) -> float:
    """Adjust mutation pressure using stagnation and species diversity."""

    if not settings.adaptive_mutation:
        return base_rate

    diversity_ratio = species_count / population_size
    rate = base_rate

    if stagnation_count >= settings.stagnation_generations:
        rate *= 1.5

    if diversity_ratio < 0.25:
        rate *= 1.25
    elif diversity_ratio >= 0.75 and stagnation_count == 0:
        rate *= 0.90

    return float(np.clip(
        rate,
        settings.min_mutation_rate,
        settings.max_mutation_rate,
    ))


def run_ecological_generations(
    initial_population: Population,
    frame: pd.DataFrame,
    habitat_frame: pd.DataFrame,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    timeframe_seconds: int,
    settings: EvolutionSettings,
    minimum_habitat_trades: int = 5,
    minimum_period_trades: int = 5,
    minimum_total_trades: int = 0,
) -> EcologicalRun:
    """Run deterministic ecological selection and reproduction cycles."""

    if initial_population.size != settings.population_size:
        raise ValueError("Initial population size must match settings")
    if minimum_total_trades < 0:
        raise ValueError("minimum_total_trades cannot be negative")

    inactive_initial = [
        organism.organism_id
        for organism in initial_population.organisms
        if not has_both_directional_signals(frame, organism.genome)
    ]
    if inactive_initial:
        raise ValueError(
            "Initial population contains genomes without at least one "
            "exclusive long and short signal; inactive organisms: "
            f"{inactive_initial[:5]}"
        )

    features = available_features(frame)
    feature_ranges = _thresholds(frame, features)

    if not feature_ranges:
        raise ValueError("Training data contains no usable features")

    rng = np.random.default_rng(settings.seed)
    previous_species: tuple[Species, ...] = ()
    population = initial_population
    history: list[dict[str, object]] = []
    best_expectancy_seen = float("-inf")
    stagnation_count = 0

    survivor_capacity = max(
        settings.elite_count,
        int(np.ceil(settings.population_size * 0.50)),
    )

    for _ in range(settings.generations):
        evaluation = evaluate_ecological_population(
            population,
            frame,
            habitat_frame,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            timeframe_seconds=timeframe_seconds,
            minimum_habitat_trades=minimum_habitat_trades,
            minimum_period_trades=minimum_period_trades,
        )

        classified_population, species = speciate_population(
            population,
            previous_species=previous_species,
        )

        species_profiles = profile_species_habitats(
            species,
            evaluation.fitness,
            minimum_trades=minimum_habitat_trades,
        )

        selection = select_ecological_survivors(
            classified_population,
            species,
            evaluation.fitness,
            capacity=survivor_capacity,
            minimum_habitat_trades=minimum_habitat_trades,
            minimum_total_trades=minimum_total_trades,
        )

        survivor_profiles = profile_species_habitats(
            selection.species,
            evaluation.fitness,
            minimum_trades=minimum_habitat_trades,
        )
        expectancy_values = [
            item.expectancy_r
            for item in evaluation.fitness.values()
        ]

        eligible_expectancies = [
            evaluation.fitness[organism_id].expectancy_r
            for organism_id, organism_evaluation in evaluation.evaluations.items()
            if (
                evaluation.fitness[organism_id].trade_count
                >= minimum_total_trades
                and len({
                    period_index
                    for profile in organism_evaluation.temporal_habitat_persistence
                    for period_index, expectancy, trade_count
                    in profile.period_expectancies
                    if (
                        trade_count >= minimum_period_trades
                        and expectancy is not None
                    )
                }) >= 2
            )
        ]

        if eligible_expectancies:
            current_best = max(eligible_expectancies)

            if current_best > best_expectancy_seen + 1e-9:
                best_expectancy_seen = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1

        next_mutation_rate = _adaptive_mutation_rate(
            settings.mutation_rate,
            stagnation_count=stagnation_count,
            species_count=len(species),
            population_size=population.size,
            settings=settings,
        )
        persistence_values = [
            item.temporal_persistence_score
            for item in evaluation.fitness.values()
        ]

        history.append({
            "mutation_rate": next_mutation_rate,
            "minimum_total_trades": minimum_total_trades,
            "stagnation_eligible_candidates": len(eligible_expectancies),
            "species_habitat_profiles": [
                {
                    "species_id": profile.species_id,
                    "preferred_habitat": profile.preferred_habitat,
                    "specialization_score": profile.specialization_score,
                    "sampled_habitats": profile.sampled_habitats,
                    "habitat_expectancy": dict(profile.habitat_expectancy),
                }
                for profile in species_profiles
            ],
            "stagnation_count": stagnation_count,
            "diversity_ratio": (
                len({
                    organism.genome.fingerprint()
                    for organism in population.organisms
                }) / population.size
            ),
            "species_diversity_ratio": len(species) / population.size,
            "generation": population.generation,
            "population_size": population.size,
            "species_count": len(species),
            "survivor_count": selection.population.size,
            "species_champion_ids": selection.species_champion_ids,
            "habitat_champion_ids": selection.habitat_champion_ids,
            "global_competition_ids": selection.global_competition_ids,
            "species_champion_count": len(selection.species_champion_ids),
            "habitat_champion_count": len(selection.habitat_champion_ids),
            "global_competition_count": len(selection.global_competition_ids),
            "extinct_species": selection.extinct_species_ids,
            "pareto_front_count": len(selection.pareto_fronts),
            "pareto_front_sizes": [
                len(front) for front in selection.pareto_fronts
            ],
            "first_front_size": len(selection.pareto_fronts[0]),
            "first_front_ratio": (
                len(selection.pareto_fronts[0]) / population.size
            ),
            "mean_expectancy_r": float(np.mean(expectancy_values)),
            "best_expectancy_r": float(max(expectancy_values)),
            "mean_novelty": float(np.mean([
                item.novelty_score
                for item in evaluation.fitness.values()
            ])),
            "mean_temporal_persistence_score": float(
                np.mean(persistence_values)
            ),
            "best_temporal_persistence_score": float(
                max(persistence_values)
            ),
        })

        previous_species = selection.species

        if len(history) == settings.generations:
            return EcologicalRun(
                final_population=classified_population,
                final_species=species,
                final_evaluation=evaluation,
                final_selection=selection,
                history=tuple(history),
                final_species_profiles=species_profiles,
            )

        population = reproduce_population(
            selection,
            target_size=settings.population_size,
            feature_ranges=feature_ranges,
            rng=rng,
            fitness=evaluation.fitness,
            mutation_rate=next_mutation_rate,
            immigrant_rate=settings.random_injection_rate,
            migration_rate=0.05,
            species_profiles=survivor_profiles,
            signal_frame=frame,
        )

    raise RuntimeError("Ecological evolution ended unexpectedly")
