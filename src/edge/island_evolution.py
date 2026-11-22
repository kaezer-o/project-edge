from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .training_checkpoint import load_training_checkpoint, save_training_checkpoint


Genome = dict[str, object]
BehaviourDescriptor = dict[str, object]
Evaluation = tuple[float, BehaviourDescriptor]
Evaluator = Callable[[Genome], Evaluation]
Reproducer = Callable[[Genome, Genome, np.random.Generator], Genome]


@dataclass(frozen=True)
class IslandEvolutionConfig:
    islands: int = 4
    population_per_island: int = 32
    elite_count: int = 2
    tournament_size: int = 3
    migration_interval: int = 10
    migrants_per_island: int = 2
    archive_capacity: int = 256
    novelty_weight: float = 0.25
    minimum_family_elites: int = 2

    def __post_init__(self) -> None:
        if self.islands < 2:
            raise ValueError("islands must be at least 2")
        if self.population_per_island < 4:
            raise ValueError("population_per_island must be at least 4")
        if not 1 <= self.elite_count < self.population_per_island:
            raise ValueError("elite_count must be between 1 and population_per_island - 1")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be at least 2")
        if self.migration_interval < 1 or self.migrants_per_island < 1:
            raise ValueError("migration settings must be positive")
        if self.archive_capacity < 1:
            raise ValueError("archive_capacity must be positive")
        if not 0.0 <= self.novelty_weight <= 1.0:
            raise ValueError("novelty_weight must be between 0 and 1")
        if self.minimum_family_elites < 0:
            raise ValueError("minimum_family_elites must be non-negative")


@dataclass
class IslandEvolutionResult:
    generation: int
    islands: list[list[Genome]]
    history: list[dict[str, object]]
    best_genome: Genome
    best_fitness: float
    runtime_metrics: list[dict[str, float | int]]


def _config_payload(config: IslandEvolutionConfig) -> dict[str, object]:
    return {
        "islands": config.islands,
        "population_per_island": config.population_per_island,
        "elite_count": config.elite_count,
        "tournament_size": config.tournament_size,
        "migration_interval": config.migration_interval,
        "migrants_per_island": config.migrants_per_island,
        "archive_capacity": config.archive_capacity,
        "novelty_weight": config.novelty_weight,
        "minimum_family_elites": config.minimum_family_elites,
    }


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones(len(values), dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank / (len(values) - 1)
        start = end
    return ranks


def _validate_descriptor(descriptor: BehaviourDescriptor) -> BehaviourDescriptor:
    if not isinstance(descriptor, dict) or set(descriptor) != {"rates", "trades", "signal_fingerprint", "behaviour_fingerprint", "families"}:
        raise ValueError("Behaviour descriptor has an invalid schema")
    rates = descriptor["rates"]
    trades = descriptor["trades"]
    signal_fingerprint = descriptor["signal_fingerprint"]
    behaviour_fingerprint = descriptor["behaviour_fingerprint"]
    families = descriptor["families"]
    if (
        not isinstance(rates, list)
        or not rates
        or not np.isfinite(np.asarray(rates, dtype=float)).all()
        or not isinstance(trades, list)
        or any(not isinstance(item, str) or not item for item in trades)
        or len(set(trades)) != len(trades)
        or not isinstance(signal_fingerprint, str)
        or not signal_fingerprint
        or not isinstance(behaviour_fingerprint, str)
        or not behaviour_fingerprint
        or not isinstance(families, list)
        or any(not isinstance(item, str) or not item for item in families)
    ):
        raise ValueError("Behaviour descriptor contains invalid rates or trade events")
    return {"rates": list(map(float, rates)), "trades": sorted(trades), "signal_fingerprint": signal_fingerprint, "behaviour_fingerprint": behaviour_fingerprint, "families": sorted(set(families))}


def _descriptor_distance(first: BehaviourDescriptor, second: BehaviourDescriptor) -> float:
    """Combine coarse signal-rate distance with exact directional trade Jaccard distance."""
    first = _validate_descriptor(first)
    second = _validate_descriptor(second)
    if first["behaviour_fingerprint"] == second["behaviour_fingerprint"]:
        return 0.0
    first_rates = np.asarray(first["rates"], dtype=float)
    second_rates = np.asarray(second["rates"], dtype=float)
    if first_rates.shape != second_rates.shape:
        raise ValueError("Behaviour descriptor rate dimensions do not match")
    rate_distance = float(np.mean(np.abs(first_rates - second_rates)))
    first_trades = set(first["trades"])
    second_trades = set(second["trades"])
    union = first_trades | second_trades
    if not union:
        trade_distance = 0.0
    else:
        trade_distance = 1.0 - len(first_trades & second_trades) / len(union)
    return 0.35 * rate_distance + 0.65 * trade_distance


def _novelty_scores(
    descriptors: list[BehaviourDescriptor],
    archive: list[BehaviourDescriptor],
    *,
    neighbours: int = 5,
) -> np.ndarray:
    if not descriptors:
        raise ValueError("At least one behaviour descriptor is required")
    current = [_validate_descriptor(item) for item in descriptors]
    archived = [_validate_descriptor(item) for item in archive]
    rate_size = len(current[0]["rates"])
    if any(len(item["rates"]) != rate_size for item in current + archived):
        raise ValueError("Behaviour archive descriptor dimensions do not match")
    reference = current + archived
    rate_vectors = [np.asarray(item["rates"], dtype=float) for item in reference]
    trade_sets = [set(item["trades"]) for item in reference]
    behaviour_ids = [str(item["behaviour_fingerprint"]) for item in reference]
    scores = np.zeros(len(current), dtype=float)
    for index in range(len(current)):
        distances = []
        for other_index in range(len(reference)):
            if other_index == index:
                continue
            if behaviour_ids[index] == behaviour_ids[other_index]:
                distances.append(0.0)
                continue
            rate_distance = float(np.mean(np.abs(
                rate_vectors[index] - rate_vectors[other_index]
            )))
            first_trades = trade_sets[index]
            second_trades = trade_sets[other_index]
            union_size = len(first_trades | second_trades)
            trade_distance = (
                0.0 if union_size == 0
                else 1.0 - len(first_trades & second_trades) / union_size
            )
            distances.append(0.35 * rate_distance + 0.65 * trade_distance)
        if distances:
            count = min(neighbours, len(distances))
            scores[index] = float(np.mean(np.partition(distances, count - 1)[:count]))
    return scores


def _select_diverse_rows(
    rows: list[dict[str, object]],
    count: int,
) -> list[dict[str, object]]:
    """Greedily select migrants with low pairwise behavioural overlap."""
    if count <= 0 or not rows:
        return []
    remaining = sorted(
        rows,
        key=lambda row: float(row["novelty"]),
        reverse=True,
    )
    selected = [remaining.pop(0)]
    while remaining and len(selected) < count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: min(
                _descriptor_distance(
                    remaining[index]["descriptor"], chosen["descriptor"]
                )
                for chosen in selected
            ),
        )
        selected.append(remaining.pop(best_index))
    return selected


def _fingerprint(genome: Genome) -> str:
    encoded = json.dumps(genome, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_island_evolution(
    initial_populations: list[list[Genome]] | None,
    *,
    evaluate: Evaluator,
    reproduce: Reproducer,
    config: IslandEvolutionConfig,
    seed: int,
    generations: int,
    checkpoint_path: str | Path,
    run_signature: str,
    resume: bool = False,
) -> IslandEvolutionResult:
    """Run resumable island evolution with trade-overlap novelty and ring migration.

    generations is the total target generation, including work already saved.
    The evaluator must use training data only; validation and final-test data
    are deliberately outside this runtime's interface.
    """
    if generations < 1:
        raise ValueError("generations must be at least one")
    checkpoint = Path(checkpoint_path)
    expected_config = _config_payload(config)

    if resume:
        state = load_training_checkpoint(checkpoint, expected_run_signature=run_signature)
        runtime_state = state["rng_state"]
        if runtime_state.get("runtime") != "island_evolution_v3":
            raise ValueError("Checkpoint does not contain compatible island-evolution state")
        if runtime_state.get("config") != expected_config:
            raise ValueError("Checkpoint evolution configuration does not match")
        islands = state["population"]
        archives = runtime_state["archives"]
        rngs = []
        for rng_state in runtime_state["island_rng_states"]:
            rng = np.random.default_rng()
            rng.bit_generator.state = rng_state
            rngs.append(rng)
        generation = int(state["generation"])
        history = list(state["history"])
        if len(islands) != config.islands or len(archives) != config.islands:
            raise ValueError("Checkpoint island count does not match configuration")
        for archive in archives:
            for descriptor in archive:
                _validate_descriptor(descriptor)
    else:
        if initial_populations is None or len(initial_populations) != config.islands:
            raise ValueError("Provide one initial population per island")
        if any(len(population) != config.population_per_island for population in initial_populations):
            raise ValueError("Every initial island must match population_per_island")
        islands = [[_json_copy(genome) for genome in population] for population in initial_populations]
        archives: list[list[BehaviourDescriptor]] = [[] for _ in range(config.islands)]
        seed_sequence = np.random.SeedSequence(seed)
        rngs = [np.random.default_rng(child) for child in seed_sequence.spawn(config.islands)]
        generation = 0
        history: list[dict[str, object]] = []

    evaluation_cache: dict[str, Evaluation] = {}
    runtime_metrics: list[dict[str, float | int]] = []
    phase_stats: dict[str, float | int] = {}

    def cached_evaluation(genome: Genome) -> Evaluation:
        key = _fingerprint(genome)
        if key not in evaluation_cache:
            evaluation_started = time.perf_counter()
            fitness_value, descriptor_value = evaluate(_json_copy(genome))
            phase_stats["evaluation_seconds"] = float(phase_stats.get("evaluation_seconds", 0.0)) + (time.perf_counter() - evaluation_started)
            phase_stats["evaluator_calls"] = int(phase_stats.get("evaluator_calls", 0)) + 1
            evaluation_cache[key] = (
                float(fitness_value),
                _validate_descriptor(descriptor_value),
            )
        else:
            phase_stats["cache_hits"] = int(phase_stats.get("cache_hits", 0)) + 1
        return evaluation_cache[key]

    while generation < generations:
        generation_started = time.perf_counter()
        phase_stats = {
            "evaluation_seconds": 0.0,
            "evaluator_calls": 0,
            "cache_hits": 0,
            "novelty_seconds": 0.0,
            "archive_seconds": 0.0,
            "history_seconds": 0.0,
            "reproduction_seconds": 0.0,
            "migration_seconds": 0.0,
        }
        evaluated_islands: list[list[dict[str, object]]] = []
        island_novelty: list[np.ndarray] = []
        generation_best = -float("inf")
        generation_best_genome: Genome | None = None

        for island_index, population in enumerate(islands):
            evaluated = [cached_evaluation(genome) for genome in population]
            fitness = np.asarray([float(row[0]) for row in evaluated], dtype=float)
            descriptors = [_validate_descriptor(row[1]) for row in evaluated]
            if not np.isfinite(fitness).all():
                raise ValueError("Evaluator returned non-finite fitness")
            if any(
                len(descriptor["rates"]) != len(descriptors[0]["rates"])
                for descriptor in descriptors
            ):
                raise ValueError("Evaluator returned inconsistent behaviour descriptors")

            novelty_started = time.perf_counter()
            novelty = _novelty_scores(descriptors, archives[island_index])
            phase_stats["novelty_seconds"] = float(phase_stats["novelty_seconds"]) + (
                time.perf_counter() - novelty_started
            )
            combined = (
                (1.0 - config.novelty_weight) * _rank(fitness)
                + config.novelty_weight * _rank(novelty)
            )
            rows = [
                {
                    "genome": _json_copy(genome),
                    "fitness": float(fitness[index]),
                    "descriptor": descriptors[index],
                    "novelty": float(novelty[index]),
                    "selection_score": float(combined[index]),
                }
                for index, genome in enumerate(population)
            ]
            evaluated_islands.append(rows)
            island_novelty.append(novelty)
            best_index = int(np.argmax(fitness))
            if fitness[best_index] > generation_best:
                generation_best = float(fitness[best_index])
                generation_best_genome = _json_copy(population[best_index])

            archive_started = time.perf_counter()
            archived_behaviours = {
                str(item["behaviour_fingerprint"]) for item in archives[island_index]
            }
            for index in np.argsort(novelty)[::-1]:
                descriptor = descriptors[int(index)]
                behaviour_id = str(descriptor["behaviour_fingerprint"])
                if novelty[int(index)] > 0 and behaviour_id not in archived_behaviours:
                    archives[island_index].append(descriptor)
                    archived_behaviours.add(behaviour_id)
            archives[island_index] = archives[island_index][-config.archive_capacity:]
            phase_stats["archive_seconds"] = float(phase_stats["archive_seconds"]) + (
                time.perf_counter() - archive_started
            )

        history_started = time.perf_counter()
        generation += 1
        genome_fingerprints = [
            _fingerprint(row["genome"])
            for rows in evaluated_islands for row in rows
        ]
        behaviour_fingerprints = [
            str(row["descriptor"]["behaviour_fingerprint"])
            for rows in evaluated_islands for row in rows
        ]
        signal_fingerprints = [
            str(row["descriptor"]["signal_fingerprint"])
            for rows in evaluated_islands for row in rows
        ]
        family_counts: dict[str, int] = {}
        for rows in evaluated_islands:
            for row in rows:
                for family in row["descriptor"]["families"]:
                    family_counts[family] = family_counts.get(family, 0) + 1
        history.append({
            "generation": generation,
            "best_fitness": generation_best,
            "best_genome_fingerprint": _fingerprint(generation_best_genome or {}),
            "mean_novelty": float(np.mean([
                score for scores in island_novelty for score in scores
            ])),
            "island_best_fitness": [
                max(float(row["fitness"]) for row in rows)
                for rows in evaluated_islands
            ],
            "unique_genomes": len(set(genome_fingerprints)),
            "unique_behaviours": len(set(behaviour_fingerprints)),
            "unique_signal_streams": len(set(signal_fingerprints)),
            "island_unique_signal_streams": [
                len({str(row["descriptor"]["signal_fingerprint"]) for row in rows})
                for rows in evaluated_islands
            ],
            "island_unique_genomes": [
                len({_fingerprint(row["genome"]) for row in rows})
                for rows in evaluated_islands
            ],
            "island_unique_behaviours": [
                len({str(row["descriptor"]["behaviour_fingerprint"]) for row in rows})
                for rows in evaluated_islands
            ],
            "strategy_family_population_counts": family_counts,
        })
        phase_stats["history_seconds"] = float(phase_stats["history_seconds"]) + (
            time.perf_counter() - history_started
        )

        # Prepare the next population even at the stopping generation so resume
        # follows the same deterministic RNG trajectory as an uninterrupted run.
        reproduction_started = time.perf_counter()
        next_islands: list[list[Genome]] = []
        for rows, rng in zip(evaluated_islands, rngs):
            ranked = sorted(
                rows, key=lambda row: float(row["selection_score"]), reverse=True
            )
            elites = sorted(
                rows, key=lambda row: float(row["fitness"]), reverse=True
            )
            next_population: list[Genome] = []
            next_behaviours: set[str] = set()
            family_elite_counts: dict[str, int] = {}

            def retain_elite(row: dict[str, object]) -> None:
                behaviour = str(row["descriptor"]["behaviour_fingerprint"])
                if behaviour in next_behaviours:
                    return
                next_population.append(_json_copy(row["genome"]))
                next_behaviours.add(behaviour)
                for family in row["descriptor"]["families"]:
                    if family not in {"hybrid_or_other", "other", "random_fallback"}:
                        family_elite_counts[family] = family_elite_counts.get(family, 0) + 1

            # Preserve the strongest overall behaviours first.
            for row in elites:
                retain_elite(row)
                if len(next_population) >= config.elite_count:
                    break

            # Then reserve a small per-island floor for each recognizable family.
            # This prevents selection pressure from deleting an entire family in
            # one generation while leaving the remaining slots free to evolve.
            if config.minimum_family_elites:
                families = sorted({
                    family
                    for row in rows
                    for family in row["descriptor"]["families"]
                    if family not in {"hybrid_or_other", "other", "random_fallback"}
                })
                for family in families:
                    for row in elites:
                        if family_elite_counts.get(family, 0) >= config.minimum_family_elites:
                            break
                        if family in row["descriptor"]["families"]:
                            retain_elite(row)
                        if len(next_population) >= config.population_per_island:
                            break
                    if len(next_population) >= config.population_per_island:
                        break

            def select_parent() -> Genome:
                count = min(config.tournament_size, len(ranked))
                indices = rng.choice(len(ranked), size=count, replace=False)
                winner = max(
                    (ranked[int(index)] for index in indices),
                    key=lambda row: float(row["selection_score"]),
                )
                return _json_copy(winner["genome"])

            while len(next_population) < config.population_per_island:
                child: Genome | None = None
                child_behaviour = ""
                for _attempt in range(12):
                    proposed = reproduce(select_parent(), select_parent(), rng)
                    if not isinstance(proposed, dict):
                        raise ValueError("reproduce must return a genome dictionary")
                    proposed = _json_copy(proposed)
                    _, descriptor = cached_evaluation(proposed)
                    proposed_behaviour = str(descriptor["behaviour_fingerprint"])
                    child = proposed
                    child_behaviour = proposed_behaviour
                    if proposed_behaviour not in next_behaviours:
                        break
                if child is None:
                    raise RuntimeError("Reproduction failed to produce a child")
                next_population.append(child)
                next_behaviours.add(child_behaviour)
            next_islands.append(next_population)
        phase_stats["reproduction_seconds"] = float(phase_stats["reproduction_seconds"]) + (
            time.perf_counter() - reproduction_started
        )

        if generation % config.migration_interval == 0:
            migration_started = time.perf_counter()
            migrants = [
                [
                    _json_copy(row["genome"])
                    for row in _select_diverse_rows(rows, config.migrants_per_island)
                ]
                for rows in evaluated_islands
            ]
            for target_index, target_population in enumerate(next_islands):
                incoming = migrants[(target_index - 1) % config.islands]
                existing_behaviours = {
                    str(cached_evaluation(genome)[1]["behaviour_fingerprint"])
                    for genome in target_population
                }
                replaceable = list(range(len(target_population) - 1, config.elite_count - 1, -1))
                for migrant in incoming:
                    if not replaceable:
                        break
                    migrant_descriptor = cached_evaluation(migrant)[1]
                    migrant_behaviour = str(migrant_descriptor["behaviour_fingerprint"])
                    if migrant_behaviour in existing_behaviours:
                        continue
                    replace_index = replaceable.pop(0)
                    removed_behaviour = str(
                        cached_evaluation(target_population[replace_index])[1]["behaviour_fingerprint"]
                    )
                    existing_behaviours.discard(removed_behaviour)
                    target_population[replace_index] = migrant
                    existing_behaviours.add(migrant_behaviour)
            phase_stats["migration_seconds"] = float(phase_stats["migration_seconds"]) + (
                time.perf_counter() - migration_started
            )
        islands = next_islands

        checkpoint_started = time.perf_counter()
        save_training_checkpoint(
            checkpoint,
            run_signature=run_signature,
            generation=generation,
            population=islands,
            rng_state={
                "runtime": "island_evolution_v3",
                "config": expected_config,
                "archives": archives,
                "island_rng_states": [rng.bit_generator.state for rng in rngs],
            },
            history=history,
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        generation_seconds = time.perf_counter() - generation_started
        evaluation_seconds = float(phase_stats["evaluation_seconds"])
        runtime_metrics.append({
            "generation": generation,
            "generation_wall_seconds": generation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "other_seconds": max(0.0, generation_seconds - evaluation_seconds - checkpoint_seconds),
            "evaluator_calls": int(phase_stats["evaluator_calls"]),
            "evaluation_cache_hits": int(phase_stats["cache_hits"]),
            "novelty_seconds": float(phase_stats["novelty_seconds"]),
            "archive_seconds": float(phase_stats["archive_seconds"]),
            "history_seconds": float(phase_stats["history_seconds"]),
            "reproduction_seconds": float(phase_stats["reproduction_seconds"]),
            "migration_seconds": float(phase_stats["migration_seconds"]),
        })
        # Keep only the next generation's cache entries. Candidate descriptors
        # contain trade events, so retaining every attempted child would grow
        # memory with the total number of generations.
        evaluation_cache = {
            _fingerprint(genome): evaluation_cache[_fingerprint(genome)]
            for population in islands
            for genome in population
            if _fingerprint(genome) in evaluation_cache
        }

    # Every genome in the final population has already been evaluated while
    # preparing that population. Reuse the generation cache instead of
    # repeating the full fitness and behaviour workload here.
    final_rows = [
        (float(cached_evaluation(genome)[0]), genome)
        for population in islands for genome in population
    ]
    best_fitness, best_genome = max(final_rows, key=lambda row: row[0])
    return IslandEvolutionResult(
        generation=generation,
        islands=islands,
        history=history,
        best_genome=_json_copy(best_genome),
        best_fitness=float(best_fitness),
        runtime_metrics=runtime_metrics,
    )
