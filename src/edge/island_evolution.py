from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .training_checkpoint import load_training_checkpoint, save_training_checkpoint


Genome = dict[str, object]
Evaluation = tuple[float, list[float]]
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


@dataclass
class IslandEvolutionResult:
    generation: int
    islands: list[list[Genome]]
    history: list[dict[str, object]]
    best_genome: Genome
    best_fitness: float


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
    }


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones(len(values), dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float) / (len(values) - 1)
    return ranks


def _novelty_scores(
    descriptors: list[list[float]],
    archive: list[list[float]],
    *,
    neighbours: int = 5,
) -> np.ndarray:
    matrix = np.asarray(descriptors, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Behaviour descriptors must be a finite rectangular matrix")
    archived = np.asarray(archive, dtype=float) if archive else np.empty((0, matrix.shape[1]))
    if archived.ndim != 2 or archived.shape[1] != matrix.shape[1]:
        raise ValueError("Behaviour archive descriptor dimensions do not match")
    reference = np.vstack((matrix, archived))
    scores = np.zeros(len(matrix), dtype=float)
    for index, descriptor in enumerate(matrix):
        distances = np.linalg.norm(reference - descriptor, axis=1)
        distances[index] = np.inf
        finite = distances[np.isfinite(distances)]
        if finite.size:
            count = min(neighbours, finite.size)
            scores[index] = float(np.mean(np.partition(finite, count - 1)[:count]))
    return scores


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
    """Run resumable island evolution with novelty ranking and ring migration.

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
        if runtime_state.get("runtime") != "island_evolution_v1":
            raise ValueError("Checkpoint does not contain island-evolution state")
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
    else:
        if initial_populations is None or len(initial_populations) != config.islands:
            raise ValueError("Provide one initial population per island")
        if any(len(population) != config.population_per_island for population in initial_populations):
            raise ValueError("Every initial island must match population_per_island")
        islands = [[_json_copy(genome) for genome in population] for population in initial_populations]
        archives: list[list[list[float]]] = [[] for _ in range(config.islands)]
        seed_sequence = np.random.SeedSequence(seed)
        rngs = [np.random.default_rng(child) for child in seed_sequence.spawn(config.islands)]
        generation = 0
        history: list[dict[str, object]] = []

    while generation < generations:
        evaluated_islands: list[list[dict[str, object]]] = []
        island_descriptors: list[list[list[float]]] = []
        island_novelty: list[np.ndarray] = []
        generation_best = -float("inf")
        generation_best_genome: Genome | None = None

        for island_index, population in enumerate(islands):
            evaluated = [evaluate(_json_copy(genome)) for genome in population]
            fitness = np.asarray([float(row[0]) for row in evaluated], dtype=float)
            descriptors = [list(map(float, row[1])) for row in evaluated]
            if not np.isfinite(fitness).all():
                raise ValueError("Evaluator returned non-finite fitness")
            if not descriptors or any(
                not descriptor or not np.isfinite(descriptor).all()
                or len(descriptor) != len(descriptors[0])
                for descriptor in descriptors
            ):
                raise ValueError("Evaluator returned invalid behaviour descriptors")

            novelty = _novelty_scores(descriptors, archives[island_index])
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
            island_descriptors.append(descriptors)
            island_novelty.append(novelty)
            best_index = int(np.argmax(fitness))
            if fitness[best_index] > generation_best:
                generation_best = float(fitness[best_index])
                generation_best_genome = _json_copy(population[best_index])

            for index in np.argsort(novelty)[::-1]:
                descriptor = descriptors[int(index)]
                if novelty[int(index)] > 0 and descriptor not in archives[island_index]:
                    archives[island_index].append(descriptor)
            archives[island_index] = archives[island_index][-config.archive_capacity:]

        generation += 1
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
        })

        if generation < generations:
            next_islands: list[list[Genome]] = []
            for rows, rng in zip(evaluated_islands, rngs):
                ranked = sorted(
                    rows, key=lambda row: float(row["selection_score"]), reverse=True
                )
                elites = sorted(
                    rows, key=lambda row: float(row["fitness"]), reverse=True
                )[:config.elite_count]
                next_population = [_json_copy(row["genome"]) for row in elites]

                def select_parent() -> Genome:
                    count = min(config.tournament_size, len(ranked))
                    indices = rng.choice(len(ranked), size=count, replace=False)
                    winner = max(
                        (ranked[int(index)] for index in indices),
                        key=lambda row: float(row["selection_score"]),
                    )
                    return _json_copy(winner["genome"])

                while len(next_population) < config.population_per_island:
                    child = reproduce(select_parent(), select_parent(), rng)
                    if not isinstance(child, dict):
                        raise ValueError("reproduce must return a genome dictionary")
                    next_population.append(_json_copy(child))
                next_islands.append(next_population)

            if generation % config.migration_interval == 0:
                migrants = []
                for rows in evaluated_islands:
                    novel_rows = sorted(
                        rows, key=lambda row: float(row["novelty"]), reverse=True
                    )
                    migrants.append([
                        _json_copy(row["genome"])
                        for row in novel_rows[:config.migrants_per_island]
                    ])
                for target_index, target_population in enumerate(next_islands):
                    incoming = migrants[(target_index - 1) % config.islands]
                    replace_count = min(
                        len(incoming),
                        config.population_per_island - config.elite_count,
                    )
                    if replace_count:
                        target_population[-replace_count:] = incoming[:replace_count]
            islands = next_islands

        save_training_checkpoint(
            checkpoint,
            run_signature=run_signature,
            generation=generation,
            population=islands,
            rng_state={
                "runtime": "island_evolution_v1",
                "config": expected_config,
                "archives": archives,
                "island_rng_states": [rng.bit_generator.state for rng in rngs],
            },
            history=history,
        )

    final_rows = [
        (float(evaluate(genome)[0]), genome)
        for population in islands for genome in population
    ]
    best_fitness, best_genome = max(final_rows, key=lambda row: row[0])
    return IslandEvolutionResult(
        generation=generation,
        islands=islands,
        history=history,
        best_genome=_json_copy(best_genome),
        best_fitness=float(best_fitness),
    )
