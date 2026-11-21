import numpy as np
import pytest

from edge.island_evolution import (
    IslandEvolutionConfig,
    run_island_evolution,
)


def evaluator(genome):
    x = float(genome["x"])
    y = float(genome["y"])
    fitness = -(x - 3.0) ** 2 - (y + 2.0) ** 2
    return fitness, [x, y]


def reproducer(first, second, rng):
    parent = first if rng.random() < 0.5 else second
    return {
        "x": float(np.clip(parent["x"] + rng.normal(0, 0.2), -10, 10)),
        "y": float(np.clip(parent["y"] + rng.normal(0, 0.2), -10, 10)),
    }


def initial_populations(config):
    return [
        [
            {"x": float(island * 2 + i), "y": float(-i)}
            for i in range(config.population_per_island)
        ]
        for island in range(config.islands)
    ]


def test_island_evolution_runs_and_checkpoints(tmp_path):
    config = IslandEvolutionConfig(
        islands=3, population_per_island=8, elite_count=2,
        migration_interval=2, migrants_per_island=1,
    )
    path = tmp_path / "islands.json"
    result = run_island_evolution(
        initial_populations(config),
        evaluate=evaluator,
        reproduce=reproducer,
        config=config,
        seed=44,
        generations=4,
        checkpoint_path=path,
        run_signature="fixture-data-config-v1",
    )

    assert result.generation == 4
    assert len(result.islands) == 3
    assert all(len(population) == 8 for population in result.islands)
    assert len(result.history) == 4
    assert path.exists()


def test_resume_matches_uninterrupted_run(tmp_path):
    config = IslandEvolutionConfig(
        islands=3, population_per_island=8, elite_count=2,
        migration_interval=2, migrants_per_island=1,
    )
    initial = initial_populations(config)
    full = run_island_evolution(
        initial, evaluate=evaluator, reproduce=reproducer, config=config,
        seed=91, generations=5, checkpoint_path=tmp_path / "full.json",
        run_signature="same-run",
    )
    partial_path = tmp_path / "partial.json"
    run_island_evolution(
        initial, evaluate=evaluator, reproduce=reproducer, config=config,
        seed=91, generations=2, checkpoint_path=partial_path,
        run_signature="same-run",
    )
    resumed = run_island_evolution(
        None, evaluate=evaluator, reproduce=reproducer, config=config,
        seed=91, generations=5, checkpoint_path=partial_path,
        run_signature="same-run", resume=True,
    )

    assert resumed.islands == full.islands
    assert resumed.history == full.history
    assert resumed.best_genome == full.best_genome
    assert resumed.best_fitness == full.best_fitness


def test_resume_rejects_changed_config(tmp_path):
    config = IslandEvolutionConfig(islands=2, population_per_island=4)
    path = tmp_path / "state.json"
    run_island_evolution(
        initial_populations(config), evaluate=evaluator, reproduce=reproducer,
        config=config, seed=1, generations=1, checkpoint_path=path,
        run_signature="signature",
    )
    changed = IslandEvolutionConfig(islands=2, population_per_island=4, novelty_weight=0.5)
    with pytest.raises(ValueError, match="configuration"):
        run_island_evolution(
            None, evaluate=evaluator, reproduce=reproducer, config=changed,
            seed=1, generations=2, checkpoint_path=path,
            run_signature="signature", resume=True,
        )


def test_run_signature_prevents_cross_dataset_resume(tmp_path):
    config = IslandEvolutionConfig(islands=2, population_per_island=4)
    path = tmp_path / "state.json"
    run_island_evolution(
        initial_populations(config), evaluate=evaluator, reproduce=reproducer,
        config=config, seed=1, generations=1, checkpoint_path=path,
        run_signature="dataset-a",
    )
    with pytest.raises(ValueError, match="run signature"):
        run_island_evolution(
            None, evaluate=evaluator, reproduce=reproducer, config=config,
            seed=1, generations=2, checkpoint_path=path,
            run_signature="dataset-b", resume=True,
        )


@pytest.mark.parametrize("kwargs", [
    {"islands": 1},
    {"population_per_island": 3},
    {"elite_count": 4, "population_per_island": 4},
    {"novelty_weight": 1.1},
])
def test_invalid_island_configuration(kwargs):
    values = {"islands": 2, "population_per_island": 4}
    values.update(kwargs)
    with pytest.raises(ValueError):
        IslandEvolutionConfig(**values)


def test_evaluator_rejects_non_finite_behaviour(tmp_path):
    config = IslandEvolutionConfig(islands=2, population_per_island=4)
    def bad_evaluator(genome):
        return 1.0, [float("nan")]
    with pytest.raises(ValueError, match="behaviour descriptors"):
        run_island_evolution(
            initial_populations(config), evaluate=bad_evaluator,
            reproduce=reproducer, config=config, seed=1, generations=1,
            checkpoint_path=tmp_path / "bad.json", run_signature="bad",
        )
