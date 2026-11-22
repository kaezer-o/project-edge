import numpy as np
import pytest

from edge.island_evolution import (
    IslandEvolutionConfig,
    _rank,
    run_island_evolution,
)


def evaluator(genome):
    x = float(genome["x"])
    y = float(genome["y"])
    fitness = -(x - 3.0) ** 2 - (y + 2.0) ** 2
    return fitness, {"rates": [x, y], "trades": [f"long:{int(x)}", f"short:{int(y)}"], "signal_fingerprint": f"{int(x)}:{int(y)}", "behaviour_fingerprint": f"{int(x)}:{int(y)}", "families": ["test"]}


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
    {"minimum_family_elites": -1},
])
def test_invalid_island_configuration(kwargs):
    values = {"islands": 2, "population_per_island": 4}
    values.update(kwargs)
    with pytest.raises(ValueError):
        IslandEvolutionConfig(**values)


def test_evaluator_rejects_non_finite_behaviour(tmp_path):
    config = IslandEvolutionConfig(islands=2, population_per_island=4)
    def bad_evaluator(genome):
        return 1.0, {"rates": [float("nan")], "trades": [], "signal_fingerprint": "bad", "behaviour_fingerprint": "bad", "families": ["test"]}
    with pytest.raises(ValueError, match="Behaviour descriptor"):
        run_island_evolution(
            initial_populations(config), evaluate=bad_evaluator,
            reproduce=reproducer, config=config, seed=1, generations=1,
            checkpoint_path=tmp_path / "bad.json", run_signature="bad",
        )


def test_rank_assigns_equal_fitness_equal_selection_rank():
    ranks = _rank(np.array([1.0, 1.0, 2.0, 3.0]))

    assert ranks[0] == ranks[1]
    assert ranks[0] == 1.0 / 6.0
    assert ranks[2] == 2.0 / 3.0
    assert ranks[3] == 1.0



def _descriptor(trades, rates=None):
    return {"rates": rates or [0.1, 0.2], "trades": list(trades), "signal_fingerprint": "|".join(sorted(trades)), "behaviour_fingerprint": "|".join(sorted(trades)), "families": ["test"]}


def test_trade_jaccard_distance_identical_trade_sets_is_zero():
    from edge.island_evolution import _descriptor_distance

    first = _descriptor(["t1|long", "t2|short"])
    second = _descriptor(["t1|long", "t2|short"])

    assert _descriptor_distance(first, second) == 0.0


def test_trade_jaccard_distance_partial_overlap():
    from edge.island_evolution import _descriptor_distance

    first = _descriptor(["t1|long", "t2|short"])
    second = _descriptor(["t2|short", "t3|long"])

    # Jaccard distance = 1 - intersection(1) / union(3).
    assert _descriptor_distance(first, second) == pytest.approx(0.65 * (2 / 3))


def test_trade_jaccard_distance_disjoint_sets_is_one_trade_component():
    from edge.island_evolution import _descriptor_distance

    first = _descriptor(["t1|long"])
    second = _descriptor(["t2|short"])

    assert _descriptor_distance(first, second) == pytest.approx(0.65)


def test_migrant_selection_prefers_behaviourally_distinct_rows():
    from edge.island_evolution import _select_diverse_rows

    rows = [
        {"genome": {"id": "a"}, "novelty": 1.0,
         "descriptor": _descriptor(["t1", "t2"])},
        {"genome": {"id": "b"}, "novelty": 0.9,
         "descriptor": _descriptor(["t1", "t2"])},
        {"genome": {"id": "c"}, "novelty": 0.1,
         "descriptor": _descriptor(["t3", "t4"])},
    ]

    selected = _select_diverse_rows(rows, 2)

    assert [row["genome"]["id"] for row in selected] == ["a", "c"]


def test_identical_signal_fingerprint_has_zero_behaviour_distance():
    from edge.island_evolution import _descriptor_distance

    first = {
        "rates": [0.1, 0.2],
        "trades": ["entry-a|long"],
        "signal_fingerprint": "same-signal",
        "behaviour_fingerprint": "same-behaviour",
        "families": ["trend"],
    }
    second = {
        "rates": [0.8, 0.9],
        "trades": ["entry-b|short"],
        "signal_fingerprint": "same-signal",
        "behaviour_fingerprint": "same-behaviour",
        "families": ["breakout"],
    }
    assert _descriptor_distance(first, second) == 0.0


def test_generation_history_reports_genome_and_behaviour_diversity(tmp_path):
    config = IslandEvolutionConfig(islands=2, population_per_island=4)
    result = run_island_evolution(
        initial_populations(config),
        evaluate=evaluator,
        reproduce=reproducer,
        config=config,
        seed=123,
        generations=1,
        checkpoint_path=tmp_path / "telemetry.json",
        run_signature="telemetry-test",
    )
    row = result.history[0]
    assert row["unique_genomes"] >= 6
    assert row["unique_behaviours"] >= 6
    assert len(row["island_unique_behaviours"]) == 2
    assert "test" in row["strategy_family_population_counts"]


def test_final_population_reuses_cached_evaluations(tmp_path):
    import json

    config = IslandEvolutionConfig(islands=2, population_per_island=4)
    evaluated = []

    def counting_evaluator(genome):
        evaluated.append(json.dumps(genome, sort_keys=True))
        return evaluator(genome)

    def clone_reproducer(first, second, rng):
        return dict(first)

    run_island_evolution(
        initial_populations(config),
        evaluate=counting_evaluator,
        reproduce=clone_reproducer,
        config=config,
        seed=321,
        generations=1,
        checkpoint_path=tmp_path / "evaluation-cache.json",
        run_signature="evaluation-cache-test",
    )

    # Initial genomes are evaluated once; their clones in the final population
    # must be served from the generation cache.
    assert len(evaluated) == 8
    assert len(set(evaluated)) == 8


def test_runtime_metrics_measure_generation_and_checkpoint(tmp_path):
    config = IslandEvolutionConfig(islands=2, population_per_island=4)
    result = run_island_evolution(
        initial_populations(config),
        evaluate=evaluator,
        reproduce=reproducer,
        config=config,
        seed=456,
        generations=2,
        checkpoint_path=tmp_path / "runtime-metrics.json",
        run_signature="runtime-metrics-test",
    )

    assert len(result.runtime_metrics) == 2
    for row in result.runtime_metrics:
        assert row["generation_wall_seconds"] > 0
        assert row["evaluation_seconds"] > 0
        assert row["checkpoint_seconds"] > 0
        assert row["other_seconds"] >= 0
        assert row["evaluator_calls"] > 0
        assert row["evaluation_cache_hits"] >= 0
        for phase in (
            "novelty_seconds",
            "archive_seconds",
            "history_seconds",
            "reproduction_seconds",
            "migration_seconds",
        ):
            assert row[phase] >= 0



def test_optimized_novelty_scores_match_descriptor_distance_reference():
    from edge.island_evolution import _descriptor_distance, _novelty_scores

    descriptors = [
        _descriptor(["a|long", "b|short"], [0.1, 0.2]),
        _descriptor(["b|short", "c|long"], [0.2, 0.3]),
        _descriptor(["x|short"], [0.8, 0.9]),
    ]
    archive = [
        _descriptor(["a|long", "x|short"], [0.4, 0.5]),
        _descriptor(["z|long"], [0.9, 0.1]),
    ]
    actual = _novelty_scores(descriptors, archive, neighbours=3)
    reference = []
    combined = descriptors + archive
    for index, descriptor in enumerate(descriptors):
        distances = [
            _descriptor_distance(descriptor, other)
            for other_index, other in enumerate(combined)
            if other_index != index
        ]
        reference.append(float(np.mean(np.partition(distances, 2)[:3])))
    assert actual == pytest.approx(reference)


def test_minimum_family_elites_prevent_family_extinction(tmp_path):
    config = IslandEvolutionConfig(
        islands=2,
        population_per_island=12,
        elite_count=2,
        migration_interval=10,
        migrants_per_island=1,
        minimum_family_elites=2,
    )
    initial = [
        [
            {"family": family, "id": island * 100 + index}
            for family in ("trend", "breakout", "mean_reversion")
            for index in range(4)
        ]
        for island in range(config.islands)
    ]

    def family_evaluator(genome):
        family = str(genome["family"])
        identifier = int(genome["id"])
        fitness = 10.0 if family == "trend" else 1.0
        return fitness, {
            "rates": [identifier / 1000.0],
            "trades": [f"entry-{identifier}"],
            "signal_fingerprint": f"signal-{identifier}",
            "behaviour_fingerprint": f"behaviour-{identifier}",
            "families": [family],
        }

    def trend_only_reproducer(_first, _second, rng):
        return {"family": "trend", "id": int(rng.integers(10_000, 1_000_000))}

    result = run_island_evolution(
        initial,
        evaluate=family_evaluator,
        reproduce=trend_only_reproducer,
        config=config,
        seed=2026,
        generations=2,
        checkpoint_path=tmp_path / "family-preservation.json",
        run_signature="family-preservation-test",
    )

    counts = result.history[1]["strategy_family_population_counts"]
    assert counts["trend"] >= 2
    assert counts["breakout"] >= 4
    assert counts["mean_reversion"] >= 4
