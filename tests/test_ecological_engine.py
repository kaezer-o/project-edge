import numpy as np
import pandas as pd
import pytest

import edge.ecological_engine as engine
from edge.ecology import Organism, Population
from edge.evolution import Condition, EvolutionSettings, StrategyGenome 


def make_genome():
    return StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.2),),
        short_conditions=(Condition("rsi_centered", "<", -0.2),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )


def make_frames():
    index = pd.date_range(
        "2024-01-01", periods=40, freq="4h", tz="UTC"
    )

    frame = pd.DataFrame(
    {
        "close": np.linspace(1.0, 1.1, 40),
        "rsi_centered": np.linspace(-1.0, 1.0, 40),
    },
    index=index,
    )

    habitat = pd.DataFrame(
        {
            "habitat_id": ["trend_up"] * 40,
            "habitat_valid": [True] * 40,
        },
        index=index,
    )

    return frame, habitat


def make_population():
    organisms = tuple(
        Organism.born(
            make_genome(),
            generation=0,
            birth_index=i,
        )
        for i in range(3)
    )

    return Population(
        population_id="test-population",
        generation=0,
        organisms=organisms,
    )


def test_initial_ecological_population_is_unique_and_directionally_active(
    monkeypatch,
):
    frame, _ = make_frames()
    valid_first = StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.2),),
        short_conditions=(Condition("rsi_centered", "<", -0.2),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )
    valid_second = StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.4),),
        short_conditions=(Condition("rsi_centered", "<", -0.4),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )
    inactive = StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.2),),
        short_conditions=(Condition("rsi_centered", ">", 0.2),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )
    generated = iter((inactive, valid_first, valid_first, valid_second))
    monkeypatch.setattr(
        engine,
        "random_genome",
        lambda rng, ranges: next(generated),
    )

    population = engine.initialize_ecological_population(
        frame,
        population_id="initialization-test",
        population_size=2,
        feature_ranges={"rsi_centered": (-0.4, 0.0, 0.4)},
        seed=7,
        max_attempts_per_organism=3,
    )

    from edge.evolution import has_both_directional_signals

    fingerprints = [
        organism.genome.fingerprint()
        for organism in population.organisms
    ]
    assert len(fingerprints) == len(set(fingerprints)) == 2
    assert all(
        has_both_directional_signals(frame, organism.genome)
        for organism in population.organisms
    )


def test_generation_loop_rejects_inactive_initial_genomes():
    frame, habitat = make_frames()
    inactive = StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.2),),
        short_conditions=(Condition("rsi_centered", ">", 0.2),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )
    population = Population(
        population_id="inactive-initial-population",
        generation=0,
        organisms=tuple(
            Organism.born(inactive, generation=0, birth_index=i)
            for i in range(8)
        ),
    )

    with pytest.raises(ValueError, match="exclusive long and short signal"):
        engine.run_ecological_generations(
            population,
            frame,
            habitat,
            execution=None,
            risk=None,
            pip_size=0.0001,
            price_basis="bid",
            timeframe_seconds=14400,
            settings=EvolutionSettings(
                population_size=8,
                generations=2,
                elite_count=2,
                tournament_size=2,
            ),
        )


def test_under_sampled_strategies_do_not_reset_stagnation(monkeypatch):
    frame, habitat = make_frames()
    population = Population(
        population_id="under-sampled-stagnation-test",
        generation=0,
        organisms=tuple(
            Organism.born(make_genome(), generation=0, birth_index=i)
            for i in range(8)
        ),
    )

    def fake_run_genome(frame, genome, **kwargs):
        return (
            {
                "expectancy_r": 0.50,
                "max_realized_drawdown_pct": 0.05,
            },
            pd.DataFrame({
                "entry_time": [frame.index[5], frame.index[15]],
                "net_r": [0.8, 0.2],
                "direction": ["long", "short"],
                "bars_held": [2, 3],
            }),
            pd.DataFrame(),
        )

    monkeypatch.setattr(engine, "_run_genome", fake_run_genome)
    result = engine.run_ecological_generations(
        population,
        frame,
        habitat,
        execution=None,
        risk=None,
        pip_size=0.0001,
        price_basis="bid",
        timeframe_seconds=14400,
        settings=EvolutionSettings(
            population_size=8,
            generations=2,
            elite_count=2,
            tournament_size=2,
        ),
        minimum_habitat_trades=1,
        minimum_period_trades=1,
        minimum_total_trades=3,
    )

    assert tuple(
        item["stagnation_count"] for item in result.history
    ) == (1, 2)

def test_population_evaluation_handles_zero_trade_organisms(monkeypatch):
    frame, habitat = make_frames()
    population = make_population()

    def fake_run_genome(frame, genome, **kwargs):
        empty_trades = pd.DataFrame(
            columns=["entry_time", "net_r"]
        )
        return (
            {
                "expectancy_r": None,
                "max_realized_drawdown_pct": None,
            },
            empty_trades,
            pd.DataFrame(),
        )

    monkeypatch.setattr(engine, "_run_genome", fake_run_genome)

    result = engine.evaluate_ecological_population(
        population,
        frame,
        habitat,
        execution=None,
        risk=None,
        pip_size=0.0001,
        price_basis="bid",
        timeframe_seconds=14400,
    )

    assert len(result.fitness) == 3
    assert len(result.evaluations) == 3

    for organism in population.organisms:
        fitness = result.fitness[organism.organism_id]

        assert fitness.expectancy_r == -1.0
        assert fitness.max_drawdown_pct == 1.0
        assert fitness.novelty_score == 0.0
        assert fitness.habitat_outcomes[0].trade_count == 0
        assert fitness.habitat_outcomes[0].expectancy_r is None


def test_population_evaluation_preserves_organism_identity(monkeypatch):
    frame, habitat = make_frames()
    population = make_population()

    def fake_run_genome(frame, genome, **kwargs):
        return (
            {
                "expectancy_r": 0.25,
                "max_realized_drawdown_pct": 0.05,
            },
            pd.DataFrame(columns=["entry_time", "net_r"]),
            pd.DataFrame(),
        )

    monkeypatch.setattr(engine, "_run_genome", fake_run_genome)

    result = engine.evaluate_ecological_population(
        population,
        frame,
        habitat,
        execution=None,
        risk=None,
        pip_size=0.0001,
        price_basis="bid",
        timeframe_seconds=14400,
    )

    assert set(result.fitness) == {
        organism.organism_id for organism in population.organisms
    }

    assert set(result.evaluations) == set(result.fitness)

@pytest.mark.parametrize(
    ("trade_indices", "expected_stagnation", "split_habitats"),
    [
        ((5, 6), (1, 2), True,),    # One qualified period: cannot reset stagnation
        ((5, 15), (0, 1), True,),   # Two qualified periods: can reset stagnation
    ],
)

def test_ecological_generation_loop_runs_multiple_generations(
    monkeypatch,
    trade_indices,
    expected_stagnation,
    split_habitats,
):
    frame, habitat = make_frames()

    if split_habitats:
        habitat.loc[frame.index[4], "habitat_id"] = "trend_up"
        habitat.loc[frame.index[5], "habitat_id"] = "trend_down"

    organisms = tuple(
        Organism.born(
            make_genome(),
            generation=0,
            birth_index=i,
        )
        for i in range(8)
    )

    population = Population(
        population_id="generation-test",
        generation=0,
        organisms=organisms,
    )

    def fake_run_genome(frame, genome, **kwargs):
        return (
            {
                "expectancy_r": 0.25,
                "max_realized_drawdown_pct": 0.05,
            },
            pd.DataFrame({
                 "entry_time": [frame.index[i] for i in trade_indices],
                 "net_r": [0.3, 0.2],
                 "direction": ["long", "short"],
                 "bars_held": [2, 3],
            }),
            pd.DataFrame(),
        )

    monkeypatch.setattr(engine, "_run_genome", fake_run_genome)
    
    real_reproduce = engine.reproduce_population
    reproduction_profile_species_ids = []

    def tracked_reproduce(selection, **kwargs):
        profile_ids = {
            profile.species_id
            for profile in kwargs["species_profiles"]
        }
        survivor_species_ids = {
            item.species_id for item in selection.species
        }

        reproduction_profile_species_ids.append(profile_ids)

        assert profile_ids == survivor_species_ids

        return real_reproduce(selection, **kwargs)

    monkeypatch.setattr(
        engine,
        "reproduce_population",
        tracked_reproduce,
    )

    settings = EvolutionSettings(
        population_size=8,
        generations=2,
        elite_count=2,
        tournament_size=2,
    )

    result = engine.run_ecological_generations(
        population,
        frame,
        habitat,
        execution=None,
        risk=None,
        pip_size=0.0001,
        price_basis="bid",
        timeframe_seconds=14400,
        settings=settings,
        minimum_habitat_trades=1,
        minimum_period_trades=1,
    )

    assert len(result.history) == 2
    assert result.history[0]["generation"] == 0
    assert result.history[1]["generation"] == 1

    assert result.final_population.size == 8
    assert result.final_population.generation == 1
    assert len(result.final_species) >= 1
    assert result.final_selection.population.size == 4
    assert len(result.final_evaluation.fitness) == 8
    assert len(result.final_species_profiles) == len(result.final_species)

    assert any(
        profile.preferred_habitat == "trend_up"
        for profile in result.final_species_profiles
    )

    assert any(
        profile["preferred_habitat"] == "trend_up"
        for profile in result.history[-1]["species_habitat_profiles"]
    )
    assert len(reproduction_profile_species_ids) == 1
    assert result.history[0]["mean_temporal_persistence_score"] == pytest.approx(1.0)
    assert result.history[0]["best_temporal_persistence_score"] == pytest.approx(1.0)
    assert tuple(
        item["stagnation_count"] for item in result.history
    ) == expected_stagnation


def test_adaptive_mutation_increases_after_stagnation():
    settings = EvolutionSettings()

    rate = engine._adaptive_mutation_rate(
        settings.mutation_rate,
        stagnation_count=settings.stagnation_generations,
        species_count=8,
        population_size=32,
        settings=settings,
    )

    assert rate > settings.mutation_rate


def test_adaptive_mutation_increases_with_low_diversity():
    settings = EvolutionSettings()

    rate = engine._adaptive_mutation_rate(
        settings.mutation_rate,
        stagnation_count=0,
        species_count=2,
        population_size=32,
        settings=settings,
    )

    assert rate > settings.mutation_rate


def test_adaptive_mutation_respects_configured_bounds():
    settings = EvolutionSettings()

    rate = engine._adaptive_mutation_rate(
        settings.mutation_rate,
        stagnation_count=100,
        species_count=1,
        population_size=32,
        settings=settings,
    )

    assert settings.min_mutation_rate <= rate <= settings.max_mutation_rate

def test_population_evaluation_tracks_temporal_habitat_performance(monkeypatch):
    frame, habitat = make_frames()
    population = make_population()

    trades = pd.DataFrame({
        "entry_time": [
            frame.index[1],
            frame.index[2],
            frame.index[21],
            frame.index[22],
        ],
        "net_r": [0.4, 0.2, -0.3, -0.1],
        "direction": ["long", "long", "short", "short"],
        "bars_held": [2, 3, 2, 3],
    })

    def fake_run_genome(frame, genome, **kwargs):
        return (
            {
                "expectancy_r": 0.05,
                "max_realized_drawdown_pct": 0.03,
            },
            trades.copy(),
            pd.DataFrame(),
        )

    monkeypatch.setattr(engine, "_run_genome", fake_run_genome)

    result = engine.evaluate_ecological_population(
        population,
        frame,
        habitat,
        execution=None,
        risk=None,
        pip_size=0.0001,
        price_basis="bid",
        timeframe_seconds=14400,
        temporal_windows=2,
        minimum_period_trades=2,
    )

    for organism in population.organisms:
        evaluation = result.evaluations[organism.organism_id]
        temporal = evaluation.temporal_habitat_performance

        assert len(temporal.periods) == 2
        assert temporal.total_trades == 4
        assert temporal.unassigned_trades == 0

        first, second = temporal.periods

        assert first.outcomes[0].expectancy_r == pytest.approx(0.3)
        assert second.outcomes[0].expectancy_r == pytest.approx(-0.2)

        assert first.classified_trades == 2
        assert second.classified_trades == 2
        persistence = evaluation.temporal_habitat_persistence

        assert len(persistence) == 1

        profile = persistence[0]

        assert profile.habitat_id == "trend_up"
        assert profile.qualified_periods == 2
        assert profile.profitable_periods == 1
        assert profile.persistence_ratio == pytest.approx(0.5)
        assert profile.weighted_expectancy_r == pytest.approx(0.05)
        assert profile.worst_period_expectancy_r == pytest.approx(-0.2)
        assert result.fitness[organism.organism_id].trade_count == 4
        assert (
            result.fitness[organism.organism_id].temporal_persistence_score
            == pytest.approx(0.5)
        )

def test_population_evaluation_keeps_individual_persistence_scores(monkeypatch):
    frame, habitat = make_frames()
    population = make_population()

    patterns = [
        [0.2, 0.2, 0.2, 0.2],
        [0.2, -0.2, 0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2],
    ]

    calls = iter(patterns)

    def fake_run_genome(frame, genome, **kwargs):
        values = next(calls)

        trades = pd.DataFrame({
            "entry_time": [
                frame.index[1],
                frame.index[11],
                frame.index[21],
                frame.index[31],
            ],
            "net_r": values,
            "direction": ["long", "short", "long", "short"],
            "bars_held": [2, 2, 2, 2],
        })

        return (
            {
                "expectancy_r": float(np.mean(values)),
                "max_realized_drawdown_pct": 0.02,
            },
            trades,
            pd.DataFrame(),
        )

    monkeypatch.setattr(engine, "_run_genome", fake_run_genome)

    result = engine.evaluate_ecological_population(
        population,
        frame,
        habitat,
        execution=None,
        risk=None,
        pip_size=0.0001,
        price_basis="bid",
        timeframe_seconds=14400,
        temporal_windows=4,
        minimum_period_trades=1,
    )

    scores = [
        result.fitness[item.organism_id].temporal_persistence_score
        for item in population.organisms
    ]

    assert scores == pytest.approx([1.0, 0.5, 0.0])

def test_ecological_generations_are_reproducible_and_preserve_invariants(monkeypatch):
    frame, habitat = make_frames()

    initial_organisms = tuple(
        Organism.born(
            make_genome(),
            generation=0,
            birth_index=i,
        )
        for i in range(8)
    )
    population = Population(
        population_id="reproducibility-test",
        generation=0,
        organisms=initial_organisms,
    )

    def fake_run_genome(frame, genome, **kwargs):
        threshold = genome.long_conditions[0].threshold
        expectancy = 0.2 + threshold * 0.01

        return (
            {
                "expectancy_r": expectancy,
                "max_realized_drawdown_pct": 0.05,
            },
            pd.DataFrame({
                "entry_time": [frame.index[5], frame.index[10]],
                "net_r": [0.3, 0.2],
                "direction": ["long", "short"],
                "bars_held": [2, 3],
            }),
            pd.DataFrame(),
        )

    monkeypatch.setattr(engine, "_run_genome", fake_run_genome)

    settings = EvolutionSettings(
        population_size=8,
        generations=3,
        elite_count=2,
        tournament_size=2,
        seed=4217,
    )

    def run():
        return engine.run_ecological_generations(
            population,
            frame,
            habitat,
            execution=None,
            risk=None,
            pip_size=0.0001,
            price_basis="bid",
            timeframe_seconds=14400,
            settings=settings,
            minimum_habitat_trades=1,
            minimum_period_trades=1,
        )

    first = run()
    second = run()

    def population_signature(result):
        return tuple(
            (
                organism.organism_id,
                organism.genome.fingerprint(),
                organism.generation,
                organism.parent_ids,
                organism.species_id,
            )
            for organism in result.final_population.organisms
        )

    assert population_signature(first) == population_signature(second)
    assert first.history == second.history
    assert first.final_species == second.final_species

    assert len(first.history) == settings.generations
    assert first.final_population.size == settings.population_size
    assert first.final_population.generation == settings.generations - 1

    organisms = first.final_population.organisms
    organism_ids = {item.organism_id for item in organisms}
    assert len(organism_ids) == settings.population_size

    species_ids = [item.species_id for item in first.final_species]
    assert len(species_ids) == len(set(species_ids))

    assigned_ids = [
        member_id
        for species in first.final_species
        for member_id in species.member_ids
    ]
    assert len(assigned_ids) == len(set(assigned_ids))
    assert set(assigned_ids) == organism_ids

    species_by_id = {
        species.species_id: set(species.member_ids)
        for species in first.final_species
    }
    for organism in organisms:
        assert organism.species_id in species_by_id
        assert organism.organism_id in species_by_id[organism.species_id]

    for entry in first.history:
        assert entry["population_size"] == settings.population_size
        assert 0.0 <= entry["diversity_ratio"] <= 1.0
        assert 0.0 <= entry["species_diversity_ratio"] <= 1.0
        assert entry["species_count"] <= settings.population_size


def test_ecological_lineage_references_valid_previous_generation_parents(monkeypatch):
    frame, habitat = make_frames()

    population = Population(
        population_id="lineage-test",
        generation=0,
        organisms=tuple(
            Organism.born(make_genome(), generation=0, birth_index=i)
            for i in range(8)
        ),
    )

    def fake_run_genome(frame, genome, **kwargs):
        return (
            {
                "expectancy_r": 0.25,
                "max_realized_drawdown_pct": 0.05,
            },
            pd.DataFrame({
                "entry_time": [frame.index[5], frame.index[10]],
                "net_r": [0.3, 0.2],
                "direction": ["long", "short"],
                "bars_held": [2, 3],
            }),
            pd.DataFrame(),
        )
    monkeypatch.setattr(engine, "_run_genome", fake_run_genome)

    real_speciate = engine.speciate_population
    speciation_runs = []

    def tracked_speciate(population, **kwargs):
        classified, species = real_speciate(population, **kwargs)
        speciation_runs.append((population, classified, species))
        return classified, species

    monkeypatch.setattr(engine, "speciate_population", tracked_speciate)

    real_reproduce = engine.reproduce_population
    transitions = []

    def tracked_reproduce(selection, **kwargs):
        parent_pool = selection.population
        next_population = real_reproduce(selection, **kwargs)

        transitions.append((parent_pool, next_population))
        return next_population

    monkeypatch.setattr(engine, "reproduce_population", tracked_reproduce)

    settings = EvolutionSettings(
        population_size=8,
        generations=3,
        elite_count=2,
        tournament_size=2,
        random_injection_rate=0.25,
        seed=4217,
    )

    result = engine.run_ecological_generations(
        population,
        frame,
        habitat,
        execution=None,
        risk=None,
        pip_size=0.0001,
        price_basis="bid",
        timeframe_seconds=14400,
        settings=settings,
        minimum_habitat_trades=1,
        minimum_period_trades=1,
    )

    assert len(transitions) == settings.generations - 1

    for parent_pool, next_population in transitions:
        parent_ids = {
            organism.organism_id
            for organism in parent_pool.organisms
        }

        previous_ids = {
            organism.organism_id
            for organism in next_population.organisms
        }

        assert next_population.generation == parent_pool.generation + 1
        assert next_population.size == settings.population_size

        for organism in next_population.organisms:
            if organism.organism_id in parent_ids:
                assert organism.parent_ids == next(
                    item.parent_ids
                    for item in parent_pool.organisms
                    if item.organism_id == organism.organism_id
                )
                assert organism.age_generations >= 1
            else:
                assert organism.generation == next_population.generation
                assert len(organism.parent_ids) <= 2
                assert set(organism.parent_ids).issubset(parent_ids)
                assert organism.organism_id not in parent_ids

    assert result.final_population.generation == settings.generations - 1

    assert len(speciation_runs) == settings.generations

    for index, (_, next_population) in enumerate(transitions):
        next_input, classified, species = speciation_runs[index + 1]

        assert next_input is next_population

        species_by_id = {
            item.species_id: item
            for item in species
        }

        assigned_ids = set()

        for item in species:
            assigned_ids.update(item.member_ids)

        assert assigned_ids == {
            organism.organism_id
            for organism in classified.organisms
        }

        for organism in classified.organisms:
            assert organism.species_id in species_by_id
            assert organism.organism_id in (
                species_by_id[organism.species_id].member_ids
            )

        previous_ids = {
            organism.organism_id
            for organism in transitions[index][0].organisms
        }

        immigrants = [
            organism
            for organism in next_population.organisms
            if organism.organism_id not in previous_ids
            and not organism.parent_ids
        ]

        for immigrant in immigrants:
            classified_immigrant = next(
                item
                for item in classified.organisms
                if item.organism_id == immigrant.organism_id
            )
            assert classified_immigrant.species_id is not None
            
