import numpy as np
import pandas as pd
import pytest

from edge.ecology import (
    Habitat,
    HabitatModel,
    HabitatOutcome,
    OrganismFitness,
    Species,
    classify_habitats,
    fit_habitat_model,
    profile_species_habitats,
    SpeciesHabitatProfile,
    TemporalHabitatPerformanceSummary,
    TemporalHabitatPeriod,
    measure_temporal_habitat_performance,
    TemporalHabitatPersistence,
    profile_temporal_habitats,
    _parent_selection_probabilities,
)

from dataclasses import replace


def sample_market_data(rows: int = 100) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01", periods=rows, freq="4h", tz="UTC"
    )
    return pd.DataFrame(
        {
            "trend_efficiency_10": np.linspace(0.05, 0.95, rows),
            "ema_gap_atr": np.linspace(-1.5, 1.5, rows),
            "volatility_ratio_10_50": np.linspace(0.2, 2.0, rows),
            "compression_ratio_8_32": np.linspace(0.1, 0.9, rows),
        },
        index=index,
    )


def test_habitat_model_fits_training_prefix_only():
    frame = sample_market_data()

    original = fit_habitat_model(frame, fit_end=70)

    altered = frame.copy()
    altered.iloc[70:] = 9999.0

    modified = fit_habitat_model(altered, fit_end=70)

    assert original == modified
    assert original.fit_rows == 70


def test_habitat_model_serialization_round_trip():
    model = fit_habitat_model(sample_market_data(), fit_end=80)

    restored = HabitatModel.from_dict(model.to_dict())

    assert restored == model


def test_habitat_classification_supports_overlapping_conditions():
    model = HabitatModel(
        trend_efficiency_low=0.30,
        trend_efficiency_high=0.60,
        ema_gap_abs_min=0.50,
        volatility_low=0.80,
        volatility_high=1.20,
        compression_low=0.30,
        expansion_high=0.70,
        fit_rows=100,
        fit_start_utc="2024-01-01T00:00:00+00:00",
        fit_end_utc="2024-01-17T12:00:00+00:00",
    )

    frame = pd.DataFrame(
        {
            "trend_efficiency_10": [0.8, 0.8, 0.1, 0.45, np.nan],
            "ema_gap_atr": [0.8, -0.8, 0.1, 0.8, 0.0],
            "volatility_ratio_10_50": [1.5, 0.5, 1.0, 1.0, 1.0],
            "compression_ratio_8_32": [0.9, 0.2, 0.5, 0.5, 0.5],
        },
        index=pd.date_range("2025-01-01", periods=5, freq="4h", tz="UTC"),
    )

    result = classify_habitats(frame, model)

    assert result.iloc[0]["habitat"] == Habitat(
        "trend_up", "high_volatility", "expansion"
    )
    assert result.iloc[1]["habitat"] == Habitat(
        "trend_down", "low_volatility", "compression"
    )
    assert result.iloc[2]["habitat"] == Habitat(
        "range", "normal_volatility", "balanced"
    )
    assert result.iloc[3]["habitat_regime"] == "transitional"
    assert result.iloc[4]["habitat"] is None
    assert not result.iloc[4]["habitat_valid"]


def test_habitat_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="Invalid habitat regime"):
        Habitat("unknown_regime", "normal_volatility", "balanced")


def test_habitat_model_requires_enough_training_rows():
    frame = sample_market_data(40)

    with pytest.raises(ValueError, match="at least 30"):
        fit_habitat_model(frame, fit_end=20)

from edge.evolution import Condition, StrategyGenome
from edge.ecology import (
    Organism,
    Population,
    genetic_distance,
    speciate_population,
)


def sample_genomes():
    base = StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.2),),
        short_conditions=(Condition("rsi_centered", "<", -0.2),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )

    similar = StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.21),),
        short_conditions=(Condition("rsi_centered", "<", -0.21),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )

    different = StrategyGenome(
        long_conditions=(Condition("ema_gap_atr", ">", 0.8),),
        short_conditions=(Condition("volatility_ratio_10_50", ">", 1.5),),
        stop_atr=3.0,
        reward_risk=4.0,
        max_holding_bars=160,
    )

    return base, similar, different


def test_organism_identity_is_deterministic():
    base, _, _ = sample_genomes()

    first = Organism.born(base, generation=2, birth_index=4)
    second = Organism.born(base, generation=2, birth_index=4)
    third = Organism.born(base, generation=2, birth_index=5)

    assert first.organism_id == second.organism_id
    assert first.organism_id != third.organism_id


def test_organism_preserves_parent_ancestry():
    base, _, _ = sample_genomes()

    organism = Organism.born(
        base,
        generation=3,
        parent_ids=("parent-a", "parent-b"),
    )

    assert organism.parent_ids == ("parent-a", "parent-b")
    assert organism.generation == 3
    assert organism.age_generations == 0


def test_genetic_distance_is_symmetric_and_bounded():
    base, similar, different = sample_genomes()

    close_distance = genetic_distance(base, similar)
    far_distance = genetic_distance(base, different)

    assert genetic_distance(base, base) == pytest.approx(0.0)
    assert close_distance == pytest.approx(genetic_distance(similar, base))
    assert 0.0 <= close_distance <= 1.0
    assert 0.0 <= far_distance <= 1.0
    assert far_distance > close_distance


def test_similar_organisms_form_the_same_species():
    base, similar, different = sample_genomes()

    organisms = (
        Organism.born(base, 0, birth_index=0),
        Organism.born(similar, 1, birth_index=1),
        Organism.born(different, 1, birth_index=2),
    )

    population = Population(
        population_id="population-1",
        generation=1,
        organisms=organisms,
    )

    updated, species = speciate_population(
        population,
        distance_threshold=0.25,
    )

    assert len(species) == 2
    assert sorted(item.size for item in species) == [1, 2]

    assigned = {
        item.organism_id: item.species_id
        for item in updated.organisms
    }

    assert assigned[organisms[0].organism_id] == assigned[organisms[1].organism_id]
    assert assigned[organisms[0].organism_id] != assigned[organisms[2].organism_id]


def test_speciation_is_deterministic_for_same_population():
    base, similar, different = sample_genomes()

    organisms = tuple(
        Organism.born(genome, 1, birth_index=i)
        for i, genome in enumerate((base, similar, different))
    )

    population = Population("population-2", 1, organisms)

    first_population, first_species = speciate_population(population)
    second_population, second_species = speciate_population(population)

    assert first_species == second_species
    assert first_population == second_population

from edge.ecology import (
    HabitatOutcome,
    OrganismFitness,
    pareto_dominates,
    pareto_fronts,
)


def test_speciation_preserves_existing_species_identity():
    base, similar, different = sample_genomes()

    first_population = Population(
        "generation-0",
        0,
        (Organism.born(base, 0, birth_index=0),),
    )
    first_population, first_species = speciate_population(first_population)

    original_species = first_species[0]

    second_population = Population(
        "generation-1",
        1,
        (
            replace(
     		      Organism.born(similar, 1, birth_index=1),
    		      species_id=original_species.species_id,
            ),
        ),
    )

    updated, species = speciate_population(
        second_population,
        previous_species=first_species,
    )

    assert len(species) == 1
    assert species[0].species_id == original_species.species_id
    assert species[0].founder_organism_id == original_species.founder_organism_id
    assert species[0].founder_genome == original_species.founder_genome
    assert species[0].origin_generation == 0
    assert species[0].last_active_generation == 1
    assert updated.organisms[0].species_id == original_species.species_id

def test_species_identity_survives_founder_extinction():
    base, similar, _ = sample_genomes()

    founder = Organism.born(base, 0, birth_index=0)
    population_0 = Population("generation-0", 0, (founder,))
    population_0, species_0 = speciate_population(population_0)

    descendant = replace(
        Organism.born(
            similar,
            1,
            parent_ids=(founder.organism_id,),
            birth_index=0,
        ),
        species_id=species_0[0].species_id,
    )

    # The founder is absent from this generation.
    population_1 = Population("generation-1", 1, (descendant,))

    updated_population, species_1 = speciate_population(
        population_1,
        previous_species=species_0,
    )

    assert updated_population.size == 1
    assert species_1[0].species_id == species_0[0].species_id
    assert species_1[0].founder_organism_id == founder.organism_id
    assert species_1[0].member_ids == (descendant.organism_id,)
    assert species_1[0].origin_generation == 0
    assert species_1[0].last_active_generation == 1

def test_fully_extinct_species_is_removed():
    base, _, _ = sample_genomes()

    founder = Organism.born(base, 0, birth_index=0)
    population_0 = Population("generation-0", 0, (founder,))
    _, species_0 = speciate_population(population_0)

    population_1 = Population("generation-1", 1, ())

    updated_population, species_1 = speciate_population(
        population_1,
        previous_species=species_0,
    )

    assert updated_population.size == 0
    assert species_1 == ()


def test_genetically_distant_organism_establishes_new_species():
    base, _, different = sample_genomes()

    first_population = Population(
        "generation-0",
        0,
        (Organism.born(base, 0, birth_index=0),),
    )
    first_population, first_species = speciate_population(first_population)

    original_species = first_species[0]

    second_population = Population(
        "generation-1",
        1,
        (
            replace(
    		Organism.born(different, 1, birth_index=1),
    		species_id=original_species.species_id,
	    ),
        ),
    )

    updated, species = speciate_population(
        second_population,
        distance_threshold=0.25,
        previous_species=first_species,
    )

    assert len(species) == 1
    assert species[0].species_id != original_species.species_id
    assert species[0].founder_organism_id == updated.organisms[0].organism_id
    assert species[0].origin_generation == 1
    assert updated.organisms[0].species_id == species[0].species_id



def test_organism_reassigns_to_closest_existing_species():
    base, _, different = sample_genomes()

    founder_a = Organism.born(base, 0, birth_index=0)
    founder_b = Organism.born(different, 0, birth_index=1)

    population_0 = Population(
        "generation-0",
        0,
        (founder_a, founder_b),
    )
    population_0, species_0 = speciate_population(
        population_0,
        distance_threshold=0.25,
    )

    species_a = next(
        item for item in species_0
        if item.founder_organism_id == founder_a.organism_id
    )
    species_b = next(
        item for item in species_0
        if item.founder_organism_id == founder_b.organism_id
    )

    reassigned_organism = replace(
        Organism.born(different, 1, birth_index=2),
        species_id=species_a.species_id,
    )

    population_1 = Population(
        "generation-1",
        1,
        (reassigned_organism,),
    )

    updated, species_1 = speciate_population(
        population_1,
        distance_threshold=0.25,
        previous_species=species_0,
    )

    assert len(species_1) == 1
    assert species_1[0].species_id == species_b.species_id
    assert species_1[0].founder_organism_id == founder_b.organism_id
    assert updated.organisms[0].species_id == species_b.species_id



def test_speciation_rejects_generated_species_id_collision(monkeypatch):
    base, _, different = sample_genomes()

    founder = Organism.born(base, 0, birth_index=0)
    population_0 = Population("generation-0", 0, (founder,))
    _, species_0 = speciate_population(population_0)

    colliding_species = replace(
        species_0[0],
        species_id="sp-collision000",
    )

    organism = Organism.born(different, 1, birth_index=1)
    population_1 = Population("generation-1", 1, (organism,))

    class FakeDigest:
        def hexdigest(self):
            return "collision000000000000"

    monkeypatch.setattr(
        "edge.ecology.hashlib.sha256",
        lambda value: FakeDigest(),
    )

    with pytest.raises(RuntimeError, match="Species identity collision"):
        speciate_population(
            population_1,
            distance_threshold=0.25,
            previous_species=(colliding_species,),
        )



def test_speciation_rejects_collision_between_new_species(monkeypatch):
    base, _, different = sample_genomes()

    first = Organism.born(base, 1, birth_index=0)
    second = Organism.born(different, 1, birth_index=1)

    population = Population(
        "generation-1",
        1,
        (first, second),
    )

    class FakeDigest:
        def hexdigest(self):
            return "collision000000000000"

    monkeypatch.setattr(
        "edge.ecology.hashlib.sha256",
        lambda value: FakeDigest(),
    )

    with pytest.raises(RuntimeError, match="Species identity collision"):
        speciate_population(population)



def test_equal_distance_reassignment_uses_species_id_tiebreak(monkeypatch):
    base, _, different = sample_genomes()

    founder_a = Organism.born(base, 0, birth_index=0)
    founder_b = Organism.born(different, 0, birth_index=1)

    population_0 = Population(
        "generation-0",
        0,
        (founder_a, founder_b),
    )
    _, species_0 = speciate_population(population_0)

    organism = Organism.born(base, 1, birth_index=2)

    population_1 = Population(
        "generation-1",
        1,
        (organism,),
    )

    monkeypatch.setattr(
        "edge.ecology.genetic_distance",
        lambda genome_a, genome_b, feature_scales=None: 0.1,
    )

    updated, species_1 = speciate_population(
        population_1,
        distance_threshold=0.25,
        previous_species=species_0,
    )

    expected_species = min(
        species_0,
        key=lambda item: item.species_id,
    )

    assert len(species_1) == 1
    assert species_1[0].species_id == expected_species.species_id
    assert updated.organisms[0].species_id == expected_species.species_id



def test_speciation_is_independent_of_population_order():
    base, similar, different = sample_genomes()

    organisms = (
        Organism.born(base, 0, birth_index=0),
        Organism.born(similar, 0, birth_index=1),
        Organism.born(different, 0, birth_index=2),
    )

    first_population = Population("first", 0, organisms)
    second_population = Population("second", 0, tuple(reversed(organisms)))

    _, first_species = speciate_population(first_population)
    _, second_species = speciate_population(second_population)

    first_membership = {
        organism.organism_id: organism.species_id
        for organism in first_population.organisms
    }
    second_membership = {
        organism.organism_id: organism.species_id
        for organism in second_population.organisms
    }

    assert first_membership == second_membership


def test_speciation_clustering_is_independent_of_organism_ids():
    base, _, _ = sample_genomes()

    middle = replace(
        base,
        long_conditions=(
            replace(base.long_conditions[0], threshold=0.6),
        ),
        short_conditions=(
            replace(base.short_conditions[0], threshold=-0.6),
        ),
    )

    end = replace(
        base,
        long_conditions=(
            replace(base.long_conditions[0], threshold=1.0),
        ),
        short_conditions=(
            replace(base.short_conditions[0], threshold=-1.0),
        ),
    )

    organisms = (
        replace(Organism.born(base, 0, birth_index=0), organism_id="a"),
        replace(Organism.born(middle, 0, birth_index=1), organism_id="b"),
        replace(Organism.born(end, 0, birth_index=2), organism_id="c"),
    )

    renamed = (
        replace(organisms[0], organism_id="z"),
        replace(organisms[1], organism_id="a"),
        replace(organisms[2], organism_id="b"),
    )

    def get_partition(items):
        population = Population("test", 0, items)
        updated, species = speciate_population(population)

        return frozenset(
            frozenset(
                organism.genome.fingerprint()
                for organism in updated.organisms
                if organism.species_id == item.species_id
            )
            for item in species
        )

    assert get_partition(organisms) == get_partition(renamed)



def test_speciation_clustering_is_stable_across_randomized_id_changes():
    rng = np.random.default_rng(42)
    base, _, _ = sample_genomes()

    organisms = []

    for i in range(30):
        genome = replace(
            base,
            long_conditions=(
                replace(
                    base.long_conditions[0],
                    threshold=float(rng.uniform(-1.0, 1.0)),
                ),
            ),
            short_conditions=(
                replace(
                    base.short_conditions[0],
                    threshold=float(rng.uniform(-1.0, 1.0)),
                ),
            ),
            stop_atr=float(rng.uniform(1.0, 3.0)),
            reward_risk=float(rng.uniform(1.0, 4.0)),
            max_holding_bars=int(rng.integers(5, 50)),
        )

        organism = Organism.born(genome, 0, birth_index=i)
        organisms.append(
            replace(organism, organism_id=f"original-{i:02d}")
        )

    renamed = tuple(
        replace(
            organism,
            organism_id=f"renamed-{29 - i:02d}",
        )
        for i, organism in enumerate(organisms)
    )

    def get_partition(items):
        population = Population("randomized-test", 0, items)
        updated, species = speciate_population(population)

        return frozenset(
            frozenset(
                organism.genome.fingerprint()
                for organism in updated.organisms
                if organism.species_id == item.species_id
            )
            for item in species
        )

    original_partition = get_partition(tuple(organisms))

    shuffled_indices = rng.permutation(len(organisms))

    shuffled = tuple(
        organisms[int(i)] for i in shuffled_indices
    )

    assert original_partition == get_partition(renamed)
    assert original_partition == get_partition(shuffled)



def test_speciation_does_not_chain_cluster_beyond_founder_distance():
    base, _, _ = sample_genomes()

    middle = replace(
        base,
        long_conditions=(
            replace(base.long_conditions[0], threshold=0.6),
        ),
        short_conditions=(
            replace(base.short_conditions[0], threshold=-0.6),
        ),
    )

    end = replace(
        base,
        long_conditions=(
            replace(base.long_conditions[0], threshold=1.0),
        ),
        short_conditions=(
            replace(base.short_conditions[0], threshold=-1.0),
        ),
    )

    organism_a = replace(
        Organism.born(base, 0, birth_index=0),
        organism_id="a",
    )
    organism_b = replace(
        Organism.born(middle, 0, birth_index=1),
        organism_id="b",
    )
    organism_c = replace(
        Organism.born(end, 0, birth_index=2),
        organism_id="c",
    )

    assert genetic_distance(base, middle) < 0.25
    assert genetic_distance(middle, end) < 0.25
    assert genetic_distance(base, end) > 0.25

    population = Population(
        "chain-test",
        0,
        (organism_a, organism_b, organism_c),
    )

    updated_population, species = speciate_population(
        population,
        distance_threshold=0.25,
    )

    membership = {
        organism.organism_id: organism.species_id
        for organism in updated_population.organisms
    }

    assert membership["a"] == membership["b"]
    assert membership["a"] != membership["c"]
    assert len(species) == 2

def test_species_identity_breaks_after_multigeneration_genetic_drift():
    base, _, _ = sample_genomes()

    middle = replace(
        base,
        long_conditions=(
            replace(base.long_conditions[0], threshold=0.6),
        ),
        short_conditions=(
            replace(base.short_conditions[0], threshold=-0.6),
        ),
    )

    end = replace(
        base,
        long_conditions=(
            replace(base.long_conditions[0], threshold=1.0),
        ),
        short_conditions=(
            replace(base.short_conditions[0], threshold=-1.0),
        ),
    )

    organism_a = Organism.born(base, 0, birth_index=0)
    population_0 = Population("generation-0", 0, (organism_a,))
    population_0, species_0 = speciate_population(population_0)

    organism_b = replace(
        Organism.born(middle, 1, birth_index=0),
        species_id=species_0[0].species_id,
    )
    population_1 = Population("generation-1", 1, (organism_b,))
    population_1, species_1 = speciate_population(
        population_1,
        previous_species=species_0,
    )

    organism_c = replace(
        Organism.born(end, 2, birth_index=0),
        species_id=species_1[0].species_id,
    )
    population_2 = Population("generation-2", 2, (organism_c,))
    population_2, species_2 = speciate_population(
        population_2,
        previous_species=species_1,
    )

    assert genetic_distance(base, middle) < 0.25
    assert genetic_distance(middle, end) < 0.25
    assert genetic_distance(base, end) > 0.25

    assert species_1[0].species_id == species_0[0].species_id
    assert species_2[0].species_id != species_0[0].species_id
    assert species_2[0].founder_organism_id == organism_c.organism_id


def test_species_identity_survives_multigeneration_branching():
    base, similar, different = sample_genomes()

    founder = Organism.born(base, 0, birth_index=0)
    population_0 = Population("generation-0", 0, (founder,))
    population_0, species_0 = speciate_population(population_0)

    original_species_id = species_0[0].species_id

    original_descendant = replace(
        Organism.born(
            similar,
            1,
            parent_ids=(founder.organism_id,),
            birth_index=0,
        ),
        species_id=original_species_id,
    )
    divergent_descendant = Organism.born(
        different,
        1,
        parent_ids=(founder.organism_id,),
        birth_index=1,
    )

    population_1 = Population(
        "generation-1",
        1,
        (original_descendant, divergent_descendant),
    )
    population_1, species_1 = speciate_population(
        population_1,
        previous_species=species_0,
    )

    original_species = next(
        item for item in species_1
        if item.species_id == original_species_id
    )
    new_species = next(
        item for item in species_1
        if item.species_id != original_species_id
    )

    original_descendant_2 = replace(
        Organism.born(base, 2, birth_index=0),
        species_id=original_species_id,
    )
    divergent_descendant_2 = replace(
        Organism.born(different, 2, birth_index=1),
        species_id=new_species.species_id,
    )

    population_2 = Population(
        "generation-2",
        2,
        (divergent_descendant_2, original_descendant_2),
    )
    population_2, species_2 = speciate_population(
        population_2,
        previous_species=species_1,
    )

    assert len(species_2) == 2

    original_species_2 = next(
        item for item in species_2
        if item.species_id == original_species_id
    )
    new_species_2 = next(
        item for item in species_2
        if item.species_id == new_species.species_id
    )

    assert original_species_2.founder_organism_id == founder.organism_id
    assert original_species_2.origin_generation == 0

    assert new_species_2.founder_organism_id == divergent_descendant.organism_id
    assert new_species_2.origin_generation == 1

    assert set(original_species_2.member_ids) == {
        original_descendant_2.organism_id
    }
    assert set(new_species_2.member_ids) == {
        divergent_descendant_2.organism_id
    }


def test_new_species_founders_do_not_chain_cluster():
    base, similar, different = sample_genomes()

    organisms = (
        Organism.born(base, 0, birth_index=0),
        Organism.born(similar, 0, birth_index=1),
        Organism.born(different, 0, birth_index=2),
    )

    population = Population("generation-0", 0, organisms)

    _, species = speciate_population(
        population,
        distance_threshold=0.25,
    )

    assert sum(item.size for item in species) == len(organisms)

def test_adaptability_ignores_under_sampled_habitats():
    fitness = OrganismFitness(
        organism_id="org-1",
        expectancy_r=0.10,
        max_drawdown_pct=0.02,
        complexity=4,
        novelty_score=0.8,
        habitat_outcomes=(
            HabitatOutcome("trend_up", 0.20, 20),
            HabitatOutcome("range", -0.10, 15),
            HabitatOutcome("high_volatility", -1.0, 2),
        ),
    )

    assert fitness.adaptability_score(minimum_trades=5) == pytest.approx(0.5)


def test_fitness_objectives_reward_robustness_and_simplicity():
    fitness = OrganismFitness(
        organism_id="org-2",
        expectancy_r=0.15,
        max_drawdown_pct=0.03,
        complexity=3,
        novelty_score=0.7,
        habitat_outcomes=(
            HabitatOutcome("trend_up", 0.2, 20),
            HabitatOutcome("range", 0.1, 20),
        ),
        temporal_persistence_score=0.75,
    )

    assert fitness.objectives() == pytest.approx(
        (0.15, -0.03, 1.0, 0.2, -3.0, 0.7, 0.75)
    )


def test_niche_performance_rewards_well_sampled_specialists():
    fitness = OrganismFitness(
        organism_id="specialist-1",
        expectancy_r=0.08,
        max_drawdown_pct=0.04,
        complexity=3,
        novelty_score=0.5,
        habitat_outcomes=(
            HabitatOutcome("trend_up", 0.32, 20),
            HabitatOutcome("range", -0.12, 20),
            HabitatOutcome("high_volatility", 0.90, 2),
        ),
    )

    assert fitness.niche_performance_score(minimum_trades=5) == pytest.approx(0.32)
    assert fitness.adaptability_score(minimum_trades=5) == pytest.approx(0.5)


def test_pareto_dominance_requires_improvement_without_regression():
    assert pareto_dominates(
        (0.2, -0.01, 0.8),
        (0.1, -0.02, 0.5),
    )

    assert not pareto_dominates(
        (0.2, -0.04, 0.8),
        (0.1, -0.02, 0.5),
    )

    assert not pareto_dominates(
        (0.2, -0.02, 0.5),
        (0.2, -0.02, 0.5),
    )


def test_pareto_fronts_preserve_different_strengths():
    objectives = {
        "high_return": (0.3, -0.08, 0.5),
        "low_drawdown": (0.1, -0.01, 0.8),
        "balanced": (0.2, -0.03, 0.7),
        "dominated": (0.05, -0.10, 0.2),
    }

    fronts = pareto_fronts(objectives)

    assert set(fronts[0]) == {
        "high_return", "low_drawdown", "balanced"
    }
    assert fronts[1] == ["dominated"]


def test_pareto_fronts_are_deterministic():
    objectives = {
        "b": (0.2, -0.02),
        "a": (0.2, -0.02),
        "c": (0.1, -0.05),
    }

    assert pareto_fronts(objectives) == [
        ["a", "b"],
        ["c"],
    ]

from edge.ecology import select_ecological_survivors


def build_competing_population(distance_threshold=0.25):
    base, similar, different = sample_genomes()

    organisms = tuple(
        Organism.born(genome, 0, birth_index=i)
        for i, genome in enumerate((base, similar, different))
    )

    population = Population("competition-test", 0, organisms)
    updated, species = speciate_population(
        population,
        distance_threshold=distance_threshold,
    )

    return updated, species


def build_fitness(population):
    records = [
        (0.30, 0.01, 4, 0.4),
        (0.10, 0.04, 8, 0.2),
        (-0.20, 0.08, 6, 0.9),
    ]

    return {
        organism.organism_id: OrganismFitness(
            organism_id=organism.organism_id,
            expectancy_r=records[i][0],
            max_drawdown_pct=records[i][1],
            complexity=records[i][2],
            novelty_score=records[i][3],
        )
        for i, organism in enumerate(population.organisms)
    }


def test_parent_selection_prefers_better_pareto_rank():
    population, _ = build_competing_population()
    members = list(population.organisms)

    fitness = {
        item.organism_id: OrganismFitness(
            organism_id=item.organism_id,
            expectancy_r=0.1,
            max_drawdown_pct=0.02,
            complexity=4,
            novelty_score=0.5,
        )
        for item in members
    }

    ranks = {
        members[0].organism_id: 0,
        members[1].organism_id: 1,
        members[2].organism_id: 2,
    }

    probabilities = _parent_selection_probabilities(
        members, fitness, ranks, {}
    )

    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert np.isclose(probabilities.sum(), 1.0)


def test_parent_selection_rewards_crowding_distance():
    population, _ = build_competing_population()
    members = list(population.organisms)

    fitness = {
        item.organism_id: OrganismFitness(
            organism_id=item.organism_id,
            expectancy_r=0.1,
            max_drawdown_pct=0.02,
            complexity=4,
            novelty_score=0.5,
        )
        for item in members
    }

    ranks = {item.organism_id: 0 for item in members}
    crowding = {
        members[0].organism_id: 0.0,
        members[1].organism_id: 1.0,
        members[2].organism_id: float("inf"),
    }

    probabilities = _parent_selection_probabilities(
        members, fitness, ranks, crowding
    )

    assert probabilities[2] > probabilities[1] > probabilities[0]


@pytest.mark.parametrize(
    "distance",
    [
        float("nan"),
        -1.0,
        float("-inf"),
    ],
)
def test_parent_selection_rejects_invalid_crowding_distance(distance):
    population, _ = build_competing_population()
    members = list(population.organisms)

    fitness = {
        item.organism_id: OrganismFitness(
            organism_id=item.organism_id,
            expectancy_r=0.1,
            max_drawdown_pct=0.02,
            complexity=4,
            novelty_score=0.5,
        )
        for item in members
    }

    ranks = {item.organism_id: 0 for item in members}
    crowding = {
        members[0].organism_id: distance,
    }

    with pytest.raises(ValueError, match="Invalid crowding distance"):
        _parent_selection_probabilities(
            members, fitness, ranks, crowding
        )


def test_negative_expectancy_survivor_retains_parent_probability():
    population, _ = build_competing_population()
    members = list(population.organisms)

    fitness = {
        item.organism_id: OrganismFitness(
            organism_id=item.organism_id,
            expectancy_r=-0.5 if i == 2 else 0.2,
            max_drawdown_pct=0.02,
            complexity=4,
            novelty_score=0.5,
        )
        for i, item in enumerate(members)
    }

    ranks = {item.organism_id: 0 for item in members}

    probabilities = _parent_selection_probabilities(
        members, fitness, ranks, {}
    )

    assert probabilities[2] > 0
    assert np.all(probabilities > 0)
    assert np.isclose(probabilities.sum(), 1.0)


def test_selection_preserves_species_when_capacity_allows():
    population, species = build_competing_population()
    fitness = build_fitness(population)

    result = select_ecological_survivors(
        population,
        species,
        fitness,
        capacity=2,
    )

    assert result.population.size == 2
    assert len(result.species) == 2
    assert result.extinct_species_ids == ()


def test_selection_allows_species_extinction_when_capacity_is_limited():
    population, species = build_competing_population(distance_threshold=0.0)
    fitness = build_fitness(population)

    assert len(species) == 3

    result = select_ecological_survivors(
        population,
        species,
        fitness,
        capacity=2,
    )

    assert result.population.size == 2
    assert len(result.species) == 2
    assert len(result.extinct_species_ids) == 1


def test_species_champion_requires_minimum_total_trade_sample():
    population, species = build_competing_population(
        distance_threshold=1.0,
    )
    assert len(species) == 1
    first, second, third = population.organisms

    fitness = {
        first.organism_id: OrganismFitness(
            organism_id=first.organism_id,
            expectancy_r=0.90,
            max_drawdown_pct=0.50,
            complexity=4,
            novelty_score=0.1,
            trade_count=1,
        ),
        second.organism_id: OrganismFitness(
            organism_id=second.organism_id,
            expectancy_r=-0.01,
            max_drawdown_pct=0.01,
            complexity=4,
            novelty_score=0.1,
            trade_count=30,
        ),
        third.organism_id: OrganismFitness(
            organism_id=third.organism_id,
            expectancy_r=-0.20,
            max_drawdown_pct=0.08,
            complexity=8,
            novelty_score=0.1,
            trade_count=30,
        ),
    }

    result = select_ecological_survivors(
        population,
        species,
        fitness,
        capacity=1,
        minimum_total_trades=30,
    )

    assert result.survivor_ids == (second.organism_id,)


def test_selection_rejects_missing_fitness_records():
    population, species = build_competing_population()
    fitness = build_fitness(population)

    fitness.pop(next(iter(fitness)))

    with pytest.raises(ValueError, match="Fitness records"):
        select_ecological_survivors(
            population,
            species,
            fitness,
            capacity=2,
        )


def test_selection_is_deterministic():
    population, species = build_competing_population()
    fitness = build_fitness(population)

    first = select_ecological_survivors(
        population, species, fitness, capacity=2
    )
    second = select_ecological_survivors(
        population, species, fitness, capacity=2
    )

    assert first == second

from edge.ecology import reproduce_population


FEATURE_RANGES = {
    "rsi_centered": (-0.2, 0.0, 0.2),
    "ema_gap_atr": (-0.8, 0.0, 0.8),
    "volatility_ratio_10_50": (0.5, 1.0, 1.5),
}


def prepare_reproduction():
    population, species = build_competing_population()
    fitness = build_fitness(population)

    selection = select_ecological_survivors(
        population,
        species,
        fitness,
        capacity=2,
    )
    return selection


def test_reproduction_preserves_population_size_and_ages_survivors():
    selection = prepare_reproduction()

    next_population = reproduce_population(
        selection,
        target_size=10,
        feature_ranges=FEATURE_RANGES,
        rng=np.random.default_rng(42),
    )

    assert next_population.size == 10
    assert next_population.generation == selection.population.generation + 1

    survivors = {
        item.organism_id: item
        for item in selection.population.organisms
    }

    for organism in next_population.organisms:
        if organism.organism_id in survivors:
            assert organism.age_generations == survivors[organism.organism_id].age_generations + 1


def test_offspring_record_parent_ancestry():
    selection = prepare_reproduction()

    next_population = reproduce_population(
        selection,
        target_size=8,
        feature_ranges=FEATURE_RANGES,
        rng=np.random.default_rng(43),
        immigrant_rate=0.0,
        migration_rate=1.0,
        mutation_rate=0.0,
    )

    survivors = {
        item.organism_id: item
        for item in selection.population.organisms
    }

    offspring = [
        item for item in next_population.organisms
        if item.organism_id not in survivors
    ]

    assert offspring

    for child in offspring:
        assert child.generation == next_population.generation
        assert len(child.parent_ids) == 2
        assert all(parent in survivors for parent in child.parent_ids)

        parent_species = {
            survivors[parent].species_id
            for parent in child.parent_ids
        }
        assert len(parent_species) == 2


def test_random_immigration_creates_parentless_organisms():
    selection = prepare_reproduction()

    next_population = reproduce_population(
        selection,
        target_size=8,
        feature_ranges=FEATURE_RANGES,
        rng=np.random.default_rng(44),
        immigrant_rate=1.0,
    )

    survivors = {
        item.organism_id
        for item in selection.population.organisms
    }

    immigrants = [
        item for item in next_population.organisms
        if item.organism_id not in survivors
    ]

    assert immigrants
    assert all(item.parent_ids == () for item in immigrants)
    assert all(item.species_id is None for item in immigrants)


def test_reproduction_repairs_inactive_immigrants_when_signal_frame_is_given(
    monkeypatch,
):
    from edge import evolution
    from edge.evolution import has_both_directional_signals

    selection = prepare_reproduction()
    index = pd.date_range(
        "2025-01-01", periods=4, freq="4h", tz="UTC"
    )
    signal_frame = pd.DataFrame(
        {
            "rsi_centered": [0.8, -0.8, 0.0, 0.1],
            "ema_gap_atr": [1.2, 0.0, 0.0, 0.0],
            "volatility_ratio_10_50": [1.0, 2.0, 1.0, 1.0],
        },
        index=index,
    )
    inactive = StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.2),),
        short_conditions=(Condition("rsi_centered", ">", 0.2),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )
    active = StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.4),),
        short_conditions=(Condition("rsi_centered", "<", -0.4),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )
    mutation_calls = 0

    def return_inactive(rng, ranges):
        return inactive

    def repair_to_active(rng, genome, ranges):
        nonlocal mutation_calls
        mutation_calls += 1
        return active

    monkeypatch.setattr(evolution, "random_genome", return_inactive)
    monkeypatch.setattr(evolution, "mutate_genome", repair_to_active)

    next_population = reproduce_population(
        selection,
        target_size=selection.population.size + 1,
        feature_ranges=FEATURE_RANGES,
        rng=np.random.default_rng(45),
        mutation_rate=0.0,
        immigrant_rate=1.0,
        migration_rate=0.0,
        signal_frame=signal_frame,
    )

    immigrant = next_population.organisms[-1]
    assert immigrant.parent_ids == ()
    assert immigrant.genome == active
    assert has_both_directional_signals(signal_frame, immigrant.genome)
    assert mutation_calls == 1


def test_reproduction_repairs_inactive_crossover_offspring_when_signal_frame_is_given(
    monkeypatch,
):
    from edge import evolution
    from edge.evolution import has_both_directional_signals

    selection = prepare_reproduction()
    index = pd.date_range(
        "2025-01-01", periods=4, freq="4h", tz="UTC"
    )
    signal_frame = pd.DataFrame(
        {
            "rsi_centered": [0.8, -0.8, 0.0, 0.1],
            "ema_gap_atr": [1.2, 0.0, 0.0, 0.0],
            "volatility_ratio_10_50": [1.0, 2.0, 1.0, 1.0],
        },
        index=index,
    )
    inactive = StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.2),),
        short_conditions=(Condition("rsi_centered", ">", 0.2),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )
    active = StrategyGenome(
        long_conditions=(Condition("rsi_centered", ">", 0.4),),
        short_conditions=(Condition("rsi_centered", "<", -0.4),),
        stop_atr=1.5,
        reward_risk=2.0,
        max_holding_bars=10,
    )
    mutation_calls = 0

    monkeypatch.setattr(
        evolution,
        "crossover",
        lambda rng, parent_a, parent_b: inactive,
    )

    def repair_to_active(rng, genome, ranges):
        nonlocal mutation_calls
        mutation_calls += 1
        return active

    monkeypatch.setattr(evolution, "mutate_genome", repair_to_active)

    next_population = reproduce_population(
        selection,
        target_size=selection.population.size + 1,
        feature_ranges=FEATURE_RANGES,
        rng=np.random.default_rng(46),
        mutation_rate=0.0,
        immigrant_rate=0.0,
        migration_rate=0.0,
        signal_frame=signal_frame,
    )

    child = next_population.organisms[-1]
    assert child.parent_ids
    assert child.genome == active
    assert has_both_directional_signals(signal_frame, child.genome)
    assert mutation_calls == 1


def test_reproduction_is_deterministic_with_same_seed():
    selection = prepare_reproduction()

    first = reproduce_population(
        selection,
        target_size=12,
        feature_ranges=FEATURE_RANGES,
        rng=np.random.default_rng(99),
    )

    second = reproduce_population(
        selection,
        target_size=12,
        feature_ranges=FEATURE_RANGES,
        rng=np.random.default_rng(99),
    )

    assert first == second


def test_reproduction_rejects_incomplete_feature_ranges():
    selection = prepare_reproduction()

    with pytest.raises(ValueError, match="omit features"):
        reproduce_population(
            selection,
            target_size=8,
            feature_ranges={"rsi_centered": (-0.2, 0.2)},
            rng=np.random.default_rng(45),
        )

def test_reproduction_accepts_duplicate_after_five_dedup_attempts(
    monkeypatch,
):
    from edge import evolution

    selection = prepare_reproduction()
    survivor_genome = selection.population.organisms[0].genome
    mutation_calls = 0

    def duplicate_crossover(rng, parent_a, parent_b):
        return survivor_genome

    def unchanged_mutation(rng, genome, feature_ranges):
        nonlocal mutation_calls
        mutation_calls += 1
        return genome

    monkeypatch.setattr(
        evolution, "crossover", duplicate_crossover
    )
    monkeypatch.setattr(
        evolution, "mutate_genome", unchanged_mutation
    )

    next_population = reproduce_population(
        selection,
        target_size=selection.population.size + 1,
        feature_ranges=FEATURE_RANGES,
        rng=np.random.default_rng(123),
        mutation_rate=0.0,
        immigrant_rate=0.0,
        migration_rate=0.0,
    )

    assert next_population.size == selection.population.size + 1
    assert mutation_calls == 5
    assert (
        next_population.organisms[-1].genome.fingerprint()
        == survivor_genome.fingerprint()
    )

from edge.ecology import measure_habitat_performance


def test_habitat_performance_uses_signal_candle_not_entry_candle():
    index = pd.date_range(
        "2025-01-01", periods=5, freq="4h", tz="UTC"
    )

    habitats = pd.DataFrame(
        {
            "habitat_id": [
                "trend",
                "trend",
                "range",
                None,
                "transitional",
            ],
            "habitat_valid": [True, True, True, False, True],
        },
        index=index,
    )

    trades = pd.DataFrame(
        {
            "entry_time": [
                index[1],
                index[2],
                index[3],
                index[4],
            ],
            "net_r": [0.2, -0.1, 0.3, 0.5],
        }
    )

    result = measure_habitat_performance(
        trades,
        habitats,
        timeframe_seconds=4 * 60 * 60,
    )

    outcomes = {
        item.habitat_id: item
        for item in result.outcomes
    }

    assert outcomes["trend"].trade_count == 2
    assert outcomes["trend"].expectancy_r == pytest.approx(0.05)

    assert outcomes["range"].trade_count == 1
    assert outcomes["range"].expectancy_r == pytest.approx(0.3)

    assert outcomes["transitional"].trade_count == 0
    assert outcomes["transitional"].expectancy_r is None

    assert result.classified_trades == 3
    assert result.unclassified_trades == 1


def test_habitat_performance_rejects_invalid_trade_values():
    index = pd.date_range(
        "2025-01-01", periods=3, freq="4h", tz="UTC"
    )
    habitats = pd.DataFrame(
        {
            "habitat_id": ["trend", "trend", "trend"],
            "habitat_valid": [True, True, True],
        },
        index=index,
    )
    trades = pd.DataFrame(
        {
            "entry_time": [index[1]],
            "net_r": [np.nan],
        }
    )

    with pytest.raises(ValueError, match="net_r"):
        measure_habitat_performance(
            trades,
            habitats,
            timeframe_seconds=14400,
        )


def test_habitat_performance_rejects_unsorted_habitat_index():
    index = pd.DatetimeIndex(
        [
            "2025-01-01T04:00:00Z",
            "2025-01-01T00:00:00Z",
        ]
    )
    habitats = pd.DataFrame(
        {
            "habitat_id": ["trend", "range"],
            "habitat_valid": [True, True],
        },
        index=index,
    )

    with pytest.raises(ValueError, match="sorted and unique"):
        measure_habitat_performance(
            pd.DataFrame(columns=["entry_time", "net_r"]),
            habitats,
            timeframe_seconds=14400,
        )

def test_habitat_alignment_uses_previous_observed_candle_after_weekend():
    index = pd.DatetimeIndex(
        [
            "2025-01-03T12:00:00Z",  # Friday signal
            "2025-01-03T16:00:00Z",  # Friday entry
            "2025-01-06T00:00:00Z",  # Monday entry
        ]
    )

    habitats = pd.DataFrame(
        {
            "habitat_id": ["trend", "range", "transitional"],
            "habitat_valid": [True, True, True],
        },
        index=index,
    )

    trades = pd.DataFrame(
        {
            "entry_time": [index[1], index[2]],
            "net_r": [0.3, -0.2],
        }
    )

    result = measure_habitat_performance(
        trades,
        habitats,
        timeframe_seconds=14400,
    )

    outcomes = {item.habitat_id: item for item in result.outcomes}

    assert outcomes["trend"].expectancy_r == pytest.approx(0.3)
    assert outcomes["range"].expectancy_r == pytest.approx(-0.2)
    assert outcomes["transitional"].trade_count == 0
    assert result.classified_trades == 2
    assert result.unclassified_trades == 0

def test_species_habitat_profile_identifies_preferred_habitat():
    group = Species(
        species_id="species_1",
        founder_organism_id="org_1",
        founder_genome=None,
        member_ids=("org_1", "org_2"),
        origin_generation=0,
        last_active_generation=2,
    )

    fitness = {
        "org_1": OrganismFitness(
            organism_id="org_1",
            expectancy_r=0.2,
            max_drawdown_pct=2.0,
            complexity=4,
            novelty_score=0.3,
            habitat_outcomes=(
                HabitatOutcome("trend", 0.4, 10),
                HabitatOutcome("range", -0.1, 10),
            ),
        ),
        "org_2": OrganismFitness(
            organism_id="org_2",
            expectancy_r=0.3,
            max_drawdown_pct=3.0,
            complexity=5,
            novelty_score=0.4,
            habitat_outcomes=(
                HabitatOutcome("trend", 0.6, 10),
                HabitatOutcome("range", -0.2, 10),
            ),
        ),
    }

    profiles = profile_species_habitats((group,), fitness)

    assert len(profiles) == 1

    profile = profiles[0]

    assert profile.species_id == "species_1"
    assert profile.preferred_habitat == "trend"
    assert profile.sampled_habitats == 2
    assert dict(profile.habitat_expectancy) == {
        "range": pytest.approx(-0.15),
        "trend": pytest.approx(0.5),
    }
    assert 0.0 <= profile.specialization_score <= 1.0


def test_species_habitat_profile_ignores_insufficient_samples():
    group = Species(
        species_id="species_2",
        founder_organism_id="org_3",
        founder_genome=None,
        member_ids=("org_3",),
        origin_generation=0,
        last_active_generation=0,
    )

    fitness = {
        "org_3": OrganismFitness(
            organism_id="org_3",
            expectancy_r=0.2,
            max_drawdown_pct=2.0,
            complexity=3,
            novelty_score=0.1,
            habitat_outcomes=(
                HabitatOutcome("trend", 0.5, 10),
                HabitatOutcome("range", 0.8, 2),
            ),
        )
    }

    profiles = profile_species_habitats(
        (group,),
        fitness,
        minimum_trades=5,
    )

    assert profiles[0].preferred_habitat == "trend"
    assert profiles[0].sampled_habitats == 1
    assert dict(profiles[0].habitat_expectancy) == {"trend": 0.5}


def test_species_habitat_profile_handles_no_valid_habitat_data():
    group = Species(
        species_id="species_3",
        founder_organism_id="org_4",
        founder_genome=None,
        member_ids=("org_4",),
        origin_generation=0,
        last_active_generation=0,
    )

    fitness = {
        "org_4": OrganismFitness(
            organism_id="org_4",
            expectancy_r=0.0,
            max_drawdown_pct=0.0,
            complexity=2,
            novelty_score=0.0,
            habitat_outcomes=(),
        )
    }

    profiles = profile_species_habitats((group,), fitness)

    assert profiles[0].preferred_habitat is None
    assert profiles[0].sampled_habitats == 0
    assert profiles[0].specialization_score == 0.0


def test_species_habitat_profile_weights_expectancy_by_trade_count():
    group = Species(
        species_id="species_weighted",
        founder_organism_id="org_1",
        founder_genome=None,
        member_ids=("org_1", "org_2"),
        origin_generation=0,
        last_active_generation=1,
    )

    fitness = {
        "org_1": OrganismFitness(
            organism_id="org_1",
            expectancy_r=0.2,
            max_drawdown_pct=2.0,
            complexity=3,
            novelty_score=0.1,
            habitat_outcomes=(
                HabitatOutcome("trend", 0.4, 10),
            ),
        ),
        "org_2": OrganismFitness(
            organism_id="org_2",
            expectancy_r=0.3,
            max_drawdown_pct=2.0,
            complexity=3,
            novelty_score=0.1,
            habitat_outcomes=(
                HabitatOutcome("trend", 0.6, 30),
            ),
        ),
    }

    profiles = profile_species_habitats((group,), fitness)

    assert dict(profiles[0].habitat_expectancy)["trend"] == pytest.approx(0.55)


def test_species_habitat_profile_rejects_missing_member_fitness():
    group = Species(
        species_id="species_missing",
        founder_organism_id="org_1",
        founder_genome=None,
        member_ids=("org_1",),
        origin_generation=0,
        last_active_generation=0,
    )

    with pytest.raises(ValueError, match="Missing fitness records"):
        profile_species_habitats((group,), {})

def test_habitat_guided_migration_prefers_complementary_niches():
    population, _ = build_competing_population(distance_threshold=0.0)
    base_fitness = build_fitness(population)

    habitat_expectancies = (
        ("trend", 0.5),
        ("range", 0.4),
        ("trend", 0.8),
    )

    fitness = {}

    for organism, (habitat_id, expectancy) in zip(
        population.organisms,
        habitat_expectancies,
    ):
        original = base_fitness[organism.organism_id]

        fitness[organism.organism_id] = OrganismFitness(
            organism_id=original.organism_id,
            expectancy_r=original.expectancy_r,
            max_drawdown_pct=original.max_drawdown_pct,
            complexity=original.complexity,
            novelty_score=original.novelty_score,
            habitat_outcomes=(
                HabitatOutcome(habitat_id, expectancy, 20),
            ),
        )

    selection = select_ecological_survivors(
        population,
        speciate_population(population, distance_threshold=0.0)[1],
        fitness,
        capacity=3,
    )

    profiles = profile_species_habitats(
        selection.species,
        fitness,
    )

    next_population = reproduce_population(
        selection,
        target_size=12,
        feature_ranges=FEATURE_RANGES,
        rng=np.random.default_rng(123),
        mutation_rate=0.0,
        immigrant_rate=0.0,
        migration_rate=1.0,
        species_profiles=profiles,
    )

    survivors = {
        item.organism_id: item
        for item in selection.population.organisms
    }

    preferred_habitats = {
        profile.species_id: profile.preferred_habitat
        for profile in profiles
    }

    offspring = [
        item for item in next_population.organisms
        if item.organism_id not in survivors
    ]

    assert offspring

    for child in offspring:
        assert len(child.parent_ids) == 2

        parent_habitats = {
            preferred_habitats[survivors[parent_id].species_id]
            for parent_id in child.parent_ids
        }

        assert len(parent_habitats) == 2

def test_species_representatives_take_priority_over_habitat_champions():
    population, species = build_competing_population(
        distance_threshold=0.0,
    )
    assert len(species) == 3

    first, second, third = population.organisms

    fitness = {
        first.organism_id: OrganismFitness(
            organism_id=first.organism_id,
            expectancy_r=0.3,
            max_drawdown_pct=0.01,
            complexity=4,
            novelty_score=0.4,
        ),
        second.organism_id: OrganismFitness(
            organism_id=second.organism_id,
            expectancy_r=0.2,
            max_drawdown_pct=0.02,
            complexity=5,
            novelty_score=0.3,
        ),
        third.organism_id: OrganismFitness(
            organism_id=third.organism_id,
            expectancy_r=0.1,
            max_drawdown_pct=0.03,
            complexity=6,
            novelty_score=0.2,
            habitat_outcomes=(
                HabitatOutcome("range", 0.9, 10),
            ),
            temporal_habitat_persistence=(
                TemporalHabitatPersistence(
                    habitat_id="range",
                    period_expectancies=(
                        (0, 0.8, 5),
                        (1, 1.0, 5),
                    ),
                    qualified_periods=2,
                    profitable_periods=2,
                    persistence_ratio=1.0,
                    weighted_expectancy_r=0.9,
                    worst_period_expectancy_r=0.8,
                ),
            ),
        ),
    }

    result = select_ecological_survivors(
        population,
        species,
        fitness,
        capacity=2,
        minimum_habitat_trades=5,
    )
    survivor_ids = {
        organism.organism_id
        for organism in result.population.organisms
    }

    assert len(survivor_ids) == 2
    assert len(result.species) == 2
    assert len(result.extinct_species_ids) == 1

    species_champions = set(result.species_champion_ids)
    habitat_champions = set(result.habitat_champion_ids)
    global_competition = set(result.global_competition_ids)

    assert len(species_champions) == 1
    assert len(global_competition) == 1
    assert habitat_champions == set()

    assert species_champions.isdisjoint(habitat_champions)
    assert species_champions.isdisjoint(global_competition)
    assert habitat_champions.isdisjoint(global_competition)

    assert (
        species_champions
        | habitat_champions
        | global_competition
    ) == survivor_ids

    assert len({
        organism.species_id
        for organism in result.population.organisms
    }) == 2

def test_selection_preserves_habitat_champion_when_capacity_allows():
    population, species = build_competing_population(
        distance_threshold=1.0,
    )

    assert len(species) == 1

    first, second, third = population.organisms

    fitness = {
        first.organism_id: OrganismFitness(
            organism_id=first.organism_id,
            expectancy_r=0.3,
            max_drawdown_pct=0.01,
            complexity=4,
            novelty_score=0.4,
            habitat_outcomes=(
                HabitatOutcome("trend", 0.9, 10),
                HabitatOutcome("range", 0.8, 4),
            ),
        ),
        second.organism_id: OrganismFitness(
            organism_id=second.organism_id,
            expectancy_r=0.1,
            max_drawdown_pct=0.04,
            complexity=8,
            novelty_score=0.2,
            habitat_outcomes=(
                HabitatOutcome("range", 0.5, 10),
            ),
            temporal_habitat_persistence=(
                TemporalHabitatPersistence(
                    habitat_id="range",
                    period_expectancies=(
                        (0, 0.4, 5),
                        (1, 0.6, 5),
                    ),
                    qualified_periods=2,
                    profitable_periods=2,
                    persistence_ratio=1.0,
                    weighted_expectancy_r=0.5,
                    worst_period_expectancy_r=0.4,
                ),
            ),
        ),
        third.organism_id: OrganismFitness(
            organism_id=third.organism_id,
            expectancy_r=-0.2,
            max_drawdown_pct=0.08,
            complexity=8,
            novelty_score=0.1,
        ),
    }

    result = select_ecological_survivors(
        population,
        species,
        fitness,
        capacity=2,
        minimum_habitat_trades=5,
    )
    survivor_ids = {
        organism.organism_id
        for organism in result.population.organisms
    }

    assert first.organism_id in survivor_ids
    assert second.organism_id in survivor_ids
    assert third.organism_id not in survivor_ids

    assert len(result.species) == 1
    assert result.extinct_species_ids == ()

    species_champions = set(result.species_champion_ids)
    habitat_champions = set(result.habitat_champion_ids)
    global_competition = set(result.global_competition_ids)

    assert species_champions.isdisjoint(habitat_champions)
    assert species_champions.isdisjoint(global_competition)
    assert habitat_champions.isdisjoint(global_competition)

    assert (
        species_champions
        | habitat_champions
        | global_competition
    ) == survivor_ids

    assert len(species_champions) == 1
    assert len(habitat_champions) == 1
    assert len(global_competition) == 0

    assert first.organism_id in species_champions
    assert second.organism_id in habitat_champions

def test_temporal_habitat_performance_tracks_changing_expectancy():
    index = pd.date_range(
        "2025-01-01",
        periods=8,
        freq="4h",
        tz="UTC",
    )

    habitats = pd.DataFrame(
        {
            "habitat_id": ["trend"] * 8,
            "habitat_valid": [True] * 8,
        },
        index=index,
    )

    trades = pd.DataFrame({
        "entry_time": [
            index[1],
            index[2],
            index[5],
            index[6],
        ],
        "net_r": [0.4, 0.2, -0.3, -0.1],
    })

    result = measure_temporal_habitat_performance(
        trades,
        habitats,
        timeframe_seconds=14400,
        windows=2,
    )

    assert len(result.periods) == 2

    first, second = result.periods

    assert first.start_time_utc == index[0].isoformat()
    assert second.start_time_utc == index[4].isoformat()

    assert len(first.outcomes) == 1
    assert first.outcomes[0].habitat_id == "trend"
    assert first.outcomes[0].expectancy_r == pytest.approx(0.3)
    assert first.outcomes[0].trade_count == 2

    assert len(second.outcomes) == 1
    assert second.outcomes[0].habitat_id == "trend"
    assert second.outcomes[0].expectancy_r == pytest.approx(-0.2)
    assert second.outcomes[0].trade_count == 2

    assert result.total_trades == 4
    assert result.unassigned_trades == 0


def test_temporal_habitat_performance_accounts_for_unmatched_trades():
    index = pd.date_range(
        "2025-01-01",
        periods=8,
        freq="4h",
        tz="UTC",
    )

    habitats = pd.DataFrame(
        {
            "habitat_id": ["trend"] * 8,
            "habitat_valid": [True] * 8,
        },
        index=index,
    )

    trades = pd.DataFrame({
        "entry_time": [
            index[1],
            index[-1] + pd.Timedelta(hours=4),
        ],
        "net_r": [0.3, 0.2],
    })

    result = measure_temporal_habitat_performance(
        trades,
        habitats,
        timeframe_seconds=14400,
        windows=2,
    )

    assert result.total_trades == 2
    assert result.unassigned_trades == 1
    assert sum(
        period.classified_trades
        for period in result.periods
    ) == 1


def test_temporal_habitat_performance_requires_multiple_windows():
    index = pd.date_range(
        "2025-01-01",
        periods=8,
        freq="4h",
        tz="UTC",
    )

    habitats = pd.DataFrame(
        {
            "habitat_id": ["trend"] * 8,
            "habitat_valid": [True] * 8,
        },
        index=index,
    )

    with pytest.raises(ValueError, match="windows must be >= 2"):
        measure_temporal_habitat_performance(
            pd.DataFrame(columns=["entry_time", "net_r"]),
            habitats,
            timeframe_seconds=14400,
            windows=1,
        )

def test_temporal_habitat_profile_measures_persistence():
    periods = (
        TemporalHabitatPeriod(
            0, "2025-01-01T00:00:00+00:00",
            "2025-01-01T12:00:00+00:00",
            (HabitatOutcome("trend", 0.3, 10),), 10, 0,
        ),
        TemporalHabitatPeriod(
            1, "2025-01-02T00:00:00+00:00",
            "2025-01-02T12:00:00+00:00",
            (HabitatOutcome("trend", -0.2, 10),), 10, 0,
        ),
        TemporalHabitatPeriod(
            2, "2025-01-03T00:00:00+00:00",
            "2025-01-03T12:00:00+00:00",
            (HabitatOutcome("trend", 0.1, 5),), 5, 0,
        ),
    )

    summary = TemporalHabitatPerformanceSummary(
        periods=periods,
        total_trades=25,
        unassigned_trades=0,
    )

    profiles = profile_temporal_habitats(
        summary,
        minimum_period_trades=5,
    )

    profile = profiles[0]

    assert profile.habitat_id == "trend"
    assert profile.qualified_periods == 3
    assert profile.profitable_periods == 2
    assert profile.persistence_ratio == pytest.approx(2 / 3)
    assert profile.weighted_expectancy_r == pytest.approx(0.06)
    assert profile.worst_period_expectancy_r == pytest.approx(-0.2)


def test_temporal_habitat_profile_excludes_insufficient_samples():
    periods = (
        TemporalHabitatPeriod(
            0, "2025-01-01T00:00:00+00:00",
            "2025-01-01T12:00:00+00:00",
            (HabitatOutcome("range", 0.8, 2),), 2, 0,
        ),
        TemporalHabitatPeriod(
            1, "2025-01-02T00:00:00+00:00",
            "2025-01-02T12:00:00+00:00",
            (HabitatOutcome("range", -0.2, 8),), 8, 0,
        ),
    )

    summary = TemporalHabitatPerformanceSummary(
        periods=periods,
        total_trades=10,
        unassigned_trades=0,
    )

    profile = profile_temporal_habitats(
        summary,
        minimum_period_trades=5,
    )[0]

    assert profile.qualified_periods == 1
    assert profile.profitable_periods == 0
    assert profile.persistence_ratio == 0.0
    assert profile.weighted_expectancy_r == pytest.approx(-0.2)
    assert profile.worst_period_expectancy_r == pytest.approx(-0.2)

def test_temporal_persistence_is_seventh_pareto_objective():
    stable = OrganismFitness(
        organism_id="stable",
        expectancy_r=0.15,
        max_drawdown_pct=0.03,
        complexity=3,
        novelty_score=0.7,
        temporal_persistence_score=0.8,
    )

    unstable = OrganismFitness(
        organism_id="unstable",
        expectancy_r=0.15,
        max_drawdown_pct=0.03,
        complexity=3,
        novelty_score=0.7,
        temporal_persistence_score=0.3,
    )

    assert len(stable.objectives()) == 7
    assert stable.objectives()[-1] == 0.8
    assert unstable.objectives()[-1] == 0.3

    assert pareto_dominates(
        stable.objectives(),
        unstable.objectives(),
    )


def test_temporal_persistence_rejects_invalid_scores():
    with pytest.raises(ValueError, match="between 0 and 1"):
        OrganismFitness(
            organism_id="invalid",
            expectancy_r=0.1,
            max_drawdown_pct=0.02,
            complexity=2,
            novelty_score=0.5,
            temporal_persistence_score=1.5,
        )
def test_habitat_champion_requires_multiple_profitable_periods():
    population, species = build_competing_population(
        distance_threshold=1.0,
    )

    first, second, third = population.organisms

    fitness = {
        first.organism_id: OrganismFitness(
            organism_id=first.organism_id,
            expectancy_r=0.3,
            max_drawdown_pct=0.01,
            complexity=4,
            novelty_score=0.4,
        ),
        second.organism_id: OrganismFitness(
            organism_id=second.organism_id,
            expectancy_r=0.1,
            max_drawdown_pct=0.04,
            complexity=8,
            novelty_score=0.2,
            habitat_outcomes=(
                HabitatOutcome("range", 0.5, 10),
            ),
            temporal_habitat_persistence=(
                TemporalHabitatPersistence(
                    habitat_id="range",
                    period_expectancies=(
                        (0, 0.8, 5),
                        (1, -0.1, 5),
                    ),
                    qualified_periods=2,
                    profitable_periods=1,
                    persistence_ratio=0.5,
                    weighted_expectancy_r=0.35,
                    worst_period_expectancy_r=-0.1,
                ),
            ),
        ),
        third.organism_id: OrganismFitness(
            organism_id=third.organism_id,
            expectancy_r=-0.2,
            max_drawdown_pct=0.08,
            complexity=8,
            novelty_score=0.1,
        ),
    }

    result = select_ecological_survivors(
        population,
        species,
        fitness,
        capacity=2,
        minimum_habitat_trades=5,
    )

    assert second.organism_id not in result.habitat_champion_ids
def test_temporal_habitat_performance_handles_zero_trades():
    from edge.ecology import measure_temporal_habitat_performance

    index = pd.date_range(
        "2025-01-01",
        periods=8,
        freq="4h",
        tz="UTC",
    )

    habitat_frame = pd.DataFrame({
        "habitat_id": ["range"] * 8,
        "habitat_valid": [True] * 8,
    }, index=index)

    result = measure_temporal_habitat_performance(
        pd.DataFrame(),
        habitat_frame,
        timeframe_seconds=4 * 60 * 60,
        windows=4,
    )

    assert result.total_trades == 0
    assert result.unassigned_trades == 0
    assert len(result.periods) == 4

    for period in result.periods:
        assert period.classified_trades == 0
        assert period.unclassified_trades == 0
        assert len(period.outcomes) == 1
        assert period.outcomes[0].habitat_id == "range"
        assert period.outcomes[0].trade_count == 0
        assert period.outcomes[0].expectancy_r is None
