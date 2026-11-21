import json

import pytest

from edge.training_checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)


def sample_state():
    return {
        "run_signature": "eurusd-h4-datahash-confighash",
        "generation": 17,
        "population": [{"version": 1, "fingerprint": "abc"}],
        "rng_state": {"bit_generator": "PCG64", "state": {"state": 123, "inc": 456}},
        "history": [{"generation": 17, "best_fitness": 0.25}],
    }


def test_checkpoint_round_trip(tmp_path):
    state = sample_state()
    path = save_training_checkpoint(tmp_path / "state.json", **state)

    restored = load_training_checkpoint(
        path, expected_run_signature=state["run_signature"]
    )

    assert restored["generation"] == state["generation"]
    assert restored["population"] == state["population"]
    assert restored["rng_state"] == state["rng_state"]
    assert restored["history"] == state["history"]
    assert restored["checksum_sha256"]


def test_checkpoint_rejects_different_run_signature(tmp_path):
    state = sample_state()
    path = save_training_checkpoint(tmp_path / "state.json", **state)

    with pytest.raises(ValueError, match="run signature"):
        load_training_checkpoint(path, expected_run_signature="different-data")


def test_checkpoint_detects_tampering(tmp_path):
    state = sample_state()
    path = save_training_checkpoint(tmp_path / "state.json", **state)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["population"][0]["fingerprint"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_training_checkpoint(
            path, expected_run_signature=state["run_signature"]
        )


@pytest.mark.parametrize("generation", [-1, True, 1.5])
def test_checkpoint_rejects_invalid_generation(tmp_path, generation):
    state = sample_state()
    state["generation"] = generation
    with pytest.raises(ValueError, match="generation"):
        save_training_checkpoint(tmp_path / "state.json", **state)


def test_checkpoint_rejects_empty_population(tmp_path):
    state = sample_state()
    state["population"] = []
    with pytest.raises(ValueError, match="population"):
        save_training_checkpoint(tmp_path / "state.json", **state)


def test_checkpoint_rejects_non_finite_json_values(tmp_path):
    state = sample_state()
    state["history"] = [{"score": float("nan")}]
    with pytest.raises(ValueError):
        save_training_checkpoint(tmp_path / "state.json", **state)
