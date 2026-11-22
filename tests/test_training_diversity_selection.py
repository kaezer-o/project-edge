from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_training_diversity_selection import (
    build_selection_audit,
    select_training_diverse,
)


def test_training_selection_keeps_highest_fitness_representative_of_exact_duplicates():
    candidates = [
        {"rank": 1, "training_fitness": 1.0},
        {"rank": 2, "training_fitness": 2.0},
        {"rank": 3, "training_fitness": 0.5},
    ]
    signals = {
        1: np.array([0, 1, -1, 0]),
        2: np.array([0, 1, -1, 0]),
        3: np.array([0, 0, 1, -1]),
    }
    result = select_training_diverse(candidates, signals, similarity_threshold=1.0)
    assert [row["rank"] for row in result["selected"]] == [2, 3]
    assert [row["rank"] for row in result["rejected_as_redundant"]] == [1]


def test_training_selection_respects_active_signal_similarity_threshold():
    candidates = [
        {"rank": 1, "training_fitness": 3.0},
        {"rank": 2, "training_fitness": 2.0},
        {"rank": 3, "training_fitness": 1.0},
    ]
    signals = {
        1: np.array([1, 1, 1, 0]),
        2: np.array([1, 1, 0, 0]),
        3: np.array([-1, 0, 0, -1]),
    }
    result = select_training_diverse(candidates, signals, similarity_threshold=0.65)
    assert [row["rank"] for row in result["selected"]] == [1, 3]
    assert result["rejected_as_redundant"][0]["rank"] == 2


def test_training_selection_limit_is_respected():
    candidates = [{"rank": i, "training_fitness": float(10-i)} for i in range(1, 5)]
    signals = {i: np.eye(4, dtype=np.int8)[i - 1] for i in range(1, 5)}
    result = select_training_diverse(candidates, signals, limit=2)
    assert result["selected_count"] == 2


def test_training_selection_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="similarity_threshold"):
        select_training_diverse([], {}, similarity_threshold=1.1)


def test_selection_audit_rejects_final_test_evaluated_report():
    report = {
        "symbol": "EURUSD",
        "timeframe": "H4",
        "source_timeframe": "M1",
        "data": {
            "train_bars": 4,
            "validation_bars": 3,
            "final_test_bars": 2,
            "final_test_evaluated": True,
        },
    }
    with pytest.raises(ValueError, match="final-test evaluation"):
        build_selection_audit(
            [(1, report)],
            object(),
        )
