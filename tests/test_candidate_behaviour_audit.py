from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edge.config import ExecutionConfig, RiskConfig

from scripts import audit_candidate_behaviour as audit_module
from scripts.audit_candidate_behaviour import (
    build_behaviour_audit,
    signal_similarity,
    trade_entry_events,
    trade_overlap,
)


def test_signal_similarity_distinguishes_long_short_and_inactive_bars():
    left = np.array([0, 1, 1, -1, 0, -1], dtype=np.int8)
    right = np.array([0, 1, -1, -1, 1, 0], dtype=np.int8)
    result = signal_similarity(left, right)
    assert result["bars"] == 6
    assert result["long_signal_jaccard"] == pytest.approx(1 / 3)
    assert result["short_signal_jaccard"] == pytest.approx(1 / 3)
    assert result["active_signal_jaccard"] == pytest.approx(3 / 5)
    assert result["directional_conflict_rate"] == pytest.approx(1 / 6)


def test_signal_similarity_does_not_treat_two_inactive_streams_as_matching_activity():
    result = signal_similarity(np.zeros(5, dtype=np.int8), np.zeros(5, dtype=np.int8))
    assert result["exact_signal_agreement"] == 1.0
    assert result["active_signal_jaccard"] == 0.0


def test_signal_similarity_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="identically sized"):
        signal_similarity(np.array([0, 1]), np.array([0]))


def test_trade_entry_events_normalize_timezone_and_direction():
    trades = pd.DataFrame({
        "entry_time": [pd.Timestamp("2025-01-01T00:00:00Z")],
        "direction": ["LONG"],
    })
    assert trade_entry_events(trades) == {"2025-01-01T00:00:00+00:00|long"}


def test_trade_overlap_handles_empty_sets():
    assert trade_overlap(set(), set())["entry_jaccard"] == 0.0
    assert trade_overlap({"a"}, set())["entry_jaccard"] == 0.0


def _report(seed, *, final_test=False):
    return {
        "symbol": "EURUSD",
        "timeframe": "H4",
        "source_timeframe": "M1",
        "data": {
            "train_bars": 4,
            "validation_bars": 3,
            "final_test_bars": 3,
            "final_test_evaluated": final_test,
        },
        "candidates": [],
    }


def test_audit_rejects_reports_that_evaluated_final_test():
    with pytest.raises(ValueError, match="final-test evaluation"):
        build_behaviour_audit(
            [(1, _report(1)), (2, _report(2, final_test=True))],
            pd.DataFrame(index=range(4)),
            pd.DataFrame(index=range(3)),
            execution=ExecutionConfig(),
            risk=RiskConfig(),
            pip_size=0.0001,
            price_basis="bid",
        )


def test_audit_rejects_different_partitions():
    first, second = _report(1), _report(2)
    second["data"]["validation_bars"] = 4
    with pytest.raises(ValueError, match="partition metadata"):
        build_behaviour_audit(
            [(1, first), (2, second)],
            pd.DataFrame(index=range(4)),
            pd.DataFrame(index=range(3)),
            execution=ExecutionConfig(),
            risk=RiskConfig(),
            pip_size=0.0001,
            price_basis="bid",
        )


def test_audit_builds_cross_seed_signal_and_trade_comparisons(monkeypatch):
    first, second = _report(1), _report(2)
    first["candidates"] = [{}]
    second["candidates"] = [{}]

    def fake_candidate(seed, rank, candidate, train, validation, **kwargs):
        signal = np.array([0, 1, -1, 0], dtype=np.int8) if seed == 1 else np.array([0, 1, 0, -1], dtype=np.int8)
        validation_signal = np.array([0, 1, 0], dtype=np.int8) if seed == 1 else np.array([0, 0, -1], dtype=np.int8)
        return {
            "seed": seed,
            "rank": rank,
            "validation_gate_pass": False,
            "rule_fingerprint": f"rule-{seed}",
            "risk_parameters": {"stop_atr": 2.0, "reward_risk": 2.0, "max_holding_bars": 20},
            "training": {"metrics": {"trades": 2}, "signal": signal, "trade_events": {"entry-a"}},
            "validation": {"metrics": {"trades": 1}, "signal": validation_signal, "trade_events": {"entry-v"}},
        }

    monkeypatch.setattr(audit_module, "_candidate_behaviour", fake_candidate)
    audit = build_behaviour_audit(
        [(1, first), (2, second)],
        pd.DataFrame(index=range(4)),
        pd.DataFrame(index=range(3)),
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="bid",
    )
    assert audit["scope"]["cross_seed_pair_count"] == 1
    pair = audit["cross_seed_pairs"][0]
    assert pair["training_signal_similarity"]["long_signal_jaccard"] == 1.0
    assert pair["training_trade_overlap"]["entry_jaccard"] == 1.0
    assert pair["validation_trade_overlap"]["entry_jaccard"] == 1.0
