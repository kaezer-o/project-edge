from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edge.config import ExecutionConfig, RiskConfig

from scripts import audit_candidate_redundancy as audit_module
from scripts.audit_candidate_redundancy import (
    _duplicate_groups,
    _pair_summary,
    build_redundancy_audit,
)


def test_duplicate_groups_only_returns_repeated_fingerprints():
    rows = [
        {"rank": 1, "signal": "same"},
        {"rank": 2, "signal": "same"},
        {"rank": 3, "signal": "unique"},
        {"rank": 4, "signal": "same"},
    ]
    assert _duplicate_groups(rows, "signal") == [
        {"fingerprint": "same", "ranks": [1, 2, 4], "size": 3}
    ]


def test_pair_summary_handles_empty_pair_list():
    result = _pair_summary(
        [], signal_field="training_signal_similarity",
        trade_field="training_trade_overlap",
    )
    assert result["pair_count"] == 0
    assert result["mean_active_signal_jaccard"] == 0.0
    assert result["mean_trade_entry_jaccard"] == 0.0


def _report(*, final_test=False):
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
        "candidates": [
            {"program": {}},
            {"program": {}},
            {"program": {}},
        ],
    }


def test_intra_run_audit_counts_exact_signal_and_trade_duplicates(monkeypatch):
    def fake_candidate(seed, rank, candidate, train, validation, **kwargs):
        signal = np.array([0, 1, -1, 0], dtype=np.int8)
        validation_signal = np.array([0, 1, 0], dtype=np.int8)
        if rank == 3:
            signal = np.array([0, 0, -1, 1], dtype=np.int8)
            validation_signal = np.array([0, 0, -1], dtype=np.int8)
        events = {"entry-a"} if rank in (1, 2) else {"entry-b"}
        return {
            "seed": seed,
            "rank": rank,
            "validation_gate_pass": rank in (1, 3),
            "training": {"signal": signal, "trade_events": events},
            "validation": {"signal": validation_signal, "trade_events": events},
        }

    monkeypatch.setattr(audit_module, "_candidate_behaviour", fake_candidate)
    monkeypatch.setattr(audit_module, "_classify_strategy_families", lambda program: ["other"])
    monkeypatch.setattr(
        audit_module.StrategyProgram,
        "from_dict",
        staticmethod(lambda value: object()),
    )
    report = _report()
    audit = build_redundancy_audit(
        [(2033, report)],
        pd.DataFrame(index=range(4)),
        pd.DataFrame(index=range(3)),
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="bid",
    )
    summary = audit["per_seed"]["2033"]
    assert audit["scope"]["within_seed_pair_count"] == 3
    assert summary["candidate_count"] == 3
    assert summary["validation_qualified_count"] == 2
    assert summary["unique_training_signal_streams"] == 2
    assert summary["unique_validation_signal_streams"] == 2
    assert summary["exact_duplicate_training_signal_groups"][0]["ranks"] == [1, 2]
    assert summary["exact_duplicate_training_trade_groups"][0]["ranks"] == [1, 2]
    duplicate_pair = next(
        row for row in summary["pairs"]
        if row["rank_a"] == 1 and row["rank_b"] == 2
    )
    assert duplicate_pair["exact_training_signal_match"] is True
    assert duplicate_pair["exact_validation_trade_match"] is True
    assert duplicate_pair["both_validation_qualified"] is False
    assert audit["scope"]["final_test_evaluated"] is False


def test_intra_run_audit_rejects_final_test_evaluation(monkeypatch):
    with pytest.raises(ValueError, match="final-test evaluation"):
        build_redundancy_audit(
            [(1, _report()), (2, _report(final_test=True))],
            pd.DataFrame(index=range(4)),
            pd.DataFrame(index=range(3)),
            execution=ExecutionConfig(),
            risk=RiskConfig(),
            pip_size=0.0001,
            price_basis="bid",
        )


def test_intra_run_audit_rejects_mismatched_partition_metadata():
    first, second = _report(), _report()
    second["data"]["train_bars"] = 5
    with pytest.raises(ValueError, match="partition metadata"):
        build_redundancy_audit(
            [(1, first), (2, second)],
            pd.DataFrame(index=range(4)),
            pd.DataFrame(index=range(3)),
            execution=ExecutionConfig(),
            risk=RiskConfig(),
            pip_size=0.0001,
            price_basis="bid",
        )
