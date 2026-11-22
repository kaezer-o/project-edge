from __future__ import annotations

import pytest

from scripts.audit_candidate_reproducibility import (
    build_audit,
    classify_motifs,
    normalize_expression,
)


def feature(name):
    return {"op": "feature", "value": name, "args": []}


def const(value):
    return {"op": "const", "value": value, "args": []}


def node(op, *args, value=None):
    return {"op": op, "value": value, "args": list(args)}


def test_normalization_ignores_threshold_values_but_preserves_features():
    first = node("gt", feature("rsi_14"), const(30))
    second = node("gt", feature("rsi_14"), const(70))
    assert normalize_expression(first) == normalize_expression(second)
    assert normalize_expression(first) != normalize_expression(
        node("gt", feature("ema_20"), const(30))
    )


def test_normalization_orders_commutative_operands():
    left = node("and", node("gt", feature("close"), feature("ema_20")),
                node("lt", feature("rsi_14"), const(40)))
    right = node("and", node("lt", feature("rsi_14"), const(75)),
                 node("gt", feature("close"), feature("ema_20")))
    assert normalize_expression(left) == normalize_expression(right)


def test_motif_classifier_reports_side_specific_motifs():
    long_rule = node(
        "and",
        node("cross_above", feature("rsi_14"), const(30)),
        node("gt", feature("close"), feature("ema_20")),
        node("gt", feature("atr_14"), node("rolling_mean", feature("atr_14"), value=20)),
    )
    short_rule = node("gt", feature("ema_10"), feature("ema_40"))
    result = classify_motifs({"long_entry": long_rule, "short_entry": short_rule})
    assert "rsi_threshold_cross" in result["long_entry"]
    assert "price_vs_ema_filter" in result["long_entry"]
    assert "atr_vs_rolling_atr" in result["long_entry"]
    assert "ema_direction_with_atr_regime_gate" in result["long_entry"]
    assert "ema_pair_comparison" in result["short_entry"]


def _candidate(*, threshold=30, entry_fingerprint="same-event-hash", passed=True):
    return {
        "fingerprint": f"genome-{threshold}",
        "program": {
            "long_entry": node("cross_above", feature("rsi_14"), const(threshold)),
            "short_entry": node("cross_below", feature("rsi_14"), const(70)),
            "stop_atr": 2.0,
            "reward_risk": 2.0,
            "max_holding_bars": 24,
        },
        "validation": {
            "pass": passed,
            "passed_windows": 2,
            "metrics": {"trades": 40, "expectancy_r": 0.1, "profit_factor": 1.2},
            "trade_event_diagnostics": {"entry_event_fingerprint": entry_fingerprint},
            "one_and_half_x_cost_pass": passed,
            "cost_stress": [{"cost_multiplier": 2, "pass": passed}],
        },
    }


def _report(seed, candidate, *, final_test=False):
    return {
        "symbol": "EURUSD",
        "timeframe": "H4",
        "data": {
            "train_bars": 4703,
            "validation_bars": 1568,
            "final_test_bars": 1568,
            "final_test_evaluated": final_test,
        },
        "status": "validated_candidate",
        "candidate_count": 1,
        "validated_candidate_count": int(candidate["validation"]["pass"]),
        "candidates": [candidate],
        "runtime": {
            "generation": 30,
            "history": [{
                "unique_genomes": 128,
                "unique_behaviours": 128,
                "unique_signal_streams": 60,
                "strategy_family_class_population_counts": {"trend": 20},
                "population_motif_diagnostics": {"motif_population_counts": {"rsi_threshold_cross": 80}},
            }],
        },
    }


def test_audit_detects_exact_rule_and_event_matches_across_seeds():
    audit = build_audit([
        (1, _report(1, _candidate(threshold=30))),
        (2, _report(2, _candidate(threshold=70))),
    ])
    assert len(audit["exact_cross_seed_rule_matches"]) == 1
    assert len(audit["exact_cross_seed_validation_entry_matches"]) == 1
    assert audit["scope"]["final_test_evaluated"] is False
    assert audit["candidate_records"][0]["rule_structure_fingerprint"] == (
        audit["candidate_records"][1]["rule_structure_fingerprint"]
    )


def test_audit_does_not_claim_approximate_event_overlap():
    audit = build_audit([
        (1, _report(1, _candidate(entry_fingerprint="hash-a"))),
        (2, _report(2, _candidate(entry_fingerprint="hash-b"))),
    ])
    assert audit["exact_cross_seed_validation_entry_matches"] == []
    assert any("approximate cross-seed Jaccard" in item for item in audit["limitations"])


def test_audit_rejects_final_test_evaluation():
    with pytest.raises(ValueError, match="final-test evaluation"):
        build_audit([
            (1, _report(1, _candidate())),
            (2, _report(2, _candidate(), final_test=True)),
        ])


def test_audit_rejects_different_data_partitions():
    other = _report(2, _candidate())
    other["data"]["validation_bars"] += 1
    with pytest.raises(ValueError, match="partition metadata"):
        build_audit([(1, _report(1, _candidate())), (2, other)])
