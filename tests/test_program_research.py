import json

import pytest
import numpy as np
import pandas as pd

from edge.config import ExecutionConfig, RiskConfig
from edge.program_research import (
    _combine_training_fitness,
    _diverse_ranked,
    _fitness,
    _minimum_window_trades,
    _program_structure,
    _validation_check,
    run_program_research,
)


def _market_frame(rows: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(73)
    index = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    close = 1.10 + np.cumsum(rng.normal(0.0, 0.00015, rows))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + rng.uniform(0.00002, 0.00012, rows),
            "low": np.minimum(open_, close) - rng.uniform(0.00002, 0.00012, rows),
            "close": close,
        },
        index=index,
    )


def test_program_research_runs_evolution_and_writes_reproducible_report(tmp_path):
    report = run_program_research(
        _market_frame(),
        timeframe="M15",
        source_timeframe="M15",
        symbol="EURUSD",
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="mid",
        population_size=8,
        generations=2,
        seed=17,
        min_train_trades=1,
        min_validation_trades=1,
        output_dir=tmp_path,
    )

    path = tmp_path / "program_evolution_report.json"
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["engine"] == "open_ended_expression_program_evolution"
    assert saved["evolution"]["population_size"] == 8
    assert len(saved["evolution"]["generation_history"]) == 2
    assert "best_training_robustness" in saved["evolution"]["generation_history"][0]
    assert "unique_rule_structures" in saved["evolution"]["generation_history"][0]
    assert saved["candidate_count"] > 0
    assert saved["data"]["train_bars"] == 900
    assert saved["data"]["validation_bars"] == 300
    assert saved["data"]["final_test_bars"] == 300
    assert saved["final_test"] is None
    assert saved["final_test_skipped"] is True
    for candidate in saved["candidates"]:
        assert candidate["program"]["version"] == 1
        assert candidate["fingerprint"]


def test_program_research_rejects_target_timeframe_below_source(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="lower than source"):
        run_program_research(
            _market_frame(),
            timeframe="M5",
            source_timeframe="M15",
            symbol="EURUSD",
            execution=ExecutionConfig(),
            risk=RiskConfig(),
            pip_size=0.0001,
            price_basis="mid",
            population_size=8,
            generations=2,
            output_dir=tmp_path,
        )



def test_training_robust_fitness_penalizes_weak_regimes():
    consistent = _combine_training_fitness(2.0, [2.0, 2.0, 2.0])
    concentrated = _combine_training_fitness(3.0, [3.0, 2.0, -1.0])
    insufficient = _combine_training_fitness(3.0, [3.0, 2.0, -1e9])

    assert consistent > concentrated
    assert insufficient == -1e9


def test_diverse_ranking_limits_threshold_only_variants():
    from edge.program import Expr, StrategyProgram

    def program(threshold, feature="ema_gap_atr"):
        return StrategyProgram(
            long_entry=Expr(
                "gt",
                (Expr("feature", value=feature), Expr("const", value=threshold)),
            ),
            short_entry=Expr(
                "lt",
                (Expr("feature", value="rsi_14"), Expr("const", value=30.0)),
            ),
        )

    first = program(0.5)
    threshold_variant = program(1.0)
    different_structure = program(0.5, feature="price_change_atr")
    rows = [
        (3.0, first, {}, {}),
        (2.0, threshold_variant, {}, {}),
        (1.0, different_structure, {}, {}),
    ]

    selected = _diverse_ranked(rows, 3, max_per_structure=1)

    assert len(selected) == 2
    assert _program_structure(selected[0][1]) != _program_structure(selected[1][1])



def test_training_minimum_trades_are_distributed_across_windows():
    assert _minimum_window_trades(30, 3) == 10
    assert _minimum_window_trades(15, 3) == 5
    assert _minimum_window_trades(1, 3) == 5


def test_training_fitness_reuses_precomputed_aggregate_result(monkeypatch):
    import edge.program_research as research
    from edge.program import Expr, StrategyProgram

    frame = pd.DataFrame(
        {"ema_gap_atr": np.zeros(90), "rsi_14": np.full(90, 50.0)},
        index=pd.date_range("2025-01-01", periods=90, freq="h", tz="UTC"),
    )
    program = StrategyProgram(
        long_entry=Expr("gt", (Expr("feature", value="ema_gap_atr"), Expr("const", value=0.0))),
        short_entry=Expr("lt", (Expr("feature", value="rsi_14"), Expr("const", value=30.0))),
    )
    aggregate_metrics = {
        "trades": 30, "expectancy_r": 0.1,
        "max_realized_drawdown_pct": 0.01, "profit_factor": 1.2,
    }
    calls = []

    def fake_evaluate(candidate_frame, candidate_program, **kwargs):
        calls.append(len(candidate_frame))
        return aggregate_metrics, pd.DataFrame()

    monkeypatch.setattr(research, "_evaluate", fake_evaluate)
    score, metrics, robustness = research._training_fitness(
        frame, program, execution=ExecutionConfig(), risk=RiskConfig(),
        pip_size=0.0001, price_basis="mid", min_train_trades=3,
        aggregate_result=(aggregate_metrics, pd.DataFrame()),
    )
    assert len(calls) == 3
    assert calls == [30, 30, 30]
    assert metrics is aggregate_metrics
    assert robustness["aggregate_fitness"] > -1e8
    assert score > -1e8




def test_validation_reports_two_of_three_pass_separately_from_strict_pass(monkeypatch):
    import edge.program_research as research
    from edge.program import Expr, StrategyProgram

    index = pd.date_range("2025-01-01", periods=90, freq="h", tz="UTC")
    frame = pd.DataFrame({"close": np.ones(90)}, index=index)
    program = StrategyProgram(
        long_entry=Expr("gt", (Expr("feature", value="ema_gap_atr"), Expr("const", value=0.0))),
        short_entry=Expr("lt", (Expr("feature", value="rsi_14"), Expr("const", value=30.0))),
    )

    def fake_evaluate(candidate_frame, candidate_program, *, execution, risk, pip_size, price_basis):
        is_third_window = len(candidate_frame) == 30 and candidate_frame.index[0] == index[60]
        metrics = {
            "trades": 10,
            "expectancy_r": -0.1 if is_third_window else 0.1,
            "profit_factor": 0.8 if is_third_window else 1.2,
            "max_realized_drawdown_pct": 0.01,
        }
        return metrics, pd.DataFrame()

    monkeypatch.setattr(research, "_evaluate", fake_evaluate)
    result = _validation_check(
        frame,
        program,
        execution=ExecutionConfig(spread_pips=0.8),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="mid",
        min_trades=5,
    )

    assert result["passed_windows"] == 2
    assert result["pass"] is True
    assert result["strict_pass"] is False


def test_cost_stress_report_explains_trade_set_changes(monkeypatch):
    import edge.program_research as research
    from edge.program import Expr, StrategyProgram

    index = pd.date_range("2025-01-01", periods=30, freq="4h", tz="UTC")
    frame = pd.DataFrame({"close": np.ones(30)}, index=index)
    program = StrategyProgram(
        long_entry=Expr(
            "gt", (Expr("feature", value="ema_gap_atr"), Expr("const", value=0.0))
        ),
        short_entry=Expr(
            "lt", (Expr("feature", value="rsi_14"), Expr("const", value=30.0))
        ),
    )

    def fake_evaluate(candidate_frame, candidate_program, *, execution, risk, pip_size, price_basis):
        stressed = execution.spread_pips > 0.8
        entry_positions = (1, 3) if stressed else (1, 2)
        trades = pd.DataFrame({
            "entry_time": [index[position] for position in entry_positions],
            "direction": ["long", "short"],
        })
        metrics = {
            "trades": 2,
            "expectancy_r": 0.12 if stressed else 0.10,
            "profit_factor": 1.2 if stressed else 1.1,
            "max_realized_drawdown_pct": 0.01,
        }
        return metrics, trades

    monkeypatch.setattr(research, "_evaluate", fake_evaluate)
    result = _validation_check(
        frame,
        program,
        execution=ExecutionConfig(spread_pips=0.8),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="bid",
        min_trades=1,
    )

    diagnostic = result["cost_stress"][0]["trade_set_diagnostics"]
    baseline_events = result["trade_event_diagnostics"]
    assert len(baseline_events["entry_event_fingerprint"]) == 64
    assert baseline_events["long_entries"] == 1
    assert baseline_events["short_entries"] == 1
    assert len(result["windows"]) == 3
    assert result["strict_pass"] is True
    for window in result["windows"]:
        assert len(window["trade_event_diagnostics"]["entry_event_fingerprint"]) == 64
        assert window["trade_event_diagnostics"]["long_entries"] == 1
        assert window["trade_event_diagnostics"]["short_entries"] == 1
    assert diagnostic["baseline_trade_count"] == 2
    assert diagnostic["stressed_trade_count"] == 2
    assert diagnostic["shared_entry_events"] == 1
    assert diagnostic["baseline_only_entry_events"] == 1
    assert diagnostic["stress_only_entry_events"] == 1
    assert diagnostic["entry_event_jaccard_similarity"] == pytest.approx(1 / 3)
    assert diagnostic["expectancy_delta_r"] == pytest.approx(0.02)
    assert diagnostic["profit_factor_delta"] == pytest.approx(0.1)



def test_training_fitness_keeps_non_finite_profit_factor_out_of_score():
    from edge.program import Expr, StrategyProgram

    program = StrategyProgram(
        long_entry=Expr("const", value=True),
        short_entry=Expr("const", value=False),
    )
    base = {
        "trades": 30,
        "expectancy_r": 0.1,
        "max_realized_drawdown_pct": 0.01,
    }
    missing = _fitness(base, program, 10)
    for invalid_profit_factor in (float("nan"), float("inf"), float("-inf"), -1.0):
        metrics = {**base, "profit_factor": invalid_profit_factor}
        assert _fitness(metrics, program, 10) == missing
    assert np.isfinite(missing)


def test_training_fitness_rejects_non_finite_core_metrics():
    from edge.program import Expr, StrategyProgram

    program = StrategyProgram(
        long_entry=Expr("const", value=True),
        short_entry=Expr("const", value=False),
    )
    for key in ("trades", "expectancy_r", "max_realized_drawdown_pct"):
        metrics = {
            "trades": 30,
            "expectancy_r": 0.1,
            "max_realized_drawdown_pct": 0.01,
            "profit_factor": 1.5,
        }
        metrics[key] = float("nan")
        assert _fitness(metrics, program, 10) == -1e9
