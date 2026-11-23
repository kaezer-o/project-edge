import json

import numpy as np
import pandas as pd

from edge.config import ExecutionConfig, RiskConfig
from edge.island_evolution import IslandEvolutionConfig
import edge.island_program_research as island_program_research
from edge.program import Expr, StrategyProgram
from edge.island_program_research import run_island_program_research


def _market_frame(rows: int = 750) -> pd.DataFrame:
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


def test_island_program_research_runs_and_resumes_without_final_test(tmp_path):
    config = IslandEvolutionConfig(
        islands=2,
        population_per_island=4,
        elite_count=1,
        tournament_size=2,
        migration_interval=1,
        migrants_per_island=1,
        archive_capacity=16,
        novelty_weight=0.25,
    )
    kwargs = {
        "raw_frame": _market_frame(),
        "timeframe": "M15",
        "source_timeframe": "M15",
        "symbol": "EURUSD",
        "execution": ExecutionConfig(),
        "risk": RiskConfig(),
        "pip_size": 0.0001,
        "price_basis": "mid",
        "seed": 17,
        "min_train_trades": 1,
        "min_validation_trades": 1,
        "checkpoint_path": tmp_path / "checkpoint.json",
        "output_dir": tmp_path / "report",
        "config": config,
    }

    first = run_island_program_research(
        **kwargs, generations=1, resume=False
    )
    assert first["runtime"]["generation"] == 1
    assert first["runtime"]["total_population"] == 8
    family_counts = first["runtime"]["initial_strategy_family_counts"]
    assert sum(family_counts.values()) == 8
    assert set(first["runtime"]["available_strategy_families"]) == {
        "trend", "mean_reversion", "breakout",
        "volatility_expansion", "volatility_contraction",
    }
    assert "close" in first["data"]["features"]
    assert first["final_test"] is None
    assert first["final_test_skipped"] is True
    assert first["finalist_selection"]["similarity_filter_enabled"] is False
    assert (tmp_path / "checkpoint.json").exists()

    resumed = run_island_program_research(
        **kwargs, generations=2, resume=True
    )
    assert resumed["runtime"]["generation"] == 2
    assert resumed["runtime"]["resumed"] is True
    assert resumed["data"]["train_bars"] == 450
    assert resumed["data"]["validation_bars"] == 150
    assert resumed["data"]["final_test_bars"] == 150
    assert resumed["final_test"] is None

    saved = json.loads(
        (tmp_path / "report" / "island_program_evolution_report.json").read_text()
    )
    assert saved["engine"] == "resumable_multi_island_program_evolution"
    assert saved["runtime"]["generation"] == 2
    assert len(saved["runtime"]["history"]) == 2


def test_island_program_resume_rejects_changed_configuration(tmp_path):
    config = IslandEvolutionConfig(
        islands=2, population_per_island=4, elite_count=1, tournament_size=2
    )
    kwargs = {
        "raw_frame": _market_frame(),
        "timeframe": "M15",
        "source_timeframe": "M15",
        "symbol": "EURUSD",
        "execution": ExecutionConfig(),
        "risk": RiskConfig(),
        "pip_size": 0.0001,
        "price_basis": "mid",
        "seed": 19,
        "min_train_trades": 1,
        "min_validation_trades": 1,
        "checkpoint_path": tmp_path / "checkpoint.json",
        "output_dir": tmp_path / "report",
        "config": config,
    }
    run_island_program_research(**kwargs, generations=1)
    changed = IslandEvolutionConfig(
        islands=2, population_per_island=4, elite_count=1, tournament_size=2,
        novelty_weight=0.5,
    )
    import pytest

    with pytest.raises(ValueError, match="run signature"):
        run_island_program_research(
            **{**kwargs, "config": changed}, generations=2, resume=True
        )



def test_program_structure_diagnostics_normalizes_constants_and_reports_asymmetry():
    from edge.island_program_research import _program_structure_diagnostics

    feature = lambda name: Expr("feature", value=name)
    first = StrategyProgram(
        long_entry=Expr("and", (
            Expr("cross_above", (feature("rsi_14"), Expr("const", value=30))),
            Expr("gt", (feature("close"), feature("ema_20"))),
        )),
        short_entry=Expr("cross_below", (feature("rsi_14"), Expr("const", value=70))),
    )
    second = StrategyProgram(
        long_entry=Expr("and", (
            Expr("cross_above", (feature("rsi_14"), Expr("const", value=35))),
            Expr("gt", (feature("close"), feature("ema_20"))),
        )),
        short_entry=Expr("cross_below", (feature("rsi_14"), Expr("const", value=65))),
    )

    first_result = _program_structure_diagnostics(first)
    second_result = _program_structure_diagnostics(second)

    assert first_result["fingerprint"] == second_result["fingerprint"]
    assert first_result["long_nodes"] == 7
    assert first_result["short_nodes"] == 3
    assert first_result["long_depth"] == 3
    assert first_result["short_depth"] == 2
    assert first_result["features"] == ["close", "ema_20", "rsi_14"]
    assert first_result["long_only_features"] == ["close", "ema_20"]
    assert first_result["short_only_features"] == []
    assert first_result["long_motifs"] == ["price_vs_ema_filter", "rsi_threshold_cross"]
    assert first_result["short_motifs"] == ["rsi_threshold_cross"]
    assert first_result["motifs"] == ["price_vs_ema_filter", "rsi_threshold_cross"]


def test_structural_motif_taxonomy_recognizes_volatility_and_donchian_patterns():
    from edge.island_program_research import _classify_structural_motifs

    feature = lambda name: Expr("feature", value=name)
    long_rule = Expr("and", (
        Expr("gt", (feature("close"), feature("ema_20"))),
        Expr("gt", (feature("atr_14"), Expr("rolling_mean", (feature("atr_14"),), value=30))),
    ))
    short_rule = Expr("lt", (feature("close"), feature("donchian_low_20")))

    assert _classify_structural_motifs(long_rule) == [
        "atr_vs_rolling_atr",
        "ema_direction_with_atr_regime_gate",
        "price_vs_ema_filter",
    ]
    assert _classify_structural_motifs(short_rule) == ["donchian_low_breakout"]


def test_behaviour_descriptor_uses_comparable_rate_scales(monkeypatch):
    frame = pd.DataFrame(
        {"open": np.ones(90),
         "high": np.ones(90), "low": np.ones(90), "close": np.ones(90)},
        index=pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC"),
    )
    program = StrategyProgram(
        long_entry=Expr("gt", (Expr("feature", value="close"), Expr("feature", value="open"))),
        short_entry=Expr("lt", (Expr("feature", value="close"), Expr("feature", value="open"))),
    )
    signals = pd.Series(0, index=frame.index, dtype="int8")
    signals.iloc[:10] = 1
    signals.iloc[30:40] = -1
    monkeypatch.setattr(island_program_research, "evaluate_program", lambda _program, _frame: signals)
    entries = pd.DatetimeIndex([frame.index[5], frame.index[35], frame.index[65]])
    monkeypatch.setattr(
        island_program_research,
        "_evaluate",
        lambda *_args, **_kwargs: ({"trades": 3}, pd.DataFrame({"entry_time": entries, "direction": ["long", "short", "long"]})),
    )

    descriptor = island_program_research._behaviour_descriptor(
        frame,
        program,
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="mid",
    )

    assert len(descriptor["rates"]) == 9
    assert descriptor["rates"][0] == 10 / 30
    assert descriptor["rates"][4] == 10 / 30
    assert descriptor["rates"][2] == 1 / 30
    assert descriptor["rates"][5] == 1 / 30
    assert descriptor["trades"] == [
        f"{entries[0].isoformat()}|long",
        f"{entries[1].isoformat()}|short",
        f"{entries[2].isoformat()}|long",
    ]
    assert len(descriptor["signal_fingerprint"]) == 64
    assert len(descriptor["behaviour_fingerprint"]) == 64
    assert descriptor["families"] == ["other"]
    assert descriptor["structure"]["long_nodes"] > 0
    assert descriptor["structure"]["short_nodes"] > 0
    assert len(descriptor["structure"]["fingerprint"]) == 64
    assert max(descriptor["rates"]) <= 1.0


def test_behaviour_descriptor_reuses_precomputed_signal_and_trades(monkeypatch):
    frame = pd.DataFrame(
        {"open": np.ones(30), "high": np.ones(30), "low": np.ones(30), "close": np.ones(30)},
        index=pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
    )
    program = StrategyProgram(
        long_entry=Expr("gt", (Expr("feature", value="close"), Expr("feature", value="open"))),
        short_entry=Expr("lt", (Expr("feature", value="close"), Expr("feature", value="open"))),
    )
    signal = pd.Series(0, index=frame.index, dtype="int8")
    signal.iloc[2] = 1
    trades = pd.DataFrame({"entry_time": [frame.index[3]], "direction": ["long"]})

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("precomputed results should prevent another evaluation/backtest")

    monkeypatch.setattr(island_program_research, "evaluate_program", unexpected_evaluation)
    monkeypatch.setattr(island_program_research, "_evaluate", unexpected_evaluation)
    descriptor = island_program_research._behaviour_descriptor(
        frame, program, execution=ExecutionConfig(), risk=RiskConfig(),
        pip_size=0.0001, price_basis="mid", signal=signal, trades=trades,
    )
    assert descriptor["trades"] == [f"{frame.index[3].isoformat()}|long"]



def test_semantic_family_classifier_recognizes_seed_templates():
    from edge.island_program_research import _classify_strategy_families
    from edge.program_evolution import strategy_family_program

    features = (
        "ema_10", "ema_20", "ema_40", "rsi_14", "close",
        "donchian_high_20", "donchian_low_20", "atr_14",
    )
    rng = np.random.default_rng(2026)
    expected = (
        "trend",
        "mean_reversion",
        "breakout",
        "volatility_expansion",
        "volatility_contraction",
    )
    for family in expected:
        program = strategy_family_program(family, features, rng)
        assert _classify_strategy_families(program) == [family]


def test_semantic_family_classifier_marks_multi_family_programs_as_hybrid():
    from edge.island_program_research import _classify_strategy_families
    from edge.program_evolution import strategy_family_program

    features = (
        "ema_10", "ema_20", "ema_40", "rsi_14", "close",
        "donchian_high_20", "donchian_low_20", "atr_14",
    )
    rng = np.random.default_rng(73)
    trend = strategy_family_program("trend", features, rng)
    mean_reversion = strategy_family_program("mean_reversion", features, rng)
    hybrid = StrategyProgram(
        long_entry=Expr("and", (trend.long_entry, mean_reversion.long_entry)),
        short_entry=Expr("and", (trend.short_entry, mean_reversion.short_entry)),
        stop_atr=trend.stop_atr,
        reward_risk=trend.reward_risk,
        max_holding_bars=trend.max_holding_bars,
    )
    assert _classify_strategy_families(hybrid) == ["hybrid", "mean_reversion", "trend"]


def test_semantic_family_classifier_does_not_tag_incidental_features():
    from edge.island_program_research import _classify_strategy_families

    program = StrategyProgram(
        long_entry=Expr(
            "gt", (Expr("feature", value="ema_10"), Expr("feature", value="close"))
        ),
        short_entry=Expr(
            "lt", (Expr("feature", value="ema_40"), Expr("feature", value="close"))
        ),
    )
    assert _classify_strategy_families(program) == ["other"]



def test_volatility_family_requires_matching_directional_filter_and_lookback():
    from edge.island_program_research import _classify_strategy_families

    feature = lambda name: Expr("feature", value=name)
    gate = lambda op, lookback: Expr(
        op,
        (
            feature("atr_14"),
            Expr("rolling_mean", (feature("atr_14"),), value=lookback),
        ),
    )
    long_direction = Expr("gt", (feature("close"), feature("ema_40")))
    short_direction = Expr("lt", (feature("close"), feature("ema_40")))

    valid = StrategyProgram(
        long_entry=Expr("and", (gate("gt", 20), long_direction)),
        short_entry=Expr("and", (gate("gt", 20), short_direction)),
    )
    missing_direction = StrategyProgram(
        long_entry=gate("gt", 20),
        short_entry=gate("gt", 20),
    )
    mismatched_lookback = StrategyProgram(
        long_entry=Expr("and", (gate("gt", 10), long_direction)),
        short_entry=Expr("and", (gate("gt", 20), short_direction)),
    )

    assert _classify_strategy_families(valid) == ["volatility_expansion"]
    assert _classify_strategy_families(missing_direction) == ["other"]
    assert _classify_strategy_families(mismatched_lookback) == ["other"]



def test_candidate_validation_overlap_measures_trade_event_jaccard():
    candidates = [
        {"validation": {"trade_event_diagnostics": {"events": ["a", "b", "c"]}}},
        {"validation": {"trade_event_diagnostics": {"events": ["a", "b", "d"]}}},
        {"validation": {"trade_event_diagnostics": {"events": ["x", "y"]}}},
        {"validation": {"trade_event_diagnostics": {"events": []}}},
    ]

    overlap = island_program_research._candidate_validation_overlap(candidates)

    assert overlap[0] == {
        "candidate_a": 1,
        "candidate_b": 2,
        "shared_entry_events": 2,
        "union_entry_events": 4,
        "jaccard_similarity": 0.5,
    }
    assert len(overlap) == 3
    assert all(row["candidate_a"] < row["candidate_b"] for row in overlap)



def test_training_signal_diversity_filter_rejects_redundant_finalists(monkeypatch):
    feature = lambda name: Expr("feature", value=name)
    programs = [
        StrategyProgram(
            long_entry=Expr("gt", (feature("ema_gap_atr"), Expr("const", value=value))),
            short_entry=Expr("lt", (feature("ema_gap_atr"), Expr("const", value=0))),
        )
        for value in (1, 2, 3)
    ]
    signals_by_fingerprint = {
        programs[0].fingerprint(): np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8),
        programs[1].fingerprint(): np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8),
        programs[2].fingerprint(): np.array([0, 0, 0, 0, 0, 0, -1, -1], dtype=np.int8),
    }
    entry = pd.Timestamp("2024-01-01T00:00:00Z")
    same_path = pd.DataFrame([
        {"entry_time": entry, "direction": "long", "exit_time": entry + pd.Timedelta(hours=1), "exit_reason": "target"},
        {"entry_time": entry + pd.Timedelta(hours=2), "direction": "short", "exit_time": entry + pd.Timedelta(hours=3), "exit_reason": "stop"},
    ])
    distinct_path = pd.DataFrame([
        {"entry_time": entry + pd.Timedelta(days=1), "direction": "short", "exit_time": entry + pd.Timedelta(days=1, hours=2), "exit_reason": "time"},
    ])
    trades_by_fingerprint = {
        programs[0].fingerprint(): same_path,
        programs[1].fingerprint(): same_path.copy(),
        programs[2].fingerprint(): distinct_path,
    }
    monkeypatch.setattr(
        island_program_research,
        "evaluate_program",
        lambda program, frame: pd.Series(
            signals_by_fingerprint[program.fingerprint()], index=frame.index
        ),
    )
    monkeypatch.setattr(
        island_program_research,
        "_evaluate",
        lambda _frame, program, **_kwargs: (
            {},
            trades_by_fingerprint[program.fingerprint()],
        ),
    )
    ranked = [
        (3.0, programs[0], {}, {}),
        (2.0, programs[1], {}, {}),
        (1.0, programs[2], {}, {}),
    ]

    selected, diagnostics = island_program_research._training_signal_diverse_shortlist(
        ranked,
        pd.DataFrame(index=range(8)),
        similarity_threshold=0.90,
        limit=3,
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="mid",
    )

    assert [item[1].fingerprint() for item in selected] == [
        programs[0].fingerprint(),
        programs[2].fingerprint(),
    ]
    assert diagnostics["selected_fitness_ranks"] == [1, 3]
    rejected = diagnostics["rejected_as_redundant"][0]
    assert rejected["fitness_rank"] == 2
    assert rejected["most_similar_selected_fitness_rank"] == 1
    assert rejected["max_directional_signal_jaccard"] == 1.0
    assert rejected["max_training_trade_path_jaccard"] == 1.0
    assert rejected["max_risk_profile_similarity"] == 1.0
    assert diagnostics["validation_used_for_selection"] is False
    assert diagnostics["final_test_used_for_selection"] is False



def test_training_trade_path_events_use_timezone_normalized_compact_keys():
    utc = pd.Timestamp("2024-01-01T00:00:00Z")
    equivalent_local = pd.Timestamp("2024-01-01T05:30:00+05:30")
    trades = pd.DataFrame([
        {
            "entry_time": utc,
            "direction": "LONG",
            "exit_time": utc + pd.Timedelta(hours=1),
            "exit_reason": "TARGET",
        },
        {
            "entry_time": equivalent_local,
            "direction": "long",
            "exit_time": equivalent_local + pd.Timedelta(hours=1),
            "exit_reason": "target",
        },
    ])

    events = island_program_research._training_trade_path_events(trades)

    assert len(events) == 1
    event = next(iter(events))
    assert event == (
        int(utc.value),
        "long",
        int((utc + pd.Timedelta(hours=1)).value),
        "target",
    )


def test_training_signal_filter_evaluates_duplicate_program_fingerprint_once(monkeypatch):
    program = StrategyProgram(
        long_entry=Expr("gt", (
            Expr("feature", value="ema_gap_atr"),
            Expr("const", value=1),
        )),
        short_entry=Expr("lt", (
            Expr("feature", value="ema_gap_atr"),
            Expr("const", value=0),
        )),
    )
    calls = {"signals": 0, "backtests": 0}

    def evaluate(_program, frame):
        calls["signals"] += 1
        return pd.Series(np.array([1, 0, -1], dtype=np.int8), index=frame.index)

    def backtest(_frame, _program, **_kwargs):
        calls["backtests"] += 1
        return {}, pd.DataFrame(columns=[
            "entry_time", "direction", "exit_time", "exit_reason"
        ])

    monkeypatch.setattr(island_program_research, "evaluate_program", evaluate)
    monkeypatch.setattr(island_program_research, "_evaluate", backtest)
    ranked = [
        (2.0, program, {}, {}),
        (1.0, program, {}, {}),
    ]

    selected, diagnostics = island_program_research._training_signal_diverse_shortlist(
        ranked,
        pd.DataFrame(index=range(3)),
        similarity_threshold=0.95,
        limit=2,
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="mid",
    )

    assert calls == {"signals": 1, "backtests": 1}
    assert len(selected) == 2
    assert diagnostics["selected_fitness_ranks"] == [1, 2]
    assert diagnostics["performance"] == {
        "ranked_program_count": 2,
        "unique_program_count": 1,
        "signal_evaluations": 1,
        "training_backtests": 1,
        "pairwise_comparisons": 1,
        "short_circuited_comparisons": 0,
    }

def test_training_signal_filter_keeps_same_signals_with_different_trade_paths(monkeypatch):
    feature = lambda name: Expr("feature", value=name)
    programs = [
        StrategyProgram(
            long_entry=Expr("gt", (feature("ema_gap_atr"), Expr("const", value=value))),
            short_entry=Expr("lt", (feature("ema_gap_atr"), Expr("const", value=0))),
        )
        for value in (1, 2)
    ]
    signal = np.array([1, 1, 0, -1, -1, 0], dtype=np.int8)
    monkeypatch.setattr(
        island_program_research,
        "evaluate_program",
        lambda _program, frame: pd.Series(signal, index=frame.index),
    )
    entry = pd.Timestamp("2024-01-01T00:00:00Z")
    paths = {
        programs[0].fingerprint(): pd.DataFrame([{
            "entry_time": entry,
            "direction": "long",
            "exit_time": entry + pd.Timedelta(hours=1),
            "exit_reason": "target",
        }]),
        programs[1].fingerprint(): pd.DataFrame([{
            "entry_time": entry,
            "direction": "long",
            "exit_time": entry + pd.Timedelta(hours=4),
            "exit_reason": "time",
        }]),
    }
    monkeypatch.setattr(
        island_program_research,
        "_evaluate",
        lambda _frame, program, **_kwargs: ({}, paths[program.fingerprint()]),
    )
    ranked = [(2.0, programs[0], {}, {}), (1.0, programs[1], {}, {})]

    selected, diagnostics = island_program_research._training_signal_diverse_shortlist(
        ranked,
        pd.DataFrame(index=range(len(signal))),
        similarity_threshold=0.95,
        limit=2,
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="mid",
    )

    assert len(selected) == 2
    assert diagnostics["rejected_as_redundant"] == []


def test_training_signal_filter_keeps_materially_different_risk_profiles(monkeypatch):
    feature = lambda name: Expr("feature", value=name)
    programs = [
        StrategyProgram(
            long_entry=Expr("gt", (feature("ema_gap_atr"), Expr("const", value=1))),
            short_entry=Expr("lt", (feature("ema_gap_atr"), Expr("const", value=0))),
            stop_atr=1.5,
            reward_risk=reward_risk,
            max_holding_bars=holding,
        )
        for reward_risk, holding in ((2.0, 24), (3.5, 48))
    ]
    signal = np.array([1, 1, 0, -1, -1, 0], dtype=np.int8)
    monkeypatch.setattr(
        island_program_research,
        "evaluate_program",
        lambda _program, frame: pd.Series(signal, index=frame.index),
    )
    entry = pd.Timestamp("2024-01-01T00:00:00Z")
    same_path = pd.DataFrame([{
        "entry_time": entry,
        "direction": "long",
        "exit_time": entry + pd.Timedelta(hours=1),
        "exit_reason": "target",
    }])
    monkeypatch.setattr(
        island_program_research,
        "_evaluate",
        lambda _frame, _program, **_kwargs: ({}, same_path.copy()),
    )
    ranked = [(2.0, programs[0], {}, {}), (1.0, programs[1], {}, {})]

    selected, diagnostics = island_program_research._training_signal_diverse_shortlist(
        ranked,
        pd.DataFrame(index=range(len(signal))),
        similarity_threshold=0.95,
        limit=2,
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="mid",
    )

    assert len(selected) == 2
    assert diagnostics["rejected_as_redundant"] == []


def test_directional_signal_jaccard_distinguishes_opposite_directions():
    left = np.array([1, 0, -1, 0], dtype=np.int8)
    right = np.array([-1, 0, 1, 0], dtype=np.int8)

    assert island_program_research._directional_signal_jaccard(left, right) == 0.0


def test_training_signal_diversity_filter_validates_threshold():
    import pytest

    with pytest.raises(ValueError, match="between zero and one"):
        island_program_research._training_signal_diverse_shortlist(
            [],
            pd.DataFrame(),
            similarity_threshold=1.01,
            limit=20,
            execution=ExecutionConfig(),
            risk=RiskConfig(),
            pip_size=0.0001,
            price_basis="mid",
        )


def test_finalist_limit_cli_defaults_to_50_and_accepts_custom_value():
    from edge.cli import _parser

    parser = _parser()

    default_args = parser.parse_args([
        "evolve-program-islands",
        "--input", "data.csv",
    ])
    assert default_args.finalist_limit == 50

    custom_args = parser.parse_args([
        "evolve-program-islands",
        "--input", "data.csv",
        "--finalist-limit", "100",
    ])
    assert custom_args.finalist_limit == 100


def test_finalist_limit_cli_rejects_non_integer_values():
    import pytest
    from edge.cli import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args([
            "evolve-program-islands",
            "--input", "data.csv",
            "--finalist-limit", "not-an-integer",
        ])


def test_training_eligibility_filter_excludes_ineligible_and_non_finite_scores():
    programs = [
        StrategyProgram(
            long_entry=Expr("gt", (Expr("feature", value="ema_gap_atr"), Expr("const", value=value))),
            short_entry=Expr("lt", (Expr("feature", value="ema_gap_atr"), Expr("const", value=0))),
        )
        for value in (1, 2, 3, 4)
    ]
    evaluated = [
        (2.0, programs[0], {}, {"all_windows_eligible": True}),
        (-1e9, programs[1], {}, {"all_windows_eligible": False}),
        (1.0, programs[2], {}, {"all_windows_eligible": False}),
        (float("nan"), programs[3], {}, {"all_windows_eligible": True}),
    ]

    eligible, diagnostics = island_program_research._training_eligible_finalists(evaluated)

    assert [row[1].fingerprint() for row in eligible] == [programs[0].fingerprint()]
    assert diagnostics["evaluated_population_count"] == 4
    assert diagnostics["eligible_population_count"] == 1
    assert diagnostics["rejected_population_count"] == 3
    assert diagnostics["validation_used"] is False
    assert diagnostics["final_test_used"] is False
    assert [row["reason"] for row in diagnostics["rejected"]] == [
        "training_window_ineligible",
        "training_window_ineligible",
        "non_finite_or_ineligible_training_fitness",
    ]


def test_finalist_selection_report_preserves_requested_limit_above_pool_size(tmp_path):
    config = IslandEvolutionConfig(
        islands=2,
        population_per_island=4,
        elite_count=1,
        tournament_size=2,
        migration_interval=1,
        migrants_per_island=1,
        archive_capacity=16,
        novelty_weight=0.25,
    )
    report = run_island_program_research(
        raw_frame=_market_frame(),
        timeframe="M15",
        source_timeframe="M15",
        symbol="EURUSD",
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        pip_size=0.0001,
        price_basis="mid",
        generations=1,
        seed=23,
        min_train_trades=1,
        min_validation_trades=1,
        checkpoint_path=tmp_path / "checkpoint.json",
        output_dir=tmp_path / "report",
        config=config,
        finalist_similarity_threshold=0.95,
        finalist_limit=100,
    )

    selection = report["finalist_selection"]
    assert selection["requested_limit"] == 100
    assert selection["fitness_ranked_pool_count"] <= 8
    assert selection["selected_count"] <= selection["fitness_ranked_pool_count"]
    assert selection["training_eligibility"]["validation_used"] is False
    assert selection["training_eligibility"]["final_test_used"] is False
