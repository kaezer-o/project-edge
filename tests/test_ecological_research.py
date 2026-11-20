import pandas as pd

from edge.config import ResearchConfig
from edge.evolution import EvolutionSettings
from edge.ecological_research import research_ecological_timeframe
from edge.ecological_research import run_ecological_research


def test_ecological_research_rejects_insufficient_data():
    index = pd.date_range(
        "2025-01-01",
        periods=200,
        freq="4h",
        tz="UTC",
    )

    frame = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.01,
            "low": 0.99,
            "close": 1.0,
        },
        index=index,
    )

    result = research_ecological_timeframe(
        frame,
        timeframe="H4",
        symbol="EURUSD",
        execution=None,
        risk=None,
        research=ResearchConfig(),
        evolution=EvolutionSettings(),
        pip_size=0.0001,
        price_basis="bid",
    )

    assert result["status"] == "insufficient_data"
    assert result["winner"] is None

def test_ecological_research_writes_reports_for_insufficient_data(tmp_path):
    import json

    import numpy as np

    from edge.config import ExecutionConfig, RiskConfig

    index = pd.date_range(
        "2025-01-01",
        periods=200,
        freq="4h",
        tz="UTC",
    )

    frame = pd.DataFrame({
        "open": np.full(200, 1.0),
        "high": np.full(200, 1.01),
        "low": np.full(200, 0.99),
        "close": np.full(200, 1.0),
    }, index=index)

    report = run_ecological_research(
        frame,
        source_timeframe="H4",
        timeframes=["H4"],
        symbol="EURUSD",
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        research=ResearchConfig(),
        evolution=EvolutionSettings(),
        pip_size=0.0001,
        price_basis="bid",
        output_dir=tmp_path,
        run_final_test=False,
    )

    assert report["results"][0]["status"] == "insufficient_data"

    saved = json.loads(
        (tmp_path / "ecological_report.json").read_text(encoding="utf-8")
    )

    assert saved["results"][0]["status"] == "insufficient_data"
    assert (tmp_path / "ecological_timeframe_summary.csv").exists()
    assert (tmp_path / "ecological_summary.md").exists()
    assert (tmp_path / "source_gap_audit.csv").exists()


def test_final_test_diagnostics_reports_windows_and_trade_groups():
    import pandas as pd

    from edge.ecological_research import _final_test_diagnostics

    index = pd.date_range(
        "2025-01-01",
        periods=8,
        freq="4h",
        tz="UTC",
    )

    test_frame = pd.DataFrame(index=index)

    trades = pd.DataFrame({
        "entry_time": [index[0], index[2], index[5], index[7]],
        "net_r": [0.5, -1.0, 0.25, -0.5],
        "gross_r": [0.6, -0.9, 0.35, -0.4],
        "cost_r": [0.1, 0.1, 0.1, 0.1],
        "direction": ["long", "short", "long", "short"],
        "exit_reason": ["target", "stop", "data_gap_exit", "stop"],
        "gap_exit": [False, False, True, False],
        "spread_cost_pips": [0.2, 0.2, 0.2, 0.2],
        "commission_cost_pips": [0.0, 0.0, 0.0, 0.0],
        "slippage_cost_pips": [0.1, 0.1, 0.1, 0.1],
        "swap_cost_pips": [0.0, 0.0, 0.0, 0.0],
    })

    result = _final_test_diagnostics(test_frame, trades)

    assert len(result["chronological_windows"]) == 4
    assert [
        item["trades"] for item in result["chronological_windows"]
    ] == [1, 1, 1, 1]

    assert result["overall"]["trades"] == 4
    assert result["overall"]["expectancy_r"] == -0.1875
    assert result["overall"]["gap_exits"] == 1

    directions = {
        item["direction"]: item
        for item in result["by_direction"]
    }
    assert directions["long"]["expectancy_r"] == 0.375
    assert directions["short"]["expectancy_r"] == -0.75

    assert result["costs"]["separately_deducted_cost_r"] == 0.4
    assert result["unmatched_entry_timestamps"] == 0

def test_final_test_diagnostics_handles_zero_trades():
    import pandas as pd

    from edge.ecological_research import _final_test_diagnostics

    index = pd.date_range(
        "2025-01-01",
        periods=8,
        freq="4h",
        tz="UTC",
    )

    result = _final_test_diagnostics(
        pd.DataFrame(index=index),
        pd.DataFrame(),
    )

    assert result["overall"]["trades"] == 0
    assert result["unmatched_entry_timestamps"] == 0
    assert len(result["chronological_windows"]) == 4
    assert all(
        window["trades"] == 0
        for window in result["chronological_windows"]
    )

def test_validation_pool_prioritizes_qualified_training_samples():
    from edge.ecology import OrganismFitness
    from edge.ecological_research import _candidate_pool_order_key

    qualified = OrganismFitness(
        organism_id="qualified",
        expectancy_r=-0.1,
        max_drawdown_pct=2.0,
        complexity=4,
        novelty_score=0.1,
        trade_count=30,
    )
    under_sampled = OrganismFitness(
        organism_id="under-sampled",
        expectancy_r=1.5,
        max_drawdown_pct=0.1,
        complexity=1,
        novelty_score=1.0,
        trade_count=4,
    )

    qualified_key = _candidate_pool_order_key(
        "qualified", qualified, 8, 0.0, minimum_train_trades=30
    )
    under_sampled_key = _candidate_pool_order_key(
        "under-sampled", under_sampled, 0, 10.0,
        minimum_train_trades=30,
    )

    assert qualified_key < under_sampled_key


def test_candidate_eligibility_requires_all_three_windows():
    from edge.config import ResearchConfig
    from edge.ecological_research import _candidate_passes_validation

    research = ResearchConfig()

    training = {
    	"trades": 1000,
    	"expectancy_r": 0.04,
    }
    validation = {
        "trades": 1000,
        "expectancy_r": 0.04,
        "profit_factor": 1.25,
        "max_realized_drawdown_pct": 0.01,
    }

    windows = [
        {
            "enough_trades": True,
            "pass": True,
            "metrics": {"expectancy_r": 0.05},
        },
        {
            "enough_trades": True,
            "pass": True,
            "metrics": {"expectancy_r": 0.03},
        },
        {
            "enough_trades": True,
            "pass": True,
            "metrics": {"expectancy_r": 0.02},
        },
    ]

    robustness = {
        "chronological_validation_windows": windows,
        "one_and_half_x_cost_pass": True,
    }

    assert _candidate_passes_validation(
        training, validation, robustness, research
    )

    training["expectancy_r"] = -0.01

    assert not _candidate_passes_validation(
        training, validation, robustness, research
    )

    training["expectancy_r"] = 0.04

    robustness["chronological_validation_windows"][2]["pass"] = False

    assert not _candidate_passes_validation(
        training, validation, robustness, research
    )


def test_candidate_ranking_prefers_consistent_validation_performance():
    from edge.ecological_research import _candidate_rank_key

    validation_7b = {
        "profit_factor": 1.095,
        "max_realized_drawdown_pct": 0.0167,
    }
    robustness_7b = {
        "chronological_validation_windows": [
            {"metrics": {"expectancy_r": 0.0051}},
            {"metrics": {"expectancy_r": 0.1365}},
            {"metrics": {"expectancy_r": -0.0404}},
        ],
        "median_window_expectancy_r": 0.0051,
        "expectancy_block_bootstrap": {
            "ci_90_low": -0.0666,
            "ci_90_high": 0.1482,
        },
        "cost_stress": [{
            "cost_multiplier": 1.5,
            "metrics": {"expectancy_r": 0.0312},
        }],
    }

    validation_3f = {
        "profit_factor": 1.265,
        "max_realized_drawdown_pct": 0.0084,
    }
    robustness_3f = {
        "chronological_validation_windows": [
            {"metrics": {"expectancy_r": 0.0302}},
            {"metrics": {"expectancy_r": 0.0410}},
            {"metrics": {"expectancy_r": 0.0250}},
        ],
        "median_window_expectancy_r": 0.0302,
        "expectancy_block_bootstrap": {
            "ci_90_low": -0.0350,
            "ci_90_high": 0.1084,
        },
        "cost_stress": [{
            "cost_multiplier": 1.5,
            "metrics": {"expectancy_r": 0.0372},
        }],
    }

    score_7b = _candidate_rank_key(
        validation_7b, robustness_7b, complexity=4
    )
    score_3f = _candidate_rank_key(
        validation_3f, robustness_3f, complexity=5
    )

    assert score_3f > score_7b
def test_walk_forward_splits_expand_training_without_future_overlap():
    from edge.ecological_research import build_walk_forward_splits

    splits = build_walk_forward_splits(
        1000,
        folds=3,
        initial_train_fraction=0.60,
        step_fraction=0.10,
    )

    assert splits == [
        {"fold": 1, "train_end": 600, "validation_end": 700, "test_end": 800},
        {"fold": 2, "train_end": 700, "validation_end": 800, "test_end": 900},
        {"fold": 3, "train_end": 800, "validation_end": 900, "test_end": 1000},
    ]


def test_walk_forward_split_rejects_insufficient_data():
    import pytest
    from edge.ecological_research import build_walk_forward_splits

    with pytest.raises(ValueError):
        build_walk_forward_splits(100, folds=3)


def test_walk_forward_split_rejects_excessive_fold_count():
    import pytest
    from edge.ecological_research import build_walk_forward_splits

    with pytest.raises(ValueError):
        build_walk_forward_splits(
            1000,
            folds=5,
            initial_train_fraction=0.60,
            step_fraction=0.10,
        )

def test_walk_forward_runner_isolates_folds_and_exports_reports(
    tmp_path, monkeypatch
):
    import json
    import pandas as pd

    import edge.ecological_research as module
    from edge.config import ExecutionConfig, RiskConfig, ResearchConfig
    from edge.evolution import EvolutionSettings

    index = pd.date_range(
        "2020-01-01",
        periods=1000,
        freq="1min",
        tz="UTC",
    )

    source = pd.DataFrame({
        "open": 1.0,
        "high": 1.01,
        "low": 0.99,
        "close": 1.0,
    }, index=index)

    calls = []

    monkeypatch.setattr(
        module, "annotate_gaps",
        lambda frame, seconds: frame.copy(),
    )
    monkeypatch.setattr(
        module, "data_quality_summary",
        lambda frame: {"rows": len(frame)},
    )
    monkeypatch.setattr(
        module, "build_gap_audit",
        lambda frame, seconds: pd.DataFrame(columns=["gap"]),
    )
    monkeypatch.setattr(
        module, "resample_ohlc",
        lambda frame, timeframe: frame.copy(),
    )

    def fake_research(frame, **kwargs):
        train_end, validation_end = kwargs["partition_ends"]
        call_number = len(calls) + 1

        calls.append({
            "bars": len(frame),
            "partition_ends": kwargs["partition_ends"],
            "last_time": frame.index[-1],
        })

        entry_time = frame.index[validation_end]

        return {
            "timeframe": "H4",
            "status": "test_passed",
            "winner_id": f"winner-{call_number}",
            "candidates": [
                {
                    "eligible": True,
                    "rejection_reasons": [],
                },
                {
                    "eligible": False,
                    "rejection_reasons": [
                        "non_positive_training_expectancy",
                        "execution_cost_stress_failed",
                    ],
                },
            ],
            "train_bars": train_end,
            "validation_bars": validation_end - train_end,
            "test_bars": len(frame) - validation_end,
            "final_test": {
                "trades": 1,
                "expectancy_r": 0.1,
                "profit_factor": 1.2,
                "max_realized_drawdown_pct": 0.01,
            },
            "final_test_diagnostics": {},
            "_final_test_trades": pd.DataFrame({
                "entry_time": [entry_time],
                "net_r": [0.1],
            }),
            "_final_test_equity": pd.DataFrame({
                "time": [entry_time],
                "equity": [10001.0],
            }),
        }

    monkeypatch.setattr(
        module, "research_ecological_timeframe", fake_research
    )

    report = module.run_ecological_walk_forward_research(
        source,
        source_timeframe="M1",
        timeframes=["H4"],
        symbol="EURUSD",
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        research=ResearchConfig(),
        evolution=EvolutionSettings(
            population_size=8,
            generations=2,
            seed=42,
        ),
        pip_size=0.0001,
        price_basis="bid",
        output_dir=tmp_path,
        folds=3,
        initial_train_fraction=0.60,
        step_fraction=0.10,
    )

    assert calls == [
        {
            "bars": 800,
            "partition_ends": (600, 700),
            "last_time": index[799],
        },
        {
            "bars": 900,
            "partition_ends": (700, 800),
            "last_time": index[899],
        },
        {
            "bars": 1000,
            "partition_ends": (800, 900),
            "last_time": index[999],
        },
    ]

    folds = report["results"][0]["folds"]

    assert len(folds) == 3
    assert [fold["winner_id"] for fold in folds] == [
        "winner-1",
        "winner-2",
        "winner-3",
    ]

    assert report["results"][0]["aggregate_test_summary"][
        "positive_expectancy_folds"
    ] == 3

    for number in range(1, 4):
        suffix = f"fold_{number:02d}"
        assert (tmp_path / f"H4_{suffix}_test_trades.csv").exists()
        assert (tmp_path / f"H4_{suffix}_test_equity.csv").exists()

    saved = json.loads(
        (tmp_path / "ecological_walk_forward_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(saved["results"][0]["folds"]) == 3
    summary = pd.read_csv(
        tmp_path / "ecological_walk_forward_summary.csv"
    )

    assert len(summary) == 3
    assert summary["candidates_evaluated"].tolist() == [2, 2, 2]
    assert summary["eligible_candidates"].tolist() == [1, 1, 1]
    assert summary["rejected_candidates"].tolist() == [1, 1, 1]

    rejection_counts = json.loads(
        summary.iloc[0]["rejection_reason_counts"]
    )
    assert rejection_counts == {
        "execution_cost_stress_failed": 1,
        "non_positive_training_expectancy": 1,
    }
    assert saved["walk_forward_design"]["cross_fold_test_selection"] is False


def test_candidate_rejection_reasons_identify_failed_gates():
    from edge.config import ResearchConfig
    from edge.ecological_research import _candidate_rejection_reasons

    research = ResearchConfig()

    training = {
        "trades": 1000,
        "expectancy_r": -0.10,
    }
    validation = {
        "trades": 1000,
        "expectancy_r": 0.04,
        "profit_factor": 1.25,
        "max_realized_drawdown_pct": 0.01,
    }
    robustness = {
        "chronological_validation_windows": [
            {"enough_trades": True, "pass": True},
            {"enough_trades": True, "pass": True},
            {"enough_trades": True, "pass": True},
        ],
        "one_and_half_x_cost_pass": True,
    }

    reasons = _candidate_rejection_reasons(
        training, validation, robustness, research
    )

    assert reasons == ["non_positive_training_expectancy"]