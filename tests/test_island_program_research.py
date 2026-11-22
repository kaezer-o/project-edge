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
    assert first["final_test"] is None
    assert first["final_test_skipped"] is True
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
