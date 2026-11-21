import json

import numpy as np
import pandas as pd

from edge.config import ExecutionConfig, RiskConfig
from edge.program_research import run_program_research


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
