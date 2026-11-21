from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExecutionConfig, RiskConfig, TIMEFRAME_SECONDS
from .data import annotate_gaps, resample_ohlc
from .features import FEATURE_COLUMNS, add_features
from .island_evolution import (
    IslandEvolutionConfig,
    run_island_evolution,
)
from .program import StrategyProgram, evaluate_program
from .program_evolution import crossover_programs, mutate_program
from .program_research import (
    _diverse_ranked,
    _evaluate,
    _random_program,
    _training_fitness,
    _validation_check,
)


def _prepare_frames(
    raw_frame: pd.DataFrame,
    *,
    timeframe: str,
    source_timeframe: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    timeframe = timeframe.upper()
    source_timeframe = source_timeframe.upper()
    if timeframe not in TIMEFRAME_SECONDS or source_timeframe not in TIMEFRAME_SECONDS:
        raise ValueError("Unsupported timeframe")
    if TIMEFRAME_SECONDS[timeframe] < TIMEFRAME_SECONDS[source_timeframe]:
        raise ValueError("Target timeframe cannot be lower than source timeframe")
    annotated = annotate_gaps(raw_frame, TIMEFRAME_SECONDS[source_timeframe])
    market = annotated if timeframe == source_timeframe else resample_ohlc(annotated, timeframe)
    featured = add_features(market)
    n = len(featured)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    if train_end < 300 or validation_end - train_end < 100 or n - validation_end < 100:
        raise ValueError("Insufficient bars for chronological train/validation/test partitions")
    train = featured.iloc[:train_end].copy()
    validation = featured.iloc[train_end:validation_end].copy()
    test = featured.iloc[validation_end:].copy()
    features = tuple(
        column for column in FEATURE_COLUMNS
        if column in train.columns
        and pd.api.types.is_numeric_dtype(train[column])
        and train[column].notna().any()
    )
    if len(features) < 4:
        raise ValueError("Fewer than four usable numeric features in training data")
    return train, validation, test, features


def _behaviour_descriptor(
    frame: pd.DataFrame,
    program: StrategyProgram,
    *,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
) -> list[float]:
    """Describe actual directional signals and executed trades in three eras."""
    signal = evaluate_program(program, frame)
    _, trades = _evaluate(
        frame,
        program,
        execution=execution,
        risk=risk,
        pip_size=pip_size,
        price_basis=price_basis,
    )
    trade_times = (
        pd.DatetimeIndex(trades["entry_time"])
        if not trades.empty and "entry_time" in trades
        else pd.DatetimeIndex([], tz=frame.index.tz)
    )
    chunks = np.array_split(np.arange(len(frame)), 3)
    descriptor: list[float] = []
    for positions in chunks:
        if not len(positions):
            descriptor.extend((0.0, 0.0, 0.0))
            continue
        segment = signal.iloc[positions]
        start_time = frame.index[int(positions[0])]
        end_time = frame.index[int(positions[-1])]
        trade_count = int(((trade_times >= start_time) & (trade_times <= end_time)).sum())
        bars = float(len(positions))
        descriptor.extend((
            float((segment == 1).mean()),
            float((segment == -1).mean()),
            float(trade_count / bars * 1000.0),
        ))
    return descriptor


def _run_signature(
    source_data: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    source_timeframe: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    config: IslandEvolutionConfig,
    seed: int,
    min_train_trades: int,
    min_validation_trades: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(source_data, index=True).values.tobytes())
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "source_timeframe": source_timeframe,
        "data_sha256": digest.hexdigest(),
        "execution": execution.__dict__,
        "risk": risk.__dict__,
        "pip_size": pip_size,
        "price_basis": price_basis,
        "island_config": config.__dict__,
        "seed": seed,
        "min_train_trades": min_train_trades,
        "min_validation_trades": min_validation_trades,
        "fitness_version": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_island_program_research(
    raw_frame: pd.DataFrame,
    *,
    timeframe: str,
    source_timeframe: str,
    symbol: str,
    execution: ExecutionConfig,
    risk: RiskConfig,
    pip_size: float,
    price_basis: str,
    generations: int,
    seed: int,
    min_train_trades: int,
    min_validation_trades: int,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: IslandEvolutionConfig | None = None,
    resume: bool = False,
    run_final_test: bool = False,
) -> dict[str, object]:
    """Run resumable multi-island program evolution; only training reaches the evaluator."""
    if generations < 1:
        raise ValueError("generations must be at least one")
    config = config or IslandEvolutionConfig()
    timeframe = timeframe.upper()
    source_timeframe = source_timeframe.upper()
    train, validation, test, features = _prepare_frames(
        raw_frame, timeframe=timeframe, source_timeframe=source_timeframe
    )
    signature = _run_signature(
        pd.concat((train, validation, test)),
        symbol=symbol,
        timeframe=timeframe,
        source_timeframe=source_timeframe,
        execution=execution,
        risk=risk,
        pip_size=pip_size,
        price_basis=price_basis,
        config=config,
        seed=seed,
        min_train_trades=min_train_trades,
        min_validation_trades=min_validation_trades,
    )

    def evaluate(genome: dict[str, object]) -> tuple[float, list[float]]:
        program = StrategyProgram.from_dict(genome)
        score, _, _ = _training_fitness(
            train,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            min_train_trades=min_train_trades,
        )
        descriptor = _behaviour_descriptor(
            train,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
        )
        return float(score), descriptor

    def reproduce(
        first: dict[str, object],
        second: dict[str, object],
        rng: np.random.Generator,
    ) -> dict[str, object]:
        parent_a = StrategyProgram.from_dict(first)
        parent_b = StrategyProgram.from_dict(second)
        if rng.random() < 0.55:
            child = crossover_programs(parent_a, parent_b, rng)
        else:
            child = mutate_program(parent_a, features, rng)
        return child.to_dict()

    initial_populations = None
    if not resume:
        seed_sequence = np.random.SeedSequence(seed)
        initial_populations = []
        for child_seed in seed_sequence.spawn(config.islands):
            rng = np.random.default_rng(child_seed)
            population: list[dict[str, object]] = []
            seen: set[str] = set()
            while len(population) < config.population_per_island:
                candidate = _random_program(features, rng)
                fingerprint = candidate.fingerprint()
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    population.append(candidate.to_dict())
            initial_populations.append(population)

    result = run_island_evolution(
        initial_populations,
        evaluate=evaluate,
        reproduce=reproduce,
        config=config,
        seed=seed,
        generations=generations,
        checkpoint_path=checkpoint_path,
        run_signature=signature,
        resume=resume,
    )

    final_evaluated = []
    for population in result.islands:
        for genome in population:
            program = StrategyProgram.from_dict(genome)
            score, metrics, robustness = _training_fitness(
                train,
                program,
                execution=execution,
                risk=risk,
                pip_size=pip_size,
                price_basis=price_basis,
                min_train_trades=min_train_trades,
            )
            final_evaluated.append((score, program, metrics, robustness))
    shortlist = _diverse_ranked(
        final_evaluated,
        min(20, len(final_evaluated)),
        max_per_structure=1,
    )
    candidates: list[dict[str, object]] = []
    accepted: list[tuple[float, StrategyProgram, dict[str, object]]] = []
    for score, program, train_metrics, robustness in shortlist:
        validation_result = _validation_check(
            validation,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            min_trades=max(5, min_validation_trades // 3),
        )
        row = {
            "program": program.to_dict(),
            "fingerprint": program.fingerprint(),
            "training_fitness": float(score),
            "training_metrics": train_metrics,
            "training_robustness": robustness,
            "validation": validation_result,
        }
        candidates.append(row)
        if validation_result["pass"]:
            accepted.append((score, program, row))

    winner = max(accepted, key=lambda item: item[0]) if accepted else None
    final_test = None
    if winner is not None and run_final_test:
        test_metrics, _ = _evaluate(
            test,
            winner[1],
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
        )
        final_test = {
            "metrics": test_metrics,
            "warning": "Single historical holdout; not prospective or live-trading evidence.",
        }

    report: dict[str, object] = {
        "project": "Project EDGE",
        "engine": "resumable_multi_island_program_evolution",
        "version": 1,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_timeframe": source_timeframe,
        "data": {
            "train_bars": len(train),
            "validation_bars": len(validation),
            "final_test_bars": len(test),
            "features": list(features),
            "final_test_evaluated": final_test is not None,
        },
        "runtime": {
            "generation": result.generation,
            "checkpoint": str(Path(checkpoint_path)),
            "resumed": resume,
            "islands": config.islands,
            "population_per_island": config.population_per_island,
            "total_population": config.islands * config.population_per_island,
            "migration_interval": config.migration_interval,
            "migrants_per_island": config.migrants_per_island,
            "novelty_weight": config.novelty_weight,
            "history": result.history,
        },
        "candidate_count": len(candidates),
        "validated_candidate_count": len(accepted),
        "status": "validated_candidate" if winner else "no_robust_candidate",
        "candidates": candidates,
        "winner": winner[2] if winner else None,
        "final_test": final_test,
        "final_test_skipped": final_test is None,
        "validation_warning": (
            "Validation is development data if previously inspected. "
            "The final chronological partition must remain untouched until the methodology is frozen."
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "island_program_evolution_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
