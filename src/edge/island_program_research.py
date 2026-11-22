from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
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
from .program import StrategyProgram, evaluate_program, canonicalize_program
from .program_evolution import (
    available_strategy_families,
    crossover_programs,
    mutate_program,
    strategy_family_program,
)
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
    # Include close for causal Donchian breakout templates. Raw price levels
    # remain protected by the expression unit checks against scalar thresholds.
    features = tuple(
        column for column in (*FEATURE_COLUMNS, "close")
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
    signal: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Describe signal rates and exact executed entry-time/direction events."""
    if signal is None:
        signal = evaluate_program(program, frame)
    if trades is None:
        _, trades = _evaluate(
            frame,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            signal=signal,
        )
    chunks = np.array_split(np.arange(len(frame)), 3)
    rates: list[float] = []
    for positions in chunks:
        if not len(positions):
            rates.extend((0.0, 0.0, 0.0))
            continue
        segment = signal.iloc[positions]
        start_time = pd.Timestamp(frame.index[int(positions[0])])
        end_time = pd.Timestamp(frame.index[int(positions[-1])])
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize("UTC")
            end_time = end_time.tz_localize("UTC")
        else:
            start_time = start_time.tz_convert("UTC")
            end_time = end_time.tz_convert("UTC")
        if not trades.empty:
            entry_times = pd.to_datetime(trades["entry_time"], utc=True)
            trade_count = int(((entry_times >= start_time) & (entry_times <= end_time)).sum())
        else:
            trade_count = 0
        rates.extend((
            float((segment == 1).mean()),
            float((segment == -1).mean()),
            float(trade_count / len(positions)),
        ))
    trade_events: list[str] = []
    if not trades.empty:
        for trade in trades.itertuples(index=False):
            timestamp = pd.Timestamp(trade.entry_time)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            trade_events.append(f"{timestamp.isoformat()}|{str(trade.direction).lower()}")
    feature_names: set[str] = set()
    def collect_features(expr) -> None:
        if expr.op == "feature":
            feature_names.add(str(expr.value))
        for child in expr.args:
            collect_features(child)
    collect_features(program.long_entry)
    collect_features(program.short_entry)
    families = set()
    if any(name.startswith("ema_") for name in feature_names):
        families.add("trend")
    if any(name.startswith("donchian_") for name in feature_names):
        families.add("breakout")
    if any(name.startswith("rsi_") for name in feature_names):
        families.add("mean_reversion")
    if any(name.startswith("atr_") or name.endswith("_atr") for name in feature_names):
        families.add("volatility")
    if not families:
        families.add("other")
    signal_fingerprint = hashlib.sha256(
        signal.to_numpy(dtype="int8", copy=False).tobytes()
    ).hexdigest()
    unique_trade_events = sorted(set(trade_events))
    behaviour_payload = json.dumps(
        {
            "signals": signal_fingerprint,
            "trades": unique_trade_events,
            "stop_atr": program.stop_atr,
            "reward_risk": program.reward_risk,
            "max_holding_bars": program.max_holding_bars,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    behaviour_fingerprint = hashlib.sha256(behaviour_payload).hexdigest()
    return {
        "rates": rates,
        "trades": unique_trade_events,
        "signal_fingerprint": signal_fingerprint,
        "behaviour_fingerprint": behaviour_fingerprint,
        "families": sorted(families),
    }


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
        "fitness_version": 5,
        "strategy_seed_version": 1,
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

    training_results: OrderedDict[str, tuple[float, dict[str, object], dict[str, object]]] = OrderedDict()
    training_result_cache_limit = max(256, config.islands * config.population_per_island * 4)

    def evaluate(genome: dict[str, object]) -> tuple[float, dict[str, object]]:
        program = StrategyProgram.from_dict(genome)
        signal = evaluate_program(program, train)
        aggregate_result = _evaluate(
            train,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            signal=signal,
        )
        score, metrics, robustness = _training_fitness(
            train,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            min_train_trades=min_train_trades,
            aggregate_result=aggregate_result,
        )
        fingerprint = program.fingerprint()
        training_results[fingerprint] = (score, metrics, robustness)
        training_results.move_to_end(fingerprint)
        if len(training_results) > training_result_cache_limit:
            training_results.popitem(last=False)
        descriptor = _behaviour_descriptor(
            train,
            program,
            execution=execution,
            risk=risk,
            pip_size=pip_size,
            price_basis=price_basis,
            signal=signal,
            trades=aggregate_result[1],
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
        return canonicalize_program(child).to_dict()

    initial_populations = None
    initial_family_counts: dict[str, int] | None = None
    if not resume:
        initial_family_counts = {}
        seed_sequence = np.random.SeedSequence(seed)
        initial_populations = []
        family_names = available_strategy_families(features)
        if not family_names:
            raise ValueError("No supported strategy-family templates for available features")
        for island_index, child_seed in enumerate(seed_sequence.spawn(config.islands)):
            rng = np.random.default_rng(child_seed)
            population: list[dict[str, object]] = []
            seen: set[str] = set()
            while len(population) < config.population_per_island:
                slot = len(population)
                family = family_names[
                    (island_index * config.population_per_island + slot)
                    % len(family_names)
                ]
                candidate = canonicalize_program(
                    strategy_family_program(family, features, rng)
                )
                fingerprint = candidate.fingerprint()
                attempts = 0
                while fingerprint in seen and attempts < 16:
                    candidate = canonicalize_program(
                        strategy_family_program(family, features, rng)
                    )
                    fingerprint = candidate.fingerprint()
                    attempts += 1
                assigned_family = family
                if fingerprint in seen:
                    candidate = canonicalize_program(_random_program(features, rng))
                    fingerprint = candidate.fingerprint()
                    assigned_family = "random_fallback"
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    population.append(candidate.to_dict())
                    initial_family_counts[assigned_family] = (
                        initial_family_counts.get(assigned_family, 0) + 1
                    )
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
            cached = training_results.get(program.fingerprint())
            if cached is None:
                score, metrics, robustness = _training_fitness(
                    train,
                    program,
                    execution=execution,
                    risk=risk,
                    pip_size=pip_size,
                    price_basis=price_basis,
                    min_train_trades=min_train_trades,
                )
            else:
                score, metrics, robustness = cached
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
            "initial_strategy_family_counts": initial_family_counts,
            "available_strategy_families": list(available_strategy_families(features)),
            "generation_performance": result.runtime_metrics,
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
