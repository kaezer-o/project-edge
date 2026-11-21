from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import BacktestSettings
from .config import (
    ExecutionConfig, ResearchConfig, RiskConfig, SUPPORTED_TIMEFRAMES,
    TIMEFRAME_SECONDS,
)
from .data import (
    build_gap_audit, data_quality_summary, load_ohlc_csv,
    validate_source_interval, validate_source_timeframe,
)
from .providers import CSVOHLCProvider
from .research import run_research
from .evolution import EvolutionSettings
from .source_comparison import compare_ohlc_csv_files
from .evolutionary_research import run_evolutionary_research
from .frozen_replay import replay_frozen_candidate
from .quote_data import aggregate_tick_csv

from .ecological_research import (
    run_ecological_research,
    run_ecological_walk_forward_research,
)


def _add_execution_cost_arguments(parser: argparse.ArgumentParser, *, include_spread: bool = True) -> None:
    if include_spread:
        parser.add_argument("--spread-pips", type=float, default=0.8, help="fallback spread estimate when quote-side prices are unavailable")
    parser.add_argument("--slippage-pips", type=float, default=0.1, help="slippage per side")
    parser.add_argument("--commission-pips", type=float, default=0.0, help="round-trip commission, pip equivalent")
    parser.add_argument("--swap-long-pips", type=float, default=0.0, help="signed long swap per broker rollover, pip equivalent; negative is a debit")
    parser.add_argument("--swap-short-pips", type=float, default=0.0, help="signed short swap per broker rollover, pip equivalent; negative is a debit")
    parser.add_argument("--rollover-timezone", default="America/New_York", help="IANA timezone used for broker rollover")
    parser.add_argument("--rollover-local-time", default="17:00", help="broker rollover time in rollover timezone")
    parser.add_argument("--triple-swap-weekday", type=int, choices=range(7), default=None, help="triple-swap weekday: Monday=0 through Sunday=6; broker-specific")
    parser.add_argument("--triple-swap-multiplier", type=float, default=3.0)


def _execution_config_from_args(args: argparse.Namespace, *, include_spread: bool = True) -> ExecutionConfig:
    return ExecutionConfig(
        spread_pips=float(getattr(args, "spread_pips", 0.0)) if include_spread else 0.0,
        slippage_pips_per_side=float(args.slippage_pips),
        commission_round_trip_pips=float(args.commission_pips),
        swap_long_pips_per_rollover=float(args.swap_long_pips),
        swap_short_pips_per_rollover=float(args.swap_short_pips),
        rollover_timezone=str(args.rollover_timezone),
        rollover_local_time=str(args.rollover_local_time),
        triple_swap_weekday=args.triple_swap_weekday,
        triple_swap_multiplier=float(args.triple_swap_multiplier),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge",
        description="Project EDGE: Forex strategy research and validation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate an OHLC CSV file")
    validate.add_argument("--input", required=True, help="CSV containing time, open, high, low, close")

    aggregate_ticks = subparsers.add_parser(
        "aggregate-ticks", help="convert bid/ask tick CSV into observed quote-side OHLC bars"
    )
    aggregate_ticks.add_argument("--input", required=True, help="tick CSV with time, bid, ask")
    aggregate_ticks.add_argument("--timeframe", default="M1", choices=SUPPORTED_TIMEFRAMES)
    aggregate_ticks.add_argument("--pip-size", type=float, default=0.0001)
    aggregate_ticks.add_argument("--output", default="data/QUOTE_EURUSD_M1.csv")

    replay = subparsers.add_parser(
        "replay-frozen", help="replay an unchanged candidate and frozen pattern model on new bid/ask data"
    )
    replay.add_argument("--input", required=True, help="independent source quote OHLC CSV; must include paired ask OHLC")
    replay.add_argument("--report", required=True, help="original evolution_report.json containing frozen candidate")
    replay.add_argument("--provider-name", required=True, help="descriptive provider label; not authenticated by EDGE")
    replay.add_argument("--timeframe", default="H4", choices=SUPPORTED_TIMEFRAMES)
    replay.add_argument("--source-timeframe", default="M1", choices=SUPPORTED_TIMEFRAMES)
    replay.add_argument("--from", dest="from_utc", required=True, help="evaluation start UTC timestamp/date; must follow original validation")
    replay.add_argument("--to", dest="to_utc", required=True, help="evaluation end UTC timestamp/date")
    replay.add_argument("--pip-size", type=float, default=0.0001)
    replay.add_argument("--slippage-pips", type=float, default=0.1, help="slippage per side")
    replay.add_argument("--commission-pips", type=float, default=0.0, help="round-trip commission, pip equivalent")
    replay.add_argument("--swap-long-pips", type=float, default=0.0, help="signed long swap per rollover, pip equivalent; negative is a debit")
    replay.add_argument("--swap-short-pips", type=float, default=0.0, help="signed short swap per rollover, pip equivalent; negative is a debit")
    replay.add_argument("--rollover-timezone", default="America/New_York")
    replay.add_argument("--rollover-local-time", default="17:00")
    replay.add_argument("--triple-swap-weekday", type=int, choices=range(7), default=None)
    replay.add_argument("--triple-swap-multiplier", type=float, default=3.0)
    replay.add_argument("--starting-balance", type=float, default=10_000.0)
    replay.add_argument("--risk-per-trade", type=float, default=0.0025)
    replay.add_argument("--max-daily-loss", type=float, default=0.01)
    replay.add_argument("--max-trades-per-day", type=int, default=4)
    replay.add_argument("--minimum-warmup-bars", type=int, default=205)
    replay.add_argument("--output", default="results/frozen-replay")

    compare_data = subparsers.add_parser(
        "compare-data", help="compare two independent normalized OHLC CSV feeds"
    )
    compare_data.add_argument("--reference", required=True, help="reference OHLC CSV")
    compare_data.add_argument("--comparison", required=True, help="second-provider OHLC CSV")
    compare_data.add_argument("--pip-size", type=float, default=0.0001)
    compare_data.add_argument("--output", default="results/source_comparison.json")

    gap_audit = subparsers.add_parser(
        "audit-gaps", help="export a row for every detected timestamp discontinuity"
    )
    gap_audit.add_argument("--input", required=True, help="source OHLC CSV")
    gap_audit.add_argument("--source-timeframe", default="M1", choices=SUPPORTED_TIMEFRAMES)
    gap_audit.add_argument("--gap-factor", type=float, default=1.5)
    gap_audit.add_argument("--output", default="data/gap_audit.csv")

    research = subparsers.add_parser("research", help="run multi-timeframe strategy research")
    research.add_argument("--input", required=True, help="source OHLC CSV (UTC timestamps recommended)")
    research.add_argument("--symbol", default="EURUSD")
    research.add_argument("--source-timeframe", default="M1", choices=SUPPORTED_TIMEFRAMES)
    research.add_argument(
        "--timeframes",
        nargs="+",
        default=["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"],
        choices=SUPPORTED_TIMEFRAMES,
    )
    research.add_argument("--pip-size", type=float, default=0.0001)
    research.add_argument("--price-basis", choices=["mid", "bid"], default="mid", help="price basis of input OHLC candles")
    _add_execution_cost_arguments(research)
    research.add_argument("--starting-balance", type=float, default=10_000.0)
    research.add_argument("--risk-per-trade", type=float, default=0.0025)
    research.add_argument("--max-daily-loss", type=float, default=0.01)
    research.add_argument("--max-trades-per-day", type=int, default=4)
    research.add_argument("--min-train-trades", type=int, default=30)
    research.add_argument("--min-validation-trades", type=int, default=30)
    research.add_argument("--max-drawdown-filter", type=float, default=0.20)
    research.add_argument("--output", default="results")

    evolve = subparsers.add_parser(
        "evolve", help="discover strategies using genetic programming and price-pattern recognition"
    )
    evolve.add_argument("--input", required=True, help="source OHLC CSV (UTC timestamps recommended)")
    evolve.add_argument("--symbol", default="EURUSD")
    evolve.add_argument("--source-timeframe", default="M1", choices=SUPPORTED_TIMEFRAMES)
    evolve.add_argument(
        "--timeframes", nargs="+",
        default=["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"],
        choices=SUPPORTED_TIMEFRAMES,
    )
    evolve.add_argument("--pip-size", type=float, default=0.0001)
    evolve.add_argument("--price-basis", choices=["mid", "bid"], default="mid", help="price basis of input OHLC candles")
    _add_execution_cost_arguments(evolve)
    evolve.add_argument("--starting-balance", type=float, default=10_000.0)
    evolve.add_argument("--risk-per-trade", type=float, default=0.0025)
    evolve.add_argument("--max-daily-loss", type=float, default=0.01)
    evolve.add_argument("--max-trades-per-day", type=int, default=4)
    evolve.add_argument("--min-train-trades", type=int, default=30)
    evolve.add_argument("--min-validation-trades", type=int, default=30)
    evolve.add_argument("--max-drawdown-filter", type=float, default=0.20)
    evolve.add_argument("--population", type=int, default=32)
    evolve.add_argument("--generations", type=int, default=12)
    evolve.add_argument("--pattern-clusters", type=int, default=6)
    evolve.add_argument("--validation-pool", type=int, default=24)
    evolve.add_argument("--seed", type=int, default=2026)
    evolve.add_argument("--no-adaptive-mutation", action="store_true", help="disable diversity/stagnation-based mutation adaptation")
    evolve.add_argument("--memory-file", default="results/evolution-memory.json", help="genome-only memory archive; old scores are not reused")
    evolve.add_argument("--research-only", action="store_true", help="skip the final chronological holdout; use when a previous holdout has already been inspected")
    evolve.add_argument("--output", default="results/evolutionary")

    evolve_ecology = subparsers.add_parser(
        "evolve-ecology",
        help="discover strategies using ecological evolution and habitat specialization",
    )
    evolve_ecology.add_argument("--input", required=True)
    evolve_ecology.add_argument("--symbol", default="EURUSD")
    evolve_ecology.add_argument(
        "--source-timeframe", default="M1", choices=SUPPORTED_TIMEFRAMES
    )
    evolve_ecology.add_argument(
        "--timeframes",
        nargs="+",
        default=["H1", "H4", "D1"],
        choices=SUPPORTED_TIMEFRAMES,
    )
    evolve_ecology.add_argument("--pip-size", type=float, default=0.0001)
    evolve_ecology.add_argument(
        "--price-basis", choices=["mid", "bid"], default="mid"
    )

    _add_execution_cost_arguments(evolve_ecology)

    evolve_ecology.add_argument("--starting-balance", type=float, default=10_000.0)
    evolve_ecology.add_argument("--risk-per-trade", type=float, default=0.0025)
    evolve_ecology.add_argument("--max-daily-loss", type=float, default=0.01)
    evolve_ecology.add_argument("--max-trades-per-day", type=int, default=4)

    evolve_ecology.add_argument("--min-train-trades", type=int, default=30)
    evolve_ecology.add_argument("--min-validation-trades", type=int, default=30)
    evolve_ecology.add_argument("--max-drawdown-filter", type=float, default=0.20)

    evolve_ecology.add_argument("--population", type=int, default=32)
    evolve_ecology.add_argument("--generations", type=int, default=12)
    evolve_ecology.add_argument("--pattern-clusters", type=int, default=6)
    evolve_ecology.add_argument(
        "--validation-pool",
        type=int,
        default=24,
        help=(
            "maximum training-ranked shortlist size; only its top candidate "
            "receives validation confirmation"
        ),
    )
    evolve_ecology.add_argument("--seed", type=int, default=2026)

    evolve_ecology.add_argument(
        "--no-adaptive-mutation", action="store_true"
    )
    evolve_ecology.add_argument(
        "--minimum-habitat-trades", type=int, default=5
    )
    evolve_ecology.add_argument(
        "--minimum-period-trades", type=int, default=5
    )
    evolve_ecology.add_argument(
        "--research-only",
        action="store_true",
        help="skip final chronological holdout evaluation",
    )
    evolve_ecology.add_argument(
        "--walk-forward",
        action="store_true",
        help="run expanding-window walk-forward research",
    )
    evolve_ecology.add_argument("--walk-forward-folds", type=int, default=3)
    evolve_ecology.add_argument(
        "--walk-forward-initial-train-fraction",
        type=float,
        default=0.60,
    )
    evolve_ecology.add_argument(
        "--walk-forward-step-fraction",
        type=float,
        default=0.10,
    )
    evolve_ecology.add_argument(
        "--output", default="results/ecological"
    )

    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()

    if (
        args.command == "evolve-ecology"
        and args.walk_forward
        and args.research_only
    ):
        parser.error(
            "--walk-forward cannot be combined with --research-only"
        )
    if args.command == "validate":
        frame = load_ohlc_csv(args.input)
        print(json.dumps(data_quality_summary(frame), indent=2))
        return

    if args.command == "aggregate-ticks":
        report = aggregate_tick_csv(
            args.input, args.output, timeframe=args.timeframe, pip_size=args.pip_size
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "replay-frozen":
        execution = _execution_config_from_args(args, include_spread=False)
        risk = RiskConfig(
            starting_balance=args.starting_balance,
            risk_per_trade=args.risk_per_trade,
            max_daily_loss=args.max_daily_loss,
            max_trades_per_day=args.max_trades_per_day,
        )
        report = replay_frozen_candidate(
            args.input, args.report, timeframe=args.timeframe,
            source_timeframe=args.source_timeframe, provider_name=args.provider_name,
            start_utc=args.from_utc, end_utc=args.to_utc, pip_size=args.pip_size,
            execution=execution, risk=risk, output_dir=args.output,
            minimum_warmup_bars=args.minimum_warmup_bars,
        )
        print(json.dumps({
            "candidate_id": report["original_candidate"]["candidate_id"],
            "provider_name": report["provider_name_user_supplied"],
            "metrics": report["metrics"],
            "report": str((Path(args.output) / "frozen_replay_report.json").resolve()),
        }, indent=2))
        return

    if args.command == "compare-data":
        report = compare_ohlc_csv_files(
            args.reference, args.comparison, pip_size=args.pip_size, output_path=args.output
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "audit-gaps":
        frame = load_ohlc_csv(args.input)
        audit = build_gap_audit(
            frame,
            TIMEFRAME_SECONDS[args.source_timeframe],
            gap_factor=args.gap_factor,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(output_path, index=False)
        counts = (
            audit["category"].value_counts().sort_index().to_dict()
            if not audit.empty else {}
        )
        print(json.dumps({
            "gap_events": int(len(audit)),
            "category_counts": counts,
            "output": str(output_path.resolve()),
            "synthetic_bars_created": False,
        }, indent=2))
        return

    frame = CSVOHLCProvider(args.input).load(args.symbol, args.source_timeframe)
    validate_source_interval(frame, args.source_timeframe)
    validate_source_timeframe(args.source_timeframe, args.timeframes)
    execution = _execution_config_from_args(args)
    risk = RiskConfig(
        starting_balance=args.starting_balance,
        risk_per_trade=args.risk_per_trade,
        max_daily_loss=args.max_daily_loss,
        max_trades_per_day=args.max_trades_per_day,
    )
    research_config = ResearchConfig(
        minimum_train_trades=args.min_train_trades,
        minimum_validation_trades=args.min_validation_trades,
        max_drawdown_filter=args.max_drawdown_filter,
    )

    if args.command == "research":
        report = run_research(
            frame,
            source_timeframe=args.source_timeframe,
            timeframes=args.timeframes,
            symbol=args.symbol,
            execution=execution,
            risk=risk,
            research=research_config,
            settings=BacktestSettings(pip_size=args.pip_size, price_basis=args.price_basis),
            output_dir=Path(args.output),
        )

    elif args.command == "evolve-ecology":
        evolution_settings = EvolutionSettings(
            population_size=args.population,
            generations=args.generations,
            pattern_clusters=args.pattern_clusters,
            validation_pool=args.validation_pool,
            seed=args.seed,
            adaptive_mutation=not args.no_adaptive_mutation,
        )

        if args.walk_forward:
            report = run_ecological_walk_forward_research(
                frame,
                source_timeframe=args.source_timeframe,
                timeframes=args.timeframes,
                symbol=args.symbol,
                execution=execution,
                risk=risk,
                research=research_config,
                evolution=evolution_settings,
                pip_size=args.pip_size,
                price_basis=args.price_basis,
                output_dir=Path(args.output),
                folds=args.walk_forward_folds,
                initial_train_fraction=args.walk_forward_initial_train_fraction,
                step_fraction=args.walk_forward_step_fraction,
                minimum_habitat_trades=args.minimum_habitat_trades,
                minimum_period_trades=args.minimum_period_trades,
            )
        else:
            report = run_ecological_research(
                frame,
                source_timeframe=args.source_timeframe,
                timeframes=args.timeframes,
                symbol=args.symbol,
                execution=execution,
                risk=risk,
                research=research_config,
                evolution=evolution_settings,
                pip_size=args.pip_size,
                price_basis=args.price_basis,
                output_dir=Path(args.output),
                run_final_test=not args.research_only,
                minimum_habitat_trades=args.minimum_habitat_trades,
                minimum_period_trades=args.minimum_period_trades,
            )
    print(f"Research complete. Reports saved to: {Path(args.output).resolve()}")
    for result in report["results"]:
        print(f"{result['timeframe']}: {result['status']}")


if __name__ == "__main__":
    main()
