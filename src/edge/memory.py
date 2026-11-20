from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .evolution import StrategyGenome, genome_from_dict


MEMORY_VERSION = 1


def load_genome_memory(
    path: str | Path | None,
    *,
    symbol: str,
    timeframe: str,
    allowed_features: Iterable[str] | None = None,
) -> list[StrategyGenome]:
    """Load genotypes only; stored fitness and test results are never imported."""
    if path is None:
        return []
    source = Path(path)
    if not source.exists():
        return []
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read evolutionary memory {source}: {exc}") from exc
    if payload.get("memory_version") != MEMORY_VERSION:
        raise ValueError(f"Unsupported evolutionary memory version in {source}")

    allowed = set(allowed_features) if allowed_features is not None else None
    seeds: list[StrategyGenome] = []
    seen: set[str] = set()
    for item in payload.get("organisms", []):
        if str(item.get("symbol", "")).upper() != symbol.upper():
            continue
        if str(item.get("timeframe", "")).upper() != timeframe.upper():
            continue
        try:
            genome = genome_from_dict(item["genome"])
        except (KeyError, TypeError, ValueError):
            continue
        required = {c.feature for c in genome.long_conditions + genome.short_conditions}
        if (allowed is not None and not required.issubset(allowed)) or not genome.long_conditions or not genome.short_conditions:
            continue
        fingerprint = genome.fingerprint()
        if fingerprint not in seen:
            seen.add(fingerprint)
            seeds.append(genome)
    return seeds


def update_genome_memory(
    path: str | Path | None,
    *,
    symbol: str,
    timeframe: str,
    genomes: Iterable[StrategyGenome],
    limit: int = 50,
) -> dict[str, object]:
    """Persist only genomes and provenance, never validation/final-test scores."""
    if path is None:
        return {"enabled": False, "stored": 0}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot update evolutionary memory {target}: {exc}") from exc
        if payload.get("memory_version") != MEMORY_VERSION:
            raise ValueError(f"Unsupported evolutionary memory version in {target}")
    else:
        payload = {"memory_version": MEMORY_VERSION, "organisms": []}

    existing = [
        item for item in payload.get("organisms", [])
        if not (
            str(item.get("symbol", "")).upper() == symbol.upper()
            and str(item.get("timeframe", "")).upper() == timeframe.upper()
        )
    ]
    new_items = []
    seen: set[str] = set()
    for genome in genomes:
        fingerprint = genome.fingerprint()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        new_items.append({
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "fingerprint": fingerprint,
            "genome": genome.to_dict(),
            "source": "training_only_archive",
        })
        if len(new_items) >= limit:
            break
    payload["organisms"] = existing + new_items
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return {"enabled": True, "stored": len(new_items), "path": str(target.resolve()),
            "contains_fitness_scores": False, "contains_validation_or_test_metrics": False}
