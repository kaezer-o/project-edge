from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


CHECKPOINT_SCHEMA_VERSION = 1


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def save_training_checkpoint(
    path: str | Path,
    *,
    run_signature: str,
    generation: int,
    population: list[dict[str, object]],
    rng_state: dict[str, object],
    history: list[dict[str, object]] | None = None,
) -> Path:
    """Atomically persist a resumable training state.

    The caller owns serialization of genomes and the meaning of run_signature.
    A checkpoint is only resumable when the caller supplies the same signature.
    """
    if not run_signature or not isinstance(run_signature, str):
        raise ValueError("run_signature must be a non-empty string")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    if not isinstance(population, list) or not population:
        raise ValueError("population must be a non-empty list")
    if not isinstance(rng_state, dict):
        raise ValueError("rng_state must be a dictionary")

    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_signature": run_signature,
        "generation": generation,
        "population": population,
        "rng_state": rng_state,
        "history": history or [],
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["checksum_sha256"] = _checksum(payload)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    expected_run_signature: str,
) -> dict[str, Any]:
    """Load and verify a checkpoint, rejecting incompatible or corrupted state."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read training checkpoint: {source}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Training checkpoint must contain a JSON object")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported training checkpoint schema version")
    if payload.get("run_signature") != expected_run_signature:
        raise ValueError("Checkpoint run signature does not match this training run")

    checksum = payload.get("checksum_sha256")
    unsigned_payload = {key: value for key, value in payload.items() if key != "checksum_sha256"}
    if not isinstance(checksum, str) or checksum != _checksum(unsigned_payload):
        raise ValueError("Training checkpoint checksum mismatch")

    generation = payload.get("generation")
    population = payload.get("population")
    rng_state = payload.get("rng_state")
    history = payload.get("history")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("Training checkpoint has an invalid generation")
    if not isinstance(population, list) or not population:
        raise ValueError("Training checkpoint has an invalid population")
    if not isinstance(rng_state, dict) or not isinstance(history, list):
        raise ValueError("Training checkpoint has invalid runtime state")
    return payload
