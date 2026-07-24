"""Append-only prediction log written by the /predict endpoint.

Two things read this later: a human reviewer sampling it to produce
ground-truth labels for scripts/retrain.py, and anyone auditing what the
service actually returned for a given tweet at a given time.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "predictions.jsonl"


def log_prediction(record: dict[str, Any], log_path: Path = DEFAULT_LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
