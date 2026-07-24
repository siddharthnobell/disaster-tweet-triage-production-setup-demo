"""Batch ingestion: score any new tweet CSVs dropped in data/incoming/ and
write triage results to data/triaged/, reusing the exact same feature
pipeline + model as the live /predict endpoint (src.predict) so batch and
real-time scoring never disagree.

Processed input files are moved to data/incoming/processed/ so re-running
this script doesn't re-score the same tweets twice.

Run from the repo root:
    python -m scripts.ingest
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.predict import load_model, predict_df

REPO_ROOT = Path(__file__).resolve().parents[1]
INCOMING_DIR = REPO_ROOT / "data" / "incoming"
PROCESSED_DIR = INCOMING_DIR / "processed"
TRIAGED_DIR = REPO_ROOT / "data" / "triaged"


def run_ingestion(
    incoming_dir: Path = INCOMING_DIR,
    processed_dir: Path = PROCESSED_DIR,
    triaged_dir: Path = TRIAGED_DIR,
    model=None,
) -> list[Path]:
    """Score every CSV in `incoming_dir`, write results to `triaged_dir`,
    and move processed inputs to `processed_dir`. Returns the list of
    triaged output paths written. Directories/model are parameterized so
    tests can point this at a temp directory and a tiny in-memory model.
    """
    incoming_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    triaged_dir.mkdir(parents=True, exist_ok=True)

    incoming_files = sorted(incoming_dir.glob("*.csv"))
    if not incoming_files:
        print(f"no new files in {incoming_dir}")
        return []

    if model is None:
        model = load_model()

    written = []
    for path in incoming_files:
        df = pd.read_csv(path)
        if "text" not in df.columns:
            print(f"skipping {path.name}: no 'text' column")
            continue

        scored = predict_df(model, df)
        n_flagged = int(scored["label"].sum())
        print(f"{path.name}: scored {len(scored)} tweets, {n_flagged} flagged as real disasters")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = triaged_dir / f"{path.stem}_{timestamp}.csv"
        scored.to_csv(out_path, index=False)
        print(f"  -> {out_path}")
        written.append(out_path)

        path.rename(processed_dir / path.name)

    return written


if __name__ == "__main__":
    run_ingestion()
