"""Section A entry point: clean data/raw/train.csv, build the 5 handcrafted
features via src.feature_pipeline, and save the result to
data/processed/train_features.csv.

Run from the repo root:
    python -m scripts.build_features
"""
import sys
from pathlib import Path

import pandas as pd

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.feature_pipeline import FEATURE_COLUMNS, clean_and_featurize

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "train.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "train_features.csv"


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    print(f"loaded {len(raw)} raw rows from {RAW_PATH}")

    featurized = clean_and_featurize(raw, dedupe=True)
    print(f"{len(featurized)} rows after dedup + featurization "
          f"({len(raw) - len(featurized)} dropped)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    featurized.to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")

    print("\ntarget distribution:")
    print(featurized["target"].value_counts())

    print("\nfeature summary:")
    print(featurized[FEATURE_COLUMNS].describe())

    print("\nsample rows:")
    cols = ["text", "cleaned_text"] + FEATURE_COLUMNS + ["target"]
    with pd.option_context("display.max_colwidth", 60):
        print(featurized[cols].sample(5, random_state=42).to_string())


if __name__ == "__main__":
    main()
