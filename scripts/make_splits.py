"""Build the stratified train/val/test split used by every Section B model.

Run from the repo root:
    python -m scripts.make_splits
"""
from pathlib import Path

import pandas as pd

from src.data_split import make_splits

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = REPO_ROOT / "data" / "processed" / "train_features.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "splits"


def main() -> None:
    df = pd.read_csv(IN_PATH)
    train, val, test = make_splits(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        rate = split["target"].mean()
        print(f"{name}: {len(split)} rows, positive rate {rate:.3f}")


if __name__ == "__main__":
    main()
