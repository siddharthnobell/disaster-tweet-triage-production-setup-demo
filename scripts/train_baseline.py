"""Train the TF-IDF baseline (Section B) and evaluate on val + test.

Run from the repo root:
    python -m scripts.train_baseline
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.evaluate import save_metrics
from src.representations import BaselineRepresentation
from src.train import fit_and_evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "data" / "processed" / "splits"
MODEL_PATH = REPO_ROOT / "models" / "baseline.joblib"
METRICS_PATH = REPO_ROOT / "metrics" / "baseline.json"


def main() -> None:
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")

    representation = BaselineRepresentation()
    results, clf = fit_and_evaluate(representation, train, {"val": val, "test": test})

    for name, metrics in results.items():
        print(f"\n{name} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"representation": representation, "classifier": clf}, MODEL_PATH)
    save_metrics(results, METRICS_PATH)

    print(f"\nsaved model -> {MODEL_PATH}")
    print(f"saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
