"""Retraining pipeline (Section D): expand the training pool with newly
labeled tweets - e.g. a human reviewer's corrections after
scripts/monitor_batch.py or a spot-check of logs/predictions.jsonl flagged
something - retrain the baseline representation + classifier, and only
replace the deployed model if it doesn't regress on the frozen test
holdout. Same promotion-margin gate as Section B (src.promotion.should_promote),
so "should we ship this" is decided identically whether the new model came
from trying a different representation or from more training data.

`val` and `test` are never touched by retraining: they stay exactly the
splits carved out by scripts/make_splits.py, so every retrain is judged
against the same yardstick as every previous model. Only `train` grows,
by folding in every CSV under data/new_labels/.

Run from the repo root:
    python -m scripts.retrain
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.evaluate import save_metrics
from src.feature_pipeline import build_features, dedupe_labels
from src.promotion import should_promote
from src.representations import BaselineRepresentation
from src.train import fit_and_evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "data" / "processed" / "splits"
NEW_LABELS_DIR = REPO_ROOT / "data" / "new_labels"
MODEL_PATH = REPO_ROOT / "models" / "baseline.joblib"
METRICS_PATH = REPO_ROOT / "metrics" / "baseline.json"
ARCHIVE_MODEL_DIR = REPO_ROOT / "models" / "archive"
ARCHIVE_METRICS_DIR = REPO_ROOT / "metrics" / "archive"
CANDIDATE_METRICS_DIR = REPO_ROOT / "metrics" / "retrain_candidates"


def load_expanded_train(train_path: Path, new_labels_dir: Path) -> pd.DataFrame:
    """The original train split, plus every new-labels CSV folded in and
    featurized the same way as the original data. Duplicate texts (e.g. a
    tweet that shows up in both the original data and a review batch) are
    resolved by the same majority-vote rule used in Section A.
    """
    train = pd.read_csv(train_path)
    new_batches = sorted(new_labels_dir.glob("*.csv")) if new_labels_dir.exists() else []
    if not new_batches:
        return train

    new_rows = [build_features(pd.read_csv(p)) for p in new_batches]
    combined = pd.concat([train] + new_rows, ignore_index=True)
    deduped = dedupe_labels(combined)
    print(
        f"expanded train: {len(train)} original + {sum(len(r) for r in new_rows)} new "
        f"labels -> {len(deduped)} rows after dedupe"
    )
    return deduped


def run_retraining(
    split_dir: Path = SPLIT_DIR,
    new_labels_dir: Path = NEW_LABELS_DIR,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    archive_model_dir: Path = ARCHIVE_MODEL_DIR,
    archive_metrics_dir: Path = ARCHIVE_METRICS_DIR,
    candidate_metrics_dir: Path = CANDIDATE_METRICS_DIR,
) -> dict:
    """Run one retraining cycle. Returns a summary dict with the decision,
    so both `main()` and tests can inspect the outcome without parsing
    stdout.
    """
    train = load_expanded_train(split_dir / "train.csv", new_labels_dir)
    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")

    representation = BaselineRepresentation()
    results, clf = fit_and_evaluate(representation, train, {"val": val, "test": test})

    for name, metrics in results.items():
        print(f"\n{name} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    current_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    current_f1: Optional[float] = current_metrics["test"]["f1"] if current_metrics else None
    new_f1 = results["test"]["f1"]

    promoted = current_f1 is None or should_promote(new_f1, current_f1)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if promoted:
        if metrics_path.exists() and model_path.exists():
            archive_model_dir.mkdir(parents=True, exist_ok=True)
            archive_metrics_dir.mkdir(parents=True, exist_ok=True)
            model_path.rename(archive_model_dir / f"baseline_{timestamp}.joblib")
            metrics_path.rename(archive_metrics_dir / f"baseline_{timestamp}.json")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"representation": representation, "classifier": clf}, model_path)
        save_metrics(results, metrics_path)
        current_display = f"{current_f1:.4f}" if current_f1 is not None else "n/a"
        print(f"\ntest F1: current={current_display} new={new_f1:.4f} -> PROMOTED")
    else:
        candidate_metrics_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(results, candidate_metrics_dir / f"baseline_{timestamp}.json")
        print(f"\ntest F1: current={current_f1:.4f} new={new_f1:.4f} -> NOT PROMOTED (kept existing model)")

    return {"promoted": promoted, "current_f1": current_f1, "new_f1": new_f1, "metrics": results}


if __name__ == "__main__":
    run_retraining()
