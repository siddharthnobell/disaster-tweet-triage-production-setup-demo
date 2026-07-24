"""Shared fit + evaluate logic, reused by scripts/train_baseline.py,
scripts/train_candidate.py, and scripts/retrain.py so all three train a
representation/classifier pair and score it against a set of splits the
exact same way.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluate import compute_metrics
from src.model_config import make_classifier


def fit_and_evaluate(
    representation: Any, train: pd.DataFrame, eval_splits: dict[str, pd.DataFrame]
) -> tuple[dict[str, dict], Any]:
    """Fit `representation` + a classifier on `train`, then score on every
    split in `eval_splits` (e.g. {"val": val_df, "test": test_df}).

    Returns (metrics_by_split, fitted_classifier). `representation` is
    fitted in place.
    """
    X_train = representation.fit_transform(train)
    y_train = train["target"]

    clf = make_classifier()
    clf.fit(X_train, y_train)

    results = {}
    for name, split in eval_splits.items():
        X = representation.transform(split)
        y_true = split["target"]
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]
        results[name] = compute_metrics(y_true, y_pred, y_prob)

    return results, clf
