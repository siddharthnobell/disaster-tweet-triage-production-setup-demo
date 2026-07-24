"""Shared inference core, used by both the FastAPI service (app/main.py)
and the batch ingestion script (scripts/ingest.py) so real-time and batch
scoring can never compute predictions differently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from src.feature_pipeline import build_features

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "baseline.joblib"
DECISION_THRESHOLD = 0.5


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load a {"representation": ..., "classifier": ...} artifact saved by
    scripts/train_baseline.py or scripts/train_candidate.py.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"no model artifact at {model_path} - "
            "run `python -m scripts.train_baseline` first"
        )
    return joblib.load(model_path)


def predict_df(model: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Score raw tweets (must have a `text` column; `keyword` optional) and
    return the input rows with `probability` and `label` columns appended.

    Runs the exact same feature_pipeline.build_features step used at
    training time, then the model's own fitted representation, so serving
    can't silently drift from training.
    """
    working = df.copy()
    if "keyword" not in working.columns:
        working["keyword"] = None

    featurized = build_features(working)
    features = model["representation"].transform(featurized)
    probabilities = model["classifier"].predict_proba(features)[:, 1]

    out = df.copy()
    out["probability"] = probabilities
    out["label"] = (probabilities >= DECISION_THRESHOLD).astype(int)
    return out


def predict_one(model: dict[str, Any], text: str, keyword: Optional[str] = None) -> dict[str, Any]:
    """Score a single tweet. Thin wrapper around predict_df for the
    real-time /predict endpoint.
    """
    df = pd.DataFrame({"text": [text], "keyword": [keyword]})
    row = predict_df(model, df).iloc[0]
    return {"label": int(row["label"]), "probability": float(row["probability"])}
