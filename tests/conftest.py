import pandas as pd
import pytest

from src.feature_pipeline import build_features
from src.model_config import make_classifier
from src.representations import BaselineRepresentation


@pytest.fixture
def tiny_baseline_model():
    """A small but real fitted {representation, classifier} model, so tests
    exercise the actual feature_pipeline -> representation -> classifier
    path without depending on the committed (gitignored) models/baseline.joblib
    artifact produced by scripts/train_baseline.py.
    """
    train = pd.DataFrame({
        "keyword": ["fire", "fire", "flood", None, None, None],
        "text": [
            "massive fire burning through the forest right now",
            "wildfire evacuation orders issued for the whole valley",
            "flood waters rising fast, roads are closed",
            "lol this traffic jam is a disaster",
            "what a lovely sunny day for a picnic",
            "I love this new song so much",
        ],
        "target": [1, 1, 1, 0, 0, 0],
    })
    train = build_features(train)
    representation = BaselineRepresentation(min_df=1)
    X = representation.fit_transform(train)
    clf = make_classifier()
    clf.fit(X, train["target"])
    return {"representation": representation, "classifier": clf}
