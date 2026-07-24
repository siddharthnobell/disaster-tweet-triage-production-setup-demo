import numpy as np
import pandas as pd
import pytest

from src.feature_pipeline import FEATURE_COLUMNS
from src.representations import BaselineRepresentation


def _toy_df():
    return pd.DataFrame({
        "cleaned_text": [
            "fire spreading fast near downtown",
            "lol traffic is such a disaster today",
            "earthquake felt across the city",
            "what a wonderful sunny morning",
        ],
        "word_count": [5, 7, 5, 5],
        "has_url": [0, 0, 1, 0],
        "hashtag_count": [0, 1, 0, 0],
        "mention_count": [0, 0, 0, 0],
        "has_keyword": [1, 1, 1, 0],
    })


def test_baseline_fit_transform_shape():
    df = _toy_df()
    rep = BaselineRepresentation(min_df=1)
    X = rep.fit_transform(df)
    assert X.shape[0] == len(df)
    assert X.shape[1] == len(rep.vectorizer.vocabulary_) + len(FEATURE_COLUMNS)


def test_baseline_transform_after_fit_uses_same_vocabulary():
    df = _toy_df()
    rep = BaselineRepresentation(min_df=1)
    rep.fit(df)
    unseen = pd.DataFrame({
        "cleaned_text": ["completely novel words never seen before"],
        "word_count": [6],
        "has_url": [0],
        "hashtag_count": [0],
        "mention_count": [0],
        "has_keyword": [0],
    })
    X = rep.transform(unseen)
    assert X.shape[1] == len(rep.vectorizer.vocabulary_) + len(FEATURE_COLUMNS)


def test_baseline_handcrafted_features_are_scaled():
    df = _toy_df()
    rep = BaselineRepresentation(min_df=1)
    rep.fit(df)
    scaled = rep.scaler.transform(df[FEATURE_COLUMNS])
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-8)


@pytest.mark.slow
def test_candidate_representation_smoke():
    pytest.importorskip("sentence_transformers")
    from src.representations import CandidateRepresentation

    try:
        rep = CandidateRepresentation()
    except Exception:
        pytest.skip("MiniLM model unavailable (no network / not cached)")

    df = _toy_df()
    X = rep.fit_transform(df)
    assert X.shape == (len(df), 384 + len(FEATURE_COLUMNS))
