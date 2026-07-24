import pandas as pd
import pytest

from src.predict import load_model, predict_df, predict_one


def test_predict_df_adds_label_and_probability_columns(tiny_baseline_model):
    new = pd.DataFrame({
        "text": ["another wildfire spreading through the valley"],
        "keyword": ["fire"],
    })
    scored = predict_df(tiny_baseline_model, new)
    assert "label" in scored.columns
    assert "probability" in scored.columns
    assert scored["label"].iloc[0] in (0, 1)
    assert 0.0 <= scored["probability"].iloc[0] <= 1.0


def test_predict_df_works_without_keyword_column(tiny_baseline_model):
    new = pd.DataFrame({"text": ["a totally unrelated happy tweet"]})
    scored = predict_df(tiny_baseline_model, new)
    assert len(scored) == 1
    assert "label" in scored.columns


def test_predict_df_preserves_original_columns(tiny_baseline_model):
    new = pd.DataFrame({"id": [99], "text": ["fire spreading"], "keyword": ["fire"]})
    scored = predict_df(tiny_baseline_model, new)
    assert "id" in scored.columns
    assert scored["id"].iloc[0] == 99


def test_predict_one_returns_label_and_probability(tiny_baseline_model):
    result = predict_one(tiny_baseline_model, "evacuation ordered as wildfire spreads", keyword="fire")
    assert set(result.keys()) == {"label", "probability"}
    assert result["label"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0


def test_load_model_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_baseline"):
        load_model(tmp_path / "does_not_exist.joblib")
