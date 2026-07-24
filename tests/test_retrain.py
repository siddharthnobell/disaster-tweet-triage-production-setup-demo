import json

import joblib
import pandas as pd

from scripts.retrain import load_expanded_train, run_retraining
from src.feature_pipeline import build_features


def _raw_df(rows):
    return pd.DataFrame(rows)


def _write_split(path, rows):
    build_features(_raw_df(rows)).to_csv(path, index=False)


def _setup_splits(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir()

    _write_split(split_dir / "train.csv", {
        "keyword": ["fire", "fire", "flood", None, None, None, None, None],
        "text": [
            "massive fire burning through the forest right now",
            "wildfire evacuation orders issued for the whole valley",
            "flood waters rising fast, roads are closed",
            "lol this traffic jam is a disaster",
            "what a lovely sunny day for a picnic",
            "I love this new song so much",
            "having brunch with friends this morning",
            "my cat is sleeping on the couch again",
        ],
        "target": [1, 1, 1, 0, 0, 0, 0, 0],
    })
    _write_split(split_dir / "val.csv", {
        "keyword": ["fire", None],
        "text": ["evacuation ordered as wildfire spreads across the hills", "beautiful weather for a walk today"],
        "target": [1, 0],
    })
    _write_split(split_dir / "test.csv", {
        "keyword": ["flood", None],
        "text": ["flash flood warning issued for the river valley", "just finished a great book"],
        "target": [1, 0],
    })
    return split_dir


def test_load_expanded_train_without_new_labels_returns_original(tmp_path):
    split_dir = _setup_splits(tmp_path)
    empty_new_labels = tmp_path / "new_labels"
    train = pd.read_csv(split_dir / "train.csv")

    result = load_expanded_train(split_dir / "train.csv", empty_new_labels)
    assert len(result) == len(train)


def test_load_expanded_train_folds_in_new_batches(tmp_path):
    split_dir = _setup_splits(tmp_path)
    new_labels_dir = tmp_path / "new_labels"
    new_labels_dir.mkdir()
    pd.DataFrame({
        "keyword": ["fire"],
        "text": ["tornado tears through downtown leaving destruction"],
        "target": [1],
    }).to_csv(new_labels_dir / "batch_001.csv", index=False)

    train = pd.read_csv(split_dir / "train.csv")
    result = load_expanded_train(split_dir / "train.csv", new_labels_dir)
    assert len(result) == len(train) + 1
    assert "cleaned_text" in result.columns


def test_run_retraining_promotes_when_no_existing_metrics(tmp_path):
    split_dir = _setup_splits(tmp_path)
    outcome = run_retraining(
        split_dir=split_dir,
        new_labels_dir=tmp_path / "new_labels",
        model_path=tmp_path / "models" / "baseline.joblib",
        metrics_path=tmp_path / "metrics" / "baseline.json",
        archive_model_dir=tmp_path / "models" / "archive",
        archive_metrics_dir=tmp_path / "metrics" / "archive",
        candidate_metrics_dir=tmp_path / "metrics" / "retrain_candidates",
    )
    assert outcome["promoted"] is True
    assert outcome["current_f1"] is None
    assert (tmp_path / "models" / "baseline.joblib").exists()
    assert (tmp_path / "metrics" / "baseline.json").exists()


def test_run_retraining_does_not_promote_when_current_is_better(tmp_path):
    split_dir = _setup_splits(tmp_path)
    model_path = tmp_path / "models" / "baseline.joblib"
    metrics_path = tmp_path / "metrics" / "baseline.json"
    model_path.parent.mkdir(parents=True)
    metrics_path.parent.mkdir(parents=True)
    joblib.dump({"placeholder": True}, model_path)
    metrics_path.write_text(json.dumps({"val": {"f1": 0.99}, "test": {"f1": 0.999}}))

    outcome = run_retraining(
        split_dir=split_dir,
        new_labels_dir=tmp_path / "new_labels",
        model_path=model_path,
        metrics_path=metrics_path,
        archive_model_dir=tmp_path / "models" / "archive",
        archive_metrics_dir=tmp_path / "metrics" / "archive",
        candidate_metrics_dir=tmp_path / "metrics" / "retrain_candidates",
    )
    assert outcome["promoted"] is False
    # existing model/metrics untouched
    assert joblib.load(model_path) == {"placeholder": True}
    assert json.loads(metrics_path.read_text())["test"]["f1"] == 0.999
    candidates = list((tmp_path / "metrics" / "retrain_candidates").glob("*.json"))
    assert len(candidates) == 1


def test_run_retraining_archives_old_model_on_promotion(tmp_path):
    split_dir = _setup_splits(tmp_path)
    model_path = tmp_path / "models" / "baseline.joblib"
    metrics_path = tmp_path / "metrics" / "baseline.json"
    archive_model_dir = tmp_path / "models" / "archive"
    archive_metrics_dir = tmp_path / "metrics" / "archive"
    model_path.parent.mkdir(parents=True)
    metrics_path.parent.mkdir(parents=True)
    joblib.dump({"placeholder": True}, model_path)
    metrics_path.write_text(json.dumps({"val": {"f1": 0.0}, "test": {"f1": 0.0}}))

    outcome = run_retraining(
        split_dir=split_dir,
        new_labels_dir=tmp_path / "new_labels",
        model_path=model_path,
        metrics_path=metrics_path,
        archive_model_dir=archive_model_dir,
        archive_metrics_dir=archive_metrics_dir,
        candidate_metrics_dir=tmp_path / "metrics" / "retrain_candidates",
    )
    assert outcome["promoted"] is True
    assert model_path.exists()
    assert joblib.load(model_path) != {"placeholder": True}
    assert len(list(archive_model_dir.glob("*.joblib"))) == 1
    assert len(list(archive_metrics_dir.glob("*.json"))) == 1
