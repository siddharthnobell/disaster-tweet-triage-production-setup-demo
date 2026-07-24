import json

import pytest
from fastapi.testclient import TestClient

import app.main as app_main


@pytest.fixture
def logged_predictions(monkeypatch):
    """Capture calls to log_prediction instead of writing to logs/predictions.jsonl."""
    records = []
    monkeypatch.setattr(app_main, "log_prediction", lambda record: records.append(record))
    return records


@pytest.fixture
def client(monkeypatch, tiny_baseline_model, logged_predictions):
    monkeypatch.setattr(app_main, "load_model", lambda: tiny_baseline_model)
    with TestClient(app_main.app) as test_client:
        yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "baseline"


def test_predict_returns_label_and_probability(client):
    response = client.post(
        "/predict",
        json={"text": "wildfire evacuation ordered as flames spread", "keyword": "fire"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_name"] == "baseline"


def test_predict_without_keyword(client):
    response = client.post("/predict", json={"text": "just a normal happy tweet"})
    assert response.status_code == 200


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_logs_the_request_and_result(client, logged_predictions):
    client.post("/predict", json={"text": "wildfire spreads through the valley", "keyword": "fire"})
    assert len(logged_predictions) == 1
    entry = logged_predictions[0]
    assert entry["text"] == "wildfire spreads through the valley"
    assert entry["keyword"] == "fire"
    assert entry["label"] in (0, 1)
    assert 0.0 <= entry["probability"] <= 1.0


def test_log_prediction_writes_jsonl_line(tmp_path):
    from src.prediction_log import log_prediction

    log_path = tmp_path / "predictions.jsonl"
    log_prediction({"text": "test tweet", "label": 1, "probability": 0.9}, log_path=log_path)
    log_prediction({"text": "second tweet", "label": 0, "probability": 0.1}, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["text"] == "test tweet"
    assert "timestamp" in first


def test_health_reports_503_when_model_missing(monkeypatch):
    def _raise():
        raise FileNotFoundError("no model artifact - run training first")

    monkeypatch.setattr(app_main, "load_model", _raise)
    with TestClient(app_main.app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 503
