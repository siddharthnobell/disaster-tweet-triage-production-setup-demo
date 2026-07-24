"""FastAPI service for real-time disaster-tweet triage.

Serves the model promoted in Section B (the TF-IDF baseline - see
metrics/baseline.json and metrics/candidate.json for why). Loading only
this model means the service never has to import torch/sentence-transformers
at all, a real deployment-size win from the candidate not being promoted.

Run from the repo root:
    uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from app.schemas import HealthResponse, PredictResponse, TweetRequest
from src.predict import DEFAULT_MODEL_PATH, load_model, predict_one
from src.prediction_log import log_prediction

MODEL_STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        MODEL_STATE["model"] = load_model()
        MODEL_STATE["name"] = DEFAULT_MODEL_PATH.stem
        MODEL_STATE["error"] = None
    except FileNotFoundError as exc:
        MODEL_STATE["model"] = None
        MODEL_STATE["name"] = None
        MODEL_STATE["error"] = str(exc)
    yield
    MODEL_STATE.clear()


app = FastAPI(title="Disaster Tweet Triage", lifespan=lifespan)


def _require_model() -> dict[str, Any]:
    model = MODEL_STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail=MODEL_STATE.get("error", "model not loaded"))
    return model


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    _require_model()
    return HealthResponse(status="ok", model_name=MODEL_STATE["name"])


@app.post("/predict", response_model=PredictResponse)
def predict(request: TweetRequest) -> PredictResponse:
    model = _require_model()
    result = predict_one(model, request.text, request.keyword)
    log_prediction({
        "text": request.text,
        "keyword": request.keyword,
        "label": result["label"],
        "probability": result["probability"],
        "model_name": MODEL_STATE["name"],
    })
    return PredictResponse(**result, model_name=MODEL_STATE["name"])
