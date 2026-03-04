"""FastAPI route handlers for all 5 DriftPilot endpoints."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Request
from mlflow import MlflowClient

from src.drift.monitor import DriftEvent
from src.models.registry import load_champion
from src.serving.schemas import (
    DetectorStatus,
    DriftResetResponse,
    DriftStatusResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    model_loaded = (
        hasattr(request.app.state, "model") and request.app.state.model is not None
    )
    return HealthResponse(status="ok", model_loaded=model_loaded)


@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    state = request.app.state
    df = pd.DataFrame([body.features])
    prediction = float(state.model.predict(df)[0])

    if body.y_true is not None:
        state.retrainer.ingest({**body.features, "OT": body.y_true})
        event = state.monitor.update(prediction, body.y_true)
        if isinstance(event, DriftEvent):
            result = state.retrainer.handle(event)
            if result.promoted:
                state.model = load_champion(state.model_name)

    return PredictResponse(prediction=prediction)


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    state = request.app.state
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name=state.model_name, alias="champion")
    run = client.get_run(mv.run_id)
    val_rmse = run.data.metrics.get("val_rmse")
    return ModelInfoResponse(
        name=state.model_name,
        version=mv.version,
        alias="champion",
        val_rmse=val_rmse,
    )


@router.get("/drift/status", response_model=DriftStatusResponse)
async def drift_status(request: Request) -> DriftStatusResponse:
    monitor = request.app.state.monitor
    detectors = [DetectorStatus(name=det.name) for det in monitor._detectors]
    return DriftStatusResponse(
        row_index=monitor.row_index,
        drift_count=monitor.drift_count,
        detectors=detectors,
    )


@router.post("/drift/reset", response_model=DriftResetResponse)
async def drift_reset(request: Request) -> DriftResetResponse:
    request.app.state.monitor.reset()
    return DriftResetResponse(reset=True, message="All detectors reset.")
