"""FastAPI route handlers for AdaptCast serving API."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from mlflow.exceptions import MlflowException

from src.drift.monitor import DriftEvent
from src.models.registry import load_champion, register_champion
from src.serving.schemas import (
    DetectorStatus,
    DriftResetResponse,
    DriftStatusResponse,
    HealthResponse,
    MLflowRunInfo,
    MLflowRunsResponse,
    ModelInfoResponse,
    ModelRollbackRequest,
    ModelRollbackResponse,
    ModelVersionInfo,
    ModelVersionsResponse,
    PredictHistoryResponse,
    PredictRequest,
    PredictResponse,
    PredictionPoint,
    StreamStartRequest,
    StreamStartResponse,
    StreamStatusResponse,
    StreamStopResponse,
)

router = APIRouter()

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_TARGET_COL = "OT"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    model_loaded = (
        hasattr(request.app.state, "model") and request.app.state.model is not None
    )
    return HealthResponse(status="ok", model_loaded=model_loaded)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    state = request.app.state
    df = pd.DataFrame([body.features])
    prediction = float(state.model.predict(df)[0])

    if body.y_true is not None:
        state.retrainer.ingest({**body.features, _TARGET_COL: body.y_true})
        event = state.monitor.update(prediction, body.y_true)

        state.prediction_history.append(
            PredictionPoint(
                row_index=state.monitor.row_index - 1,
                prediction=prediction,
                actual=body.y_true,
            )
        )

        if isinstance(event, DriftEvent):
            state.last_drift_event = event
            result = state.retrainer.handle(event)
            if result.promoted:
                async with state.model_lock:
                    state.model = load_champion(state.model_name)
    else:
        state.prediction_history.append(
            PredictionPoint(
                row_index=state.monitor.row_index,
                prediction=prediction,
                actual=None,
            )
        )

    return PredictResponse(prediction=prediction)


# ---------------------------------------------------------------------------
# Model info / versions / rollback
# ---------------------------------------------------------------------------

@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    state = request.app.state
    try:
        mv = state.mlflow_client.get_model_version_by_alias(
            name=state.model_name, alias="champion"
        )
        run = state.mlflow_client.get_run(mv.run_id)
        val_rmse = run.data.metrics.get("val_rmse")
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}")
    return ModelInfoResponse(
        name=state.model_name,
        version=mv.version,
        alias="champion",
        val_rmse=val_rmse,
    )


@router.get("/model/versions", response_model=ModelVersionsResponse)
async def model_versions(request: Request) -> ModelVersionsResponse:
    state = request.app.state
    try:
        champion_mv = state.mlflow_client.get_model_version_by_alias(
            name=state.model_name, alias="champion"
        )
        champion_version = champion_mv.version
        all_versions = state.mlflow_client.search_model_versions(
            f"name='{state.model_name}'"
        )
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}")

    versions: list[ModelVersionInfo] = []
    for mv in sorted(all_versions, key=lambda v: int(v.version)):
        try:
            run = state.mlflow_client.get_run(mv.run_id)
            val_rmse = run.data.metrics.get("val_rmse")
        except MlflowException:
            val_rmse = None
        versions.append(
            ModelVersionInfo(
                version=mv.version,
                val_rmse=val_rmse,
                run_id=mv.run_id[:8],
                is_champion=(mv.version == champion_version),
            )
        )
    return ModelVersionsResponse(versions=versions)


@router.post("/model/rollback", response_model=ModelRollbackResponse)
async def model_rollback(body: ModelRollbackRequest, request: Request) -> ModelRollbackResponse:
    state = request.app.state
    try:
        mv = state.mlflow_client.get_model_version(
            name=state.model_name, version=body.version
        )
        register_champion(run_id=mv.run_id, model_name=state.model_name)
        async with state.model_lock:
            state.model = load_champion(state.model_name)
    except (MlflowException, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ModelRollbackResponse(
        success=True,
        version=body.version,
        message=f"Rolled back to version {body.version}.",
    )


# ---------------------------------------------------------------------------
# MLflow run history
# ---------------------------------------------------------------------------

@router.get("/mlflow/runs", response_model=MLflowRunsResponse)
async def get_mlflow_runs(request: Request) -> MLflowRunsResponse:
    state = request.app.state
    try:
        experiments = state.mlflow_client.search_experiments(
            filter_string=f"name = '{state.experiment_name}'"
        )
        if not experiments:
            return MLflowRunsResponse(runs=[])
        exp_id = experiments[0].experiment_id
        raw_runs = state.mlflow_client.search_runs(
            experiment_ids=[exp_id],
            order_by=["start_time ASC"],
        )
        runs: list[MLflowRunInfo] = []
        for r in raw_runs:
            val_rmse = r.data.metrics.get("val_rmse") or r.data.metrics.get("rmse")
            start_ms = r.info.start_time
            start_iso = (
                datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
                if start_ms
                else ""
            )
            runs.append(
                MLflowRunInfo(
                    run_id=r.info.run_id[:8],
                    start_time=start_iso,
                    val_rmse=val_rmse,
                    status=r.info.status,
                )
            )
        runs.sort(key=lambda x: x.start_time)
        return MLflowRunsResponse(runs=runs)
    except MlflowException as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}")


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

@router.get("/drift/status", response_model=DriftStatusResponse)
async def drift_status(request: Request) -> DriftStatusResponse:
    state = request.app.state
    monitor = state.monitor
    last_event: DriftEvent | None = state.last_drift_event
    last_triggered_names = set(last_event.triggered_detectors) if last_event else set()

    detectors = [
        DetectorStatus(name=det.name, last_triggered=det.name in last_triggered_names)
        for det in monitor._detectors
    ]
    return DriftStatusResponse(
        row_index=monitor.row_index,
        drift_count=monitor.drift_count,
        last_drift_row=last_event.row_index if last_event else None,
        detectors=detectors,
    )


@router.post("/drift/reset", response_model=DriftResetResponse)
async def drift_reset(request: Request) -> DriftResetResponse:
    state = request.app.state
    state.monitor.reset()
    state.last_drift_event = None
    return DriftResetResponse(reset=True, message="All detectors reset.")


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------

@router.get("/predictions/history", response_model=PredictHistoryResponse)
async def predictions_history(request: Request) -> PredictHistoryResponse:
    points = list(request.app.state.prediction_history)
    return PredictHistoryResponse(points=points)


# ---------------------------------------------------------------------------
# Stream control
# ---------------------------------------------------------------------------

async def _stream_worker(state, test_path: Path, delay_seconds: float) -> None:
    """Background task: stream test data row-by-row through the prediction pipeline."""
    try:
        df = pd.read_parquet(test_path)
        state.stream_total_rows = len(df)
        cols = list(df.columns)

        for row_tuple in df.itertuples(index=False):
            if not state.stream_running:
                break

            row = dict(zip(cols, row_tuple))
            features = {k: v for k, v in row.items() if k != _TARGET_COL}
            y_true = row.get(_TARGET_COL)

            # Predict
            df_row = pd.DataFrame([features])
            prediction = float(state.model.predict(df_row)[0])

            # Store history
            state.prediction_history.append(
                PredictionPoint(
                    row_index=state.monitor.row_index,
                    prediction=prediction,
                    actual=y_true,
                )
            )

            # Drift detection + retraining
            if y_true is not None:
                state.retrainer.ingest({**features, _TARGET_COL: y_true})
                event = state.monitor.update(prediction, y_true)
                if isinstance(event, DriftEvent):
                    state.last_drift_event = event
                    result = state.retrainer.handle(event)
                    if result.promoted:
                        async with state.model_lock:
                            state.model = load_champion(state.model_name)

            await asyncio.sleep(delay_seconds)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        print(f"[stream] Worker error: {exc}")
    finally:
        state.stream_running = False


@router.post("/stream/start", response_model=StreamStartResponse)
async def stream_start(body: StreamStartRequest, request: Request) -> StreamStartResponse:
    state = request.app.state

    if state.stream_running:
        return StreamStartResponse(started=False, message="Stream is already running.")

    test_path = _PROJECT_ROOT / "data/processed/test.parquet"
    if not test_path.exists():
        raise HTTPException(status_code=404, detail=f"Test data not found: {test_path}")

    state.stream_running = True
    state.stream_task = asyncio.create_task(
        _stream_worker(state, test_path, body.delay_seconds)
    )
    return StreamStartResponse(started=True, message="Stream started.")


@router.post("/stream/stop", response_model=StreamStopResponse)
async def stream_stop(request: Request) -> StreamStopResponse:
    state = request.app.state
    if not state.stream_running:
        return StreamStopResponse(stopped=False, message="Stream is not running.")
    state.stream_running = False
    if state.stream_task and not state.stream_task.done():
        state.stream_task.cancel()
    return StreamStopResponse(stopped=True, message="Stream stopped.")


@router.get("/stream/status", response_model=StreamStatusResponse)
async def stream_status(request: Request) -> StreamStatusResponse:
    state = request.app.state
    return StreamStatusResponse(
        running=state.stream_running,
        rows_processed=state.monitor.row_index,
        total_rows=state.stream_total_rows,
    )
