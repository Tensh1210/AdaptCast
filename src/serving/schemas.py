"""Pydantic v2 request/response models for the AdaptCast serving API."""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictRequest(BaseModel):
    features: dict[str, float]
    y_true: float | None = None


class PredictResponse(BaseModel):
    prediction: float


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    alias: str
    val_rmse: float | None = None


class ModelVersionInfo(BaseModel):
    version: str
    val_rmse: float | None
    run_id: str
    is_champion: bool


class ModelVersionsResponse(BaseModel):
    versions: list[ModelVersionInfo]


class ModelRollbackRequest(BaseModel):
    version: str


class ModelRollbackResponse(BaseModel):
    success: bool
    version: str
    message: str


class DetectorStatus(BaseModel):
    name: str
    last_triggered: bool = False


class DriftStatusResponse(BaseModel):
    row_index: int
    drift_count: int
    last_drift_row: int | None
    detectors: list[DetectorStatus]


class DriftResetResponse(BaseModel):
    reset: bool
    message: str


class StreamStartRequest(BaseModel):
    delay_seconds: float = 0.01


class StreamStartResponse(BaseModel):
    started: bool
    message: str


class StreamStopResponse(BaseModel):
    stopped: bool
    message: str


class StreamStatusResponse(BaseModel):
    running: bool
    rows_processed: int
    total_rows: int


class PredictionPoint(BaseModel):
    row_index: int
    prediction: float
    actual: float | None


class PredictHistoryResponse(BaseModel):
    points: list[PredictionPoint]


class MLflowRunInfo(BaseModel):
    run_id: str
    start_time: str
    val_rmse: float | None
    status: str


class MLflowRunsResponse(BaseModel):
    runs: list[MLflowRunInfo]
