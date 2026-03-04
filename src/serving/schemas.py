"""Pydantic v2 request/response models for the DriftPilot serving API."""
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


class DetectorStatus(BaseModel):
    name: str


class DriftStatusResponse(BaseModel):
    row_index: int
    drift_count: int
    detectors: list[DetectorStatus]


class DriftResetResponse(BaseModel):
    reset: bool
    message: str
