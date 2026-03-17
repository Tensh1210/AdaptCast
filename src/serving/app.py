"""FastAPI application factory with lifespan startup for AdaptCast serving."""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mlflow import MlflowClient

from src.drift.monitor import DriftMonitor
from src.drift.retrainer import DriftRetrainer
from src.models.registry import load_champion
from src.serving.routes import router

_PROJECT_ROOT = Path(__file__).parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = _PROJECT_ROOT / "configs/serving.yaml"
    if not config_path.exists():
        raise RuntimeError(f"[serving] Config not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    model_name: str = cfg["mlflow"]["model_name"]
    tracking_uri: str = cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    try:
        app.state.model = load_champion(model_name)
    except RuntimeError as exc:
        raise RuntimeError(f"[serving] Startup failed — cannot load champion: {exc}") from exc

    val_path = _PROJECT_ROOT / "data/processed/val.parquet"
    if not val_path.exists():
        raise RuntimeError(f"[serving] Validation data not found: {val_path}. Run preprocess first.")
    val_df = pd.read_parquet(val_path)

    app.state.model_name = model_name
    app.state.experiment_name = cfg["mlflow"]["experiment_name"]
    app.state.monitor = DriftMonitor()
    app.state.retrainer = DriftRetrainer(val_df, model_name)
    app.state.model_lock = asyncio.Lock()
    app.state.mlflow_client = MlflowClient(tracking_uri)
    app.state.prediction_history: deque = deque(maxlen=500)
    app.state.last_drift_event = None
    app.state.stream_running = False
    app.state.stream_task = None
    app.state.stream_total_rows = 0

    yield

    # Shutdown — cancel any running stream task
    if app.state.stream_task and not app.state.stream_task.done():
        app.state.stream_task.cancel()


def create_app() -> FastAPI:
    application = FastAPI(title="AdaptCast", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
