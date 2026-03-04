"""FastAPI application factory with lifespan startup for DriftPilot serving."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI

from src.drift.monitor import DriftMonitor
from src.drift.retrainer import DriftRetrainer
from src.models.registry import load_champion
from src.serving.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(Path("configs/serving.yaml")) as fh:
        cfg = yaml.safe_load(fh)

    model_name: str = cfg["mlflow"]["model_name"]
    tracking_uri: str = cfg["mlflow"]["tracking_uri"]

    mlflow.set_tracking_uri(tracking_uri)
    app.state.model = load_champion(model_name)
    app.state.monitor = DriftMonitor()
    val_df = pd.read_parquet("data/processed/val.parquet")
    app.state.retrainer = DriftRetrainer(val_df, model_name)
    app.state.model_name = model_name

    yield


def create_app() -> FastAPI:
    application = FastAPI(title="DriftPilot", lifespan=lifespan)
    application.include_router(router)
    return application


app = create_app()
