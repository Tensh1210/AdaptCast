"""Tests for src/models/baseline.py and src/models/registry.py."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import mlflow
import pytest
from mlflow import MlflowClient

from src.data.download import download_data
from src.data.preprocess import run_preprocessing
from src.models.baseline import train_baseline
from src.models.registry import load_champion, register_champion


@pytest.fixture(scope="session")
def mlflow_tracking_uri(tmp_path_factory) -> str:
    """Return a session-scoped temporary MLflow tracking URI."""
    mlruns_dir = tmp_path_factory.mktemp("mlruns")
    db_path = mlruns_dir / "mlflow.db"
    uri = f"sqlite:///{db_path.as_posix()}"  # sqlite URI avoids filesystem-store deprecation
    mlflow.set_tracking_uri(uri)
    return uri


@pytest.fixture(scope="session")
def trained_model_and_run(mlflow_tracking_uri):
    """Download data, preprocess, and train the baseline once per session."""
    download_data()
    train_df, val_df, _ = run_preprocessing()
    model, run_id = train_baseline(train_df, val_df, experiment_name="test-driftpilot")
    return model, run_id


def test_train_returns_model_and_run_id(trained_model_and_run):
    model, run_id = trained_model_and_run
    assert isinstance(model, lgb.Booster), "Expected a lgb.Booster instance"
    assert isinstance(run_id, str) and run_id, "Expected a non-empty run_id string"


def test_metrics_logged_to_mlflow(trained_model_and_run, mlflow_tracking_uri):
    _, run_id = trained_model_and_run
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    metrics = client.get_run(run_id).data.metrics
    assert "val_rmse" in metrics, f"val_rmse not found in logged metrics: {metrics}"


def test_register_and_load_champion(trained_model_and_run, mlflow_tracking_uri):
    _, run_id = trained_model_and_run
    model_name = "test-driftpilot-forecaster"
    register_champion(run_id=run_id, model_name=model_name)
    loaded = load_champion(model_name=model_name)
    assert loaded is not None, "load_champion() returned None"
