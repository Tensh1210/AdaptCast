"""Async integration tests for the AdaptCast FastAPI service."""
from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from src.serving.app import create_app


@pytest.fixture
def mock_model():
    m = MagicMock()
    m.predict.return_value = np.array([42.5])
    return m


@pytest.fixture
def mock_monitor():
    m = MagicMock()
    m.row_index = 0
    m.drift_count = 0
    fake_det = MagicMock()
    fake_det.name = "ADWIN"
    m._detectors = [fake_det]
    return m


@pytest.fixture
def mock_retrainer():
    return MagicMock()


@pytest.fixture
def mock_mlflow_client():
    m = MagicMock()
    mock_mv = MagicMock()
    mock_mv.version = "3"
    mock_mv.run_id = "abc12345"
    mock_run = MagicMock()
    mock_run.data.metrics = {"val_rmse": 0.123}
    m.get_model_version_by_alias.return_value = mock_mv
    m.get_run.return_value = mock_run
    return m


@pytest.fixture
async def client(mock_model, mock_monitor, mock_retrainer, mock_mlflow_client):
    app = create_app()
    # httpx.ASGITransport does not trigger ASGI lifespan events, so inject
    # mock state directly instead of relying on the lifespan context.
    app.state.model = mock_model
    app.state.monitor = mock_monitor
    app.state.retrainer = mock_retrainer
    app.state.model_name = "test-model"
    app.state.experiment_name = "test-experiment"
    app.state.mlflow_client = mock_mlflow_client
    app.state.prediction_history = deque(maxlen=500)
    app.state.last_drift_event = None
    app.state.stream_running = False
    app.state.stream_task = None
    app.state.stream_total_rows = 0
    import asyncio
    app.state.model_lock = asyncio.Lock()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


async def test_predict_no_y_true(client, mock_monitor):
    resp = await client.post("/predict", json={"features": {"x": 1.0}})
    assert resp.status_code == 200
    assert resp.json()["prediction"] == pytest.approx(42.5)
    mock_monitor.update.assert_not_called()


async def test_predict_with_y_true(client, mock_monitor, mock_retrainer):
    mock_monitor.update.return_value = None  # no drift event
    resp = await client.post(
        "/predict", json={"features": {"x": 1.0}, "y_true": 40.0}
    )
    assert resp.status_code == 200
    mock_monitor.update.assert_called_once()
    mock_retrainer.ingest.assert_called_once()


async def test_model_info(client):
    resp = await client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "test-model"
    assert data["version"] == "3"
    assert data["val_rmse"] == pytest.approx(0.123)


async def test_drift_status(client):
    resp = await client.get("/drift/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["row_index"] == 0
    assert data["last_drift_row"] is None
    assert any(d["name"] == "ADWIN" for d in data["detectors"])


async def test_drift_reset(client, mock_monitor):
    resp = await client.post("/drift/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["reset"] is True
    mock_monitor.reset.assert_called_once()
