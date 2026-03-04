"""Centralised data-fetching layer for the DriftPilot dashboard.

All external calls live here so components stay pure presentation logic.
Each function is cached with ``@st.cache_data(ttl=...)`` and returns a safe
fallback value when FastAPI or MLflow is unreachable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
import yaml
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "serving.yaml"


def load_config() -> dict:
    """Read serving.yaml once; cached at module level."""
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


_cfg = load_config()
_PORT = _cfg["api"]["port"]
BASE_URL = f"http://localhost:{_PORT}"
_MLFLOW_URI = _cfg["mlflow"]["tracking_uri"]
_EXPERIMENT = _cfg["mlflow"]["experiment_name"]
_REFRESH = _cfg["dashboard"]["refresh_interval_seconds"]

# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------


@st.cache_data(ttl=_REFRESH)
def fetch_health() -> dict:
    """GET /health — returns service health dict or a safe unreachable fallback."""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("fetch_health failed: %s", exc)
        return {"status": "unreachable", "model_loaded": False}


@st.cache_data(ttl=_REFRESH)
def fetch_drift_status() -> dict:
    """GET /drift/status — returns drift state dict or a zero-filled fallback."""
    try:
        resp = httpx.get(f"{BASE_URL}/drift/status", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("fetch_drift_status failed: %s", exc)
        return {
            "drift_detected": False,
            "drift_count": 0,
            "row_index": 0,
            "detectors": {},
        }


@st.cache_data(ttl=_REFRESH)
def fetch_model_info() -> dict | None:
    """GET /model/info — returns model metadata dict or None when unavailable."""
    try:
        resp = httpx.get(f"{BASE_URL}/model/info", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("fetch_model_info failed: %s", exc)
        return None


@st.cache_data(ttl=_REFRESH)
def fetch_mlflow_runs() -> list[dict[str, Any]]:
    """Query MLflow for all runs in the experiment, sorted oldest → newest.

    Each dict contains:
        run_id    – first 8 characters of the MLflow run ID
        start_time – ISO 8601 string (UTC)
        val_rmse  – float or None
        status    – MLflow run status string
    """
    try:
        client = MlflowClient(tracking_uri=_MLFLOW_URI)
        experiments = client.search_experiments(filter_string=f"name = '{_EXPERIMENT}'")
        if not experiments:
            return []
        exp_id = experiments[0].experiment_id
        mlflow_runs = client.search_runs(
            experiment_ids=[exp_id],
            order_by=["start_time ASC"],
        )
        runs: list[dict[str, Any]] = []
        for r in mlflow_runs:
            val_rmse = r.data.metrics.get("val_rmse") or r.data.metrics.get("rmse")
            start_ms = r.info.start_time  # epoch milliseconds
            import datetime
            start_iso = (
                datetime.datetime.utcfromtimestamp(start_ms / 1000).isoformat()
                if start_ms
                else ""
            )
            runs.append(
                {
                    "run_id": r.info.run_id[:8],
                    "start_time": start_iso,
                    "val_rmse": val_rmse,
                    "status": r.info.status,
                }
            )
        # Already ordered ASC by MLflow query, but re-sort defensively
        runs.sort(key=lambda x: x["start_time"])
        return runs
    except Exception as exc:
        logger.warning("fetch_mlflow_runs failed: %s", exc)
        return []
