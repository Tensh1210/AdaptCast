"""Centralised data-fetching layer for the AdaptCast dashboard.

All external calls live here so components stay pure presentation logic.
Each function is cached with ``@st.cache_data(ttl=...)`` and returns a safe
fallback value when FastAPI is unreachable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
import yaml

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
_REFRESH = _cfg["dashboard"]["refresh_interval_seconds"]
REFRESH_INTERVAL: int = _REFRESH

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
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.warning("fetch_health failed: %s", exc)
        return {"status": "unreachable", "model_loaded": False}


@st.cache_data(ttl=_REFRESH)
def fetch_drift_status() -> dict:
    """GET /drift/status — returns drift state dict or a zero-filled fallback."""
    try:
        resp = httpx.get(f"{BASE_URL}/drift/status", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.warning("fetch_drift_status failed: %s", exc)
        return {
            "drift_count": 0,
            "row_index": 0,
            "last_drift_row": None,
            "detectors": [],
        }


@st.cache_data(ttl=_REFRESH)
def fetch_model_versions() -> list[dict]:
    """GET /model/versions — returns all registered model versions."""
    try:
        resp = httpx.get(f"{BASE_URL}/model/versions", timeout=3.0)
        resp.raise_for_status()
        return resp.json().get("versions", [])
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.warning("fetch_model_versions failed: %s", exc)
        return []


@st.cache_data(ttl=1)
def fetch_stream_status() -> dict:
    """GET /stream/status — returns stream running state and progress (TTL=1s)."""
    try:
        resp = httpx.get(f"{BASE_URL}/stream/status", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.warning("fetch_stream_status failed: %s", exc)
        return {"running": False, "rows_processed": 0, "total_rows": 0}


@st.cache_data(ttl=_REFRESH)
def fetch_prediction_history() -> list[dict]:
    """GET /predictions/history — returns recent prediction/actual pairs."""
    try:
        resp = httpx.get(f"{BASE_URL}/predictions/history", timeout=3.0)
        resp.raise_for_status()
        return resp.json().get("points", [])
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.warning("fetch_prediction_history failed: %s", exc)
        return []


@st.cache_data(ttl=_REFRESH)
def fetch_mlflow_runs() -> list[dict[str, Any]]:
    """GET /mlflow/runs — fetch all MLflow run history via the API."""
    try:
        resp = httpx.get(f"{BASE_URL}/mlflow/runs", timeout=3.0)
        resp.raise_for_status()
        return resp.json().get("runs", [])
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.warning("fetch_mlflow_runs failed: %s", exc)
        return []
