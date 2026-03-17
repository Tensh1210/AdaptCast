"""AdaptCast Streamlit dashboard — entry point.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import time
from datetime import datetime

import httpx
import streamlit as st

from src.dashboard.data_loader import (
    BASE_URL,
    REFRESH_INTERVAL,
    fetch_drift_status,
    fetch_health,
    fetch_mlflow_runs,
    fetch_model_versions,
    fetch_prediction_history,
    fetch_stream_status,
)
from src.dashboard.components.drift_gauge import render_drift_gauge
from src.dashboard.components.forecast_chart import render_forecast_chart
from src.dashboard.components.model_timeline import render_model_timeline
from src.dashboard.components.prediction_chart import render_prediction_chart

# ---------------------------------------------------------------------------
# Page config (runs once on initial load)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AdaptCast", layout="wide")

# ---------------------------------------------------------------------------
# Fetch all data once per script run
# ---------------------------------------------------------------------------

health = fetch_health()
stream_status = fetch_stream_status()
drift_status = fetch_drift_status()
runs = fetch_mlflow_runs()
history = fetch_prediction_history()
model_versions = fetch_model_versions()

api_ok = health.get("status") == "ok"
stream_running: bool = stream_status.get("running", False)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("AdaptCast")
    st.markdown("---")

    # Service health badges
    if api_ok:
        st.success("API: ok")
    else:
        st.error(f"API: {health.get('status', 'unknown')}")

    if health.get("model_loaded", False):
        st.success("Model: loaded")
    else:
        st.warning("Model: not loaded")

    # Active champion indicator
    champion = next((v for v in model_versions if v.get("is_champion")), None)
    if champion:
        rmse_str = (
            f" — RMSE {champion['val_rmse']:.4f}"
            if champion.get("val_rmse") is not None
            else ""
        )
        st.caption(f"Active: v{champion['version']}{rmse_str}")

    if not api_ok:
        st.warning("Showing cached / fallback data.")

    st.markdown("---")

    # --- Stream control ---
    st.subheader("Stream Control")
    rows_processed: int = stream_status.get("rows_processed", 0)
    total_rows: int = stream_status.get("total_rows", 0)

    if stream_running:
        st.info(f"Streaming… {rows_processed:,} / {total_rows:,} rows")
        if total_rows > 0:
            st.progress(min(rows_processed / total_rows, 1.0))
        if st.button("Stop Stream", type="primary"):
            try:
                httpx.post(f"{BASE_URL}/stream/stop", timeout=5.0)
            except httpx.RequestError:
                st.error("Could not reach API.")
            st.cache_data.clear()
            st.rerun()
    else:
        if rows_processed > 0 and rows_processed >= total_rows > 0:
            st.success(f"Stream complete ({rows_processed:,} rows).")
        _speed_options = {"Slow (0.05s)": 0.05, "Normal (0.01s)": 0.01, "Fast (0s)": 0.0}
        speed_label = st.radio("Speed", list(_speed_options.keys()), index=1, horizontal=True)
        delay = _speed_options[speed_label]
        if st.button("Start Stream", type="primary", disabled=not api_ok):
            try:
                httpx.post(
                    f"{BASE_URL}/stream/start",
                    json={"delay_seconds": delay},
                    timeout=5.0,
                )
            except httpx.RequestError:
                st.error("Could not reach API.")
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # --- Drift control ---
    st.subheader("Drift Detectors")
    if st.button("Reset Detectors", disabled=not api_ok):
        try:
            httpx.post(f"{BASE_URL}/drift/reset", timeout=5.0)
        except httpx.RequestError:
            st.error("Could not reach API.")
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # --- Model rollback ---
    st.subheader("Model Rollback")
    if model_versions:
        version_options = {
            f"v{v['version']} — RMSE {v['val_rmse']:.4f}{' ✓' if v['is_champion'] else ''}"
            if v.get("val_rmse") is not None
            else f"v{v['version']}{' ✓' if v['is_champion'] else ''}": v["version"]
            for v in model_versions
        }
        selected_label = st.selectbox("Select version", list(version_options.keys()))
        selected_version = version_options[selected_label]
        champion_version = next(
            (v["version"] for v in model_versions if v["is_champion"]), None
        )
        if selected_version != champion_version:
            if st.button("Set as Champion", disabled=not api_ok):
                try:
                    resp = httpx.post(
                        f"{BASE_URL}/model/rollback",
                        json={"version": selected_version},
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        st.success(f"Rolled back to v{selected_version}.")
                    else:
                        st.error(resp.json().get("detail", "Rollback failed."))
                except httpx.RequestError:
                    st.error("Could not reach API.")
                st.cache_data.clear()
                st.rerun()
        else:
            st.caption("Already champion.")
    else:
        st.caption("No versions available.")

    st.markdown("---")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()

    if stream_running:
        st.caption(f"Auto-refresh every {REFRESH_INTERVAL}s")
    else:
        st.caption("Auto-refresh paused (stream idle).")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("AdaptCast — Live Monitoring Dashboard")

# Row 1: Drift status + Actual vs Predicted
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Drift Status")
    render_drift_gauge(drift_status)

with col2:
    st.subheader("Actual vs Predicted")
    render_prediction_chart(history, drift_status)

st.markdown("---")

# Row 2: RMSE history
st.subheader("RMSE History")
render_forecast_chart(runs, model_versions)

st.markdown("---")

# Row 3: Model run timeline
st.subheader("Model Run Timeline")
render_model_timeline(runs, model_versions)

# ---------------------------------------------------------------------------
# Auto-refresh: only when stream is actively running
# ---------------------------------------------------------------------------

if stream_running:
    time.sleep(REFRESH_INTERVAL)
    st.cache_data.clear()
    st.rerun()
