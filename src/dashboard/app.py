"""DriftPilot Streamlit dashboard — entry point.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

from src.dashboard.data_loader import (
    fetch_drift_status,
    fetch_health,
    fetch_mlflow_runs,
    load_config,
)
from src.dashboard.components.drift_gauge import render_drift_gauge
from src.dashboard.components.forecast_chart import render_forecast_chart
from src.dashboard.components.model_timeline import render_model_timeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DriftPilot", layout="wide")

cfg = load_config()
refresh_interval: int = cfg["dashboard"]["refresh_interval_seconds"]

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------

health = fetch_health()
drift_status = fetch_drift_status()
runs = fetch_mlflow_runs()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("DriftPilot")
    st.markdown("---")

    # Service health badge
    if health.get("status") == "ok":
        st.success("API: ok")
    else:
        st.error(f"API: {health.get('status', 'unknown')}")

    # Model loaded indicator
    model_loaded: bool = health.get("model_loaded", False)
    if model_loaded:
        st.success("Model: loaded")
    else:
        st.warning("Model: not loaded")

    st.markdown("---")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    # Manual refresh button
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"Auto-refresh every {refresh_interval}s")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("DriftPilot — Live Monitoring Dashboard")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Drift Status")
    render_drift_gauge(drift_status)

with col2:
    st.subheader("RMSE History")
    render_forecast_chart(runs)

st.subheader("Model Run Timeline")
render_model_timeline(runs)

# ---------------------------------------------------------------------------
# Auto-refresh loop
# ---------------------------------------------------------------------------

time.sleep(refresh_interval)
st.rerun()
