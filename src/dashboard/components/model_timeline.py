"""MLflow run history table component."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_model_timeline(runs: list[dict]) -> None:
    """Render a dataframe table of MLflow runs sorted oldest → newest.

    Args:
        runs: List of run dicts from ``fetch_mlflow_runs()``.
              Expected keys: run_id, start_time, val_rmse, status.
    """
    if not runs:
        st.info("No MLflow runs found.")
        return

    df = pd.DataFrame(
        [
            {
                "Start Time": r.get("start_time", ""),
                "Run ID": r.get("run_id", ""),
                "val_rmse": r.get("val_rmse"),
                "Status": r.get("status", ""),
            }
            for r in runs
        ]
    )

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Most recent run = current champion")
