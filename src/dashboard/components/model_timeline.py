"""MLflow run history table component."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_model_timeline(runs: list[dict], model_versions: list[dict]) -> None:
    """Render a dataframe table of MLflow runs sorted oldest → newest.

    Args:
        runs: List of run dicts from ``fetch_mlflow_runs()``.
              Expected keys: run_id, start_time, val_rmse, status.
        model_versions: List of version dicts from ``fetch_model_versions()``.
                        Used to mark the actual champion run.
    """
    if not runs:
        st.info("No MLflow runs found.")
        return

    champion_run_id = next(
        (v.get("run_id") for v in model_versions if v.get("is_champion")),
        None,
    )

    df = pd.DataFrame(
        [
            {
                "Start Time (UTC)": r.get("start_time", ""),
                "Run ID": r.get("run_id", ""),
                "val_rmse": (
                    f"{r['val_rmse']:.4f}" if r.get("val_rmse") is not None else "—"
                ),
                "Status": r.get("status", ""),
                "Champion": "✓" if r.get("run_id") == champion_run_id else "",
            }
            for r in runs
        ]
    )

    st.dataframe(df, use_container_width=True, hide_index=True)
    caption = f"Showing {len(runs)} run(s)."
    if champion_run_id:
        caption += f" Champion: {champion_run_id}."
    st.caption(caption)
