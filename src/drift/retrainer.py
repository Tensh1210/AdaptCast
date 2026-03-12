"""DriftRetrainer: dispatches online or full-retrain on DriftEvents, applies promotion gate."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from src.drift.monitor import DriftEvent
from src.models.baseline import evaluate, train_baseline
from src.models.online import OnlineForecaster
from src.models.registry import get_champion_rmse, register_champion

TARGET_COL = "OT"
_PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_CONFIG = _PROJECT_ROOT / "configs/model.yaml"


@dataclass
class PromotionResult:
    """Outcome of a retraining attempt.

    Attributes:
        promoted: True if the challenger beat the champion by ≥ 5 %.
        challenger_rmse: Validation RMSE of the challenger model.
        champion_rmse: Validation RMSE of the current champion (None if no champion exists).
        mode: "online" (Mode A) or "full_retrain" (Mode B).
        run_id: MLflow run_id of the challenger (None for online mode if not logged).
    """

    promoted: bool
    challenger_rmse: float
    champion_rmse: float | None
    mode: str
    run_id: str | None = field(default=None)


class DriftRetrainer:
    """Reacts to DriftEvents by retraining and evaluating a challenger model.

    Two modes (selected by DriftEvent.severity):
    - Mode A  "online"       — 1 detector fired: incremental river update (fast).
    - Mode B  "full_retrain" — ≥ 2 detectors fired: full LightGBM retrain (thorough).

    Promotion gate: ``challenger_rmse < champion_rmse × 0.95`` (5 % improvement).

    The validation set used for champion/challenger comparison starts as the last
    ``val_window_size`` rows of the provided ``val_df``.  Once the buffer contains
    enough data (> 2 × val_window_size rows), evaluation automatically shifts to a
    rolling window drawn from the most recent buffer rows, keeping the gate aligned
    with the current data distribution.

    Args:
        val_df: Initial validation DataFrame (features + target).
            Sliced to the last ``val_window_size`` rows at construction time.
        model_name: Registered model name in the MLflow Model Registry.
        config_path: Path to ``configs/model.yaml``.
        experiment_name: MLflow experiment name for full-retrain runs.
    """

    def __init__(
        self,
        val_df: pd.DataFrame,
        model_name: str,
        config_path: Path = MODEL_CONFIG,
        experiment_name: str = "driftpilot",
    ) -> None:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)

        retrain_window = cfg["training"]["retrain_window_size"]
        self._val_window = cfg["training"]["val_window_size"]

        self._model_name = model_name
        self._experiment_name = experiment_name
        self._static_val_df = val_df.iloc[-self._val_window:]
        self._buffer: deque[dict] = deque(maxlen=retrain_window)
        self._online: OnlineForecaster | None = None

    def ingest(self, row: dict) -> None:
        """Buffer every incoming stream row. Call this for each row regardless of drift."""
        self._buffer.append(row)

    def handle(self, event: DriftEvent) -> PromotionResult:
        """Dispatch to Mode A or Mode B based on event severity.

        Args:
            event: The DriftEvent emitted by DriftMonitor.

        Returns:
            PromotionResult describing whether the challenger was promoted.
        """
        print(
            f"[retrainer] DriftEvent at row {event.row_index} | "
            f"detectors={event.triggered_detectors} | severity={event.severity}"
        )
        if event.severity == "high":
            return self._full_retrain()
        return self._online_update()

    @staticmethod
    def should_promote(challenger_rmse: float, champion_rmse: float) -> bool:
        """Return True iff challenger is ≥ 5 % better than champion."""
        return challenger_rmse < champion_rmse * 0.95

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_champion_rmse(self) -> float:
        """Return champion RMSE, or inf if no champion is registered yet."""
        rmse = get_champion_rmse(self._model_name)
        return rmse if rmse is not None else float("inf")

    def _get_current_val_df(self) -> pd.DataFrame:
        """Return a rolling validation window aligned to the current data distribution.

        If the buffer has more than 2 × val_window rows, the most recent
        val_window rows are used so that the promotion gate reflects recent
        behaviour rather than the (potentially stale) initial validation set.
        Falls back to the static validation set when the buffer is too small.
        """
        if len(self._buffer) > self._val_window * 2:
            recent = pd.DataFrame(list(self._buffer)).iloc[-self._val_window:]
            for col in recent.columns:
                recent[col] = pd.to_numeric(recent[col], errors="coerce")
            recent = recent.dropna()
            if len(recent) >= self._val_window // 2:
                return recent
        return self._static_val_df

    def _online_update(self) -> PromotionResult:
        """Mode A: incremental HoeffdingAdaptiveTree update on buffered rows."""
        if self._online is None:
            self._online = OnlineForecaster()

        grace_period = self._online._params["grace_period"]
        if len(self._buffer) < grace_period:
            print(
                f"[retrainer] Mode A — buffer has {len(self._buffer)} rows, "
                f"less than grace_period={grace_period}; predictions may be naive."
            )

        val_df = self._get_current_val_df()
        feature_cols = [c for c in val_df.columns if c != TARGET_COL]

        # Learn on every buffered row
        for row in self._buffer:
            x = {k: v for k, v in row.items() if k != TARGET_COL}
            y = row[TARGET_COL]
            self._online.learn_one(x, y)

        challenger_rmse = self._online.evaluate_on_df(val_df)["rmse"]
        champion_rmse = self._get_champion_rmse()
        promoted = False
        run_id: str | None = None

        if self.should_promote(challenger_rmse, champion_rmse):
            run_id = self._online.log_to_mlflow(
                experiment_name=self._experiment_name,
                metrics={"val_rmse": challenger_rmse},
            )
            register_champion(run_id=run_id, model_name=self._model_name)
            promoted = True
            print(
                f"[retrainer] Mode A — PROMOTED online model "
                f"(RMSE {challenger_rmse:.4f} vs champion {champion_rmse:.4f})"
            )
        else:
            print(
                f"[retrainer] Mode A — rejected (RMSE {challenger_rmse:.4f} "
                f"vs champion {champion_rmse:.4f}, need < {champion_rmse * 0.95:.4f})"
            )

        return PromotionResult(
            promoted=promoted,
            challenger_rmse=challenger_rmse,
            champion_rmse=champion_rmse if champion_rmse != float("inf") else None,
            mode="online",
            run_id=run_id,
        )

    def _full_retrain(self) -> PromotionResult:
        """Mode B: full LightGBM retrain on the rolling data buffer."""
        if not self._buffer:
            print("[retrainer] Mode B — buffer empty, skipping retrain.")
            champion_rmse = self._get_champion_rmse()
            return PromotionResult(
                promoted=False,
                challenger_rmse=float("inf"),
                champion_rmse=champion_rmse if champion_rmse != float("inf") else None,
                mode="full_retrain",
            )

        buffer_df = pd.DataFrame(list(self._buffer))

        # Ensure column types are numeric (dicts from stream may carry mixed types)
        initial_rows = len(buffer_df)
        for col in buffer_df.columns:
            buffer_df[col] = pd.to_numeric(buffer_df[col], errors="coerce")
        buffer_df = buffer_df.dropna()
        rows_dropped = initial_rows - len(buffer_df)
        if rows_dropped > 0:
            print(
                f"[retrainer] Mode B — dropped {rows_dropped}/{initial_rows} rows "
                "after numeric type coercion."
            )

        if buffer_df.empty:
            print("[retrainer] Mode B — buffer produced empty DataFrame after dropna.")
            champion_rmse = self._get_champion_rmse()
            return PromotionResult(
                promoted=False,
                challenger_rmse=float("inf"),
                champion_rmse=champion_rmse if champion_rmse != float("inf") else None,
                mode="full_retrain",
            )

        val_df = self._get_current_val_df()

        # When using the rolling buffer val, exclude those rows from training
        # to avoid data leakage between train and validation sets.
        if val_df is not self._static_val_df and len(buffer_df) > len(val_df):
            train_df = buffer_df.iloc[: -len(val_df)]
        else:
            train_df = buffer_df

        model, run_id = train_baseline(
            train_df=train_df,
            val_df=val_df,
            experiment_name=self._experiment_name,
        )

        # Retrieve val_rmse logged by train_baseline
        client = mlflow.MlflowClient()
        challenger_rmse = client.get_run(run_id).data.metrics.get("val_rmse", float("inf"))
        champion_rmse = self._get_champion_rmse()
        promoted = False

        if self.should_promote(challenger_rmse, champion_rmse):
            register_champion(run_id=run_id, model_name=self._model_name)
            promoted = True
            print(
                f"[retrainer] Mode B — PROMOTED LightGBM retrain "
                f"(RMSE {challenger_rmse:.4f} vs champion {champion_rmse:.4f})"
            )
        else:
            print(
                f"[retrainer] Mode B — rejected (RMSE {challenger_rmse:.4f} "
                f"vs champion {champion_rmse:.4f}, need < {champion_rmse * 0.95:.4f})"
            )

        return PromotionResult(
            promoted=promoted,
            challenger_rmse=challenger_rmse,
            champion_rmse=champion_rmse if champion_rmse != float("inf") else None,
            mode="full_retrain",
            run_id=run_id,
        )
