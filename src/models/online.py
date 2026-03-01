"""Online incremental forecaster wrapping river's HoeffdingAdaptiveTreeRegressor."""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from river.tree import HoeffdingAdaptiveTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_COL = "OT"


class RiverModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapper so river models can be registered in the model registry.

    Loaded via ``mlflow.pyfunc.load_model()``. ``predict()`` accepts a DataFrame
    and returns a numpy array — same contract as the LightGBM pyfunc flavour.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        with open(context.artifacts["river_model"], "rb") as fh:
            self._river_model = pickle.load(fh)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> np.ndarray:
        return np.array(
            [
                self._river_model.predict_one(row.to_dict()) or 0.0
                for _, row in model_input.iterrows()
            ]
        )


class OnlineForecaster:
    """Incremental one-step-ahead forecaster using HoeffdingAdaptiveTreeRegressor.

    Designed for use after a drift event: call ``learn_one()`` on buffered rows,
    then ``evaluate_on_df()`` to compare against the LightGBM champion.

    Args:
        grace_period: Minimum samples before a split is attempted.
        leaf_prediction: Prediction strategy — "adaptive", "mean", or "model".
        model_selector_decay: Decay factor for the adaptive leaf selector.
    """

    def __init__(
        self,
        grace_period: int = 100,
        leaf_prediction: str = "adaptive",
        model_selector_decay: float = 0.95,
    ) -> None:
        self._params = dict(
            grace_period=grace_period,
            leaf_prediction=leaf_prediction,
            model_selector_decay=model_selector_decay,
        )
        self._model = HoeffdingAdaptiveTreeRegressor(**self._params)
        self.rows_seen: int = 0

    def learn_one(self, x: dict, y: float) -> None:
        """Incrementally update the model with one (features, target) pair."""
        self._model.learn_one(x, y)
        self.rows_seen += 1

    def predict_one(self, x: dict) -> float:
        """Predict for a single feature dict. Returns 0.0 before warm-up."""
        result = self._model.predict_one(x)
        return float(result) if result is not None else 0.0

    def evaluate_on_df(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COL,
    ) -> dict:
        """Pure inference pass — does NOT update the model.

        Args:
            df: DataFrame containing feature columns and the target column.
            target_col: Name of the target column.

        Returns:
            Dict with keys ``rmse``, ``mae``, ``r2``.
        """
        feature_cols = [c for c in df.columns if c != target_col]
        preds = np.array(
            [self.predict_one(row.to_dict()) for _, row in df[feature_cols].iterrows()]
        )
        y_true = df[target_col].values
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
            "mae": float(mean_absolute_error(y_true, preds)),
            "r2": float(r2_score(y_true, preds)),
        }

    def log_to_mlflow(
        self,
        experiment_name: str = "driftpilot",
        metrics: dict | None = None,
    ) -> str:
        """Pickle the river model, log it as an MLflow pyfunc artifact, and return run_id.

        Args:
            experiment_name: MLflow experiment to log into.
            metrics: Optional dict of metrics to log alongside the model.

        Returns:
            MLflow run_id of the logged run.
        """
        mlflow.set_experiment(experiment_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "river_model.pkl"
            with open(model_path, "wb") as fh:
                pickle.dump(self._model, fh)

            with mlflow.start_run() as run:
                if metrics:
                    mlflow.log_metrics(metrics)
                mlflow.log_param("rows_seen", self.rows_seen)
                mlflow.log_param("model_type", "HoeffdingAdaptiveTreeRegressor")
                mlflow.pyfunc.log_model(
                    python_model=RiverModelWrapper(),
                    artifact_path="model",
                    artifacts={"river_model": str(model_path)},
                )
                run_id = run.info.run_id

        return run_id
