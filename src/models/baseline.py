"""LightGBM baseline model: time-series CV, MLflow tracking, and final fit."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

TARGET_COL = "OT"
_PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = _PROJECT_ROOT / "configs/model.yaml"


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"[baseline] Config not found: {config_path}")
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return df[feature_cols], df[TARGET_COL]


def evaluate(model: lgb.Booster, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute RMSE, MAE, and R² for a fitted model on a given dataset.

    Args:
        model: A fitted LightGBM Booster.
        X: Feature DataFrame.
        y: True target values.

    Returns:
        Dict with keys ``rmse``, ``mae``, ``r2``.
    """
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config_path: Path = CONFIG_PATH,
    experiment_name: str = "driftpilot",
) -> tuple[lgb.Booster, str]:
    """Train a LightGBM baseline model with time-series CV and log to MLflow.

    Args:
        train_df: Training split (features + target).
        val_df: Validation split (features + target).
        config_path: Path to ``configs/model.yaml``.
        experiment_name: MLflow experiment name.

    Returns:
        A tuple of ``(fitted_booster, mlflow_run_id)``.
    """
    cfg = _load_config(config_path)
    lgbm_params = cfg["lgbm"]
    cv_splits = cfg["training"]["cv_splits"]

    X_train, y_train = _split_xy(train_df)
    X_val, y_val = _split_xy(val_df)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(lgbm_params)

        # Walk-forward cross-validation on the training set
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_rmses: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

            fold_model = lgb.train(
                {**lgbm_params, "verbosity": -1},
                dtrain,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            fold_metrics = evaluate(fold_model, X_va, y_va)
            cv_rmses.append(fold_metrics["rmse"])
            mlflow.log_metric(f"cv_rmse_fold_{fold_idx}", fold_metrics["rmse"])

        cv_rmse = float(np.mean(cv_rmses))
        mlflow.log_metric("cv_rmse", cv_rmse)
        print(f"[baseline] CV RMSE (mean over {cv_splits} folds): {cv_rmse:.4f}")

        # Final fit on the full training set with early stopping on the validation set
        dtrain_full = lgb.Dataset(X_train, label=y_train)
        dval_full = lgb.Dataset(X_val, label=y_val, reference=dtrain_full)
        model = lgb.train(
            {**lgbm_params, "verbosity": -1},
            dtrain_full,
            num_boost_round=lgbm_params["n_estimators"],
            valid_sets=[dval_full],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        # Evaluate on validation set
        val_metrics = evaluate(model, X_val, y_val)
        mlflow.log_metric("val_rmse", val_metrics["rmse"])
        mlflow.log_metric("val_mae", val_metrics["mae"])
        mlflow.log_metric("val_r2", val_metrics["r2"])
        print(
            f"[baseline] Val RMSE={val_metrics['rmse']:.4f} "
            f"MAE={val_metrics['mae']:.4f} R²={val_metrics['r2']:.4f}"
        )

        mlflow.lightgbm.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    print(f"[baseline] MLflow run_id: {run_id}")
    return model, run_id


if __name__ == "__main__":
    from src.data.download import download_data
    from src.data.preprocess import run_preprocessing
    from src.models.registry import register_champion

    with open(_PROJECT_ROOT / "configs/serving.yaml") as fh:
        serving_cfg = yaml.safe_load(fh)

    mlflow.set_tracking_uri(serving_cfg["mlflow"]["tracking_uri"])
    model_name = serving_cfg["mlflow"]["model_name"]

    download_data()
    train_df, val_df, _ = run_preprocessing()
    _, run_id = train_baseline(train_df, val_df)
    register_champion(run_id, model_name)
