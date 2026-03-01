"""MLflow model registry helpers: register, load, and inspect the champion model."""
from __future__ import annotations

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient


def register_champion(run_id: str, model_name: str) -> None:
    """Register the model artifact from *run_id* and tag it as the champion.

    Args:
        run_id: MLflow run ID whose ``model`` artifact will be registered.
        model_name: Registered model name in the MLflow Model Registry.
    """
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = result.version

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version,
    )
    print(
        f"[registry] Registered {model_name} v{version} "
        f"from run {run_id} as 'champion'."
    )


def load_champion(model_name: str) -> mlflow.pyfunc.PyFuncModel:
    """Load the model currently aliased as 'champion'.

    Args:
        model_name: Registered model name in the MLflow Model Registry.

    Returns:
        A loaded ``mlflow.pyfunc.PyFuncModel`` ready for inference.
    """
    model_uri = f"models:/{model_name}@champion"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"[registry] Loaded champion model: {model_name}@champion")
    return model


def get_champion_rmse(model_name: str) -> float | None:
    """Retrieve the ``val_rmse`` metric from the champion model's training run.

    Args:
        model_name: Registered model name in the MLflow Model Registry.

    Returns:
        The ``val_rmse`` float, or ``None`` if the metric is not found.
    """
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias="champion")
        run = client.get_run(mv.run_id)
        return run.data.metrics.get("val_rmse")
    except Exception:
        return None
