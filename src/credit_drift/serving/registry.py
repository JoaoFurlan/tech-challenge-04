"""Resolve and load the champion from the MLflow Model Registry, with its decision settings."""
import json
import time
from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient

DEFAULT_COST_RATIO = 5.0


@dataclass(frozen=True)
class LoadedModel:
    model: object
    name: str
    version: str
    alias: str
    run_id: str
    threshold: float  # flag a prediction as high risk when its probability reaches this
    cost_ratio: float
    loaded_at: float


def decision_threshold(cost_ratio: float) -> float:
    """Cost-optimal cut-off for calibrated probabilities: flag when p > C_FP / (C_FP + C_FN)."""
    return 1 / (1 + cost_ratio)


def champion_window_weeks(version: str, name: str = "credit_scoring") -> list[int]:
    """The production weeks a version was trained on, kept as a registry tag by the retraining step (none for v1)."""
    tags = MlflowClient().get_model_version(name, str(version)).tags or {}
    return json.loads(tags.get("window_weeks", "[]"))


def load_champion(name: str = "credit_scoring", alias: str = "champion") -> LoadedModel:
    """Look the alias up first, then load that exact version, so the model and its metadata always match."""
    version = MlflowClient().get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
    tags = version.tags or {}
    ratio = float(tags.get("cost_ratio", DEFAULT_COST_RATIO))
    return LoadedModel(
        model=model, name=name, version=str(version.version), alias=alias, run_id=version.run_id,
        threshold=float(tags.get("decision_threshold", decision_threshold(ratio))), cost_ratio=ratio,
        loaded_at=time.time())
