"""MLflow experiment helpers shared by the training stages."""
import mlflow
from mlflow import MlflowClient

QUICK_PREFIX = "quick_"


def activate_experiment(name: str) -> None:
    """Make `name` the active experiment.

    Smoke tests (`--quick`) write to throwaway `quick_*` experiments that get deleted afterwards, and MLflow refuses to
    reuse the name of a deleted experiment. So a deleted `quick_*` experiment is restored first. A deleted real
    experiment is never restored silently: that failure should be seen.
    """
    existing = MlflowClient().get_experiment_by_name(name)
    if existing is not None and existing.lifecycle_stage == "deleted" and name.startswith(QUICK_PREFIX):
        MlflowClient().restore_experiment(existing.experiment_id)
    mlflow.set_experiment(name)
