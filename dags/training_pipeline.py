"""Training pipeline: the reproducible path from the Reference data to a registered champion.

Run on demand (no schedule), normally once; the retraining pipeline later reuses these steps on another data window.
Every task calls the project's own virtualenv, so the DAG itself stays thin and the logic stays in `src/`.

  check_data -> contracts -> feature_engineering -> model_selection -> hyperparameter_tuning -> champion

The champion step registers the model and tags the decision threshold on the new version.

Trigger with `quick = true` for a smoke test (10k rows, 3 trials, separate MLflow experiments and model name).
"""
from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PROJECT = "/opt/credit-drift"
PYTHON = f"{PROJECT}/.venv/bin/python"
QUICK = "{{ '--quick' if params.quick else '' }}"
TRIALS = "{% if params.trials %}--trials {{ params.trials }}{% endif %}"

with DAG(
    dag_id="training_pipeline",
    description="Validate the Reference data, run the staged MLflow experiments, register the champion",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    params={
        "quick": Param(False, type="boolean", description="Smoke test: 10k rows, 3 trials, separate names"),
        "trials": Param(None, type=["null", "integer"], description="Override Optuna trials per family"),
    },
    tags=["training", "mlflow"],
    doc_md=__doc__,
) as dag:
    check_data = BashOperator(
        task_id="check_data",
        bash_command=(f"cd {PROJECT} && for f in train calibration test; do "
                      "test -s data/reference/$f.parquet || { echo \"missing data/reference/$f.parquet\"; exit 1; }; "
                      "done && echo 'reference data present'"),
    )
    contracts = BashOperator(
        task_id="contracts", bash_command=f"cd {PROJECT} && {PYTHON} -m credit_drift.contracts.validate_reference")

    def stage(task_id: str, name: str, extra: str = "") -> BashOperator:
        return BashOperator(
            task_id=task_id,
            bash_command=f"cd {PROJECT} && {PYTHON} -m credit_drift.training.run {name} {QUICK} {extra}".strip(),
        )

    feature_engineering = stage("feature_engineering", "features")
    model_selection = stage("model_selection", "models")
    hyperparameter_tuning = stage("hyperparameter_tuning", "tune", TRIALS)
    champion = stage("champion", "select")
    check_data >> contracts >> feature_engineering >> model_selection >> hyperparameter_tuning >> champion
