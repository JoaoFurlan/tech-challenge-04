"""Retraining pipeline: candidates from the labelled weeks, a promotion gate, and the promotion itself.

Started by `production_monitoring` (which waits for it) when the retraining rule fires, or by hand with `run_week`.

  decide -> build_candidates -> promote

* decide: the trigger (first major data-drift alert, later a performance drop about unseen weeks).
* build_candidates: A = the champion recalibrated, B = the champion's recipe refitted on Reference + labelled weeks
  with recency weights, then recalibrated. Both are registered; the better one becomes `challenger`; the paired
  bootstrap gate runs on a held-out slice.
* promote: only if the gate passed, move the `champion` alias and tell the model server to reload.
Labels arrive a week late, so a run for week N trains on labels through week N-1 only.
"""
from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PROJECT = "/opt/credit-drift"
PYTHON = f"{PROJECT}/.venv/bin/python"
WEEK = "{{ params.run_week | int }}"

with DAG(
    dag_id="retraining_pipeline",
    description="Candidates from labelled weeks, a paired-bootstrap gate, promotion and a server reload",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    params={"run_week": Param(None, type=["integer", "string"],  # the triggering DAG passes it as text
                              description="The simulated week (4-13) whose run triggered this retrain"),
            "trigger": Param(None, type=["null", "string"], description="Override the trigger (manual runs)")},
    tags=["retraining", "mlflow"],
    doc_md=__doc__,
) as dag:
    decide = BashOperator(
        task_id="decide",
        bash_command=(f"cd {PROJECT} && {PYTHON} -m credit_drift.retraining.policy --week {WEEK} | tail -n 1"),
        do_xcom_push=True)
    trigger = "{{ params.trigger or ti.xcom_pull(task_ids='decide') or 'manual' }}"
    build = BashOperator(
        task_id="build_candidates",
        bash_command=(f"cd {PROJECT} && {PYTHON} -m credit_drift.retraining.run build --run-week {WEEK} "
                      f"--through-week {{{{ (params.run_week | int) - 1 }}}} --trigger {trigger}"))
    promote = BashOperator(
        task_id="promote", bash_command=f"cd {PROJECT} && {PYTHON} -m credit_drift.retraining.run promote --run-week {WEEK}")
    decide >> build >> promote
