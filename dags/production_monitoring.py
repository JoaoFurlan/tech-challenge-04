"""Production monitoring: one run per simulated week, the project's "sustentacao e confiabilidade" loop.

Scheduled daily from 2026-01-01 to 2026-01-13 with catch-up, one run at a time, so January N is simulated week N and the
13 weeks flow in order. (The data has no real timestamps; time is synthetic by design.)

  gate -> choose_path -> score_batch -> drift_report ---+
          |                                             |
          +--------> quarantine_batch ------------------+--> evaluate_alerts -> log_mlflow -> push_metrics
  gate -> label_performance ----------------------------+

* gate: the batch against the data contract (Pandera, then Great Expectations).
* choose_path: a rejected batch is quarantined and never scored (week 9 by design); the run still finishes and alerts.
* score_batch: through the model server over HTTP, like production. drift_report: PSI/KS/chi-square against the two
  baselines plus an Evidently HTML report.
* label_performance: labels arrive one week late, so run N finishes week N-1: AUC-PR, Brier, observed vs predicted rate,
  14 slices over two weeks, and any comparison models.
* evaluate_alerts: alert rules and the week-over-week diagnosis. log_mlflow, push_metrics: MLflow and the Pushgateway.

Pass the `week` parameter to rerun one specific week by hand.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.sdk import DAG, Param

PROJECT = "/opt/credit-drift"
PYTHON = f"{PROJECT}/.venv/bin/python"
WEEK = "{{ params.week if params.week else data_interval_start.day }}"


def choose_path(week: str) -> str:
    gate = json.loads(Path(f"{PROJECT}/artifacts/monitoring/week_{int(week):02d}/gate.json").read_text())
    return "score_batch" if gate["passed"] else "quarantine_batch"


def step(task_id: str, name: str, **kwargs) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT} && {PYTHON} -m credit_drift.monitoring.weekly {name} --week {WEEK}",
        **kwargs,
    )


with DAG(
    dag_id="production_monitoring",
    description="One run per simulated week: contract, scoring, drift, labels, alerts, metrics",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    end_date=datetime(2026, 1, 13),
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=True,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=20)},
    params={"week": Param(None, type=["null", "integer"], minimum=1, maximum=13,
                          description="Rerun one specific week by hand (scheduled runs use the date: January N = week N)")},
    tags=["monitoring", "drift"],
    doc_md=__doc__,
) as dag:
    gate = step("gate", "gate")
    choose = BranchPythonOperator(task_id="choose_path", python_callable=choose_path, op_kwargs={"week": WEEK})
    score_batch = step("score_batch", "score")
    drift_report = step("drift_report", "drift")
    quarantine_batch = step("quarantine_batch", "quarantine")
    label_performance = step("label_performance", "performance")
    evaluate_alerts = step("evaluate_alerts", "evaluate", trigger_rule="none_failed_min_one_success")
    log_mlflow = step("log_mlflow", "log")
    push_metrics = step("push_metrics", "push")

    gate >> choose >> [score_batch, quarantine_batch]
    score_batch >> drift_report
    gate >> label_performance
    [drift_report, quarantine_batch, label_performance] >> evaluate_alerts >> log_mlflow >> push_metrics
