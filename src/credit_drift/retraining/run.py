"""Retraining and promotion, as two steps the `retraining_pipeline` DAG calls (docs/decisions.md, #18).

  python -m credit_drift.retraining.run build   --run-week 5 --through-week 4 --trigger data_drift_major
  python -m credit_drift.retraining.run promote --run-week 5

`build` makes candidates A (champion recalibrated) and B (retrained, then recalibrated), registers both, gives the
better one the `challenger` alias and runs the gate. `promote` moves the `champion` alias only if the gate passed,
keeps the old champion under `baseline` (the first one, once) and `previous`, and tells the model server to reload.
Nothing here retrains by hand: a notebook only reads what these steps logged.
"""
import argparse
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
import requests
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

import credit_drift
from credit_drift.data import TARGET
from credit_drift.monitoring.weekly import ROOT, write_json
from credit_drift.retraining.candidates import build_candidates, features_of, pick
from credit_drift.retraining.gate import paired_gate
from credit_drift.retraining.policy import load_state, save_state
from credit_drift.retraining.window import build_window
from credit_drift.serving.registry import champion_window_weeks, load_champion
from credit_drift.split import REFERENCE_DIR
from credit_drift.training.champion import MODEL_NAME, tag_decision_settings
from credit_drift.training.metrics import DEFAULT_RATIO
from credit_drift.training.tracking import activate_experiment

EXPERIMENT = "retraining"


def _write_event(event: dict) -> None:
    folder = ROOT / f"week_{event['run_week']:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "retraining.json", event)


def _register(model, name: str, X_sample, **tags) -> str:
    signature = mlflow.models.infer_signature(X_sample, model.predict_proba(X_sample)[:, 1])
    info = mlflow.sklearn.log_model(model, name=name, registered_model_name=MODEL_NAME, signature=signature,
                                    serialization_format="cloudpickle",
                                    code_paths=[str(Path(credit_drift.__file__).parent)])
    version = str(info.registered_model_version)
    tag_decision_settings(MODEL_NAME, version, DEFAULT_RATIO, **tags)
    return version


def step_build(run_week: int, through_week: int, trigger: str) -> int:
    champion = load_champion()
    window = build_window(through_week, pd.read_parquet(REFERENCE_DIR / "train.parquet"))
    threshold = champion.threshold
    candidates = build_candidates(champion.model, window, threshold)
    chosen = pick(candidates)

    X_gate, y_gate = features_of(window.gate), window.gate[TARGET].to_numpy()
    p_champion = champion.model.predict_proba(X_gate)[:, 1]
    gates = {k: paired_gate(y_gate, p_champion, c["model"].predict_proba(X_gate)[:, 1], threshold, threshold)
             for k, c in candidates.items()}  # both are reported; only the chosen one is acted on
    base_weeks, trained_weeks = champion_window_weeks(champion.version), sorted(set(window.weeks))
    activate_experiment(EXPERIMENT)
    versions = {}
    with mlflow.start_run(run_name=f"week_{run_week:02d}_{trigger}") as run:
        mlflow.log_params({"run_week": run_week, "through_week": through_week, "trigger": trigger,
                           "champion_version": champion.version, "chosen": chosen})
        mlflow.log_dict(window.describe(), "window.json")
        for key, c in candidates.items():
            mlflow.log_metrics({f"{key}_{m}": v for m, v in gates[key].items() if isinstance(v, float)}
                               | {f"{key}_crossfit_cost": c["calibration"]["crossfit_cost"]})
            versions[key] = _register(
                c["model"], f"candidate_{key}", X_gate.head(50), candidate=key,
                calibration=c["calibration"]["calibration"], labels_through_week=str(through_week), trigger=trigger,
                base_champion=champion.version, window_weeks=json.dumps(trained_weeks if key == "B" else base_weeks),
                gate_passed=str(gates[key]["passed"]), selected=str(key == chosen))
        mlflow.log_dict({"candidates": {k: c["calibration"] for k, c in candidates.items()}, "gates": gates,
                         "chosen": chosen, "versions": versions}, "retraining.json")
    MlflowClient().set_registered_model_alias(MODEL_NAME, "challenger", versions[chosen])

    event = {"run_week": run_week, "through_week": through_week, "trigger": trigger,
             "from_version": champion.version, "chosen": chosen, "versions": versions, "gate": gates[chosen],
             "all_gates": gates, "window": window.describe(), "run_id": run.info.run_id, "promoted": False}
    save_state([*load_state(), event])
    _write_event(event)
    g = gates[chosen]
    print(f"week {run_week:02d}: retrain on labels through week {through_week} ({trigger}); candidate {chosen} "
          f"-> challenger v{versions[chosen]}; gate {'PASSED' if g['passed'] else 'failed'}: cost gain "
          f"{g['cost_gain']:+.4f} CI [{g['cost_gain_ci'][0]:+.4f}, {g['cost_gain_ci'][1]:+.4f}], AUC-PR diff "
          f"{g['auc_pr_diff']:+.3f} CI [{g['auc_pr_diff_ci'][0]:+.3f}, {g['auc_pr_diff_ci'][1]:+.3f}]")
    return 0


def step_promote(run_week: int) -> int:
    events = load_state()
    event = next((e for e in reversed(events) if e["run_week"] == run_week), None)
    if event is None:
        print(f"week {run_week:02d}: no retraining event to promote")
        return 0
    if not event["gate"]["passed"]:
        print(f"week {run_week:02d}: gate failed: v{event['versions'][event['chosen']]} stays the challenger, "
              f"v{event['from_version']} keeps serving")
        return 0
    client = MlflowClient()
    new, old = event["versions"][event["chosen"]], event["from_version"]
    try:
        client.get_model_version_by_alias(MODEL_NAME, "baseline")
    except MlflowException:
        client.set_registered_model_alias(MODEL_NAME, "baseline", old)  # the first champion stays the comparison baseline
    client.set_registered_model_alias(MODEL_NAME, "previous", old)
    client.set_registered_model_alias(MODEL_NAME, "champion", new)
    client.delete_registered_model_alias(MODEL_NAME, "challenger")
    url = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8010")
    answer = requests.post(url + "/reload", headers={"x-reload-token": os.environ.get("RELOAD_TOKEN", "dev-reload-token")},
                           timeout=120)
    answer.raise_for_status()
    event.update(promoted=True, to_version=new, reload=answer.json())
    save_state(events)
    _write_event(event)
    print(f"week {run_week:02d}: promoted v{new} (was v{old}); server reloaded: {answer.json()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["build", "promote"])
    parser.add_argument("--run-week", type=int, required=True)
    parser.add_argument("--through-week", type=int)
    parser.add_argument("--trigger", default="manual")
    args = parser.parse_args()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    if args.step == "build":
        return step_build(args.run_week, args.through_week or args.run_week - 1, args.trigger)
    return step_promote(args.run_week)


if __name__ == "__main__":
    raise SystemExit(main())
