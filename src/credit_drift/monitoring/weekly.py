"""One simulated week of the production loop, as small steps an orchestrator calls one by one.

  python -m credit_drift.monitoring.weekly gate --week 5         # contract: Pandera + Great Expectations
  python -m credit_drift.monitoring.weekly score --week 5        # score the batch through the model server (HTTP)
  python -m credit_drift.monitoring.weekly drift --week 5        # PSI/KS/chi-square vs the baselines + Evidently HTML
  python -m credit_drift.monitoring.weekly quarantine --week 9   # set a rejected batch aside
  python -m credit_drift.monitoring.weekly performance --week 5  # labels of week 4 (one-week delay): metrics + slices
  python -m credit_drift.monitoring.weekly evaluate --week 5     # alert rules + diagnosis
  python -m credit_drift.monitoring.weekly log --week 5          # MLflow run
  python -m credit_drift.monitoring.weekly push --week 5         # Prometheus Pushgateway

Steps talk through small files in artifacts/monitoring/week_NN/, so each one can be rerun alone and inspected.
Labels for week N arrive a week late, so run N finishes week N-1: `performance` and the label-dependent part of
`evaluate` and `push` describe week N-1, while everything else describes week N.
"""
import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from credit_drift.contracts.check import check_batch
from credit_drift.contracts.pandera_gate import validate_labels
from credit_drift.contracts.rules import WEEKLY_BATCH_ROWS
from credit_drift.data import ID_COL, TARGET
from credit_drift.monitoring import alerts as alert_rules
from credit_drift.monitoring.performance import (
    compute_baseline,
    overall_metrics,
    pooled_slice_performance,
)
from credit_drift.split import DATA_DIR, PRODUCTION_DIR, REFERENCE_DIR

ROOT = DATA_DIR.parent / "artifacts" / "monitoring"
REPORTS_DIR = DATA_DIR.parent / "reports" / "drift"
SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8010")
PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "http://127.0.0.1:9091")
CHUNK = 5_000  # well under the server's 10,000-record limit
COMPARISON_ALIASES = ("baseline", "challenger")  # models scored offline next to the champion, once they exist


# ---------------------------------------------------------------- small file helpers
def week_dir(week: int) -> Path:
    path = ROOT / f"week_{week:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clean(value):
    """JSON cannot hold NaN or numpy types: turn them into null and plain numbers."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_clean(payload), indent=1), encoding="utf-8")


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def features_path(week: int) -> Path:
    return PRODUCTION_DIR / "weeks" / f"week_{week:02d}_features.parquet"


def labels_path(week: int) -> Path:
    return PRODUCTION_DIR / "weeks" / f"week_{week:02d}_labels.parquet"


# ---------------------------------------------------------------- steps that describe week N
def step_gate(week: int) -> int:
    """The contract at the door. A rejected batch is not scored; the DAG branches on `passed`."""
    batch = pd.read_parquet(features_path(week))
    report = check_batch(batch, f"week_{week:02d}", expected_rows=WEEKLY_BATCH_ROWS)
    blocking = [{"column": f.column, "rule": f.rule, "count": f.count} for f in report.pandera.blocking]
    warnings_ = [{"column": f.column, "rule": f.rule, "count": f.count} for f in report.gx.warnings]
    write_json(week_dir(week) / "gate.json", {
        "week": week, "passed": report.passed, "rows": len(batch), "blocking": blocking,
        "gx_blocking_rules": len(report.gx.blocking), "warnings": warnings_})
    print(f"week {week:02d}: contract {'PASSED' if report.passed else 'REJECTED'} | "
          f"blocking {len(blocking)} (Pandera) / {len(report.gx.blocking)} (GX) | warnings {len(warnings_)}")
    for f in blocking:
        print(f"   blocking: {f['column']} {f['rule']} ({f['count']} rows)")
    return 0


def step_quarantine(week: int) -> int:
    """Set the rejected batch aside, keeping it for the data owner; nothing downstream sees it."""
    target = ROOT / "quarantine"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(features_path(week), target / f"week_{week:02d}_features.parquet")
    write_json(week_dir(week) / "quarantine.json", {"week": week, "stored_in": str(target)})
    print(f"week {week:02d}: batch quarantined in {target}; it is not scored and has no drift report")
    return 0


def step_score(week: int) -> int:
    """Score through the model server over HTTP, like production, and keep what it answered."""
    import requests

    gate = read_json(week_dir(week) / "gate.json")
    if not gate or not gate["passed"]:
        print(f"week {week:02d}: refusing to score a batch that did not pass the contract")
        return 3
    batch = pd.read_parquet(features_path(week))
    started, predictions, version, threshold = time.perf_counter(), [], None, None
    for start in range(0, len(batch), CHUNK):
        part = batch.iloc[start:start + CHUNK]
        response = requests.post(f"{SERVER_URL}/predict", json={"records": json.loads(part.to_json(orient="records"))},
                                 timeout=120)
        response.raise_for_status()
        body = response.json()
        predictions += body["predictions"]
        version, threshold = body["model"]["version"], body["decision_threshold"]
    scores = pd.DataFrame(predictions).rename(columns={"id": ID_COL})
    assert (scores[ID_COL].to_numpy() == batch[ID_COL].to_numpy()).all(), "server answers out of order"
    scores.to_parquet(week_dir(week) / "scores.parquet", index=False)
    write_json(week_dir(week) / "score.json", {
        "week": week, "model_version": version, "threshold": threshold, "rows": len(scores),
        "mean_score": scores["probability"].mean(), "flag_rate": scores["high_risk"].mean(),
        "seconds": round(time.perf_counter() - started, 2)})
    print(f"week {week:02d}: scored {len(scores)} rows with model v{version} in {time.perf_counter() - started:.1f}s | "
          f"mean {scores['probability'].mean():.4f} | flagged {scores['high_risk'].mean():.1%}")
    return 0


def _load_champion():
    from credit_drift.serving.registry import load_champion

    return load_champion()


def step_drift(week: int) -> int:
    from credit_drift.monitoring.context import analyze, build_context

    loaded = _load_champion()
    batch = pd.read_parquet(features_path(week))
    scores = pd.read_parquet(week_dir(week) / "scores.parquet")["probability"].to_numpy()
    tables, summaries = analyze(build_context(loaded.model), batch, scores, week,
                                REPORTS_DIR / f"week_{week:02d}_vs_champion_window.html")
    table = tables["champion_window"]
    table.to_parquet(week_dir(week) / "drift_features.parquet", index=False)
    top = table.query("monitored and feature != 'SCORE'").nlargest(5, "psi")
    write_json(week_dir(week) / "drift.json", {
        **summaries["champion_window"], "model_version": loaded.version,
        "reference_summary": summaries["reference"],
        "top_features": [{"feature": r.feature, "psi": r.psi, "ks": r.ks_stat} for r in top.itertuples()]})
    s = summaries["champion_window"]
    print(f"week {week:02d}: monitored PSI>=0.10: {s['n_psi_moderate']} (>=0.25: {s['n_psi_major']}) | "
          f"max {s['max_psi']:.3f} {s['max_psi_feature']} | score PSI {s['score_psi']:.3f} KS {s['score_ks']:.3f}")
    return 0


# ---------------------------------------------------------------- steps that describe week N-1 (labels arrived)
def ensure_baseline(loaded) -> dict:
    """The champion's own validation baseline, computed once per model version on the untouched Reference test part."""
    path = ROOT / f"baseline_v{loaded.version}.json"
    existing = read_json(path)
    if existing:
        return existing
    baseline = compute_baseline(loaded.model, pd.read_parquet(REFERENCE_DIR / "train.parquet"),
                                pd.read_parquet(REFERENCE_DIR / "test.parquet"), loaded.threshold, loaded.version)
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(path, baseline)
    return baseline


def labelled_week(week: int) -> pd.DataFrame | None:
    """Features + label + the probability the server gave, for a week whose labels have arrived; else None."""
    scores_file, lab_file = week_dir(week) / "scores.parquet", labels_path(week)
    if not scores_file.exists() or not lab_file.exists():
        return None
    labels = pd.read_parquet(lab_file)
    contract = validate_labels(labels)
    if not contract.passed:
        raise SystemExit(f"week {week:02d}: the label contract failed: {[(f.column, f.rule) for f in contract.blocking]}")
    scores = pd.read_parquet(scores_file)
    merged = pd.read_parquet(features_path(week)).merge(labels, on=ID_COL, how="inner").merge(
        scores[[ID_COL, "probability"]], on=ID_COL, how="inner")
    if len(merged) != len(scores):
        raise SystemExit(f"week {week:02d}: labels do not match the scored rows ({len(merged)} of {len(scores)})")
    return merged


def comparison_models() -> list[tuple[str, str]]:
    """Registered versions held under the offline aliases (the pre-promotion baseline, a shadow challenger)."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client, found = MlflowClient(), []
    for alias in COMPARISON_ALIASES:
        with contextlib.suppress(MlflowException):  # the alias does not exist (yet): nothing to compare
            found.append((alias, client.get_model_version_by_alias("credit_scoring", alias).version))
    return found


def step_performance(week: int) -> int:
    """Finish week N-1: its labels have matured. Overall metrics, 14 slices over two weeks, comparison models."""
    prev = week - 1
    note = {"run_week": week, "for_week": prev, "available": False, "reason": ""}
    frame = labelled_week(prev) if prev >= 1 else None
    if frame is None:
        note["reason"] = "no previous week" if prev < 1 else "no labels or no scores for the previous week"
        write_json(week_dir(week) / "labels_backfill.json", note)
        print(f"week {week:02d}: nothing to backfill ({note['reason']})")
        return 0

    loaded = _load_champion()
    baseline = ensure_baseline(loaded)
    threshold = (read_json(week_dir(prev) / "score.json") or {}).get("threshold", loaded.threshold)
    metrics = overall_metrics(frame[TARGET].to_numpy(), frame["probability"].to_numpy(), threshold)

    labelled = {prev: frame}
    if prev > 1 and (earlier := labelled_week(prev - 1)) is not None:
        labelled[prev - 1] = earlier  # the slices pool two weeks (docs/decisions.md, #5)
    window = sorted(labelled)
    t = alert_rules.load_thresholds()
    slices, pooled = pooled_slice_performance([labelled[w] for w in window], baseline["slice_cuts"])
    overall_drop = baseline["overall"]["auc_pr"] - pooled["auc_pr"]
    flagged = alert_rules.flagged_slices(slices, baseline["slices"], overall_drop, t)

    comparison = []
    for alias, version in comparison_models():
        import mlflow

        other = mlflow.sklearn.load_model(f"models:/credit_scoring/{version}")
        p = other.predict_proba(pd.read_parquet(features_path(prev)))[:, 1]
        comparison.append({"alias": alias, "version": version,
                           **overall_metrics(frame[TARGET].to_numpy(), p, threshold)})

    write_json(week_dir(prev) / "performance.json", {
        "week": prev, "computed_in_run": week, "model_version": loaded.version, **metrics,
        "baseline_auc_pr": baseline["overall"]["auc_pr"], "auc_pr_drop": baseline["overall"]["auc_pr"] - metrics["auc_pr"],
        "slice_window_weeks": window, "slice_window_overall": pooled, "slice_overall_drop": overall_drop,
        "slices": slices, "flagged_slices": flagged, "comparison": comparison})
    note.update(available=True)
    write_json(week_dir(week) / "labels_backfill.json", note)
    print(f"week {week:02d}: backfilled week {prev:02d} | AUC-PR {metrics['auc_pr']:.3f} "
          f"(baseline {baseline['overall']['auc_pr']:.3f}, drop {baseline['overall']['auc_pr'] - metrics['auc_pr']:+.3f}) | "
          f"Brier {metrics['brier']:.4f} | predicted {metrics['mean_pred']:.3f} vs observed {metrics['observed_rate']:.3f} | "
          f"flagged slices: {flagged or 'none'}")
    return 0


# ---------------------------------------------------------------- alerts and diagnosis
def _drift_table(week: int) -> pd.DataFrame | None:
    path = week_dir(week) / "drift_features.parquet"
    if not path.exists():
        return None
    table = pd.read_parquet(path)
    return table[table["monitored"] & (table["feature"] != "SCORE")]


def _diagnose_week(week: int, t: dict, baseline: dict | None) -> dict | None:
    """Week-over-week diagnosis of one week, final once its labels are in."""
    gate = read_json(week_dir(week) / "gate.json")
    if gate and not gate["passed"]:
        return {"week": week, "cause": alert_rules.diagnose(blocked=True, features_changed=False, score_changed=False,
                                                              performance_dropped=None, calibration_off=None),
                "signals": {}}
    drift, perf = read_json(week_dir(week) / "drift.json"), read_json(week_dir(week) / "performance.json")
    table = _drift_table(week)
    if drift is None or table is None or perf is None or baseline is None:
        return None  # not enough information yet (labels pending)
    prev_drift = read_json(week_dir(week - 1) / "drift.json") if week > 1 else None
    features = alert_rules.features_changed(_drift_table(week - 1) if week > 1 else None, table, t)
    score = alert_rules.score_changed(prev_drift, drift, t)
    dropped = (baseline["overall"]["auc_pr"] - perf["auc_pr"]) >= t["performance"]["auc_pr_drop"]
    gap = abs(perf["mean_pred"] - perf["observed_rate"]) / max(perf["observed_rate"], 1e-9)
    off = gap >= t["performance"]["calibration_gap_relative"]
    cause = alert_rules.diagnose(blocked=False, features_changed=features, score_changed=score,
                                 performance_dropped=dropped, calibration_off=off)
    return {"week": week, "cause": cause, "signals": {"features_changed": features, "score_changed": score,
                                                      "performance_dropped": dropped, "calibration_off": off}}


def step_evaluate(week: int) -> int:
    t = alert_rules.load_thresholds()
    gate = read_json(week_dir(week) / "gate.json")
    drift = read_json(week_dir(week) / "drift.json")
    found = alert_rules.batch_alerts(week, gate, drift, t)

    prev = week - 1
    loaded_baseline = None
    perf = read_json(week_dir(prev) / "performance.json") if prev >= 1 else None
    if perf:
        loaded_baseline = read_json(ROOT / f"baseline_v{perf['model_version']}.json")
        found += alert_rules.performance_alerts(perf, loaded_baseline, perf["flagged_slices"], t)

    diagnoses = {}
    if not gate["passed"]:
        diagnoses[week] = _diagnose_week(week, t, None)  # a rejected batch is diagnosed immediately: no labels needed
    if perf:
        final = _diagnose_week(prev, t, loaded_baseline)  # week N-1 is final now that its labels are in
        if final:
            diagnoses[prev] = final
    for w, d in diagnoses.items():
        write_json(week_dir(w) / "diagnosis.json", d)
    write_json(week_dir(week) / "alerts.json", {"week": week, "alerts": found, "diagnoses": diagnoses})
    for a in found:
        print(f"ALERT [{a['severity']}] week {a['about_week']:02d} {a['rule']}: {a['detail']}")
    for w, d in sorted(diagnoses.items()):
        print(f"diagnosis week {w:02d}: {d['cause']} {d['signals'] or ''}")
    if not found:
        print(f"week {week:02d}: no alerts")
    return 0


# ---------------------------------------------------------------- MLflow and Prometheus
def step_log(week: int) -> int:
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("production_monitoring")
    directory = week_dir(week)
    gate, drift, score = (read_json(directory / n) for n in ("gate.json", "drift.json", "score.json"))
    alerts = (read_json(directory / "alerts.json") or {"alerts": []})["alerts"]
    perf = read_json(week_dir(week - 1) / "performance.json") if week > 1 else None
    with mlflow.start_run(run_name=f"week {week:02d}"):
        mlflow.set_tags({"week": str(week), "blocked": str(not gate["passed"]).lower(),
                         "alerts": ",".join(sorted({a["rule"] for a in alerts})) or "none"})
        mlflow.log_params({"week": week, "model_version": (score or drift or {}).get("model_version", "none")})
        metrics = {"rows": gate["rows"], "contract_warnings": len(gate["warnings"]), "alerts": len(alerts)}
        if drift:
            metrics.update({k: drift[k] for k in ("n_psi_moderate", "n_psi_major", "max_psi", "mean_psi", "score_psi",
                                                  "score_ks", "mean_score")})
        if score:
            metrics["flag_rate"] = score["flag_rate"]
        if perf:
            metrics.update({f"labels_week_{k}": perf[k] for k in ("auc_pr", "auc_roc", "brier", "mean_pred",
                                                                  "observed_rate", "auc_pr_drop")})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        for name in ("gate.json", "score.json", "drift.json", "alerts.json", "drift_features.parquet"):
            if (directory / name).exists():
                mlflow.log_artifact(str(directory / name))
        report = REPORTS_DIR / f"week_{week:02d}_vs_champion_window.html"
        if report.exists():
            mlflow.log_artifact(str(report))
        if perf:
            mlflow.log_artifact(str(week_dir(week - 1) / "performance.json"), artifact_path="previous_week")
    print(f"week {week:02d}: logged to MLflow experiment production_monitoring")
    return 0


def _gauge(registry, name: str, doc: str, labels: tuple = ()):
    from prometheus_client import Gauge

    return Gauge(name, doc, labels, registry=registry)


def _set(gauge, value, **labels) -> None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return
    (gauge.labels(**labels) if labels else gauge).set(float(value))


def step_push(week: int) -> int:
    """Push to the Pushgateway, one group per week and stage (batch = describes week N, labels = describes week N-1)."""
    from prometheus_client import CollectorRegistry, push_to_gateway

    directory = week_dir(week)
    gate, drift, score = (read_json(directory / n) for n in ("gate.json", "drift.json", "score.json"))
    alerts_doc = read_json(directory / "alerts.json") or {"alerts": [], "diagnoses": {}}

    batch = CollectorRegistry()
    _set(_gauge(batch, "batch_blocked", "1 if the contract rejected the batch"), 0 if gate["passed"] else 1)
    _set(_gauge(batch, "batch_rows", "Rows in the weekly batch"), gate["rows"])
    _set(_gauge(batch, "batch_contract_warnings", "Contract warnings on the batch"), len(gate["warnings"]))
    if score:
        _set(_gauge(batch, "score_mean", "Mean predicted default probability"), score["mean_score"])
        _set(_gauge(batch, "score_flag_rate", "Share of applicants at or above the decision threshold"),
             score["flag_rate"])
        _set(_gauge(batch, "model_version_info", "Model version that scored the batch", ("version",)), 1,
             version=score["model_version"])
    if drift:
        for key, doc in (("n_psi_moderate", "Monitored features at PSI >= 0.10"),
                         ("n_psi_major", "Monitored features at PSI >= 0.25"), ("max_psi", "Worst monitored PSI"),
                         ("mean_psi", "Mean monitored PSI"), ("score_psi", "PSI of the model score"),
                         ("score_ks", "KS statistic of the model score")):
            _set(_gauge(batch, f"drift_{key}", doc), drift[key])
        table = pd.read_parquet(directory / "drift_features.parquet")
        table = table[table["monitored"]]
        psi, ks = (_gauge(batch, "drift_feature_psi", "PSI per monitored feature", ("feature",)),
                   _gauge(batch, "drift_feature_ks", "KS statistic per monitored numeric feature", ("feature",)))
        for r in table.itertuples():
            _set(psi, r.psi, feature=r.feature)
            _set(ks, r.ks_stat, feature=r.feature)
    firing = _gauge(batch, "alert_firing", "1 for each alert raised about this week", ("rule", "severity"))
    for a in alerts_doc["alerts"]:
        if a["about_week"] == week:
            _set(firing, 1, rule=a["rule"], severity=a["severity"])
    cause = (alerts_doc["diagnoses"].get(str(week)) or {}).get("cause")
    if cause == "technical":
        _set(_gauge(batch, "diagnosis", "Diagnosed cause of the week, 1 on the chosen cause", ("cause",)), 1, cause=cause)
    push_to_gateway(PUSHGATEWAY_URL, job="production_monitoring", registry=batch,
                    grouping_key={"week": str(week), "stage": "batch"})

    prev, perf = week - 1, (read_json(week_dir(week - 1) / "performance.json") if week > 1 else None)
    pushed_labels = False
    if perf:
        labels = CollectorRegistry()
        for key, doc in (("auc_pr", "AUC-PR of the week"), ("auc_roc", "AUC-ROC of the week"),
                         ("brier", "Brier score of the week"), ("mean_pred", "Mean predicted default rate"),
                         ("observed_rate", "Observed default rate"), ("flag_rate", "Share flagged high risk"),
                         ("baseline_auc_pr", "Champion's validation AUC-PR"), ("auc_pr_drop", "Baseline minus AUC-PR")):
            _set(_gauge(labels, f"model_{key}", doc), perf[key])
        s_now, s_base = (_gauge(labels, "slice_auc_pr", "AUC-PR per slice over two weeks", ("slice",)),
                         _gauge(labels, "slice_auc_pr_baseline", "Champion's validation AUC-PR per slice", ("slice",)))
        s_flag = _gauge(labels, "slice_flagged", "1 if the slice fell well below its baseline", ("slice",))
        baseline = read_json(ROOT / f"baseline_v{perf['model_version']}.json")
        for name, entry in perf["slices"].items():
            _set(s_now, entry["auc_pr"], slice=name)
            _set(s_base, baseline["slices"][name]["auc_pr"], slice=name)
            _set(s_flag, 1 if name in perf["flagged_slices"] else 0, slice=name)
        firing = _gauge(labels, "alert_firing", "1 for each alert raised about this week", ("rule", "severity"))
        for a in alerts_doc["alerts"]:
            if a["about_week"] == prev:
                _set(firing, 1, rule=a["rule"], severity=a["severity"])
        prev_cause = (alerts_doc["diagnoses"].get(str(prev)) or {}).get("cause")
        if prev_cause:
            _set(_gauge(labels, "diagnosis", "Diagnosed cause of the week, 1 on the chosen cause", ("cause",)), 1,
                 cause=prev_cause)
        push_to_gateway(PUSHGATEWAY_URL, job="production_monitoring", registry=labels,
                        grouping_key={"week": str(prev), "stage": "labels"})
        pushed_labels = True
    print(f"week {week:02d}: pushed the batch metrics" + (f" and the labels metrics of week {prev:02d}" if pushed_labels else ""))
    return 0


STEPS = {"gate": step_gate, "quarantine": step_quarantine, "score": step_score, "drift": step_drift,
         "performance": step_performance, "evaluate": step_evaluate, "log": step_log, "push": step_push}


def main() -> int:
    warnings.filterwarnings("ignore")
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    return STEPS[args.step](args.week)


if __name__ == "__main__":
    sys.exit(main())
