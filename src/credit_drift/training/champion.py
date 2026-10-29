"""Choose the champion (docs/decisions.md, #8), calibrate it, evaluate it once on test, register it."""
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold

import credit_drift
from credit_drift.data import TARGET
from credit_drift.training.describe import NOTE_TAG, describe_champion
from credit_drift.training.metrics import (
    COST_RATIOS,
    DEFAULT_RATIO,
    bootstrap_ci,
    cost_at_threshold,
    prior_brier,
    summarize,
)
from credit_drift.training.model import CalibratedModel, apply_calibrator, fit_calibrator

MODEL_NAME = "credit_scoring"
CALIBRATION_METHODS = ("isotonic", "platt")


def tag_decision_settings(model_name: str, version: str, cost_ratio: float = DEFAULT_RATIO, **extra: str) -> dict:
    """Store the cost ratio and the decision threshold it implies on the registered model version, so the serving
    app reads them from the registry and any change leaves a trace there."""
    tags = {"cost_ratio": f"{cost_ratio:g}", "decision_threshold": f"{1 / (1 + cost_ratio):.6f}", **extra}
    client = MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(model_name, str(version), key, value)
    return tags


def shortlist(results: list[dict]) -> list[dict]:
    """Cells within one standard error of the best cross-validated AUC-PR."""
    best = max(results, key=lambda r: r["cv_auc_pr_mean"])
    floor = best["cv_auc_pr_mean"] - best["cv_auc_pr_se"]
    return [r for r in results if r["cv_auc_pr_mean"] >= floor]


def pick_champion(results: list[dict], ratio: float = DEFAULT_RATIO) -> dict:
    """Lowest out-of-fold expected cost among the shortlisted cells."""
    return min(shortlist(results), key=lambda r: r[f"oof_cost_{int(ratio)}"])


def choose_calibration(raw: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[str, dict]:
    """Cross-fitted Brier on the calibration part picks the method; nothing here touches the test set."""
    scores = {}
    for method in CALIBRATION_METHODS:
        pred = np.zeros(len(y))
        for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(raw):
            calibrator = fit_calibrator(method, raw[train_idx], y[train_idx])
            pred[valid_idx] = apply_calibrator(method, calibrator, raw[valid_idx])
        scores[method] = float(brier_score_loss(y, pred))
    return min(scores, key=scores.get), scores


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"], bins, duplicates="drop")
    return frame.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")).reset_index(drop=True)


def select_and_register(results_dir: Path, reference_dir: Path, tracking_experiment: str,
                        model_name: str = MODEL_NAME) -> dict:
    results = [json.loads(p.read_text()) for p in sorted(results_dir.glob("cell_*.json"))]
    if not results:
        raise SystemExit(f"no tuned cells in {results_dir}; run the tune stage first")
    champion = pick_champion(results)
    short = shortlist(results)
    sensitivity = {int(r): pick_champion(results, r)["variant"] + "/" + pick_champion(results, r)["family"] for r in COST_RATIOS}

    cal = pd.read_parquet(reference_dir / "calibration.parquet")
    test = pd.read_parquet(reference_dir / "test.parquet")
    X_cal, y_cal = cal.drop(columns=[TARGET]), cal[TARGET].to_numpy()
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET].to_numpy()

    def load(r: dict):
        return joblib.load(results_dir / f"cell_{r['variant']}_{r['family']}.joblib")

    pipeline = load(champion)
    raw_cal = pipeline.predict_proba(X_cal)[:, 1]
    method, cv_brier = choose_calibration(raw_cal, y_cal)
    model = CalibratedModel(pipeline, fit_calibrator(method, raw_cal, y_cal), method)

    raw_test = pipeline.predict_proba(X_test)[:, 1]
    cal_test = model.predict_proba(X_test)[:, 1]
    raw_metrics, cal_metrics = summarize(y_test, raw_test), summarize(y_test, cal_test)
    ci = bootstrap_ci(y_test, cal_test)
    threshold = 1 / (1 + DEFAULT_RATIO)  # decision-theoretic cut-off for calibrated probabilities
    gate = {"brier": cal_metrics["brier"], "prior_brier": prior_brier(y_test), "passed": cal_metrics["brier"] < prior_brier(y_test)}

    # Reported for transparency only, after the choice is made: test never selects a model.
    table = []
    for r in results:
        p = load(r).predict_proba(X_test)[:, 1]
        table.append({"cell": f"{r['variant']}/{r['family']}", "cv_auc_pr": r["cv_auc_pr_mean"],
                      "shortlisted": r in short, "chosen": r is champion, **{f"test_{k}": v for k, v in summarize(y_test, p).items()}})
    cells_on_test = pd.DataFrame(table).sort_values("cv_auc_pr", ascending=False)

    mlflow.set_experiment(tracking_experiment)
    with mlflow.start_run(run_name="champion_selection") as run:
        mlflow.set_tag(NOTE_TAG, describe_champion(
            f"{champion['variant']}/{champion['family']}", method, DEFAULT_RATIO,
            ", ".join(f"{r['variant']}/{r['family']}" for r in short)))
        mlflow.log_params({
            "champion_cell": f"{champion['variant']}/{champion['family']}", "calibration": method,
            "cost_ratio": DEFAULT_RATIO, "shortlist": ",".join(f"{r['variant']}/{r['family']}" for r in short)})
        mlflow.log_metrics({f"test_raw_{k}": v for k, v in raw_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in cal_metrics.items()})
        mlflow.log_metrics({"test_auc_pr_ci_low": ci[0], "test_auc_pr_ci_high": ci[1],
                            "test_cost_at_threshold": cost_at_threshold(y_test, cal_test, threshold),
                            "test_default_rate": float(y_test.mean())})
        mlflow.log_text(cells_on_test.to_csv(index=False), "cells_on_test.csv")
        mlflow.log_text(reliability_table(y_test, raw_test).to_csv(index=False), "reliability_raw.csv")
        mlflow.log_text(reliability_table(y_test, cal_test).to_csv(index=False), "reliability_calibrated.csv")
        mlflow.log_dict({"cost_sensitivity": sensitivity, "calibration_cv_brier": cv_brier, "gate": gate}, "selection.json")
        if not gate["passed"]:
            raise SystemExit(f"champion fails the Brier gate: {gate}")
        signature = mlflow.models.infer_signature(X_test.head(50), cal_test[:50])
        info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=model_name, signature=signature,
            serialization_format="cloudpickle",  # our own models from our own registry
            code_paths=[str(Path(credit_drift.__file__).parent)])
        version = info.registered_model_version
        MlflowClient().set_registered_model_alias(model_name, "champion", version)
        tag_decision_settings(model_name, version, calibration=method,
                              training_cell=f"{champion['variant']}/{champion['family']}")

    baseline = {"model": model_name, "version": version, "run_id": run.info.run_id,
                "cell": f"{champion['variant']}/{champion['family']}", "calibration": method,
                "test": cal_metrics, "test_auc_pr_ci": ci, "default_rate": float(y_test.mean()),
                "cost_sensitivity": sensitivity}
    (results_dir / "champion_baseline.json").write_text(json.dumps(baseline, indent=1))
    return baseline
