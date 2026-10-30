"""Helpers for the retraining notebooks (02, 03): read what the pipeline logged, score later weeks with several model
versions on the SAME batches, and put bootstrap intervals on the gaps. Offline only; notebooks never train or promote.
"""
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, brier_score_loss

from credit_drift.data import ID_COL, TARGET
from credit_drift.drift import schedule as sch
from credit_drift.drift.simulate import segment_mask
from credit_drift.retraining.run import EXPERIMENT
from credit_drift.split import PRODUCTION_DIR, REFERENCE_DIR
from credit_drift.training.metrics import DEFAULT_RATIO, cost_at_threshold

MODEL = "credit_scoring"
THRESHOLD = 1 / (1 + DEFAULT_RATIO)


def retraining_runs() -> list[dict]:
    """One record per retraining run (what the DAG logged), oldest first."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([experiment.experiment_id], max_results=100)
    out = []
    for run in runs:
        detail = mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/retraining.json")
        out.append({"run_id": run.info.run_id, "run_week": int(run.data.params["run_week"]),
                    "through_week": int(run.data.params["through_week"]), "trigger": run.data.params["trigger"], **detail})
    return sorted(out, key=lambda r: r["run_week"])


def load_batch(week: int) -> pd.DataFrame:
    folder = PRODUCTION_DIR / "weeks"
    features = pd.read_parquet(folder / f"week_{week:02d}_features.parquet")
    return features.merge(pd.read_parquet(folder / f"week_{week:02d}_labels.parquet"), on=ID_COL).assign(week=week)


def score_weeks(versions: dict[str, str], weeks: list[int]) -> pd.DataFrame:
    """Every model scores the same rows: columns week, TARGET, one probability column per name, plus the segment flag
    (age <= 45 and income <= the training median, on the pre-drift income: ground truth, so offline only)."""
    models = {name: mlflow.sklearn.load_model(f"models:/{MODEL}/{v}") for name, v in versions.items()}
    cutoff = float(pd.read_parquet(REFERENCE_DIR / "train.parquet")["AMT_INCOME_TOTAL"].median())
    mult = {s.week: s.income_mult for s in sch.SCHEDULE}
    frames = []
    for week in weeks:
        batch = load_batch(week)
        X = batch.drop(columns=[TARGET, "week"])
        original = X.assign(AMT_INCOME_TOTAL=X["AMT_INCOME_TOTAL"] / mult[week])
        frame = pd.DataFrame({"week": week, TARGET: batch[TARGET].to_numpy(),
                              "in_segment": segment_mask(original, cutoff)})
        for name, model in models.items():
            frame[name] = model.predict_proba(X)[:, 1]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def metrics(frame: pd.DataFrame, name: str) -> dict:
    y, p = frame[TARGET].to_numpy(), frame[name].to_numpy()
    return {"auc_pr": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "cost": cost_at_threshold(y, p, THRESHOLD), "mean_pred": float(p.mean()), "observed": float(y.mean())}


def per_week(scored: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = [{"week": w, "model": n, **metrics(group, n)} for w, group in scored.groupby("week") for n in names]
    return pd.DataFrame(rows)


def gap(frame: pd.DataFrame, base: str, other: str, n: int = 1000, seed: int = 42) -> dict:
    """other - base on the same rows, with a paired bootstrap 95% interval. Cost and Brier: negative is better."""
    rng = np.random.default_rng(seed)
    y, p0, p1 = frame[TARGET].to_numpy(), frame[base].to_numpy(), frame[other].to_numpy()
    draws = {"auc_pr": [], "brier": [], "cost": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        draws["auc_pr"].append(average_precision_score(y[i], p1[i]) - average_precision_score(y[i], p0[i]))
        draws["brier"].append(brier_score_loss(y[i], p1[i]) - brier_score_loss(y[i], p0[i]))
        draws["cost"].append(cost_at_threshold(y[i], p1[i], THRESHOLD) - cost_at_threshold(y[i], p0[i], THRESHOLD))
    out = {"rows": len(y), "defaulters": int(y.sum())}
    for key, values in draws.items():
        low, high = np.quantile(values, [0.025, 0.975])
        out[key] = float(np.mean(values))
        out[f"{key}_low"], out[f"{key}_high"] = float(low), float(high)
    return out


def reliability(frame: pd.DataFrame, name: str, bins: int = 10) -> pd.DataFrame:
    table = pd.DataFrame({"y": frame[TARGET], "p": frame[name]})
    table["bin"] = pd.qcut(table["p"], bins, duplicates="drop")
    return table.groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean")).reset_index(drop=True)
