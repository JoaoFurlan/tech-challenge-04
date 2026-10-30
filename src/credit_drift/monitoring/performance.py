"""Label-dependent metrics: overall performance of a week, per-slice performance, and the champion's baseline.

Labels arrive one week late, so these run for week N-1 inside the run of week N. The baseline is the champion on the
untouched Reference test part, computed once per model version.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credit_drift.data import TARGET
from credit_drift.monitoring.slices import slice_cuts, slice_masks
from credit_drift.training.metrics import cost_at_threshold

MIN_DEFAULTERS = 30  # below this a slice's AUC-PR is not reported (too few positives to mean anything)
N_BOOTSTRAP = 200


def _auc_pr(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def overall_metrics(y: np.ndarray, p: np.ndarray, threshold: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    boot = [average_precision_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOTSTRAP))]
    flagged = p >= threshold
    return {
        "rows": len(y), "defaulters": int(y.sum()), "observed_rate": float(y.mean()),
        "mean_pred": float(p.mean()), "auc_pr": _auc_pr(y, p),
        "auc_pr_ci_low": float(np.quantile(boot, 0.025)), "auc_pr_ci_high": float(np.quantile(boot, 0.975)),
        "auc_roc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
        "flag_rate": float(flagged.mean()),
        "precision_at_threshold": float(y[flagged].mean()) if flagged.any() else float("nan"),
        "cost_at_threshold": cost_at_threshold(y, p, threshold),
    }


def rolling_auc_pr(history: list[dict]) -> dict:
    """AUC-PR over the last labelled weeks: the mean of the weekly values and its standard error.

    One week holds ~400 defaulters, so a week's AUC-PR wobbles by about +-0.04; averaging a few weeks shrinks that.
    The standard error of each week comes from its bootstrap interval (width / 3.92); weeks are independent samples."""
    values = [h["auc_pr"] for h in history]
    ses = [(h["auc_pr_ci_high"] - h["auc_pr_ci_low"]) / 3.92 for h in history]
    return {"weeks": [h["week"] for h in history], "mean": float(np.mean(values)),
            "se": float(np.sqrt(np.sum(np.square(ses))) / len(values))}


def slice_metrics(y: np.ndarray, p: np.ndarray, masks: dict[str, np.ndarray], seed: int = 0) -> dict[str, dict]:
    """AUC-PR per slice with a bootstrap interval; NaN when the slice has too few defaulters."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, mask in masks.items():
        ys, ps = y[mask], p[mask]
        entry = {"rows": int(mask.sum()), "defaulters": int(ys.sum()), "auc_pr": float("nan"),
                 "ci_low": float("nan"), "ci_high": float("nan")}
        if ys.sum() >= MIN_DEFAULTERS:
            entry["auc_pr"] = _auc_pr(ys, ps)
            boot = [_auc_pr(ys[i], ps[i]) for i in (rng.integers(0, len(ys), len(ys)) for _ in range(N_BOOTSTRAP))]
            entry["ci_low"], entry["ci_high"] = float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))
        out[name] = entry
    return out


def compute_baseline(model, train: pd.DataFrame, test: pd.DataFrame, threshold: float, version: str) -> dict:
    """The champion's own validation baseline: overall and per slice, on the Reference test part."""
    cuts = slice_cuts(train)
    p = model.predict_proba(test.drop(columns=TARGET))[:, 1]
    y = test[TARGET].to_numpy()
    return {"model_version": version, "decision_threshold": threshold, "slice_cuts": cuts,
            "overall": overall_metrics(y, p, threshold), "slices": slice_metrics(y, p, slice_masks(test, cuts))}


def pooled_slice_performance(frames: list[pd.DataFrame], cuts: dict[str, float]) -> tuple[dict, dict]:
    """Slice and overall AUC-PR over a window of labelled weeks (each frame: features + TARGET + probability)."""
    pooled = pd.concat(frames, ignore_index=True)
    y, p = pooled[TARGET].to_numpy(), pooled["probability"].to_numpy()
    masks = slice_masks(pooled, cuts)
    return slice_metrics(y, p, masks), {"rows": len(pooled), "auc_pr": _auc_pr(y, p)}
