"""The two candidates every retrain builds (CLAUDE.md, "Retraining & recalibration"):

A: the current champion's own pipeline, recalibrated on the recent calibration slice (the cheap fix).
B: the champion's recipe (same preprocessing, same estimator and settings) refitted on the training window with
   recency weights, then recalibrated on the same slice.

Which one goes forward is decided WITHOUT the gate slice, on cross-fitted predictions over the calibration slice, so
the gate stays an unbiased test of the one candidate that reaches it.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from credit_drift.data import TARGET
from credit_drift.retraining.window import Window
from credit_drift.training.metrics import cost_at_threshold
from credit_drift.training.model import CalibratedModel, apply_calibrator, fit_calibrator

CALIBRATION_METHODS = ("isotonic", "platt")


def features_of(frame):
    return frame.drop(columns=[c for c in (TARGET, "week") if c in frame.columns])


def crossfit(method: str, raw: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Out-of-fold calibrated probabilities: each row is scored by a calibrator that never saw it."""
    pred = np.zeros(len(y))
    for train_idx, valid_idx in KFold(5, shuffle=True, random_state=seed).split(raw):
        pred[valid_idx] = apply_calibrator(method, fit_calibrator(method, raw[train_idx], y[train_idx]), raw[valid_idx])
    return pred


def calibrate(pipeline, X_cal, y_cal: np.ndarray, threshold: float) -> tuple[CalibratedModel, dict]:
    """Pick isotonic or Platt by cross-fitted Brier, fit it on the whole slice, report the cross-fitted cost."""
    raw = pipeline.predict_proba(X_cal)[:, 1]
    folds = {m: crossfit(m, raw, y_cal) for m in CALIBRATION_METHODS}
    method = min(folds, key=lambda m: brier_score_loss(y_cal, folds[m]))
    model = CalibratedModel(pipeline, fit_calibrator(method, raw, y_cal), method)
    return model, {"calibration": method, "crossfit_brier": float(brier_score_loss(y_cal, folds[method])),
                   "crossfit_cost": cost_at_threshold(y_cal, folds[method], threshold)}


def retrain_pipeline(champion: CalibratedModel, window: Window) -> Pipeline:
    """The champion's recipe on the new window: same steps and settings, refitted with the recency weights."""
    prep = clone(champion.pipeline.named_steps["prep"])
    estimator = clone(champion.pipeline.named_steps["model"])
    X, y = features_of(window.fit), window.fit[TARGET].to_numpy()
    matrix = prep.fit_transform(X, y)
    estimator.fit(matrix, y, sample_weight=window.weights)
    return Pipeline([("prep", prep), ("model", estimator)])


def build_candidates(champion: CalibratedModel, window: Window, threshold: float) -> dict[str, dict]:
    X_cal, y_cal = features_of(window.calibration), window.calibration[TARGET].to_numpy()
    out = {}
    out["A"] = dict(zip(("model", "calibration"), calibrate(champion.pipeline, X_cal, y_cal, threshold), strict=True))
    out["B"] = dict(zip(("model", "calibration"),
                        calibrate(retrain_pipeline(champion, window), X_cal, y_cal, threshold), strict=True))
    return out


def pick(candidates: dict[str, dict]) -> str:
    """Lowest cross-fitted expected cost on the calibration slice; a tie goes to A, the simpler change."""
    return min(candidates, key=lambda k: (round(candidates[k]["calibration"]["crossfit_cost"], 6), k))
