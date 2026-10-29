"""The registered model: preprocessing + estimator + probability calibration, accepting raw rows."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(method: str, raw: np.ndarray, y: np.ndarray):
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    if method == "platt":
        return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw), y)
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(method: str, calibrator, raw: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        return calibrator.predict(raw)
    return calibrator.predict_proba(_logit(raw))[:, 1]


class CalibratedModel(ClassifierMixin, BaseEstimator):
    """Fitted pipeline (raw rows -> probability) followed by a fitted calibrator."""

    def __init__(self, pipeline=None, calibrator=None, method: str = "isotonic"):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.method = method
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw = self.pipeline.predict_proba(X)[:, 1]
        p = apply_calibrator(self.method, self.calibrator, raw)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
