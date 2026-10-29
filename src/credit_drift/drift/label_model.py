"""Label model used only by the simulator: where concept drift redraws labels from.

It is an independent LightGBM (different seed and settings from the champion), Platt-calibrated on the
Reference calibration part. Independence matters: labels redrawn from the champion's own probabilities
would flatter it, while a weak label model (a logistic regression) would penalise it before any drift.
Measured: redrawing all labels from this model changes the champion's AUC-PR by +0.002 (0.2526 vs 0.2509).
"""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_drift.data import TARGET
from credit_drift.drift.schedule import LABEL_MODEL_SEED
from credit_drift.training.features import build_preprocessor
from credit_drift.training.model import apply_calibrator, fit_calibrator

EXT_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


@dataclass
class LabelModel:
    pipeline: object
    estimator: object
    calibrator: object
    ext_median: pd.Series
    ext_std: pd.Series
    b2: float  # linear coefficient of EXT_SOURCE_2, with the other two scores, on standardised inputs

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(self.pipeline.transform(X))[:, 1]
        return apply_calibrator("platt", self.calibrator, raw)

    def ext2_standardised(self, X: pd.DataFrame) -> np.ndarray:
        col = "EXT_SOURCE_2"
        return ((X[col].fillna(self.ext_median[col]) - self.ext_median[col]) / self.ext_std[col]).to_numpy()


def fit_label_model(train: pd.DataFrame, calibration: pd.DataFrame, seed: int = LABEL_MODEL_SEED) -> LabelModel:
    pipeline = build_preprocessor("B", "lightgbm", seed)
    y_train = train[TARGET].to_numpy()
    matrix = pipeline.fit_transform(train.drop(columns=TARGET), y_train)
    estimator = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.04, num_leaves=24, min_child_samples=60, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.6, reg_lambda=5, random_state=seed, verbose=-1).fit(matrix, y_train)
    raw_cal = estimator.predict_proba(pipeline.transform(calibration.drop(columns=TARGET)))[:, 1]
    calibrator = fit_calibrator("platt", raw_cal, calibration[TARGET].to_numpy())

    ext = train[EXT_COLUMNS]
    median, std = ext.median(), ext.fillna(ext.median()).std()
    z = (ext.fillna(median) - median) / std
    b2 = float(LogisticRegression(max_iter=1000).fit(z, y_train).coef_[0][EXT_COLUMNS.index("EXT_SOURCE_2")])
    return LabelModel(pipeline, estimator, calibrator, median, std, b2)
