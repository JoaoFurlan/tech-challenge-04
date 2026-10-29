"""Feature variants A / B / C (docs/decisions.md, #2 and #3). Every statistic is fitted on training rows only.

A: raw columns, median fill, one-hot; the 365243 sentinel is deliberately left untreated.
B: A plus sentinel handling, missing-value indicators, financial ratios, EXT_SOURCE aggregates, log income.
C: B minus near-duplicate columns and low-importance features (selection happens inside each fold).
Gender and the ID are never model inputs.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_drift.data import DAYS_EMPLOYED_SENTINEL, ID_COL, TARGET

EXCLUDED = (ID_COL, TARGET, "CODE_GENDER")
EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
VARIANTS = ("A", "B", "C")
SELECT_K = 60
CORR_THRESHOLD = 0.95


class DropColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_out_ = list(self.transform(X.head(2)).columns)
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in EXCLUDED if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_out_)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Variant B features. Fitted state: the income winsorisation cap and the housing block columns."""

    def __init__(self, income_quantile: float = 0.999):
        self.income_quantile = income_quantile

    def fit(self, X, y=None):
        self.income_cap_ = float(X["AMT_INCOME_TOTAL"].quantile(self.income_quantile))
        self.house_cols_ = [c for c in X.columns if c.endswith(("_AVG", "_MODE", "_MEDI"))]
        self.columns_out_ = list(self.transform(X.head(2)).columns)
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_out_)

    def transform(self, X):
        out = X.drop(columns=[c for c in EXCLUDED if c in X.columns])
        sentinel = out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
        out["DAYS_EMPLOYED_ANOM"] = sentinel.astype("int8")
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].mask(sentinel)
        out["HOUSING_BLOCK_MISSING"] = (X[self.house_cols_].isna().mean(axis=1) > 0.9).astype("int8")
        for col in ("EXT_SOURCE_1", "EXT_SOURCE_3"):
            out[f"{col}_MISSING"] = out[col].isna().astype("int8")

        income = out["AMT_INCOME_TOTAL"].clip(upper=self.income_cap_)
        out["AMT_INCOME_TOTAL"] = income
        out["LOG_INCOME"] = np.log1p(income)
        out["ANNUITY_INCOME"] = out["AMT_ANNUITY"] / income
        out["CREDIT_INCOME"] = out["AMT_CREDIT"] / income
        out["CREDIT_GOODS"] = out["AMT_CREDIT"] / out["AMT_GOODS_PRICE"]
        out["ANNUITY_CREDIT"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"]
        out["EMPLOYED_AGE"] = out["DAYS_EMPLOYED"] / out["DAYS_BIRTH"]
        ext = out[EXT_COLS]
        out["EXT_MEAN"] = ext.mean(axis=1)
        out["EXT_MIN"] = ext.min(axis=1)
        out["EXT_PROD"] = ext.prod(axis=1, skipna=False)
        return out.replace([np.inf, -np.inf], np.nan)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop the later column of every pair with |correlation| above the threshold."""

    def __init__(self, threshold: float = CORR_THRESHOLD, max_rows: int = 30_000):
        self.threshold = threshold
        self.max_rows = max_rows

    def fit(self, X, y=None):
        corr = pd.DataFrame(X[: self.max_rows]).corr().abs().to_numpy()
        upper = np.triu(np.nan_to_num(corr), k=1)
        drop = (upper > self.threshold).any(axis=0)
        self.keep_ = np.flatnonzero(~drop)
        return self

    def transform(self, X):
        return X[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)[self.keep_]


class ImportanceSelector(BaseEstimator, TransformerMixin):
    """Keep the k features with the highest gain in a small LightGBM fitted on the training fold."""

    def __init__(self, k: int = SELECT_K, seed: int = 42):
        self.k = k
        self.seed = seed

    def fit(self, X, y):
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.1, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=self.seed, n_jobs=-1, verbose=-1,
        ).fit(X, y)
        gain = model.booster_.feature_importance(importance_type="gain")
        self.keep_ = np.sort(np.argsort(gain)[::-1][: self.k])
        return self

    def transform(self, X):
        return X[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)[self.keep_]


def _to_float32(a):
    return np.asarray(a, dtype=np.float32)


def build_preprocessor(variant: str, family: str, seed: int = 42) -> Pipeline:
    """Raw frame -> float32 matrix. Fit it on training rows only (once per CV fold)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    # Trees take nulls natively in B and C; variant A and logistic regression need median fill.
    impute = variant == "A" or family == "logreg"
    numeric_steps = [("impute", SimpleImputer(strategy="median"))] if impute else []
    if family == "logreg":
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False, dtype=np.float32)
    encode = ColumnTransformer(
        [("num", numeric, make_column_selector(dtype_exclude=object)),
         ("cat", categorical, make_column_selector(dtype_include=object))],
        verbose_feature_names_out=False,
    )
    steps = [
        ("features", DropColumns() if variant == "A" else FeatureEngineer()),
        ("encode", encode),
        ("float32", FunctionTransformer(_to_float32, feature_names_out="one-to-one")),
    ]
    if variant == "C":
        steps += [("dedupe", CorrelationFilter()), ("select", ImportanceSelector(seed=seed))]
    return Pipeline(steps)
