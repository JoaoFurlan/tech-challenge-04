"""Cross-validated evaluation and Optuna tuning, shared by the three experiment stages.

Preprocessing does not depend on hyperparameters, so it is fitted once per CV fold and cached;
each evaluation only fits the estimator. Folds are identical everywhere (fair comparison).
"""
import time
from collections.abc import Callable
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from credit_drift.data import TARGET
from credit_drift.training.features import build_preprocessor
from credit_drift.training.metrics import summarize

FAMILIES = ("logreg", "lightgbm", "xgboost")
TRIALS = {"logreg": 12, "lightgbm": 25, "xgboost": 25}
N_FOLDS = 3
SEED = 42
MAX_TREES = 1000
EARLY_STOP = 50

# Reasonable fixed settings for the stages that compare features and models (no tuning yet).
DEFAULT_PARAMS = {
    "logreg": {"C": 1.0},
    "lightgbm": {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8,
                 "subsample": 0.8, "reg_alpha": 0.0, "reg_lambda": 0.0},
    "xgboost": {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1.0, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_lambda": 1.0, "gamma": 0.0},
}


@dataclass
class Fold:
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    valid_index: np.ndarray


def suggest(trial: optuna.Trial, family: str) -> dict:
    if family == "logreg":
        return {"C": trial.suggest_float("C", 1e-4, 10, log=True)}
    if family == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }


def make_estimator(family: str, params: dict, n_estimators: int | None = None, seed: int = SEED):
    """n_estimators=None means 'many trees, stop early on the validation fold'."""
    trees = n_estimators or MAX_TREES
    if family == "logreg":
        return LogisticRegression(C=params["C"], max_iter=1000)
    if family == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=trees, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1, **params)
    return xgb.XGBClassifier(
        n_estimators=trees, tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=-1,
        early_stopping_rounds=EARLY_STOP if n_estimators is None else None, **params)


def fit_predict(family: str, params: dict, fold: Fold) -> tuple[np.ndarray, int | None]:
    """Fit on the fold's training part, return validation probabilities and the best iteration."""
    est = make_estimator(family, params)
    if family == "logreg":
        est.fit(fold.X_train, fold.y_train)
        return est.predict_proba(fold.X_valid)[:, 1], None
    if family == "lightgbm":
        est.fit(
            fold.X_train, fold.y_train, eval_set=[(fold.X_valid, fold.y_valid)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        best = est.best_iteration_
    else:
        est.fit(fold.X_train, fold.y_train, eval_set=[(fold.X_valid, fold.y_valid)], verbose=False)
        best = est.best_iteration + 1
    return est.predict_proba(fold.X_valid)[:, 1], int(best)


class FoldCache:
    """Per-fold preprocessing fitted once per (variant, preprocessing kind) and reused by every run."""

    def __init__(self, X: pd.DataFrame, y: np.ndarray, seed: int = SEED):
        self.X, self.y, self.seed = X, y, seed
        self._folds: dict[tuple[str, str], list[Fold]] = {}

    def get(self, variant: str, family: str) -> list[Fold]:
        kind = "logreg" if family == "logreg" else "tree"  # the two preprocessing flavours (see features.py)
        if (variant, kind) not in self._folds:
            folds = []
            for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=self.seed).split(self.X, self.y):
                prep = build_preprocessor(variant, family, self.seed)
                X_train = prep.fit_transform(self.X.iloc[tr], self.y[tr])
                folds.append(Fold(X_train, self.y[tr], prep.transform(self.X.iloc[va]), self.y[va], va))
            self._folds[(variant, kind)] = folds
        return self._folds[(variant, kind)]


def cv_evaluate(folds: list[Fold], y: np.ndarray, family: str, params: dict,
                on_fold: Callable[[int, float], None] | None = None) -> dict:
    """Cross-validated AUC-PR (mean and standard error) plus pooled out-of-fold metrics."""
    started = time.time()
    oof = np.zeros(len(y))
    scores, iters = [], []
    for step, fold in enumerate(folds):
        proba, best = fit_predict(family, params, fold)
        oof[fold.valid_index] = proba
        scores.append(float(average_precision_score(fold.y_valid, proba)))
        iters.append(best)
        if on_fold:
            on_fold(step, float(np.mean(scores)))
    return {
        "family": family, "params": params,
        "cv_auc_pr_mean": float(np.mean(scores)),
        "cv_auc_pr_se": float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
        "fold_auc_pr": scores,
        "n_estimators": int(np.mean(iters) * 1.1) if iters[0] else None,  # a bit more data in the final fit
        **{f"oof_{k}": v for k, v in summarize(y, oof).items()},
        "seconds": round(time.time() - started, 1),
    }


def tune_family(cache: FoldCache, variant: str, family: str, n_trials: int,
                on_trial: Callable[[int, dict, float | None, bool], None] | None = None,
                seed: int = SEED) -> dict:
    """Optuna study for one family on one variant. `on_trial(number, params, cv_auc_pr, pruned)` logs each trial."""
    folds = cache.get(variant, family)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, family)

        def report(step: int, running_mean: float) -> None:
            trial.report(running_mean, step)
            if trial.should_prune():
                if on_trial:
                    on_trial(trial.number, params, running_mean, True)
                raise optuna.TrialPruned()

        result = cv_evaluate(folds, cache.y, family, params, on_fold=report)
        if on_trial:
            on_trial(trial.number, params, result["cv_auc_pr_mean"], False)
        return result["cv_auc_pr_mean"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0))
    study.optimize(objective, n_trials=n_trials)
    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    return {"params": study.best_trial.params, "n_trials": n_trials, "n_pruned": n_pruned}


def fit_final(X: pd.DataFrame, y: np.ndarray, variant: str, family: str, params: dict,
              n_estimators: int | None, seed: int = SEED) -> tuple[Pipeline, int]:
    """Refit preprocessing and estimator on all training rows; returns the pipeline and its feature count."""
    prep = build_preprocessor(variant, family, seed)
    matrix = prep.fit_transform(X, y)
    estimator = make_estimator(family, params, n_estimators=n_estimators, seed=seed).fit(matrix, y)
    return Pipeline([("prep", prep), ("model", estimator)]), int(matrix.shape[1])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df.drop(columns=[TARGET]), df[TARGET].to_numpy()
