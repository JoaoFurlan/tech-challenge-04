"""Optuna search inside one (feature variant x model family) cell, logged to MLflow.

Preprocessing does not depend on hyperparameters, so it is fitted once per CV fold and cached;
each trial only fits the estimator. Folds are identical across cells (fair comparison).
"""
import time
from dataclasses import dataclass

import lightgbm as lgb
import mlflow
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


def prepare_folds(X: pd.DataFrame, y: np.ndarray, variant: str, family: str, seed: int = SEED) -> list[Fold]:
    folds = []
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(X, y):
        prep = build_preprocessor(variant, family, seed)
        X_train = prep.fit_transform(X.iloc[train_idx], y[train_idx])
        folds.append(Fold(X_train, y[train_idx], prep.transform(X.iloc[valid_idx]), y[valid_idx], valid_idx))
    return folds


def tune_cell(X: pd.DataFrame, y: np.ndarray, variant: str, family: str, n_trials: int | None = None,
              seed: int = SEED) -> tuple[dict, Pipeline]:
    """Run the study, then recompute out-of-fold predictions for the best trial and refit on all of X.

    Must be called inside an active MLflow parent run; each trial is logged as a nested run.
    """
    started = time.time()
    n_trials = n_trials or TRIALS[family]
    folds = prepare_folds(X, y, variant, family, seed)

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, family)
        scores, iters = [], []
        with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
            mlflow.log_params(params)
            for step, fold in enumerate(folds):
                proba, best = fit_predict(family, params, fold)
                scores.append(average_precision_score(fold.y_valid, proba))
                iters.append(best)
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    mlflow.set_tag("pruned", "true")
                    mlflow.log_metric("cv_auc_pr_partial", float(np.mean(scores)))
                    raise optuna.TrialPruned()
            mean = float(np.mean(scores))
            mlflow.log_metrics({"cv_auc_pr": mean, **{f"fold_{i}_auc_pr": s for i, s in enumerate(scores)}})
        trial.set_user_attr("iters", iters)
        trial.set_user_attr("fold_scores", scores)
        return mean

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    params = best.params
    oof = np.zeros(len(y))
    fold_scores, iters = [], []
    for fold in folds:
        proba, it = fit_predict(family, params, fold)
        oof[fold.valid_index] = proba
        fold_scores.append(average_precision_score(fold.y_valid, proba))
        iters.append(it)
    mean_iter = int(np.mean(iters) * 1.1) if iters[0] else None  # a bit more data in the final fit

    prep = build_preprocessor(variant, family, seed)
    X_full = prep.fit_transform(X, y)
    estimator = make_estimator(family, params, n_estimators=mean_iter, seed=seed).fit(X_full, y)
    pipeline = Pipeline([("prep", prep), ("model", estimator)])

    result = {
        "variant": variant, "family": family, "params": params, "n_trials": n_trials,
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "cv_auc_pr_mean": float(np.mean(fold_scores)),
        "cv_auc_pr_se": float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores))),
        "n_estimators": mean_iter, "n_features": int(X_full.shape[1]),
        **{f"oof_{k}": v for k, v in summarize(y, oof).items()},
        "seconds": round(time.time() - started, 1),
    }
    return result, pipeline


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df.drop(columns=[TARGET]), df[TARGET].to_numpy()
