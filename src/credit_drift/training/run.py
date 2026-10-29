"""Training entry point: three experiment stages, each feeding the next, then the champion.

  uv run python -m credit_drift.training.run features   # feature_engineering: variants A/B/C -> best variant
  uv run python -m credit_drift.training.run models     # model_selection: families on that variant -> shortlist
  uv run python -m credit_drift.training.run tune       # hyperparameter_tuning: Optuna per shortlisted family
  uv run python -m credit_drift.training.run select     # champion: pick, calibrate, test once, register
  uv run python -m credit_drift.training.run all        # the four stages in order
Add --quick for a smoke test (small sample, 3 trials, separate experiments and model name).

Every MLflow run carries the params `model`, `feature_variant` and `trial` (0 outside tuning), the tag
`stage`, the dataset hash, and the same metrics, so runs can be sorted and filtered in the UI.
"""
import argparse
import json
import os
import sys
import time
import warnings

import joblib
import mlflow
import pandas as pd

from credit_drift.split import DATA_DIR, REFERENCE_DIR
from credit_drift.training.champion import MODEL_NAME, pick_champion, select_and_register, shortlist
from credit_drift.training.describe import NOTE_TAG, describe_run
from credit_drift.training.features import VARIANTS
from credit_drift.training.search import (
    DEFAULT_PARAMS,
    FAMILIES,
    SEED,
    TRIALS,
    FoldCache,
    cv_evaluate,
    fit_final,
    split_xy,
    tune_family,
)

ARTIFACTS = DATA_DIR.parent / "artifacts"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENTS = {
    "features": "feature_engineering", "models": "model_selection",
    "tune": "hyperparameter_tuning", "select": "champion",
}
METRIC_KEYS = ("cv_auc_pr_mean", "cv_auc_pr_se", "oof_auc_pr", "oof_auc_roc", "oof_brier",
               "oof_cost_3", "oof_cost_5", "oof_cost_10", "seconds")
PROBE_FAMILIES = ("logreg", "lightgbm")  # stage 1: one linear and one tree model, same default settings


class Context:
    def __init__(self, quick: bool):
        self.quick = quick
        self.prefix = "quick_" if quick else ""
        self.out = ARTIFACTS / ("quick" if quick else "stages")
        self.out.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(TRACKING_URI)
        train = pd.read_parquet(REFERENCE_DIR / "train.parquet")
        if quick:
            train = train.sample(10_000, random_state=SEED).reset_index(drop=True)
        self.X, self.y = split_xy(train)
        self.cache = FoldCache(self.X, self.y)
        self.dataset = mlflow.data.from_pandas(
            train, source=str(REFERENCE_DIR / "train.parquet"), name="reference_train", targets="TARGET")

    def experiment(self, stage: str) -> None:
        mlflow.set_experiment(self.prefix + EXPERIMENTS[stage])

    def save(self, name: str, payload: dict) -> None:
        (self.out / f"{name}.json").write_text(json.dumps(payload, indent=1))

    def load(self, name: str) -> dict:
        path = self.out / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"{path.name} not found: run the previous stage first")
        return json.loads(path.read_text())

    def log_run(self, name: str, stage: str, variant: str, family: str, trial: int, params: dict,
                metrics: dict, kind: str, pruned: bool = False, n_trials: int | None = None) -> None:
        with mlflow.start_run(run_name=name):
            note = describe_run(stage=stage, kind=kind, variant=variant, family=family, trial=trial,
                                params={f"hp_{k}": v for k, v in params.items()}, pruned=pruned, n_trials=n_trials)
            mlflow.set_tags({"stage": stage, "kind": kind, "pruned": str(pruned).lower(), NOTE_TAG: note})
            mlflow.log_input(self.dataset, context="training")
            mlflow.log_params({"model": family, "feature_variant": variant, "trial": trial, "seed": SEED})
            mlflow.log_params({f"hp_{k}": v for k, v in params.items()})
            mlflow.log_metrics({k: v for k, v in metrics.items() if k in METRIC_KEYS and v is not None})


def _line(result: dict, label: str) -> str:
    return (f"  {label:<18} AUC-PR {result['cv_auc_pr_mean']:.4f} +- {result['cv_auc_pr_se']:.4f} | "
            f"Brier {result['oof_brier']:.4f} | cost@5 {result['oof_cost_5']:.4f} | {result['seconds']:.0f}s")


def run_features(ctx: Context, _args) -> None:
    """Stage 1: does engineering the features help? Same default settings for every variant."""
    ctx.experiment("features")
    results = []
    for variant in VARIANTS:
        for family in PROBE_FAMILIES:
            result = cv_evaluate(ctx.cache.get(variant, family), ctx.y, family, DEFAULT_PARAMS[family])
            result["variant"] = variant
            results.append(result)
            ctx.log_run(f"{variant} / {family}", "feature_engineering", variant, family, 0,
                        result["params"], result, "candidate")
            print(_line(result, f"{variant} / {family}"), flush=True)
    best = pick_champion(results)
    print(f"-> best feature variant: {best['variant']} (via {best['family']}); passed to model selection")
    ctx.save("stage1_features", {"results": results, "variant": best["variant"]})


def run_models(ctx: Context, _args) -> None:
    """Stage 2: which families work best on the chosen variant, still with default settings?"""
    variant = ctx.load("stage1_features")["variant"]
    ctx.experiment("models")
    results = []
    for family in FAMILIES:
        result = cv_evaluate(ctx.cache.get(variant, family), ctx.y, family, DEFAULT_PARAMS[family])
        result["variant"] = variant
        results.append(result)
        ctx.log_run(f"{variant} / {family}", "model_selection", variant, family, 0,
                    result["params"], result, "candidate")
        print(_line(result, f"{variant} / {family}"), flush=True)
    families = [r["family"] for r in sorted(shortlist(results), key=lambda r: -r["cv_auc_pr_mean"])]
    print(f"-> families within 1 SE of the best, passed to tuning: {families}")
    ctx.save("stage2_models", {"results": results, "variant": variant, "families": families})


def run_tune(ctx: Context, args) -> None:
    """Stage 3: Optuna for every shortlisted family. Each trial is one flat run; a summary run holds the refit."""
    stage2 = ctx.load("stage2_models")
    variant = stage2["variant"]
    families = [args.family] if args.family else stage2["families"]
    ctx.experiment("tune")
    cells = ctx.out / "cells"
    cells.mkdir(exist_ok=True)
    for family in families:
        n_trials = 3 if ctx.quick else (args.trials or TRIALS[family])
        print(f"tuning {variant} / {family}: {n_trials} trials", flush=True)

        def log_trial(number: int, params: dict, metrics: dict, pruned: bool, family=family, n_trials=n_trials) -> None:
            ctx.log_run(f"{family} trial {number:02d}", "hyperparameter_tuning", variant, family, number + 1,
                        params, metrics, "trial", pruned, n_trials)

        study = tune_family(ctx.cache, variant, family, n_trials, on_trial=log_trial)
        # Re-evaluate the best settings for out-of-fold metrics, then refit on all training rows.
        result = cv_evaluate(ctx.cache.get(variant, family), ctx.y, family, study["params"])
        pipeline, n_features = fit_final(ctx.X, ctx.y, variant, family, study["params"], result["n_estimators"])
        result.update(variant=variant, n_features=n_features, n_trials=study["n_trials"], n_pruned=study["n_pruned"])
        ctx.log_run(f"{family} best (refit)", "hyperparameter_tuning", variant, family, 0,
                    result["params"], result, "summary")
        joblib.dump(pipeline, cells / f"cell_{variant}_{family}.joblib")
        (cells / f"cell_{variant}_{family}.json").write_text(json.dumps(result, indent=1))
        print(_line(result, f"{family} tuned") + f" | pruned {study['n_pruned']}/{n_trials}", flush=True)


def run_select(ctx: Context, _args) -> None:
    cells = ctx.out / "cells"
    baseline = select_and_register(
        cells, REFERENCE_DIR, ctx.prefix + EXPERIMENTS["select"],
        model_name=("credit_scoring_quick" if ctx.quick else MODEL_NAME))
    print(json.dumps(baseline, indent=1))


def run_all(ctx: Context, args) -> None:
    for stage in (run_features, run_models, run_tune, run_select):
        started = time.time()
        stage(ctx, args)
        print(f"[{stage.__name__} took {(time.time() - started) / 60:.1f} min]", flush=True)


STAGES = {"features": run_features, "models": run_models, "tune": run_tune, "select": run_select, "all": run_all}


def main() -> None:
    warnings.filterwarnings("ignore")
    for stream in (sys.stdout, sys.stderr):  # MLflow prints emoji; Windows consoles default to cp1252
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--quick", action="store_true", help="smoke test: 10k rows, 3 trials, separate names")
    parser.add_argument("--trials", type=int, help="override the trials per family (tune stage)")
    parser.add_argument("--family", choices=FAMILIES, help="tune only this family")
    args = parser.parse_args()
    STAGES[args.stage](Context(args.quick), args)


if __name__ == "__main__":
    main()
