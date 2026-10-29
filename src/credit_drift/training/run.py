"""Training entry point.

  uv run python -m credit_drift.training.run tune            # all 9 cells (skips finished ones)
  uv run python -m credit_drift.training.run tune --quick    # smoke test: small sample, 3 trials
  uv run python -m credit_drift.training.run select          # pick, calibrate, evaluate, register
"""
import argparse
import json
import os
import sys
import warnings

import joblib
import mlflow
import pandas as pd

from credit_drift.split import DATA_DIR, REFERENCE_DIR
from credit_drift.training.champion import MODEL_NAME, select_and_register
from credit_drift.training.features import VARIANTS
from credit_drift.training.search import FAMILIES, SEED, TRIALS, split_xy, tune_cell

ARTIFACTS = DATA_DIR.parent / "artifacts"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def run_tune(args: argparse.Namespace) -> None:
    out = ARTIFACTS / ("quick" if args.quick else "cells")
    out.mkdir(parents=True, exist_ok=True)
    experiment = "credit_scoring_quick" if args.quick else "credit_scoring"
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)

    train = pd.read_parquet(REFERENCE_DIR / "train.parquet")
    if args.quick:
        train = train.sample(10_000, random_state=SEED).reset_index(drop=True)
    X, y = split_xy(train)
    dataset = mlflow.data.from_pandas(
        train, source=str(REFERENCE_DIR / "train.parquet"), name="reference_train", targets="TARGET")

    wanted = [c.split("/") for c in args.cells] if args.cells else [(v, f) for v in VARIANTS for f in FAMILIES]
    for variant, family in wanted:
        stem = out / f"cell_{variant}_{family}"
        if stem.with_suffix(".json").exists() and not args.force:
            print(f"skip {variant}/{family} (done; use --force to redo)")
            continue
        n_trials = 3 if args.quick else (args.trials or TRIALS[family])
        print(f"tuning {variant}/{family} ({n_trials} trials)...", flush=True)
        with mlflow.start_run(run_name=f"cell_{variant}_{family}"):
            mlflow.log_input(dataset, context="training")
            mlflow.log_params({"variant": variant, "family": family, "n_trials": n_trials, "seed": SEED})
            result, pipeline = tune_cell(X, y, variant, family, n_trials=n_trials)
            mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, float)})
            mlflow.log_params({f"best_{k}": v for k, v in result["params"].items()})
        joblib.dump(pipeline, stem.with_suffix(".joblib"))
        stem.with_suffix(".json").write_text(json.dumps(result, indent=1))
        print(f"  cv AUC-PR {result['cv_auc_pr_mean']:.4f} +- {result['cv_auc_pr_se']:.4f} | "
              f"OOF Brier {result['oof_brier']:.4f} | cost@5 {result['oof_cost_5']:.4f} | {result['seconds']:.0f}s", flush=True)


def run_select(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    out = ARTIFACTS / ("quick" if args.quick else "cells")
    quick = args.quick
    baseline = select_and_register(
        out, REFERENCE_DIR, "credit_scoring_quick" if quick else "credit_scoring",
        model_name="credit_scoring_quick" if quick else MODEL_NAME)
    print(json.dumps(baseline, indent=1))


def main() -> None:
    warnings.filterwarnings("ignore")
    for stream in (sys.stdout, sys.stderr):  # MLflow prints emoji; Windows consoles default to cp1252
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    tune = sub.add_parser("tune")
    tune.add_argument("--cells", nargs="*", help="e.g. B/lightgbm (default: all nine)")
    tune.add_argument("--trials", type=int, help="override the number of trials per cell")
    tune.add_argument("--force", action="store_true")
    tune.add_argument("--quick", action="store_true")
    select = sub.add_parser("select")
    select.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    {"tune": run_tune, "select": run_select}[args.stage](args)


if __name__ == "__main__":
    main()
