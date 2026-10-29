"""Plain-language descriptions for MLflow runs (the `mlflow.note.content` tag shown on the run page).

Used when a run is logged and by the backfill command, so old and new runs read the same:
  uv run python -m credit_drift.training.describe
"""
import os

import mlflow
from mlflow import MlflowClient

NOTE_TAG = "mlflow.note.content"

VARIANT_TEXT = {
    "A": ("**Features, variant A (minimal).** Raw columns as they come: numbers filled with the median, "
          "categories one-hot encoded (rare ones grouped). The 365243 code in `DAYS_EMPLOYED` is left "
          "untouched. No engineered features."),
    "B": ("**Features, variant B (engineered).** Variant A plus: the 365243 code in `DAYS_EMPLOYED` becomes "
          "a null and a flag; a flag for the missing housing block; missing flags for `EXT_SOURCE_1/3`; "
          "ratios (annuity/income, credit/income, credit/goods price, annuity/credit, employed/age); "
          "`EXT_SOURCE` mean, min and product; income capped at the 99.9th percentile plus its log."),
    "C": ("**Features, variant C (selected).** Variant B, then columns correlated above 0.95 are dropped and "
          "only the 60 features with the highest gain in a small LightGBM are kept. Both steps are fitted "
          "inside each training fold, so they never see validation rows."),
}
FOOTER = "Gender, the ID and the target are never model inputs."
PREPROCESSING_TEXT = {
    "logreg": "Logistic regression gets median-filled, standardised numbers.",
    "tree": "Tree models keep nulls as they are (variant B and C); variant A fills them with the median.",
}
MODEL_TEXT = {"logreg": "Logistic regression (L2)", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
HOW_TO_READ = ("Rank by `cv_auc_pr_mean` (higher is better); candidates within one `cv_auc_pr_se` are tied, "
               "then the lowest `oof_cost_5` wins. `oof_brier` must stay below 0.0745.")


def _hyperparameters(params: dict) -> str:
    def short(value) -> str:
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    shown = {k.removeprefix("hp_"): short(v) for k, v in params.items() if k.startswith("hp_")}
    return ", ".join(f"{k}={v}" for k, v in shown.items()) or "defaults"


def describe_run(*, stage: str, kind: str, variant: str, family: str, trial: int, params: dict,
                 pruned: bool = False, n_trials: int | None = None) -> str:
    lines = [VARIANT_TEXT[variant], PREPROCESSING_TEXT["logreg" if family == "logreg" else "tree"] + " " + FOOTER]
    model = MODEL_TEXT[family]
    settings = _hyperparameters(params)
    if stage == "hyperparameter_tuning" and kind == "trial":
        total = f" of {n_trials}" if n_trials else ""
        lines.append(f"**Model: {model}**, Optuna trial {trial}{total} (the run name counts from 0). "
                     f"Settings: {settings}. Scored with 3-fold stratified cross-validation on the training part.")
        if pruned:
            lines.append("**Pruned:** the pruner stopped it after the first fold(s) because it was behind the "
                         "median of earlier trials. `cv_auc_pr_mean` here is only a partial score from the "
                         "folds that ran, not the 3-fold average, and the other metrics were not computed. "
                         "Filter `tags.pruned = \"false\"` before sorting.")
    elif stage == "hyperparameter_tuning":
        lines.append(f"**Model: {model}**, best Optuna settings re-evaluated on all 3 folds (full metrics) and "
                     f"refit on every training row. Settings: {settings}.")
    else:
        goal = ("which feature variant helps most" if stage == "feature_engineering"
                else "which model family works best on the chosen variant")
        lines.append(f"**Model: {model}** with fixed default settings ({settings}), no tuning. This stage asks "
                     f"{goal}. Scored with 3-fold stratified cross-validation on the training part.")
    lines.append(HOW_TO_READ)
    return "\n\n".join(lines)


def describe_champion(cell: str, calibration: str, cost_ratio: float, shortlist: str) -> str:
    return (f"**Champion selection.** Candidates within one standard error of the best AUC-PR: {shortlist}. "
            f"Among them the lowest expected cost at {cost_ratio:g}:1 (missed default : false alarm) won: "
            f"**{cell}**. Its probabilities were calibrated on a held-out calibration part ({calibration}), "
            "evaluated once on the untouched test part, and registered as `credit_scoring@champion`. "
            "Gate: the calibrated Brier score must beat the constant-default-rate guess.")


def backfill(tracking_uri: str | None = None) -> int:
    """Write descriptions on every existing run of the four experiments; returns how many were updated."""
    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()
    updated = 0
    for name in ("feature_engineering", "model_selection", "hyperparameter_tuning", "champion"):
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            continue
        runs = client.search_runs([experiment.experiment_id], max_results=5000)
        trials_per_family: dict[str, int] = {}
        for run in runs:
            if run.data.tags.get("kind") == "trial":
                family = run.data.params["model"]
                trials_per_family[family] = trials_per_family.get(family, 0) + 1
        for run in runs:
            params, tags = run.data.params, run.data.tags
            if name == "champion":
                if "champion_cell" not in params:
                    continue
                text = describe_champion(params["champion_cell"], params["calibration"],
                                         float(params["cost_ratio"]), params["shortlist"])
            else:
                text = describe_run(
                    stage=tags["stage"], kind=tags["kind"], variant=params["feature_variant"],
                    family=params["model"], trial=int(params["trial"]), params=params,
                    pruned=tags.get("pruned") == "true", n_trials=trials_per_family.get(params["model"]))
            client.set_tag(run.info.run_id, NOTE_TAG, text)
            updated += 1
    return updated


if __name__ == "__main__":
    print(f"descriptions written on {backfill()} runs")
