"""Causal reading of the reference run (README, "Causalidade"): what was injected, what the detectors named, and how much of
the performance drop the shifted inputs explain.

  python -m credit_drift.monitoring.causal --artifacts <copy of artifacts/monitoring>   # writes docs/calibration/*.json

Two offline analyses (the ground truth in `drift_spec.json` does not exist in production):

1. Injected vs detected. The simulator moves AMT_INCOME_TOTAL, DAYS_EMPLOYED, AMT_ANNUITY and the NAME_INCOME_TYPE mix. The
   model works on engineered features, so four ratios built from those columns move too (derived). Resampling the mix
   also drags correlated columns with it (collateral: nobody injected them, but they follow). Anything else the
   detector flags would be a false positive.
2. Importance reweighting. Reweight the Reference test part so its features look like production's; the champion's AUC-PR
   on the reweighted Reference is what the shifted inputs alone would give. The gap left between that and the observed
   production AUC-PR is not explained by the inputs: that is concept drift.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from credit_drift.data import TARGET
from credit_drift.drift import schedule as sch
from credit_drift.split import PRODUCTION_DIR, REFERENCE_DIR

OUT_DIR = Path(__file__).resolve().parents[3] / "docs" / "calibration"
DIRECT = {"AMT_INCOME_TOTAL", "DAYS_EMPLOYED", "AMT_ANNUITY", "NAME_INCOME_TYPE"}
DERIVED = {"ANNUITY_INCOME", "CREDIT_INCOME", "ANNUITY_CREDIT", "EMPLOYED_AGE"}
COLLATERAL = {"ORGANIZATION_TYPE", "OCCUPATION_TYPE", "DAYS_BIRTH", "DAYS_REGISTRATION", "FLAG_EMP_PHONE", "DAYS_ID_PUBLISH"}
PSI_FLAG = 0.10
COVARIATE_REGIME = (5, 6, 7)  # the inputs sit at the full regime, no concept drift yet
CONCEPT_WEEKS = (8, 10, 11, 12, 13, 14, 15, 16)


def summarize_reference_run(artifacts: Path) -> dict:
    """Aggregates of the production loop: per week the alerts, diagnosis, performance and the per-feature drift table."""
    weeks = []
    for folder in sorted(artifacts.glob("week_*")):
        week = int(folder.name.split("_")[1])
        entry = {"week": week}
        for name in ("alerts", "diagnosis", "performance", "gate"):
            path = folder / f"{name}.json"
            if path.exists():
                entry[name] = json.loads(path.read_text(encoding="utf-8"))
        table = folder / "drift_features.parquet"
        if table.exists():
            frame = pd.read_parquet(table)
            frame = frame[frame["monitored"] & (frame["feature"] != "SCORE")]
            entry["psi"] = {r.feature: round(float(r.psi), 4) for r in frame.itertuples()}
        weeks.append(entry)
    return {"weeks": weeks}


def detection_scorecard(summary: dict, psi_flag: float = PSI_FLAG) -> pd.DataFrame:
    """Per week: detected features (PSI >= flag), how many are direct, derived, collateral or outside, and recall."""
    rows = []
    positives = DIRECT | DERIVED
    for entry in summary["weeks"]:
        if "psi" not in entry:
            continue
        detected = {f for f, psi in entry["psi"].items() if psi >= psi_flag}
        injected_now = entry["week"] >= 3
        rows.append({
            "week": entry["week"], "detected": len(detected),
            "direct": len(detected & DIRECT), "derived": len(detected & DERIVED),
            "collateral": len(detected & COLLATERAL), "false_positives": sorted(detected - positives - COLLATERAL),
            "recall_of_injected": round(len(detected & positives) / len(positives), 2) if injected_now else None,
            "max_psi": max(entry["psi"].values())})
    return pd.DataFrame(rows)


def importance_weights(reference: pd.DataFrame, production: pd.DataFrame, target_rows: pd.DataFrame,
                       clip: float = 20.0) -> np.ndarray:
    """Density-ratio weights w(x) = p(production | x) / p(reference | x) for the rows of `target_rows`, from a logistic
    discriminator on the engineered numeric features (the shifts here are level shifts) plus the income-type mix."""
    numeric = [c for c in reference.columns if reference[c].dtype != object]
    def design(frame: pd.DataFrame) -> pd.DataFrame:
        part = frame[numeric].astype(float)
        dummies = pd.get_dummies(frame["NAME_INCOME_TYPE"].fillna("missing"), dtype=float)
        return pd.concat([part, dummies], axis=1)
    both = pd.concat([design(reference), design(production)], axis=0, ignore_index=True).fillna(0.0)
    X = both.to_numpy()
    scaler = StandardScaler().fit(X)
    y = np.r_[np.zeros(len(reference)), np.ones(len(production))]
    model = LogisticRegression(max_iter=500, C=1.0).fit(scaler.transform(X), y)
    target = design(target_rows).reindex(columns=both.columns, fill_value=0.0).fillna(0.0)
    p = model.predict_proba(scaler.transform(target.to_numpy()))[:, 1]
    weights = p / (1 - p) * (len(reference) / len(production))
    return np.clip(weights, 0, clip)


def _weighted_ap(y: np.ndarray, p: np.ndarray, w: np.ndarray | None) -> float:
    return float(average_precision_score(y, p, sample_weight=w))


def counterfactual(n_boot: int = 200, seed: int = 0) -> dict:
    """Split the AUC-PR drop of the covariate regime and of the concept-drift weeks into what the inputs explain and what
    they do not. Scores with the registered champion (v1)."""
    from credit_drift.serving.registry import load_champion

    champion = load_champion()
    engineer = champion.model.pipeline.named_steps["prep"].named_steps["features"]
    ref_train = pd.read_parquet(REFERENCE_DIR / "train.parquet").drop(columns=TARGET)
    ref_test = pd.read_parquet(REFERENCE_DIR / "test.parquet")
    y_ref = ref_test[TARGET].to_numpy()
    p_ref = champion.model.predict_proba(ref_test.drop(columns=TARGET))[:, 1]
    eng_train = engineer.transform(ref_train).sample(30_000, random_state=seed)
    eng_test = engineer.transform(ref_test.drop(columns=TARGET))

    production = {}
    for week in (*COVARIATE_REGIME, *CONCEPT_WEEKS):
        folder = PRODUCTION_DIR / "weeks"
        features = pd.read_parquet(folder / f"week_{week:02d}_features.parquet")
        labels = pd.read_parquet(folder / f"week_{week:02d}_labels.parquet")
        merged = features.merge(labels, on="SK_ID_CURR")
        production[week] = (engineer.transform(merged.drop(columns=TARGET)), merged[TARGET].to_numpy(),
                            champion.model.predict_proba(merged.drop(columns=TARGET))[:, 1])

    regime = pd.concat([production[w][0] for w in COVARIATE_REGIME], ignore_index=True)
    weights = importance_weights(eng_train, regime, eng_test)
    ess = float(weights.sum() ** 2 / (weights ** 2).sum())  # effective sample size of the reweighted Reference
    rng = np.random.default_rng(seed)
    pooled = {name: (np.concatenate([production[w][1] for w in weeks]), np.concatenate([production[w][2] for w in weeks]))
              for name, weeks in (("covariate_regime", COVARIATE_REGIME), ("concept_weeks", CONCEPT_WEEKS))}

    def metrics(i_ref: np.ndarray, i_prod: dict) -> dict:
        base = _weighted_ap(y_ref[i_ref], p_ref[i_ref], None)
        reweighted = _weighted_ap(y_ref[i_ref], p_ref[i_ref], weights[i_ref])
        out = {"baseline": base, "reweighted_reference": reweighted}
        for name, (y, p) in pooled.items():
            out[f"production_{name}"] = _weighted_ap(y[i_prod[name]], p[i_prod[name]], None)
        out["covariate_effect"] = out["baseline"] - out["reweighted_reference"]
        out["unexplained_covariate_regime"] = out["reweighted_reference"] - out["production_covariate_regime"]
        out["unexplained_concept_weeks"] = out["reweighted_reference"] - out["production_concept_weeks"]
        out["total_drop_concept_weeks"] = out["baseline"] - out["production_concept_weeks"]
        return out

    point = metrics(np.arange(len(y_ref)), {k: np.arange(len(v[0])) for k, v in pooled.items()})
    draws = []
    for _ in range(n_boot):
        draws.append(metrics(rng.integers(0, len(y_ref), len(y_ref)),
                             {k: rng.integers(0, len(v[0]), len(v[0])) for k, v in pooled.items()}))
    interval = {k: [float(np.quantile([d[k] for d in draws], q)) for q in (0.025, 0.975)] for k in point}
    return {"champion_version": champion.version, "effective_sample_size": ess, "reference_test_rows": len(y_ref),
            "weeks": {"covariate_regime": list(COVARIATE_REGIME), "concept_weeks": list(CONCEPT_WEEKS)},
            "point": point, "ci95": interval, "n_bootstrap": n_boot}


def injection_table() -> list[dict]:
    """What the schedule injected each week, for the README (from the code, same source as drift_spec.json)."""
    return [{"week": s.week, "income": s.income_mult, "employment": s.employment_mult, "annuity": s.annuity_mult,
             "mix_level": s.mix_level, "concept_drift": s.concept_drift, "corrupted": s.corrupted} for s in sch.SCHEDULE]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, required=True, help="a copy of artifacts/monitoring (json and drift tables)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize_reference_run(args.artifacts)
    (OUT_DIR / "reference_run.json").write_text(json.dumps(summary), encoding="utf-8")
    result = counterfactual()
    (OUT_DIR / "counterfactual.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: round(v, 4) for k, v in result["point"].items()}, indent=1), "\nESS", round(result["effective_sample_size"]))


if __name__ == "__main__":
    main()
