"""Cut the Production pool into weekly batches and inject the scheduled drift (docs/decisions.md, #10).

Run: uv run python -m credit_drift.drift.simulate

Per week the simulator picks 5,000 fresh rows (non-overlapping), then, in this order: redraws labels for the
concept-drift segment (using the ORIGINAL features), applies the numeric covariate shift in place, and, on the
corrupted week, damages the batch. Features and labels are written to separate files because in production labels
arrive later. `drift_spec.json` records what was injected: it is ground truth for offline evaluation only.
"""
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from credit_drift.contracts.corruption import corrupt_batch
from credit_drift.data import DAYS_EMPLOYED_SENTINEL, ID_COL, TARGET
from credit_drift.drift import schedule as sch
from credit_drift.drift.label_model import LabelModel, fit_label_model
from credit_drift.split import PRODUCTION_DIR, REFERENCE_DIR

WEEKS_DIR = PRODUCTION_DIR / "weeks"
SPEC_PATH = PRODUCTION_DIR / "drift_spec.json"
DOCS_SPEC_PATH = PRODUCTION_DIR.parents[1] / "docs" / "reports" / "drift_spec.json"


@dataclass
class WeekBatch:
    state: sch.WeekState
    features: pd.DataFrame
    labels: pd.DataFrame | None  # None when the batch never gets labels (the corrupted week)
    info: dict


def target_counts(base_share: pd.Series, level: float, n: int) -> pd.Series:
    """Rows per category for a week: the pool's original shares tilted by the mix weights, rounded to sum to n."""
    log_weight = base_share.index.to_series().map(sch.MIX_LOG_WEIGHTS).fillna(0.0)
    share = base_share * np.exp(level * log_weight)
    share = share / share.sum()
    counts = np.floor(share * n).astype(int)
    leftover = n - counts.sum()
    counts[(share * n - counts).sort_values(ascending=False).index[:leftover]] += 1
    return counts


def draw_rows(remaining: np.ndarray, categories: pd.Series, base_share: pd.Series, level: float, n: int,
              rng: np.random.Generator) -> np.ndarray:
    """Pick n of the still-unused rows. At level 0 it is the next slice of the shuffled pool. Above 0 each category
    contributes its exact target count, drawn uniformly without replacement from what is left, so the mix is the same
    every week no matter how much of the pool earlier weeks used (level-based drift)."""
    if level == 0:
        return remaining[:n]
    category_of = categories.iloc[remaining].to_numpy()
    picked = []
    for category, count in target_counts(base_share, level, n).items():
        candidates = remaining[category_of == category]
        picked.append(rng.choice(candidates, size=min(count, len(candidates)), replace=False))
    picked = np.concatenate(picked)
    if len(picked) < n:  # a category ran out: top up from the rest
        rest = np.setdiff1d(remaining, picked, assume_unique=True)
        picked = np.concatenate([picked, rng.choice(rest, size=n - len(picked), replace=False)])
    return np.sort(picked)


def apply_numeric_state(X: pd.DataFrame, state: sch.WeekState) -> pd.DataFrame:
    out = X.copy()
    out["AMT_INCOME_TOTAL"] = out["AMT_INCOME_TOTAL"] * state.income_mult
    out["AMT_ANNUITY"] = out["AMT_ANNUITY"] * state.annuity_mult
    employed = out["DAYS_EMPLOYED"] != DAYS_EMPLOYED_SENTINEL  # the sentinel is a code, not a duration
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].astype(float)
    out.loc[employed, "DAYS_EMPLOYED"] = out.loc[employed, "DAYS_EMPLOYED"] * state.employment_mult
    return out


def segment_mask(X: pd.DataFrame, income_cutoff: float) -> np.ndarray:
    age = -X["DAYS_BIRTH"] / 365.25
    return ((age <= sch.SEGMENT_MAX_AGE_YEARS) & (X["AMT_INCOME_TOTAL"] <= income_cutoff)).to_numpy()


def redraw_segment_labels(y: np.ndarray, p: np.ndarray, ext2: np.ndarray, segment: np.ndarray, b2: float,
                          rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Redraw the segment's labels from the label model with EXT_SOURCE_2's effect reversed and halved.

    The intercept is shifted so the segment keeps the default rate the model implied before the change
    (concept drift, not label shift). Labels outside the segment stay real."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)) + (sch.EXT_REVERSAL_FACTOR * b2 - b2) * ext2
    target_rate = p[segment].mean()
    low, high = -6.0, 6.0
    for _ in range(40):
        shift = (low + high) / 2
        rate = (1 / (1 + np.exp(-(logit[segment] + shift)))).mean()
        low, high = (shift, high) if rate < target_rate else (low, shift)
    p_new = 1 / (1 + np.exp(-(logit + shift)))
    out = y.copy()
    out[segment] = (rng.random(int(segment.sum())) < p_new[segment]).astype(int)
    return out, {
        "segment_rows": int(segment.sum()),
        "segment_default_rate_real": round(float(y[segment].mean()), 4),
        "segment_default_rate_redrawn": round(float(out[segment].mean()), 4),
        "labels_changed": int((out != y).sum()),
    }


def simulate(pool: pd.DataFrame, label_model: LabelModel, income_cutoff: float,
             states: tuple[sch.WeekState, ...] = sch.SCHEDULE, week_rows: int = sch.WEEK_ROWS,
             seed: int = sch.SEED) -> list[WeekBatch]:
    shuffled = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    remaining = np.arange(len(shuffled))
    base_share = shuffled[sch.MIX_COLUMN].value_counts(normalize=True)
    batches = []
    for state in states:  # sequential: week N depends only on earlier weeks, never on later ones
        rng = np.random.default_rng(seed + state.week)
        picked = draw_rows(remaining, shuffled[sch.MIX_COLUMN], base_share, state.mix_level, week_rows, rng)
        remaining = np.setdiff1d(remaining, picked, assume_unique=True)
        rows = shuffled.iloc[picked].reset_index(drop=True)
        y, X = rows[TARGET].to_numpy(), rows.drop(columns=TARGET)
        info: dict = {"rows": len(rows)}

        if state.concept_drift:
            y, segment_info = redraw_segment_labels(
                y, label_model.predict_proba(X), label_model.ext2_standardised(X),
                segment_mask(X, income_cutoff), label_model.b2, np.random.default_rng(seed + 1000 + state.week))
            info.update(segment_info)
        X = apply_numeric_state(X, state)

        income_shares = X[sch.MIX_COLUMN].value_counts(normalize=True)
        info.update(
            default_rate=round(float(y.mean()), 4),
            working_share=round(float(income_shares.get("Working", 0)), 4),
            pensioner_share=round(float(income_shares.get("Pensioner", 0)), 4),
            mean_income=round(float(X["AMT_INCOME_TOTAL"].mean()), 1),
            mean_annuity=round(float(X["AMT_ANNUITY"].mean()), 1),
            labels_available=not state.corrupted,
        )
        labels = pd.DataFrame({ID_COL: X[ID_COL].to_numpy(), TARGET: y})
        if state.corrupted:
            X, info["corruption"] = corrupt_batch(X, seed=seed + state.week)
            labels = None
        batches.append(WeekBatch(state, X, labels, info))
    return batches


def build_spec(batches: list[WeekBatch], label_model: LabelModel, income_cutoff: float, pool_rows: int, seed: int) -> dict:
    return {
        "note": "Ground truth of the simulation, for offline evaluation only: production has no such file.",
        "seed": seed, "week_rows": sch.WEEK_ROWS, "pool_rows": pool_rows,
        "rows_unused": pool_rows - sum(b.info["rows"] for b in batches),
        "level_based": "each week is an absolute state applied to fresh rows; weeks never stack on each other",
        "numeric_drift": "values changed in place, labels untouched (measurement/level drift)",
        "mix_drift": {"column": sch.MIX_COLUMN, "log_weights_at_full_strength": sch.MIX_LOG_WEIGHTS,
                      "mechanism": "each category gets its exact target count, drawn from unused rows, so each row keeps its own label"},
        "concept_drift": {
            "segment": f"age <= {sch.SEGMENT_MAX_AGE_YEARS} and income <= {income_cutoff:.0f} (training median), "
                       "defined on the original features",
            "label_model": "independent LightGBM, Platt-calibrated on the Reference calibration part",
            "ext_source_2_coefficient_before": round(label_model.b2, 4),
            "ext_source_2_coefficient_after": round(sch.EXT_REVERSAL_FACTOR * label_model.b2, 4),
            "intercept": "shifted each week so the segment keeps its model-implied default rate",
            "labels_outside_segment": "real and untouched",
        },
        "weeks": [{**asdict(b.state), **b.info} for b in batches],
    }


def write(batches: list[WeekBatch], spec: dict) -> None:
    WEEKS_DIR.mkdir(parents=True, exist_ok=True)
    for b in batches:
        stem = WEEKS_DIR / f"week_{b.state.week:02d}"
        b.features.to_parquet(f"{stem}_features.parquet", index=False)
        label_file = f"{stem}_labels.parquet"
        if b.labels is not None:
            b.labels.to_parquet(label_file, index=False)
        elif WEEKS_DIR.joinpath(f"week_{b.state.week:02d}_labels.parquet").exists():
            WEEKS_DIR.joinpath(f"week_{b.state.week:02d}_labels.parquet").unlink()
    text = json.dumps(spec, indent=1)
    SPEC_PATH.write_text(text, encoding="utf-8")
    DOCS_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_SPEC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    pool = pd.read_parquet(PRODUCTION_DIR / "pool.parquet")
    train = pd.read_parquet(REFERENCE_DIR / "train.parquet")
    calibration = pd.read_parquet(REFERENCE_DIR / "calibration.parquet")
    income_cutoff = float(train["AMT_INCOME_TOTAL"].median())
    label_model = fit_label_model(train, calibration)
    batches = simulate(pool, label_model, income_cutoff)
    spec = build_spec(batches, label_model, income_cutoff, len(pool), sch.SEED)
    write(batches, spec)
    table = pd.DataFrame([{"week": b.state.week, "note": b.state.note, **{
        k: b.info.get(k) for k in ("rows", "default_rate", "pensioner_share", "mean_income", "mean_annuity",
                                   "labels_changed", "labels_available")}} for b in batches])
    print(table.to_string(index=False))
    print(f"files in {WEEKS_DIR}, spec in {SPEC_PATH} (copy in {DOCS_SPEC_PATH})")


if __name__ == "__main__":
    main()
