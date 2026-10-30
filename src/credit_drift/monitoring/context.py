"""Everything a drift measurement needs about the champion and its baselines, built once and reused per week."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from credit_drift.data import TARGET
from credit_drift.monitoring.features import load_config, monitored_columns
from credit_drift.monitoring.report import REFERENCE_SAMPLE, build_report
from credit_drift.monitoring.stats import drift_table, numeric_drift, summarize
from credit_drift.split import PRODUCTION_DIR, REFERENCE_DIR


@dataclass
class DriftContext:
    engineer: object  # the champion's own feature step: monitoring sees what the model sees
    kinds: dict[str, str]
    monitored: list[str]
    reference: pd.DataFrame  # the original Reference train part, engineered (informational baseline)
    champion_window: pd.DataFrame  # the champion's training window (drives alerts); re-points on promotion
    baseline_scores: pd.Series  # the champion's scores on held-out data (training scores are overfit)
    report_baseline: pd.DataFrame  # a sample of the window plus SCORE, for the Evidently HTML
    report_columns: list[str]


def build_context(model, window_weeks: list[int] | tuple = ()) -> DriftContext:
    engineer = model.pipeline.named_steps["prep"].named_steps["features"]
    reference = engineer.transform(pd.read_parquet(REFERENCE_DIR / "train.parquet").drop(columns=TARGET))
    kinds = {c: ("categorical" if reference[c].dtype == object else "numeric") for c in reference.columns}
    monitored = [c for c in monitored_columns(load_config()) if c in kinds]
    champion_window = reference  # v1 was trained on the whole Reference train part
    if window_weeks:  # a retrained champion: equal parts Reference and the production weeks it was trained on
        production = pd.concat([engineer.transform(pd.read_parquet(
            PRODUCTION_DIR / "weeks" / f"week_{w:02d}_features.parquet")) for w in window_weeks], ignore_index=True)
        champion_window = pd.concat([reference.sample(len(production), random_state=0), production], ignore_index=True)
    held_out = pd.read_parquet(REFERENCE_DIR / "test.parquet")
    baseline_scores = pd.Series(model.predict_proba(held_out.drop(columns=TARGET))[:, 1])
    report_baseline = champion_window[monitored].sample(REFERENCE_SAMPLE, random_state=0).reset_index(drop=True)
    report_baseline["SCORE"] = baseline_scores.sample(REFERENCE_SAMPLE, random_state=0).to_numpy()
    return DriftContext(engineer, kinds, monitored, reference, champion_window, baseline_scores, report_baseline,
                        [*monitored, "SCORE"])


def analyze(ctx: DriftContext, features: pd.DataFrame, scores: np.ndarray, week: int,
            report_path: Path | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Drift tables and summaries against both baselines, plus the Evidently HTML when a path is given.

    `features` is the raw weekly batch; `scores` are the probabilities the model server returned for it."""
    scores = pd.Series(np.asarray(scores))
    current = ctx.engineer.transform(features).assign(SCORE=scores.to_numpy())
    score_row = pd.DataFrame([{"feature": "SCORE", "kind": "numeric", "monitored": True,
                               **numeric_drift(ctx.baseline_scores, scores)}])
    tables: dict[str, pd.DataFrame] = {}
    for name, base in (("reference", ctx.reference), ("champion_window", ctx.champion_window)):
        if name == "champion_window" and ctx.champion_window is ctx.reference:
            tables[name] = tables["reference"]  # the same data for the first champion: measure once
            continue
        tables[name] = pd.concat([drift_table(base, current, ctx.kinds, set(ctx.monitored)), score_row],
                                 ignore_index=True)
    summaries = {name: {"week": week, "baseline": name, "rows": len(features),
                        "mean_score": round(float(scores.mean()), 4), **summarize(table)}
                 for name, table in tables.items()}
    if report_path is not None:
        build_report(current[ctx.report_columns], ctx.report_baseline, {c: ctx.kinds[c] for c in ctx.monitored},
                     f"Week {week:02d} vs the champion's training window", report_path)
    return tables, summaries
