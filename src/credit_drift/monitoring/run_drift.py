"""Weekly drift run: validate each batch, score it with the champion, measure drift against two baselines.

  uv run python -m credit_drift.monitoring.run_drift            # all 13 weeks
  uv run python -m credit_drift.monitoring.run_drift --weeks 1 5 8

Two baselines: the original Reference (informational: how far the world has moved overall) and the champion's
training window (drives the alerts; it re-points whenever a new model is promoted). For the first champion they are
the same data, so the second is reused from the first. A batch that fails the contract is not scored and gets no
drift report, exactly as in the production loop. The model SCORE is monitored like a feature.
"""
import argparse
import os
import sys
import warnings

import mlflow
import pandas as pd

from credit_drift.contracts.pandera_gate import build_scoring_schema, validate_scoring_batch
from credit_drift.contracts.profile import load_profile
from credit_drift.data import TARGET
from credit_drift.monitoring.features import load_config, monitored_columns
from credit_drift.monitoring.report import REFERENCE_SAMPLE, build_report
from credit_drift.monitoring.stats import drift_table, numeric_drift, summarize
from credit_drift.split import DATA_DIR, PRODUCTION_DIR, REFERENCE_DIR

OUT_DIR = DATA_DIR / "monitoring"
REPORTS_DIR = DATA_DIR.parent / "reports" / "drift"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
BASELINES = ("reference", "champion_window")


def score_row(base_scores: pd.Series, scores: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{"feature": "SCORE", "kind": "numeric", "monitored": True,
                          **numeric_drift(base_scores, scores)}])


def main(weeks: list[int]) -> None:
    warnings.filterwarnings("ignore")
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model("models:/credit_scoring@champion")

    profile = load_profile()
    engineer = model.pipeline.named_steps["prep"].named_steps["features"]  # the champion's own input frame
    schema = build_scoring_schema(profile)

    reference = engineer.transform(pd.read_parquet(REFERENCE_DIR / "train.parquet").drop(columns=TARGET))
    kinds = {c: ("categorical" if reference[c].dtype == object else "numeric") for c in reference.columns}
    monitored = [c for c in monitored_columns(load_config()) if c in kinds]
    champion_window = reference  # v1 was trained on the whole Reference train part; later versions re-point this
    held_out = pd.read_parquet(REFERENCE_DIR / "test.parquet")  # never trained on: honest score distribution
    baseline_scores = pd.Series(model.predict_proba(held_out.drop(columns=TARGET))[:, 1])
    baselines = {"reference": reference, "champion_window": champion_window}
    # HTML baseline: a sample of the champion's training window, with SCORE from the held-out score sample.
    report_columns = [*monitored, "SCORE"]
    report_baseline = champion_window[monitored].sample(REFERENCE_SAMPLE, random_state=0).reset_index(drop=True)
    report_baseline["SCORE"] = baseline_scores.sample(REFERENCE_SAMPLE, random_state=0).to_numpy()

    tables, summaries = [], []
    for week in weeks:
        X = pd.read_parquet(PRODUCTION_DIR / "weeks" / f"week_{week:02d}_features.parquet")
        gate = validate_scoring_batch(X, schema)
        if not gate.passed:
            failed = sorted({f"{f.column}:{f.rule}" for f in gate.blocking})
            print(f"week {week:02d}: BLOCKED by the contract ({len(gate.blocking)} rules): {failed}")
            summaries.append({"week": week, "baseline": "all", "blocked": True, "blocked_rules": "; ".join(failed)})
            continue
        scores = pd.Series(model.predict_proba(X)[:, 1])
        current = engineer.transform(X).assign(SCORE=scores.to_numpy())
        computed: dict[str, pd.DataFrame] = {}
        for name, base in baselines.items():
            if name == "champion_window" and champion_window is reference:
                computed[name] = computed["reference"]
                continue
            computed[name] = pd.concat([drift_table(base, current, kinds, set(monitored)),
                                        score_row(baseline_scores, scores)], ignore_index=True)
        for name, table in computed.items():
            tables.append(table.assign(week=week, baseline=name))
            summaries.append({"week": week, "baseline": name, "blocked": False, "rows": len(X),
                              "mean_score": round(float(scores.mean()), 4), **summarize(table)})
        build_report(current[report_columns], report_baseline, {c: kinds[c] for c in monitored},
                     f"Week {week:02d} vs the champion's training window",
                     REPORTS_DIR / f"week_{week:02d}_vs_champion_window.html")
        top = computed["champion_window"].query("monitored and feature != 'SCORE'").nlargest(3, "psi")
        s = summaries[-1]
        print(f"week {week:02d}: monitored PSI>=0.10: {s['n_psi_moderate']:>2} (>=0.25: {s['n_psi_major']}) | "
              f"max {s['max_psi']:.3f} {s['max_psi_feature']} | score PSI {s['score_psi']:.3f} KS {s['score_ks']:.3f} | "
              f"top: {', '.join(f'{r.feature} {r.psi:.2f}' for r in top.itertuples())}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_parquet(OUT_DIR / "drift_features.parquet", index=False)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "drift_summary.csv", index=False)
    print(f"tables in {OUT_DIR}, HTML reports in {REPORTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weeks", type=int, nargs="*", default=list(range(1, 14)))
    main(parser.parse_args().weeks)
