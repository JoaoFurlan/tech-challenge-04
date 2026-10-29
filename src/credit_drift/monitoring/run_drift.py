"""Standalone drift run over the simulated weeks, without Airflow: gate, score locally, measure, report.

  uv run python -m credit_drift.monitoring.run_drift            # all 13 weeks
  uv run python -m credit_drift.monitoring.run_drift --weeks 1 5 8

The Airflow DAG `production_monitoring` does the same per week through the model server and adds labels, alerts and
metrics; this tool is the quick way to regenerate the drift tables and reports on a laptop.
Two baselines: the original Reference (informational) and the champion's training window (drives alerts; it re-points
on every promotion). A batch that fails the contract is not scored and gets no report, exactly as in production.
"""
import argparse
import os
import sys
import warnings

import mlflow
import pandas as pd

from credit_drift.contracts.pandera_gate import build_scoring_schema, validate_scoring_batch
from credit_drift.contracts.profile import load_profile
from credit_drift.monitoring.context import analyze, build_context
from credit_drift.split import DATA_DIR, PRODUCTION_DIR

OUT_DIR = DATA_DIR / "monitoring"
REPORTS_DIR = DATA_DIR.parent / "reports" / "drift"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def main(weeks: list[int]) -> None:
    warnings.filterwarnings("ignore")
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model("models:/credit_scoring@champion")
    ctx = build_context(model)
    schema = build_scoring_schema(load_profile())

    tables, summaries = [], []
    for week in weeks:
        X = pd.read_parquet(PRODUCTION_DIR / "weeks" / f"week_{week:02d}_features.parquet")
        gate = validate_scoring_batch(X, schema)
        if not gate.passed:
            failed = sorted({f"{f.column}:{f.rule}" for f in gate.blocking})
            print(f"week {week:02d}: BLOCKED by the contract ({len(gate.blocking)} rules): {failed}")
            summaries.append({"week": week, "baseline": "all", "blocked": True, "blocked_rules": "; ".join(failed)})
            continue
        computed, summary = analyze(ctx, X, model.predict_proba(X)[:, 1], week,
                                    REPORTS_DIR / f"week_{week:02d}_vs_champion_window.html")
        for name, table in computed.items():
            tables.append(table.assign(week=week, baseline=name))
            summaries.append({**summary[name], "blocked": False})
        top = computed["champion_window"].query("monitored and feature != 'SCORE'").nlargest(3, "psi")
        s = summary["champion_window"]
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
