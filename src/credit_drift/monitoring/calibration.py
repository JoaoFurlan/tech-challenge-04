"""Offline signals for calibrating the detector thresholds (step 14).

Simulates the 13 weeks for a given seed IN MEMORY (nothing in the official `data/production/weeks` changes), scores
them with the champion, and keeps every raw signal the alert rules read: per-feature PSI/KS/JS, score drift, overall
and per-slice performance, per week. The thresholds are then replayed over these signals without recomputing anything.

  python -m credit_drift.monitoring.calibration --seed 101

Only summary numbers are written (docs/calibration/seed_<seed>.json): no customer rows. The ground truth of what was
injected comes from the schedule and is used only offline; production has no such thing (docs/decisions.md, #16).
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_drift.data import TARGET
from credit_drift.drift import schedule as sch
from credit_drift.drift.label_model import fit_label_model
from credit_drift.drift.simulate import simulate
from credit_drift.monitoring.context import analyze, build_context
from credit_drift.monitoring.performance import (
    compute_baseline,
    overall_metrics,
    pooled_slice_performance,
)
from credit_drift.split import PRODUCTION_DIR, REFERENCE_DIR

OUT_DIR = Path(__file__).resolve().parents[3] / "docs" / "calibration"
KEEP = ["feature", "kind", "psi", "ks_stat", "js"]


def _round(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame[KEEP].round(5).to_json(orient="records"))


def collect(seed: int) -> dict:
    from credit_drift.serving.registry import load_champion

    started = time.perf_counter()
    pool = pd.read_parquet(PRODUCTION_DIR / "pool.parquet")
    train = pd.read_parquet(REFERENCE_DIR / "train.parquet")
    calibration = pd.read_parquet(REFERENCE_DIR / "calibration.parquet")
    income_cutoff = float(train["AMT_INCOME_TOTAL"].median())
    batches = simulate(pool, fit_label_model(train, calibration), income_cutoff, seed=seed)

    loaded = load_champion()
    ctx = build_context(loaded.model)
    baseline = compute_baseline(loaded.model, train, pd.read_parquet(REFERENCE_DIR / "test.parquet"),
                                loaded.threshold, loaded.version)
    labelled: dict[int, pd.DataFrame] = {}
    weeks = []
    for b in batches:
        week = b.state.week
        entry = {"week": week, "truth": {k: v for k, v in asdict(b.state).items() if k != "week"},
                 "default_rate": b.info["default_rate"]}
        if b.state.corrupted:  # the real loop never scores a rejected batch, nor receives its labels
            weeks.append(entry)
            continue
        p = loaded.model.predict_proba(b.features)[:, 1]
        tables, summaries = analyze(ctx, b.features, p, week)
        table = tables["champion_window"]
        monitored = table[table["monitored"] | (table["feature"] == "SCORE")]
        entry.update(drift=_round(monitored), mean_score=float(p.mean()), score_summary=summaries["champion_window"])
        frame = b.features.assign(**{TARGET: b.labels[TARGET].to_numpy(), "probability": p})
        labelled[week] = frame
        entry["performance"] = overall_metrics(frame[TARGET].to_numpy(), p, loaded.threshold)
        window = [w for w in (week - 1, week) if w in labelled]  # the slices pool two weeks, as in production
        slices, pooled = pooled_slice_performance([labelled[w] for w in window], baseline["slice_cuts"])
        both = pd.concat([labelled[w] for w in window])
        pooled["auc_roc"] = float(roc_auc_score(both[TARGET], both["probability"]))
        entry["window"] = {"weeks": window, "overall": pooled, "slices": slices}
        weeks.append(entry)
        print(f"seed {seed} week {week:02d}: AUC-PR {entry['performance']['auc_pr']:.3f} | "
              f"max PSI {summaries['champion_window']['max_psi']:.2f} | {time.perf_counter() - started:.0f}s", flush=True)
    return {"seed": seed, "model_version": loaded.version, "decision_threshold": loaded.threshold,
            "baseline": baseline, "weeks": weeks}


def load(seed: int) -> dict:
    return json.loads((OUT_DIR / f"seed_{seed}.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed == sch.SEED:
        raise SystemExit(f"seed {sch.SEED} is the official run: never calibrate on it")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = collect(args.seed)
    path = OUT_DIR / f"seed_{args.seed}.json"
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
