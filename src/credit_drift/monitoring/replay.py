"""Replay the alert rules and the diagnosis over saved weekly signals (see calibration.py), and score them against what
was injected. Offline only: the production loop reads the same rules from `alerts.py` and has no ground truth.
"""
import pandas as pd

from credit_drift.monitoring import alerts as rules
from credit_drift.monitoring.performance import rolling_auc_pr

# What was injected each week, as the detectors should see it (docs/decisions.md, #10, and the schedule).
DIAGNOSIS_TRUTH = {1: "none", 2: "none", 3: "covariate_shift", 4: "covariate_shift", 5: "covariate_shift",
                   6: "none", 7: "none", 8: "concept_drift", 9: "technical", 10: "concept_drift",
                   11: "concept_drift", 12: "concept_drift", 13: "concept_drift", 14: "concept_drift",
                   15: "concept_drift", 16: "concept_drift"}
CONTROL_WEEKS = (1, 2)
SHIFT_WEEKS = (3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16)  # inputs differ from the champion's training window
CONCEPT_WEEKS = (8, 10, 11, 12, 13, 14, 15, 16)  # P(y|x) changed in a segment (week 9's labels never arrive)


def _table(week: dict) -> pd.DataFrame | None:
    if "drift" not in week:
        return None
    table = pd.DataFrame(week["drift"])
    table["monitored"] = True
    return table[table["feature"] != "SCORE"]


def _drift_summary(week: dict, t: dict) -> dict:
    """The summary the production step writes to drift.json, recomputed so PSI bands can change."""
    table = _table(week)
    score = next(r for r in week["drift"] if r["feature"] == "SCORE")
    worst = table.loc[table["psi"].idxmax()]
    return {"n_psi_moderate": int((table["psi"] >= t["data"]["psi_moderate"]).sum()),
            "n_psi_major": int((table["psi"] >= t["data"]["psi_major"]).sum()),
            "max_psi": float(worst["psi"]), "max_psi_feature": worst["feature"],
            "score_psi": score["psi"], "score_ks": score["ks_stat"], "mean_score": week["mean_score"]}


def replay(signals: dict, t: dict) -> list[dict]:
    """One record per week: the rules that fired and the diagnosis, as the production loop would have produced."""
    baseline, weeks = signals["baseline"], {w["week"]: w for w in signals["weeks"]}
    out = []
    for number in sorted(weeks):
        week = weeks[number]
        record = {"week": number, "alerts": set(), "diagnosis": None}
        if "drift" not in week:  # the contract rejected the batch
            record["alerts"].add("batch_blocked")
            record["diagnosis"] = "technical"
            out.append(record)
            continue
        summary = _drift_summary(week, t)
        record["alerts"] |= {a["rule"] for a in rules.batch_alerts(number, {"passed": True}, summary, t)}
        history = [{"week": w, **weeks[w]["performance"]} for w in sorted(weeks)
                   if w <= number and "performance" in weeks[w]][-t["performance"].get("window_weeks", 1):]
        perf = {**week["performance"], "trend": rolling_auc_pr(history)}
        dropped = rules.performance_dropped(perf, baseline, t)
        if dropped:
            record["alerts"].add("performance_drop")
        window = week["window"]
        overall_drop = baseline["overall"]["auc_pr"] - window["overall"]["auc_pr"]
        if rules.flagged_slices(window["slices"], baseline["slices"], overall_drop, t):
            record["alerts"].add("slice_degradation")
        if (abs(perf["mean_pred"] - perf["observed_rate"]) / max(perf["observed_rate"], 1e-9)
                >= t["performance"]["calibration_gap_relative"]):
            record["alerts"].add("calibration_gap")
        previous = weeks.get(number - 1)
        prev_table = _table(previous) if previous else None
        features = rules.features_changed(prev_table, _table(week), t)
        prev_summary = {"score_ks": next(r for r in previous["drift"] if r["feature"] == "SCORE")["ks_stat"],
                        "mean_score": previous["mean_score"]} if prev_table is not None else None
        score = rules.score_changed(prev_summary, summary, t)
        gap = abs(perf["mean_pred"] - perf["observed_rate"]) / max(perf["observed_rate"], 1e-9)
        record["diagnosis"] = rules.diagnose(
            blocked=False, features_changed=features, score_changed=score, performance_dropped=dropped,
            calibration_off=gap >= t["performance"]["calibration_gap_relative"])
        out.append(record)
    return out


def scorecard(records: list[dict]) -> dict:
    """What the rules got right, week by week, against the injection."""
    by_week = {r["week"]: r for r in records}
    concept_weeks = tuple(w for w in CONCEPT_WEEKS if w in by_week)  # older signal files hold only 13 weeks
    truth_by_week = {w: t for w, t in DIAGNOSIS_TRUTH.items() if w in by_week}
    data_rules = {"data_drift_moderate", "data_drift_major"}
    first_data = next((w for w in sorted(by_week) if by_week[w]["alerts"] & data_rules), None)
    perf_hits = [w for w in concept_weeks if "performance_drop" in by_week[w]["alerts"]]
    slice_hits = [w for w in concept_weeks if "slice_degradation" in by_week[w]["alerts"]]
    wrong_diagnosis = {w: (by_week[w]["diagnosis"], truth) for w, truth in truth_by_week.items()
                       if by_week[w]["diagnosis"] != truth}
    non_concept = [w for w in by_week if w not in concept_weeks and w != 9]
    return {
        "control_false_alarms": sorted(w for w in CONTROL_WEEKS if by_week[w]["alerts"] - {"batch_blocked"}),
        "first_data_alert_week": first_data,
        "data_alert_weeks_missed": sorted(w for w in SHIFT_WEEKS if w in by_week and not by_week[w]["alerts"] & data_rules and w >= 5),
        "performance_recall": f"{len(perf_hits)}/{len(concept_weeks)}",
        "performance_false_alarms": sorted(w for w in non_concept if "performance_drop" in by_week[w]["alerts"]),
        "slice_recall": f"{len(slice_hits)}/{len(concept_weeks)}",
        "slice_false_alarms": sorted(w for w in non_concept if "slice_degradation" in by_week[w]["alerts"]),
        "diagnosis_correct": f"{len(truth_by_week) - len(wrong_diagnosis)}/{len(truth_by_week)}",
        "diagnosis_wrong": wrong_diagnosis,
    }
