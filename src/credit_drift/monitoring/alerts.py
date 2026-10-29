"""Alert rules and the week-over-week diagnosis of what kind of drift a week shows.

Every rule reads its threshold from `config/monitoring_thresholds.json`, so tuning never touches code. The diagnosis
follows the decision table in CLAUDE.md: which signals moved THIS week versus LAST week, not versus the Reference
(which stays "drifted" for ever after week 5).

  features changed | score changed | performance dropped | likely cause
  yes              | any           | any                 | covariate shift
  no               | no            | yes                 | concept drift
  no               | yes           | calibration off     | label (prior) shift
  contract failed  |               |                     | technical / schema issue
"""
import json
from pathlib import Path

import pandas as pd

THRESHOLDS_PATH = Path(__file__).resolve().parents[3] / "config" / "monitoring_thresholds.json"


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _alert(rule: str, severity: str, about_week: int, detail: str) -> dict:
    return {"rule": rule, "severity": severity, "about_week": about_week, "detail": detail}


def batch_alerts(week: int, gate: dict, drift: dict | None, t: dict) -> list[dict]:
    """Alerts that need no labels: the contract, the input distributions and the score."""
    if not gate["passed"]:
        rules = sorted({f"{f['column']}:{f['rule']}" for f in gate["blocking"]})
        return [_alert("batch_blocked", "critical", week, f"contract rejected the batch: {', '.join(rules)}")]
    out = []
    if drift:
        if drift["n_psi_major"] >= 1:
            out.append(_alert("data_drift_major", "high", week,
                              f"{drift['n_psi_major']} monitored feature(s) at PSI >= {t['data']['psi_major']}, "
                              f"worst {drift['max_psi_feature']} {drift['max_psi']:.2f}"))
        elif drift["n_psi_moderate"] >= t["data"]["moderate_features_for_alert"]:
            out.append(_alert("data_drift_moderate", "medium", week,
                              f"{drift['n_psi_moderate']} monitored features at PSI >= {t['data']['psi_moderate']}"))
        if drift["score_psi"] >= t["score"]["psi"] or drift["score_ks"] >= t["score"]["ks"]:
            out.append(_alert("score_drift", "medium", week,
                              f"score PSI {drift['score_psi']:.3f}, KS {drift['score_ks']:.3f}"))
    return out


def performance_alerts(perf: dict, baseline: dict, flagged_slices: list[str], t: dict) -> list[dict]:
    """Alerts that need labels; they describe the week whose labels just arrived."""
    week = perf["week"]
    out = []
    drop = baseline["overall"]["auc_pr"] - perf["auc_pr"]
    if drop >= t["performance"]["auc_pr_drop"]:
        out.append(_alert("performance_drop", "high", week,
                          f"AUC-PR {perf['auc_pr']:.3f} is {drop:.3f} below the champion's baseline "
                          f"{baseline['overall']['auc_pr']:.3f}"))
    gap = abs(perf["mean_pred"] - perf["observed_rate"]) / max(perf["observed_rate"], 1e-9)
    if gap >= t["performance"]["calibration_gap_relative"]:
        out.append(_alert("calibration_gap", "medium", week,
                          f"mean prediction {perf['mean_pred']:.3f} vs observed rate {perf['observed_rate']:.3f}"))
    for name in flagged_slices:
        out.append(_alert("slice_degradation", "medium", week, f"slice '{name}' fell well below its baseline"))
    return out


def flagged_slices(current: dict, baseline: dict, overall_drop: float, t: dict) -> list[str]:
    """Slices whose AUC-PR dropped by clearly more than the overall drop (the slice scan; noise-guarded by the
    minimum of defaulters inside `slice_metrics`)."""
    flagged = []
    for name, now in current.items():
        base = baseline.get(name, {}).get("auc_pr")
        if base is None or pd.isna(now["auc_pr"]) or pd.isna(base):
            continue
        if (base - now["auc_pr"]) >= overall_drop + t["slices"]["extra_drop"]:
            flagged.append(name)
    return flagged


def features_changed(previous: pd.DataFrame | None, current: pd.DataFrame, t: dict) -> bool:
    """Did any monitored feature move this week versus last week? (KS rise for numbers, JS rise for categories.)"""
    if previous is None:
        return False
    cols = ["feature", "kind", "ks_stat", "js"]
    prev = previous[cols].set_index("feature")
    cur = current[cols].set_index("feature")
    joined = cur.join(prev, rsuffix="_prev", how="inner")
    ks_rise = (joined["ks_stat"] - joined["ks_stat_prev"]).where(joined["kind"] == "numeric")
    js_rise = (joined["js"] - joined["js_prev"]).where(joined["kind"] == "categorical")
    return bool((ks_rise >= t["diagnosis"]["feature_ks_rise"]).any() or (js_rise >= t["diagnosis"]["feature_js_rise"]).any())


def score_changed(previous: dict | None, current: dict, t: dict) -> bool:
    if previous is None:
        return False
    return bool((current["score_ks"] - previous["score_ks"]) >= t["diagnosis"]["score_ks_rise"]
                or abs(current["mean_score"] - previous["mean_score"]) >= t["diagnosis"]["mean_score_move"])


def diagnose(*, blocked: bool, features_changed: bool, score_changed: bool, performance_dropped: bool | None,
             calibration_off: bool | None) -> str:
    """Decision order: technical, covariate, concept, label shift. `None` means labels are not available."""
    if blocked:
        return "technical"
    if features_changed:
        return "covariate_shift"
    if performance_dropped and not score_changed:
        return "concept_drift"
    if score_changed and calibration_off:
        return "label_shift"
    return "none"
