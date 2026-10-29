"""Slices, label-dependent metrics, alert rules, diagnosis, and the file-driven evaluate/push steps."""
import json

import numpy as np
import pandas as pd
import pytest

from credit_drift.data import DAYS_EMPLOYED_SENTINEL
from credit_drift.monitoring import alerts, stats, weekly
from credit_drift.monitoring.performance import (
    overall_metrics,
    pooled_slice_performance,
    slice_metrics,
)
from credit_drift.monitoring.slices import slice_cuts, slice_masks

T = alerts.load_thresholds()
rng = np.random.default_rng(0)


def _frame(n=4_000, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "EXT_SOURCE_1": r.uniform(0, 1, n), "EXT_SOURCE_2": r.uniform(0, 1, n), "EXT_SOURCE_3": r.uniform(0, 1, n),
        "AMT_INCOME_TOTAL": r.uniform(50_000, 300_000, n), "DAYS_BIRTH": -r.integers(20 * 365, 65 * 365, n),
        "DAYS_EMPLOYED": np.where(r.random(n) < 0.15, DAYS_EMPLOYED_SENTINEL, -r.integers(100, 9_000, n)),
        "CODE_GENDER": r.choice(["F", "M"], n, p=[0.65, 0.35]),
    })


# ---------------------------------------------------------------- slices
def test_there_are_fourteen_slices_and_halves_partition_the_known_rows():
    frame = _frame()
    frame.loc[:99, "EXT_SOURCE_1"] = np.nan
    masks = slice_masks(frame, slice_cuts(frame))
    assert len(masks) == 14
    low, high = masks["EXT_SOURCE_1 low"], masks["EXT_SOURCE_1 high"]
    assert not (low & high).any() and (low | high).sum() == frame["EXT_SOURCE_1"].notna().sum()
    assert not low[:100].any() and not high[:100].any()  # missing values sit in neither half


def test_the_employment_sentinel_is_not_a_duration():
    frame = _frame()
    masks = slice_masks(frame, slice_cuts(frame))
    sentinel = (frame["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).to_numpy()
    assert not (masks["EMPLOYMENT_YEARS low"] & sentinel).any() and not (masks["EMPLOYMENT_YEARS high"] & sentinel).any()


def test_cuts_are_fixed_from_the_training_window_not_the_batch():
    train, batch = _frame(seed=1), _frame(seed=2)
    batch["AMT_INCOME_TOTAL"] = batch["AMT_INCOME_TOTAL"] * 0.5  # the world got poorer
    cuts = slice_cuts(train)
    share_low = slice_masks(batch, cuts)["AMT_INCOME_TOTAL low"].mean()
    assert share_low > 0.9  # a drifted batch lands mostly in the "low income" half: definitions did not move


# ---------------------------------------------------------------- performance metrics
def _scored(n=4_000, signal=1.0, seed=3):
    r = np.random.default_rng(seed)
    p = np.clip(r.beta(1.2, 12, n), 0.001, 0.999)
    y = (r.random(n) < np.clip(p * signal, 0, 1)).astype(int)
    return y, p


def test_overall_metrics_have_everything_the_dashboard_needs():
    y, p = _scored()
    m = overall_metrics(y, p, threshold=1 / 6)
    assert {"auc_pr", "auc_pr_ci_low", "auc_pr_ci_high", "auc_roc", "brier", "mean_pred", "observed_rate",
            "flag_rate", "cost_at_threshold"} <= set(m)
    assert m["auc_pr_ci_low"] <= m["auc_pr"] <= m["auc_pr_ci_high"] and m["rows"] == len(y)


def test_a_slice_with_too_few_defaulters_reports_no_auc():
    y, p = _scored(n=2_000)
    masks = {"big": np.ones(len(y), dtype=bool), "tiny": np.arange(len(y)) < 40}
    result = slice_metrics(y, p, masks)
    assert not np.isnan(result["big"]["auc_pr"]) and np.isnan(result["tiny"]["auc_pr"])


def test_pooled_window_covers_all_slices():
    frames = []
    for seed in (1, 2):
        frame = _frame(seed=seed)
        frame["TARGET"], frame["probability"] = _scored(len(frame), seed=seed)
        frames.append(frame)
    slices, overall = pooled_slice_performance(frames, slice_cuts(frames[0]))
    assert len(slices) == 14 and overall["rows"] == 8_000


# ---------------------------------------------------------------- alert rules
def _drift(**kw):
    return {"n_psi_moderate": 0, "n_psi_major": 0, "max_psi": 0.01, "max_psi_feature": "x", "mean_psi": 0.005, "score_psi": 0.0,
            "score_ks": 0.01, "mean_score": 0.08, **kw}


def test_thresholds_in_the_config_match_the_statistics_module():
    assert T["data"]["psi_moderate"] == stats.PSI_MODERATE and T["data"]["psi_major"] == stats.PSI_MAJOR


def test_a_rejected_batch_raises_one_critical_alert_and_nothing_else():
    gate = {"passed": False, "blocking": [{"column": "AMT_INCOME_TOTAL", "rule": "not_nullable", "count": 600}]}
    out = alerts.batch_alerts(9, gate, None, T)
    assert [(a["rule"], a["severity"]) for a in out] == [("batch_blocked", "critical")]


@pytest.mark.parametrize(("drift", "expected"), [
    (_drift(), []),
    (_drift(n_psi_moderate=1), []),                                        # one moderate feature is not an alert
    (_drift(n_psi_moderate=2), ["data_drift_moderate"]),
    (_drift(n_psi_moderate=3, n_psi_major=1), ["data_drift_major"]),       # major wins over moderate
    (_drift(score_psi=0.12), ["score_drift"]),
    (_drift(score_ks=0.11), ["score_drift"]),
])
def test_batch_alert_rules(drift, expected):
    out = alerts.batch_alerts(5, {"passed": True, "blocking": []}, drift, T)
    assert [a["rule"] for a in out] == expected


def test_performance_alerts_use_the_drop_against_the_champions_baseline():
    baseline = {"overall": {"auc_pr": 0.246}}
    ok = {"week": 4, "auc_pr": 0.230, "mean_pred": 0.083, "observed_rate": 0.082}
    bad = {"week": 8, "auc_pr": 0.190, "mean_pred": 0.100, "observed_rate": 0.080}
    assert alerts.performance_alerts(ok, baseline, [], T) == []
    rules = [a["rule"] for a in alerts.performance_alerts(bad, baseline, ["AGE low"], T)]
    assert rules == ["performance_drop", "calibration_gap", "slice_degradation"]


def test_a_slice_is_flagged_only_when_it_falls_clearly_below_its_baseline():
    baseline = {n: {"auc_pr": 0.30} for n in "ABCDEF"} | {"G": {"auc_pr": None}}
    now = {
        "A": {"auc_pr": 0.20, "ci_high": 0.25},                  # 0.10 below, interval clearly below: flagged
        "B": {"auc_pr": 0.26, "ci_high": 0.30},                  # only 0.04 below (needs 0.04 + 0.03)
        "C": {"auc_pr": float("nan"), "ci_high": float("nan")},  # too few defaulters: never flagged
        "D": {"auc_pr": 0.20, "ci_high": 0.31},                  # a big drop, but the interval still reaches baseline
        "E": {"auc_pr": 0.20, "ci_high": float("nan")},          # no interval: judged on the drop alone
        "F": {"auc_pr": 0.21, "ci_high": 0.25},                  # also below, but see the second call
        "G": {"auc_pr": 0.10, "ci_high": 0.12},                  # no baseline to compare with
    }
    assert alerts.flagged_slices(now, baseline, overall_drop=0.04, t=T) == ["A", "E", "F"]


def test_a_good_week_never_makes_slices_look_degraded():
    """The control-week false alarm: the whole population improved, so slices that improved less are not degraded."""
    baseline = {"A": {"auc_pr": 0.30}}
    assert alerts.flagged_slices({"A": {"auc_pr": 0.29, "ci_high": 0.33}}, baseline, overall_drop=-0.057, t=T) == []
    assert alerts.flagged_slices({"A": {"auc_pr": 0.25, "ci_high": 0.28}}, baseline, overall_drop=-0.057, t=T) == ["A"]


# ---------------------------------------------------------------- week-over-week signals and the diagnosis table
def _table(ks, js):
    return pd.DataFrame([{"feature": "x", "kind": "numeric", "ks_stat": ks, "js": np.nan},
                         {"feature": "c", "kind": "categorical", "ks_stat": np.nan, "js": js}])


def test_features_changed_compares_with_last_week_not_with_the_reference():
    assert alerts.features_changed(None, _table(0.2, 0.05), T) is False                  # first week: nothing to compare
    assert alerts.features_changed(_table(0.19, 0.05), _table(0.20, 0.05), T) is False   # drifted long ago, stable now
    assert alerts.features_changed(_table(0.02, 0.01), _table(0.20, 0.01), T) is True    # a numeric feature jumped
    assert alerts.features_changed(_table(0.02, 0.01), _table(0.02, 0.05), T) is True    # the category mix moved


def test_score_changed_uses_the_ks_rise_or_the_mean_move():
    base = {"score_ks": 0.02, "mean_score": 0.084}
    assert alerts.score_changed(None, base, T) is False
    assert alerts.score_changed(base, {"score_ks": 0.021, "mean_score": 0.085}, T) is False
    assert alerts.score_changed(base, {"score_ks": 0.061, "mean_score": 0.090}, T) is True   # KS rose by 0.041
    assert alerts.score_changed(base, {"score_ks": 0.02, "mean_score": 0.100}, T) is True    # the mean moved 1.6 points


@pytest.mark.parametrize(("kwargs", "cause"), [
    ({"blocked": True, "features_changed": False, "score_changed": False, "performance_dropped": None, "calibration_off": None}, "technical"),
    ({"blocked": False, "features_changed": True, "score_changed": True, "performance_dropped": False, "calibration_off": False}, "covariate_shift"),
    ({"blocked": False, "features_changed": True, "score_changed": False, "performance_dropped": True, "calibration_off": False}, "covariate_shift"),
    ({"blocked": False, "features_changed": False, "score_changed": False, "performance_dropped": True, "calibration_off": False}, "concept_drift"),
    ({"blocked": False, "features_changed": False, "score_changed": True, "performance_dropped": False, "calibration_off": True}, "label_shift"),
    ({"blocked": False, "features_changed": False, "score_changed": False, "performance_dropped": False, "calibration_off": False}, "none"),
])
def test_diagnosis_follows_the_decision_table(kwargs, cause):
    assert alerts.diagnose(**kwargs) == cause


# ---------------------------------------------------------------- file-driven steps
def _artifacts(root, week, *, passed=True, ks=0.2, js=0.05, score_ks=0.03, mean_score=0.085, perf=None, version="1"):
    folder = root / f"week_{week:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gate.json").write_text(json.dumps({"week": week, "passed": passed, "rows": 5_000, "warnings": [],
                                                  "blocking": [] if passed else [{"column": "AMT_INCOME_TOTAL", "rule": "not_nullable", "count": 600}]}))
    if passed:
        drift = {**_drift(), "week": week, "score_ks": score_ks, "mean_score": mean_score, "model_version": version}
        (folder / "drift.json").write_text(json.dumps(drift))
        (folder / "score.json").write_text(json.dumps({"model_version": version, "mean_score": mean_score,
                                                       "flag_rate": 0.12, "threshold": 1 / 6}))
        table = pd.concat([_table(ks, js), pd.DataFrame([{"feature": "SCORE", "kind": "numeric", "ks_stat": score_ks, "js": 0.0}])])
        table = table.assign(monitored=True, psi=0.01, ks_p=0.5, chi2_p=np.nan)
        table.to_parquet(folder / "drift_features.parquet", index=False)
    if perf:
        base = {"week": week, "model_version": version, "mean_pred": 0.085, "observed_rate": 0.083, "auc_roc": 0.76,
                "brier": 0.068, "flag_rate": 0.12, "baseline_auc_pr": 0.246, "flagged_slices": [], "slices": {}, **perf}
        base["auc_pr_drop"] = 0.246 - base["auc_pr"]
        (folder / "performance.json").write_text(json.dumps(base))


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly, "ROOT", tmp_path)
    (tmp_path / "baseline_v1.json").write_text(json.dumps({"overall": {"auc_pr": 0.246}, "slices": {}}))
    return tmp_path


def _evaluate(root, week):
    weekly.step_evaluate(week)
    return json.loads((root / f"week_{week:02d}" / "alerts.json").read_text())


def test_evaluate_calls_the_week_after_a_feature_jump_a_covariate_shift(workspace):
    _artifacts(workspace, 4, ks=0.02, js=0.01, score_ks=0.02, mean_score=0.084, perf={"auc_pr": 0.25})
    _artifacts(workspace, 5, ks=0.20, js=0.05, score_ks=0.061, mean_score=0.090, perf={"auc_pr": 0.24})
    _artifacts(workspace, 6, ks=0.20, js=0.05, score_ks=0.06, mean_score=0.089)
    result = _evaluate(workspace, 6)
    assert result["diagnoses"]["5"]["cause"] == "covariate_shift"


def test_evaluate_calls_a_performance_drop_without_moved_features_concept_drift(workspace):
    _artifacts(workspace, 7, ks=0.20, js=0.05, perf={"auc_pr": 0.25})
    _artifacts(workspace, 8, ks=0.20, js=0.05, perf={"auc_pr": 0.18})  # unchanged inputs, AUC-PR 0.066 below baseline
    _artifacts(workspace, 9, ks=0.20, js=0.05)
    result = _evaluate(workspace, 9)
    assert result["diagnoses"]["8"]["cause"] == "concept_drift"
    assert any(a["rule"] == "performance_drop" and a["about_week"] == 8 for a in result["alerts"])


def test_evaluate_diagnoses_a_blocked_batch_at_once_and_says_so(workspace):
    _artifacts(workspace, 9, passed=False)
    result = _evaluate(workspace, 9)
    assert result["diagnoses"]["9"]["cause"] == "technical"
    assert [a["rule"] for a in result["alerts"]] == ["batch_blocked"]


def test_a_stable_new_normal_raises_no_diagnosis_alert(workspace):
    _artifacts(workspace, 6, ks=0.20, js=0.05, perf={"auc_pr": 0.24})
    _artifacts(workspace, 7, ks=0.205, js=0.05, perf={"auc_pr": 0.245})
    _artifacts(workspace, 8, ks=0.20, js=0.05)
    assert _evaluate(workspace, 8)["diagnoses"]["7"]["cause"] == "none"


def test_push_sends_one_group_per_week_and_stage(workspace, monkeypatch):
    from prometheus_client import generate_latest

    _artifacts(workspace, 4, perf={"auc_pr": 0.25})
    _artifacts(workspace, 5)
    (workspace / "week_04" / "performance.json").write_text(json.dumps({
        "week": 4, "model_version": "1", "auc_pr": 0.25, "auc_roc": 0.76, "brier": 0.068, "mean_pred": 0.085,
        "observed_rate": 0.083, "flag_rate": 0.12, "baseline_auc_pr": 0.246, "auc_pr_drop": -0.004,
        "flagged_slices": [], "slices": {}}))
    weekly.step_evaluate(5)
    pushed = []
    monkeypatch.setattr("prometheus_client.push_to_gateway",
                        lambda url, job, registry, grouping_key: pushed.append((job, grouping_key, generate_latest(registry).decode())))
    weekly.step_push(5)
    groups = {(g["week"], g["stage"]): text for _, g, text in pushed}
    assert set(groups) == {("5", "batch"), ("4", "labels")}
    assert "drift_max_psi" in groups[("5", "batch")] and "drift_feature_psi{feature=" in groups[("5", "batch")]
    assert "model_auc_pr 0.25" in groups[("4", "labels")] and "model_auc_pr_drop" in groups[("4", "labels")]


def test_json_helper_turns_nan_and_numpy_into_valid_json():
    cleaned = weekly._clean({"a": float("nan"), "b": np.float64(0.5), "c": [np.int64(3), float("inf")]})
    assert cleaned == {"a": None, "b": 0.5, "c": [3, None]}
    json.dumps(cleaned, allow_nan=False)
