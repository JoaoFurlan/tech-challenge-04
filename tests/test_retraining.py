"""The retraining window, the paired-bootstrap gate, the trigger rule and the promotion step."""
import numpy as np
import pandas as pd
import pytest

from credit_drift.data import ID_COL, TARGET
from credit_drift.retraining import policy, run, window
from credit_drift.retraining.gate import paired_gate


# ---------------------------------------------------------------- window
def _weeks(root, labelled, unlabelled=()):
    folder = root / "weeks"
    folder.mkdir(parents=True)
    r = np.random.default_rng(0)
    for w in (*labelled, *unlabelled):
        ids = np.arange(w * 10_000, w * 10_000 + 400)
        pd.DataFrame({ID_COL: ids, "x": r.normal(size=400)}).to_parquet(folder / f"week_{w:02d}_features.parquet")
        if w in labelled:
            pd.DataFrame({ID_COL: ids, TARGET: (r.random(400) < 0.1).astype(int)}).to_parquet(
                folder / f"week_{w:02d}_labels.parquet")


@pytest.fixture()
def production(tmp_path, monkeypatch):
    monkeypatch.setattr(window, "PRODUCTION_DIR", tmp_path)
    _weeks(tmp_path, labelled=[1, 2, 3, 4, 5, 6, 7, 8, 10], unlabelled=[9])
    return tmp_path


def test_a_week_without_labels_is_never_used_and_labels_arrive_one_week_late(production):
    assert window.labelled_weeks(11) == [1, 2, 3, 4, 5, 6, 7, 8, 10]  # week 9 never got labels
    assert window.labelled_weeks(4) == [1, 2, 3, 4]


def test_the_recent_slice_is_split_into_disjoint_fit_calibration_and_gate_parts(production):
    reference = pd.DataFrame({ID_COL: np.arange(2_000), "x": 0.0, TARGET: 0})
    w = window.build_window(10, reference)
    assert w.recent_weeks == [8, 10] and w.weeks[-1] == 10
    ids = [set(part[ID_COL]) for part in (w.calibration, w.gate)]
    recent_in_fit = set(w.fit.loc[w.fit["week"].isin(w.recent_weeks), ID_COL])
    assert not (ids[0] & ids[1]) and not (ids[0] & recent_in_fit) and not (ids[1] & recent_in_fit)
    assert len(w.calibration) + len(w.gate) + len(recent_in_fit) == 800
    assert len(w.gate) == pytest.approx(0.4 * 800, abs=2)


def test_production_rows_carry_half_the_weight_and_recent_weeks_count_more(production):
    reference = pd.DataFrame({ID_COL: np.arange(2_000), "x": 0.0, TARGET: 0})
    w = window.build_window(8, reference)
    production_rows = w.fit["week"] > 0
    assert w.weights[production_rows.to_numpy()].sum() == pytest.approx(w.weights[~production_rows.to_numpy()].sum())
    by_week = w.fit.assign(weight=w.weights).query("week > 0").groupby("week")["weight"].mean()
    assert by_week[8] > by_week[3]


def test_too_few_labelled_weeks_refuses_to_retrain(production):
    with pytest.raises(SystemExit):
        window.build_window(2, pd.DataFrame({ID_COL: [1], "x": 0.0, TARGET: 0}))


# ---------------------------------------------------------------- gate
def _scores(n=4_000, seed=0):
    r = np.random.default_rng(seed)
    y = (r.random(n) < 0.08).astype(int)
    good = np.clip(0.05 + 0.25 * y + r.normal(0, 0.08, n), 0.001, 0.999)
    return y, good, r


def test_the_gate_rejects_an_identical_model_and_a_worse_one():
    y, good, r = _scores()
    assert not paired_gate(y, good, good, 1 / 6, 1 / 6)["passed"]  # no gain: the interval cannot clear zero
    worse = np.clip(good + r.normal(0, 0.15, len(y)), 0.001, 0.999)
    assert not paired_gate(y, good, worse, 1 / 6, 1 / 6)["passed"]


def test_the_gate_accepts_a_clearly_better_model_and_reports_why():
    y, good, r = _scores()
    weak = np.clip(0.08 + 0.02 * y + r.normal(0, 0.1, len(y)), 0.001, 0.999)
    result = paired_gate(y, weak, good, 1 / 6, 1 / 6)
    assert result["passed"] and all(result["checks"].values())
    assert result["cost_gain_ci"][0] > 0 and result["auc_pr_diff"] > 0


# ---------------------------------------------------------------- trigger rule
def _alert(rule, about_week):
    return {"rule": rule, "about_week": about_week}


def test_the_first_major_drift_alert_triggers_the_first_retrain():
    assert policy.trigger_for(5, [_alert("data_drift_major", 5)], []) == "data_drift_major"
    assert policy.trigger_for(4, [_alert("data_drift_moderate", 4)], []) is None
    assert policy.trigger_for(3, [_alert("data_drift_major", 3)], []) is None  # too few labelled weeks


def test_a_labelled_week_is_tried_once_and_a_performance_drop_triggers_a_later_retrain():
    failed = {"run_week": 5, "through_week": 4, "promoted": False}
    assert policy.trigger_for(6, [_alert("data_drift_major", 6)], [failed]) is None  # drift alert alone: already tried
    assert policy.trigger_for(11, [_alert("performance_drop", 10)], [failed]) == "performance_drop"
    again = {"run_week": 11, "through_week": 10, "promoted": False}
    assert policy.trigger_for(11, [_alert("performance_drop", 10)], [failed, again]) is None
    seen = {"run_week": 5, "through_week": 4, "promoted": True}  # the champion already trained on week 4
    assert policy.trigger_for(7, [_alert("performance_drop", 4)], [seen]) is None


# ---------------------------------------------------------------- promotion
class _FakeClient:
    def __init__(self, with_baseline):
        self.aliases, self.deleted, self.with_baseline = {}, [], with_baseline

    def get_model_version_by_alias(self, name, alias):
        if alias == "baseline" and not self.with_baseline:
            raise run.MlflowException("no such alias")

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name, alias):
        self.deleted.append(alias)


@pytest.fixture()
def promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "STATE", tmp_path / "retraining.json")
    monkeypatch.setattr(run, "load_state", lambda: policy.load_state(tmp_path / "retraining.json"))
    monkeypatch.setattr(run, "save_state", lambda events: policy.save_state(events, tmp_path / "retraining.json"))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    calls = []

    class Answer:
        def raise_for_status(self):
            pass

        def json(self):
            return {"version": "3", "changed": True}

    monkeypatch.setattr(run.requests, "post", lambda url, **kw: calls.append(url) or Answer())
    return tmp_path, calls


def _event(passed):
    return {"run_week": 11, "through_week": 10, "from_version": "1", "chosen": "B", "versions": {"A": "2", "B": "3"},
            "gate": {"passed": passed}, "promoted": False}


def test_a_failed_gate_moves_no_alias_and_does_not_reload(promotion, monkeypatch):
    tmp, calls = promotion
    client = _FakeClient(with_baseline=False)
    monkeypatch.setattr(run, "MlflowClient", lambda: client)
    policy.save_state([_event(False)], tmp / "retraining.json")
    run.step_promote(11)
    assert client.aliases == {} and calls == [] and not policy.load_state(tmp / "retraining.json")[0]["promoted"]


def test_a_passed_gate_moves_champion_keeps_the_first_as_baseline_and_reloads(promotion, monkeypatch):
    tmp, calls = promotion
    client = _FakeClient(with_baseline=False)
    monkeypatch.setattr(run, "MlflowClient", lambda: client)
    policy.save_state([_event(True)], tmp / "retraining.json")
    run.step_promote(11)
    assert client.aliases == {"baseline": "1", "previous": "1", "champion": "3"} and client.deleted == ["challenger"]
    assert calls and calls[0].endswith("/reload")
    saved = policy.load_state(tmp / "retraining.json")[0]
    assert saved["promoted"] and saved["to_version"] == "3"
    assert policy.champion_labels_through(policy.load_state(tmp / "retraining.json")) == 10
