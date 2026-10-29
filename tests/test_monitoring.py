"""Drift statistics: effect size, missing values, ties, categories, and the monitored-feature config."""
import numpy as np
import pandas as pd

from credit_drift.monitoring.features import RATIOS, load_config, monitored_columns
from credit_drift.monitoring.stats import (
    PSI_MAJOR,
    PSI_MODERATE,
    categorical_drift,
    drift_table,
    numeric_drift,
    severity,
    summarize,
)
from credit_drift.training.features import FeatureEngineer

rng = np.random.default_rng(0)


def _spiky_income(n=20_000, seed=1):
    """Round-number incomes: 60% of rows sit on 8 values, the rest are continuous (like the real data)."""
    r = np.random.default_rng(seed)
    spikes = r.choice([90_000, 112_500, 135_000, 150_000, 180_000, 202_500, 225_000, 270_000], n)
    return pd.Series(np.where(r.random(n) < 0.6, spikes, r.lognormal(11.9, 0.45, n)))


def test_identical_distributions_show_no_drift():
    a, b = pd.Series(rng.normal(0, 1, 20_000)), pd.Series(rng.normal(0, 1, 5_000))
    result = numeric_drift(a, b)
    assert result["psi"] < 0.02 and result["ks_stat"] < 0.03


def test_a_real_shift_is_a_major_psi():
    a, b = pd.Series(rng.normal(0, 1, 20_000)), pd.Series(rng.normal(1.0, 1, 5_000))
    result = numeric_drift(a, b)
    assert result["psi"] > PSI_MAJOR and result["ks_stat"] > 0.3 and result["ks_p"] < 0.001


def test_psi_grows_with_the_shift_even_when_values_are_round_numbers():
    base = _spiky_income()
    psi = {m: numeric_drift(base, _spiky_income(5_000, seed=2) * m)["psi"] for m in (1.0, 0.90, 0.86, 0.80)}
    assert psi[1.0] < 0.02
    assert psi[1.0] < psi[0.90] < psi[0.86] < psi[0.80]  # without smoothing a spike on a bin edge breaks this


def test_psi_is_deterministic():
    a, b = _spiky_income(), _spiky_income(5_000, seed=3) * 0.9
    assert numeric_drift(a, b)["psi"] == numeric_drift(a, b)["psi"]


def test_a_rise_in_missing_values_counts_as_drift():
    base = pd.Series(rng.normal(0, 1, 20_000))
    current = pd.Series(rng.normal(0, 1, 5_000)).mask(rng.random(5_000) < 0.3)
    assert numeric_drift(base, current)["psi"] > PSI_MAJOR


def test_flags_are_binned_by_value():
    base = pd.Series((rng.random(20_000) < 0.1).astype(int))
    current = pd.Series((rng.random(5_000) < 0.3).astype(int))
    assert numeric_drift(base, current)["psi"] > PSI_MODERATE


def test_categorical_drift_sees_new_categories_and_small_pvalues():
    base = pd.Series(rng.choice(["a", "b", "c"], 20_000, p=[0.5, 0.3, 0.2]))
    same = pd.Series(rng.choice(["a", "b", "c"], 5_000, p=[0.5, 0.3, 0.2]))
    moved = pd.Series(rng.choice(["a", "b", "c", "new"], 5_000, p=[0.3, 0.3, 0.2, 0.2]))
    assert categorical_drift(base, same)["psi"] < 0.02 and categorical_drift(base, same)["chi2_p"] > 0.01
    result = categorical_drift(base, moved)
    assert result["psi"] > PSI_MAJOR and result["chi2_p"] < 1e-6 and result["js"] > 0.1


def test_small_mix_shift_has_a_tiny_psi_but_a_tiny_pvalue():
    """The PSI-versus-chi-square contrast the project demonstrates: a few points of mix are 'low' PSI, 'certain' chi2."""
    base = pd.Series(rng.choice(["Working", "Pensioner", "Other"], 100_000, p=[0.52, 0.18, 0.30]))
    current = pd.Series(rng.choice(["Working", "Pensioner", "Other"], 5_000, p=[0.58, 0.12, 0.30]))
    result = categorical_drift(base, current)
    assert result["psi"] < PSI_MODERATE and result["chi2_p"] < 1e-10


def test_severity_bands():
    assert [severity(x) for x in (0.05, 0.10, 0.2, 0.25, 0.9)] == ["none", "moderate", "moderate", "major", "major"]


def test_summary_counts_only_monitored_features_and_reports_the_score():
    table = pd.DataFrame([
        {"feature": "a", "kind": "numeric", "monitored": True, "psi": 0.30, "ks_stat": 0.2},
        {"feature": "b", "kind": "numeric", "monitored": True, "psi": 0.12, "ks_stat": 0.1},
        {"feature": "c", "kind": "numeric", "monitored": True, "psi": 0.01, "ks_stat": 0.0},
        {"feature": "d", "kind": "numeric", "monitored": False, "psi": 0.50, "ks_stat": 0.4},
        {"feature": "SCORE", "kind": "numeric", "monitored": True, "psi": 0.03, "ks_stat": 0.07},
    ])
    summary = summarize(table)
    assert summary["n_monitored"] == 3 and summary["n_psi_moderate"] == 2 and summary["n_psi_major"] == 1
    assert summary["max_psi_feature"] == "a" and summary["n_all_psi_moderate"] == 3
    assert summary["score_psi"] == 0.03 and summary["score_ks"] == 0.07


def test_drift_table_has_one_row_per_feature():
    base = pd.DataFrame({"x": rng.normal(size=5_000), "c": rng.choice(["a", "b"], 5_000)})
    current = pd.DataFrame({"x": rng.normal(size=1_000), "c": rng.choice(["a", "b"], 1_000)})
    table = drift_table(base, current, {"x": "numeric", "c": "categorical"}, {"x"})
    assert list(table["feature"]) == ["x", "c"] and list(table["monitored"]) == [True, False]
    assert table.loc[table.feature == "x", "chi2_p"].isna().all()  # chi-square is for categories only


def test_monitored_set_is_rule_based_deduplicated_and_excludes_id_and_gender():
    columns = monitored_columns(load_config())
    assert len(columns) == len(set(columns)) >= 25
    assert not {"SK_ID_CURR", "CODE_GENDER", "TARGET"} & set(columns)
    assert set(RATIOS) <= set(columns)  # the champion's debt-to-income style ratios are always watched


def test_the_ratios_exist_in_the_champions_feature_frame(batch):
    frame = FeatureEngineer().fit(batch).transform(batch)
    assert set(RATIOS) <= set(frame.columns)
