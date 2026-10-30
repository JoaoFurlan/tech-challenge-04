"""The causal reading of the reference run and the group-fairness check."""
import numpy as np
import pandas as pd
import pytest

from credit_drift.monitoring import causal, fairness


def _week(week, **psi):
    base = {"AMT_INCOME_TOTAL": 0.01, "ANNUITY_CREDIT": 0.01, "EXT_SOURCE_2": 0.01, "ORGANIZATION_TYPE": 0.01}
    return {"week": week, "psi": {**base, **psi}}


def test_detection_scorecard_counts_direct_derived_collateral_and_false_positives():
    summary = {"weeks": [_week(1), _week(5, AMT_INCOME_TOTAL=0.4, ANNUITY_CREDIT=0.9, ORGANIZATION_TYPE=0.2, EXT_SOURCE_2=0.3),
                         {"week": 9}]}
    table = causal.detection_scorecard(summary).set_index("week")
    assert list(table.index) == [1, 5]  # a blocked week has no drift table
    assert table.loc[1, "detected"] == 0 and pd.isna(table.loc[1, "recall_of_injected"])
    row = table.loc[5]
    assert (row["direct"], row["derived"], row["collateral"]) == (1, 1, 1)
    assert row["false_positives"] == ["EXT_SOURCE_2"]  # nobody injected it and it is not a side effect
    assert row["recall_of_injected"] == pytest.approx(2 / 8)


def test_importance_weights_favour_rows_that_look_like_production():
    rng = np.random.default_rng(0)
    make = lambda mean, n: pd.DataFrame({"x": rng.normal(mean, 1, n), "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner"], n)})
    reference, production = make(0.0, 4000), make(1.0, 4000)
    weights = causal.importance_weights(reference, production, reference)
    high = reference["x"].to_numpy() > 1.0
    assert weights[high].mean() > 3 * weights[~high].mean()
    assert weights.max() <= 20.0  # clipped


def test_group_table_reports_rates_per_group_and_the_gaps():
    rng = np.random.default_rng(1)
    n = 6000
    group = np.where(rng.random(n) < 0.6, "F", "M")
    y = (rng.random(n) < np.where(group == "M", 0.12, 0.07)).astype(int)
    p = np.clip(0.05 + 0.2 * y + rng.normal(0, 0.05, n), 0.001, 0.999)
    table = fairness.group_table(y, p, group, 1 / 6)
    assert table["F"]["rows"] + table["M"]["rows"] == n
    assert table["gaps_M_minus_F"]["flag_rate"] == pytest.approx(table["M"]["flag_rate"] - table["F"]["flag_rate"], abs=1e-4)
    assert table["M"]["observed_default_rate"] > table["F"]["observed_default_rate"]
