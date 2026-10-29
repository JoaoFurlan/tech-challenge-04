"""Training building blocks on synthetic data: features, metrics, selection rule, calibration."""
import numpy as np
import pytest

from credit_drift.data import DAYS_EMPLOYED_SENTINEL
from credit_drift.training.champion import pick_champion, shortlist
from credit_drift.training.features import FeatureEngineer, build_preprocessor
from credit_drift.training.metrics import cost_at_threshold, expected_cost, prior_brier
from credit_drift.training.model import CalibratedModel, fit_calibrator


def test_engineer_turns_the_sentinel_into_null_plus_flag(batch):
    batch.loc[:9, "DAYS_EMPLOYED"] = DAYS_EMPLOYED_SENTINEL
    out = FeatureEngineer().fit(batch).transform(batch)
    assert out["DAYS_EMPLOYED"].iloc[:10].isna().all()
    assert (out["DAYS_EMPLOYED_ANOM"].iloc[:10] == 1).all() and out["DAYS_EMPLOYED_ANOM"].iloc[10:].sum() == 0


def test_engineer_excludes_id_target_and_gender(batch):
    out = FeatureEngineer().fit(batch).transform(batch.assign(TARGET=0))
    assert not {"SK_ID_CURR", "TARGET", "CODE_GENDER"} & set(out.columns)


def test_engineer_flags_the_missing_housing_block(batch):
    house = [c for c in batch.columns if c.endswith(("_AVG", "_MODE", "_MEDI")) and batch[c].dtype != object]
    engineer = FeatureEngineer().fit(batch)
    broken = batch.copy()
    broken.loc[:19, house] = np.nan
    out = engineer.transform(broken)
    assert (out["HOUSING_BLOCK_MISSING"].iloc[:20] == 1).all() and out["HOUSING_BLOCK_MISSING"].iloc[20:].sum() == 0


def test_income_cap_comes_from_training_rows_only(batch):
    engineer = FeatureEngineer().fit(batch)
    cap = engineer.income_cap_
    other = batch.copy()
    other["AMT_INCOME_TOTAL"] = 10 * cap
    assert FeatureEngineer.transform(engineer, other)["AMT_INCOME_TOTAL"].max() == pytest.approx(cap)


def test_no_infinite_values_from_ratios(batch):
    batch.loc[:4, "AMT_GOODS_PRICE"] = 0
    out = FeatureEngineer().fit(batch).transform(batch)
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()


@pytest.mark.parametrize(("variant", "family", "nan_free"), [
    ("A", "lightgbm", True), ("B", "logreg", True), ("B", "lightgbm", False), ("C", "logreg", True)])
def test_preprocessors_fit_once_and_transform_new_rows(batch, variant, family, nan_free):
    y = np.random.default_rng(1).integers(0, 2, len(batch))
    prep = build_preprocessor(variant, family)
    train = prep.fit_transform(batch, y)
    new = prep.transform(batch.sample(50, random_state=0))
    assert train.dtype == np.float32 and new.shape[1] == train.shape[1]
    assert len(prep.get_feature_names_out()) == train.shape[1]
    if nan_free:
        assert not np.isnan(train).any()
    if variant == "C":
        assert train.shape[1] <= 60


def test_variant_a_leaves_the_sentinel_alone(batch):
    batch.loc[:9, "DAYS_EMPLOYED"] = DAYS_EMPLOYED_SENTINEL
    prep = build_preprocessor("A", "lightgbm")
    matrix = prep.fit_transform(batch, np.zeros(len(batch)))
    column = list(prep.get_feature_names_out()).index("DAYS_EMPLOYED")
    assert (matrix[:10, column] == DAYS_EMPLOYED_SENTINEL).all()


def test_expected_cost_edge_cases():
    y = np.array([0] * 90 + [1] * 10)
    assert expected_cost(y, y.astype(float)) == pytest.approx(0.0)  # perfect ranking
    # an uninformative score is no better than the trivial "flag nobody" (C_FN * pi = 0.5)
    assert expected_cost(y, np.full(100, 0.1), c_fn=5) == pytest.approx(0.5)
    assert cost_at_threshold(y, np.full(100, 0.1), threshold=0.5, c_fn=5) == pytest.approx(0.5)
    assert prior_brier(y) == pytest.approx(0.09)


def _cell(name, auc, se, cost):
    return {"variant": name, "family": "x", "cv_auc_pr_mean": auc, "cv_auc_pr_se": se, "oof_cost_5": cost, "oof_cost_3": cost}


def test_shortlist_keeps_cells_within_one_standard_error():
    cells = [_cell("a", 0.220, 0.004, 0.30), _cell("b", 0.217, 0.004, 0.31), _cell("c", 0.200, 0.004, 0.25)]
    assert {c["variant"] for c in shortlist(cells)} == {"a", "b"}


def test_champion_is_the_cheapest_shortlisted_cell_not_the_best_ranked():
    cells = [_cell("a", 0.220, 0.004, 0.30), _cell("b", 0.217, 0.004, 0.28), _cell("c", 0.200, 0.004, 0.10)]
    assert pick_champion(cells)["variant"] == "b"  # c is cheapest overall but outside the shortlist


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrated_model_returns_valid_probabilities(method):
    rng = np.random.default_rng(0)
    raw = rng.random(2000)
    y = (rng.random(2000) < raw * 0.3).astype(int)

    class Stub:
        def predict_proba(self, X):
            return np.column_stack([1 - X, X])

    model = CalibratedModel(Stub(), fit_calibrator(method, raw, y), method)
    p = model.predict_proba(raw)[:, 1]
    assert p.min() >= 0 and p.max() <= 1
    assert abs(p.mean() - y.mean()) < 0.01  # calibration pulls the mean prediction to the observed rate
