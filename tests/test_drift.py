"""Drift simulator: schedule, level-based mix, in-place numeric shifts, segment label redraw, reproducibility."""
import numpy as np
import pandas as pd
import pytest

from credit_drift.data import DAYS_EMPLOYED_SENTINEL
from credit_drift.drift import schedule as sch
from credit_drift.drift.simulate import (
    apply_numeric_state,
    draw_rows,
    redraw_segment_labels,
    segment_mask,
    simulate,
    target_counts,
)

N_POOL = 6_000
WEEK_ROWS = 200


class FakeLabelModel:
    """Stands in for the LightGBM label model: probability rises with a risk column, EXT_SOURCE_2 standardised."""

    b2 = -0.45

    def predict_proba(self, X):
        return np.clip(0.04 + 0.1 * (X["RISK"].to_numpy() > 0.5), 0.01, 0.99)

    def ext2_standardised(self, X):
        return ((X["EXT_SOURCE_2"] - 0.5) / 0.2).to_numpy()


@pytest.fixture(scope="module")
def pool():
    rng = np.random.default_rng(0)
    income_type = rng.choice(["Working", "Pensioner", "State servant", "Commercial associate"], N_POOL, p=[0.52, 0.18, 0.07, 0.23])
    days_employed = np.where(income_type == "Pensioner", DAYS_EMPLOYED_SENTINEL, -rng.integers(100, 8000, N_POOL))
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(N_POOL) + 1,
        "NAME_INCOME_TYPE": income_type,
        "AMT_INCOME_TOTAL": rng.uniform(40_000, 300_000, N_POOL),
        "AMT_ANNUITY": rng.uniform(8_000, 40_000, N_POOL),
        "AMT_CREDIT": rng.uniform(50_000, 600_000, N_POOL),
        "AMT_GOODS_PRICE": rng.uniform(50_000, 600_000, N_POOL),
        "DAYS_EMPLOYED": days_employed,
        "DAYS_BIRTH": -rng.integers(20 * 365, 65 * 365, N_POOL),
        "EXT_SOURCE_2": rng.uniform(0, 1, N_POOL),
        "RISK": rng.uniform(0, 1, N_POOL),
        "TARGET": (rng.random(N_POOL) < 0.08).astype(int),
    })


@pytest.fixture(scope="module")
def batches(pool):
    return simulate(pool, FakeLabelModel(), income_cutoff=150_000, week_rows=WEEK_ROWS)


def test_schedule_has_thirteen_weeks_with_the_planned_episodes():
    states = {s.week: s for s in sch.SCHEDULE}
    assert sorted(states) == list(range(1, 14))
    assert states[1].income_mult == states[2].income_mult == 1.0 and not states[2].concept_drift
    assert states[5].annuity_mult == pytest.approx(1.15) and states[5].income_mult == pytest.approx(0.86)
    assert [w for w, s in states.items() if s.concept_drift] == [8, 9, 10, 11, 12, 13]
    assert [w for w, s in states.items() if s.corrupted] == [9]
    levels = {(s.income_mult, s.employment_mult, s.annuity_mult, s.mix_level) for w, s in states.items() if w >= 5}
    assert len(levels) == 1  # weeks 5-13 all sit in the same regime


def test_numeric_state_changes_values_in_place_and_leaves_the_sentinel(pool):
    state = sch.SCHEDULE[4]  # week 5
    out = apply_numeric_state(pool.drop(columns="TARGET"), state)
    assert out["AMT_INCOME_TOTAL"].iloc[0] == pytest.approx(pool["AMT_INCOME_TOTAL"].iloc[0] * 0.86)
    assert out["AMT_ANNUITY"].iloc[0] == pytest.approx(pool["AMT_ANNUITY"].iloc[0] * 1.15)
    sentinel = pool["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    assert (out.loc[sentinel, "DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).all()
    employed = ~sentinel
    assert out.loc[employed, "DAYS_EMPLOYED"].iloc[0] == pytest.approx(pool.loc[employed, "DAYS_EMPLOYED"].iloc[0] * 0.85)


def test_weeks_never_share_a_customer(batches):
    ids = np.concatenate([b.features["SK_ID_CURR"].to_numpy() for b in batches if not b.state.corrupted])
    assert len(ids) == len(set(ids))


def test_mix_is_exactly_the_same_every_week_after_the_shift(batches, pool):
    base = pool["NAME_INCOME_TYPE"].value_counts(normalize=True)
    expected = target_counts(base, 1.0, WEEK_ROWS)["Pensioner"] / WEEK_ROWS
    for b in batches:
        if b.state.week >= 5:
            assert b.info["pensioner_share"] == pytest.approx(expected)  # level-based: no creep as the pool depletes
    assert expected < base["Pensioner"] - 0.03


def test_target_counts_sum_to_the_week_size():
    base = pd.Series({"Working": 0.52, "Pensioner": 0.18, "State servant": 0.07, "Commercial associate": 0.23})
    for level in (0.5, 1.0):
        assert target_counts(base, level, 5_000).sum() == 5_000
    assert target_counts(base, 1.0, 5_000)["Pensioner"] < target_counts(base, 0.5, 5_000)["Pensioner"]


def test_draw_rows_is_exact_and_uses_only_remaining_rows(pool):
    base = pool["NAME_INCOME_TYPE"].value_counts(normalize=True)
    remaining = np.arange(1_000, N_POOL)
    picked = draw_rows(remaining, pool["NAME_INCOME_TYPE"], base, 1.0, 300, np.random.default_rng(1))
    assert len(picked) == 300 and set(picked) <= set(remaining) and len(set(picked)) == 300


def test_labels_change_only_inside_the_segment_and_the_rate_is_kept(pool):
    X = pool.drop(columns="TARGET")
    y = pool["TARGET"].to_numpy()
    model = FakeLabelModel()
    segment = segment_mask(X, income_cutoff=150_000)
    out, info = redraw_segment_labels(y, model.predict_proba(X), model.ext2_standardised(X), segment, model.b2, np.random.default_rng(2))
    assert (out[~segment] == y[~segment]).all()
    assert info["segment_rows"] == segment.sum() and info["labels_changed"] > 0
    expected_rate = model.predict_proba(X)[segment].mean()
    assert abs(out[segment].mean() - expected_rate) < 0.01


def test_concept_drift_reverses_the_ext_source_2_relationship(pool):
    """With a strong label model, the segment's labels follow EXT_SOURCE_2 upward after the change."""
    X = pool.drop(columns="TARGET")
    n = len(X)
    p = np.full(n, 0.08)
    ext2 = ((X["EXT_SOURCE_2"] - 0.5) / 0.2).to_numpy()
    segment = np.ones(n, dtype=bool)
    out, _ = redraw_segment_labels(pool["TARGET"].to_numpy(), p, ext2, segment, b2=-0.45, rng=np.random.default_rng(3))
    high, low = out[ext2 > 0].mean(), out[ext2 <= 0].mean()
    assert high > low  # before the drift, a higher score meant LOWER risk


def test_corrupted_week_has_no_labels_and_a_spec(batches):
    week9 = next(b for b in batches if b.state.week == 9)
    assert week9.labels is None and week9.info["labels_available"] is False
    assert week9.info["corruption"]["null_income_rows"] > 0
    assert all(b.labels is not None for b in batches if b.state.week != 9)


def test_features_and_labels_line_up_and_features_have_no_target(batches):
    for b in batches:
        assert "TARGET" not in b.features.columns
        if b.labels is not None:
            assert (b.labels["SK_ID_CURR"].to_numpy() == b.features["SK_ID_CURR"].to_numpy()).all()


def test_earlier_weeks_do_not_change_when_more_weeks_are_added(pool):
    short = simulate(pool, FakeLabelModel(), 150_000, states=sch.SCHEDULE[:5], week_rows=WEEK_ROWS)
    full = simulate(pool, FakeLabelModel(), 150_000, states=sch.SCHEDULE, week_rows=WEEK_ROWS)
    for a, b in zip(short, full[:5], strict=True):
        pd.testing.assert_frame_equal(a.features, b.features)
        pd.testing.assert_frame_equal(a.labels, b.labels)
