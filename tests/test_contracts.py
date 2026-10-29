"""Contract tests on synthetic batches built from the committed profile (no DVC data needed)."""
import numpy as np
import pytest

from credit_drift.contracts import rules
from credit_drift.contracts.check import check_batch
from credit_drift.contracts.corruption import corrupt_batch
from credit_drift.contracts.pandera_gate import (
    build_scoring_schema,
    validate_labels,
    validate_scoring_batch,
)

N = 600


@pytest.fixture(scope="module")
def schema(profile):
    return build_scoring_schema(profile)


def _blocked(result, column, rule_fragment):
    return any(f.column == column and rule_fragment in f.rule for f in result.blocking)


def test_valid_batch_passes_pandera(batch, schema):
    assert validate_scoring_batch(batch, schema).passed


@pytest.mark.parametrize(
    ("column", "mutate", "rule"),
    [
        ("AMT_INCOME_TOTAL", lambda d: d.__setitem__("AMT_INCOME_TOTAL", d["AMT_INCOME_TOTAL"].mask(d.index < 5)), "not_nullable"),
        ("AMT_INCOME_TOTAL", lambda d: d.__setitem__("AMT_INCOME_TOTAL", d["AMT_INCOME_TOTAL"].mask(d.index < 5, 0)), "income_positive"),
        ("DAYS_BIRTH", lambda d: d.__setitem__("DAYS_BIRTH", d["DAYS_BIRTH"].mask(d.index < 5, -3_000)), "age_18_to_100"),
        ("SK_ID_CURR", lambda d: d.__setitem__("SK_ID_CURR", d["SK_ID_CURR"].mask(d.index < 5, 100_000)), "field_uniqueness"),
        ("DAYS_EMPLOYED", lambda d: d.__setitem__("DAYS_EMPLOYED", d["DAYS_EMPLOYED"].mask(d.index < 5, 1_000)), "days_employed_valid"),
        ("AMT_CREDIT", lambda d: d.__setitem__("AMT_CREDIT", d["AMT_CREDIT"].mask(d.index < 5, -1)), "credit_positive"),
        ("AMT_ANNUITY", lambda d: d.__setitem__("AMT_ANNUITY", d["AMT_ANNUITY"].mask(d.index < 5, -1)), "annuity_non_negative"),
        ("EXT_SOURCE_2", lambda d: d.__setitem__("EXT_SOURCE_2", d["EXT_SOURCE_2"].astype(str)), "numeric_dtype"),
    ],
)
def test_each_blocking_rule_fires(batch, schema, column, mutate, rule):
    mutate(batch)
    assert _blocked(validate_scoring_batch(batch, schema), column, rule)


def test_missing_column_is_blocked(batch, schema):
    result = validate_scoring_batch(batch.drop(columns=["EXT_SOURCE_2"]), schema)
    assert not result.passed


def test_sentinel_and_null_annuity_are_allowed(batch, schema):
    batch.loc[:9, "DAYS_EMPLOYED"] = rules.DAYS_EMPLOYED_SENTINEL
    batch.loc[:9, "AMT_ANNUITY"] = np.nan
    assert validate_scoring_batch(batch, schema).passed


def test_label_contract(batch):
    labeled = batch.assign(TARGET=np.random.default_rng(1).integers(0, 2, N))
    assert validate_labels(labeled).passed
    assert not validate_labels(labeled.assign(TARGET=2)).passed


def test_corruption_is_reproducible_and_documented(batch):
    a, spec_a = corrupt_batch(batch, seed=9)
    b, spec_b = corrupt_batch(batch, seed=9)
    assert spec_a == spec_b and a["SK_ID_CURR"].tolist() == b["SK_ID_CURR"].tolist()
    assert spec_a["null_income_rows"] == int(N * 0.12)


def test_both_tools_agree_on_clean_and_corrupted_batches(batch, profile):
    """Slow (builds the full Great Expectations suite), so both verdicts share one test."""
    clean = check_batch(batch, "test_clean", expected_rows=(500, 700), profile=profile)
    assert clean.pandera.passed and clean.gx.passed and not clean.gx.warnings

    bad, _ = corrupt_batch(batch, seed=9)
    report = check_batch(bad, "test_corrupted", expected_rows=(500, 700), profile=profile)
    assert not report.passed
    pandera_cols = {f.column for f in report.pandera.blocking}
    gx_cols = {f.column for f in report.gx.blocking}
    assert pandera_cols == gx_cols == {"AMT_INCOME_TOTAL", "DAYS_BIRTH", "SK_ID_CURR", "AMT_GOODS_PRICE"}


def test_warnings_do_not_block(batch, profile):
    batch.loc[:5, "AMT_INCOME_TOTAL"] = 20_000_000  # 6 of 600 rows above the ceiling: warn only
    report = check_batch(batch, "test_warning", expected_rows=(500, 700), profile=profile)
    assert report.passed
    assert any(w.column == "AMT_INCOME_TOTAL" for w in report.gx.warnings)
