import pandas as pd

from credit_drift.data import DAYS_EMPLOYED_SENTINEL, age_years, employment_years


def test_age_years():
    assert age_years(pd.DataFrame({"DAYS_BIRTH": [-365.25 * 30]})).iloc[0] == 30


def test_employment_sentinel_becomes_nan():
    df = pd.DataFrame({"DAYS_EMPLOYED": [-730.5, DAYS_EMPLOYED_SENTINEL]})
    out = employment_years(df)
    assert out.iloc[0] == 2 and out.isna().iloc[1]
