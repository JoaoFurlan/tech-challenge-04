"""Shared synthetic data: a valid scoring batch built from the committed contract profile."""
import numpy as np
import pandas as pd
import pytest

from credit_drift.contracts.profile import load_profile

N = 600


@pytest.fixture(scope="session")
def profile():
    return load_profile()


@pytest.fixture()
def batch(profile):
    """A valid scoring batch: every profiled column present, all contract rules satisfied."""
    rng = np.random.default_rng(0)
    data = {}
    for col, info in profile["columns"].items():
        if info["kind"] == "categorical":
            data[col] = np.full(N, info["categories"][0], dtype=object)
        else:
            data[col] = rng.uniform(1, 100, N)
    data["SK_ID_CURR"] = np.arange(N) + 100_000
    data["AMT_INCOME_TOTAL"] = rng.uniform(50_000, 400_000, N)
    data["AMT_CREDIT"] = rng.uniform(50_000, 800_000, N)
    data["AMT_ANNUITY"] = rng.uniform(5_000, 40_000, N)
    data["DAYS_BIRTH"] = rng.integers(-25_000, -8_000, N)
    data["DAYS_EMPLOYED"] = rng.integers(-5_000, -1, N)
    data["CODE_GENDER"] = np.where(rng.random(N) < 0.5, "F", "M")
    return pd.DataFrame(data)
