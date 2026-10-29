import numpy as np
import pandas as pd

from credit_drift.split import assert_disjoint, split_pools, summarize


def _toy(n=20_000, rate=0.08, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"SK_ID_CURR": np.arange(n), "TARGET": (rng.random(n) < rate).astype(int), "x": rng.normal(size=n)})


def test_parts_are_disjoint_and_cover_everything():
    df = _toy()
    parts = split_pools(df, production_size=5_000)
    assert_disjoint(parts)
    assert sum(len(p) for p in parts.values()) == len(df)


def test_reference_shares_and_stratification():
    parts = split_pools(_toy(), production_size=5_000)
    n_ref = sum(len(parts[k]) for k in ("train", "calibration", "test"))
    assert abs(len(parts["train"]) / n_ref - 0.70) < 0.01
    assert abs(len(parts["calibration"]) / n_ref - 0.10) < 0.01
    assert abs(len(parts["test"]) / n_ref - 0.20) < 0.01
    rates = summarize(parts).loc[["train", "calibration", "test"], "rate"]
    assert rates.max() - rates.min() < 0.002


def test_deterministic_for_a_seed():
    a, b = split_pools(_toy(), seed=7, production_size=5_000), split_pools(_toy(), seed=7, production_size=5_000)
    assert a["test"]["SK_ID_CURR"].tolist() == b["test"]["SK_ID_CURR"].tolist()
