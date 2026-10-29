"""Slices the weekly monitoring watches (docs/decisions.md, #5): 14 of them, 7 variables x 2 halves.

Cut points are the medians of the champion's training window, fixed once, so the slice definitions never drift
themselves. Rows with a missing value in a slice variable are left out of both halves of that variable.
Gender is not a model input; it is a slice only, as a fairness check.
"""
import numpy as np
import pandas as pd

from credit_drift.data import DAYS_EMPLOYED_SENTINEL, age_years

NUMERIC_VARIABLES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_INCOME_TOTAL", "AGE", "EMPLOYMENT_YEARS"]


def slice_values(df: pd.DataFrame) -> pd.DataFrame:
    """The raw-scale variables the slices are defined on."""
    employment = (-df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != DAYS_EMPLOYED_SENTINEL) / 365.25)
    return pd.DataFrame({
        "EXT_SOURCE_1": df["EXT_SOURCE_1"], "EXT_SOURCE_2": df["EXT_SOURCE_2"], "EXT_SOURCE_3": df["EXT_SOURCE_3"],
        "AMT_INCOME_TOTAL": df["AMT_INCOME_TOTAL"], "AGE": age_years(df), "EMPLOYMENT_YEARS": employment,
    }, index=df.index)


def slice_cuts(train: pd.DataFrame) -> dict[str, float]:
    return {name: float(slice_values(train)[name].median()) for name in NUMERIC_VARIABLES}


def slice_masks(df: pd.DataFrame, cuts: dict[str, float]) -> dict[str, np.ndarray]:
    values = slice_values(df)
    masks: dict[str, np.ndarray] = {}
    for name in NUMERIC_VARIABLES:
        column, known = values[name], values[name].notna()
        masks[f"{name} low"] = (known & (column <= cuts[name])).to_numpy()
        masks[f"{name} high"] = (known & (column > cuts[name])).to_numpy()
    for gender in ("F", "M"):
        masks[f"GENDER {gender}"] = (df["CODE_GENDER"] == gender).to_numpy()
    return masks
