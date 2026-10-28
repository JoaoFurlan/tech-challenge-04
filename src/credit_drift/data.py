"""Loading and light cleaning helpers shared by notebooks, DAGs and training code."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
# Home Credit encodes "not employed" (pensioners, unemployed) as this number of days.
DAYS_EMPLOYED_SENTINEL = 365243


def load_application_train(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or RAW_DIR / "application_train.csv")


def age_years(df: pd.DataFrame) -> pd.Series:
    return -df["DAYS_BIRTH"] / 365.25


def employment_years(df: pd.DataFrame) -> pd.Series:
    """Years employed, with the not-employed sentinel turned into NaN."""
    days = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != DAYS_EMPLOYED_SENTINEL)
    return -days / 365.25
