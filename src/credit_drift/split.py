"""Seeded, disjoint split of the raw data into Reference parts and the Production pool.

Reference pool  -> train (70%) / calibration (10%) / test (20%), stratified on TARGET.
Production pool -> a plain random sample (no stratification): real weekly default rates
fluctuate, and the control weeks must prove the detectors cope with that noise.
The Production pool is cut into weekly batches later, by the drift simulator.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_drift.data import ID_COL, RAW_DIR, TARGET, load_application_train

SEED = 42
PRODUCTION_SIZE = 100_000  # 13 weeks x 5,000 rows = 65,000, plus headroom for resampled drift
TRAIN_SHARE, CALIBRATION_SHARE = 0.70, 0.10  # test gets the remaining 20%

DATA_DIR = RAW_DIR.parent
REFERENCE_DIR = DATA_DIR / "reference"
PRODUCTION_DIR = DATA_DIR / "production"


def split_pools(df: pd.DataFrame, seed: int = SEED, production_size: int = PRODUCTION_SIZE) -> dict[str, pd.DataFrame]:
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    production, reference = shuffled.iloc[:production_size], shuffled.iloc[production_size:]

    train, rest = train_test_split(
        reference, test_size=1 - TRAIN_SHARE, stratify=reference[TARGET], random_state=seed
    )
    test_share_of_rest = (1 - TRAIN_SHARE - CALIBRATION_SHARE) / (1 - TRAIN_SHARE)
    calibration, test = train_test_split(
        rest, test_size=test_share_of_rest, stratify=rest[TARGET], random_state=seed
    )
    parts = {"train": train, "calibration": calibration, "test": test, "production_pool": production}
    return {name: part.reset_index(drop=True) for name, part in parts.items()}


def write_splits(parts: dict[str, pd.DataFrame]) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "calibration", "test"):
        parts[name].to_parquet(REFERENCE_DIR / f"{name}.parquet", index=False)
    parts["production_pool"].to_parquet(PRODUCTION_DIR / "pool.parquet", index=False)


def summarize(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {name: {"rows": len(p), "defaults": int(p[TARGET].sum()), "rate": round(p[TARGET].mean(), 4)}
         for name, p in parts.items()}
    ).T


def assert_disjoint(parts: dict[str, pd.DataFrame]) -> None:
    ids = [set(p[ID_COL]) for p in parts.values()]
    total = sum(len(i) for i in ids)
    assert len(set().union(*ids)) == total, "splits overlap or contain duplicate IDs"


if __name__ == "__main__":
    parts = split_pools(load_application_train())
    assert_disjoint(parts)
    write_splits(parts)
    print(summarize(parts).to_string())
