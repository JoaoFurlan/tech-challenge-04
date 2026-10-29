"""Aggregate profile of the training data (null shares, categories). No customer rows."""
import json
from pathlib import Path

import pandas as pd

from credit_drift.data import TARGET

PROFILE_PATH = Path(__file__).resolve().parents[3] / "config" / "contract_profile.json"


def build_profile(train: pd.DataFrame) -> dict:
    columns = {}
    for col in train.columns:
        if col == TARGET:
            continue
        is_text = train[col].dtype == object
        columns[col] = {
            "kind": "categorical" if is_text else "numeric",
            "null_share": round(float(train[col].isna().mean()), 6),
        }
        if is_text:
            columns[col]["categories"] = sorted(train[col].dropna().unique().tolist())
    return {"n_rows": len(train), "columns": columns}


def save_profile(profile: dict, path: Path = PROFILE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=1, ensure_ascii=False), encoding="utf-8")


def load_profile(path: Path = PROFILE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    from credit_drift.split import REFERENCE_DIR

    profile = build_profile(pd.read_parquet(REFERENCE_DIR / "train.parquet"))
    save_profile(profile)
    kinds = pd.Series([c["kind"] for c in profile["columns"].values()]).value_counts().to_dict()
    print(f"profile of {profile['n_rows']:,} rows written to {PROFILE_PATH}: {kinds}")
