"""Deliberately corrupted batch (the week-9 scenario and the Etapa 1 demo)."""
import numpy as np
import pandas as pd

from credit_drift.contracts import rules


def corrupt_batch(
    df: pd.DataFrame,
    seed: int,
    null_income_share: float = 0.12,
    under_age_share: float = 0.04,
    duplicate_id_share: float = 0.03,
    text_column: str = "AMT_GOODS_PRICE",
) -> tuple[pd.DataFrame, dict]:
    """Return a corrupted copy of a batch and a spec of what was done (ground truth, offline use only)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    null_rows = rng.choice(n, int(n * null_income_share), replace=False)
    out.iloc[null_rows, out.columns.get_loc("AMT_INCOME_TOTAL")] = np.nan

    young_rows = rng.choice(n, int(n * under_age_share), replace=False)
    ages = rng.uniform(10, rules.AGE_MIN_YEARS - 0.5, len(young_rows))
    out.iloc[young_rows, out.columns.get_loc("DAYS_BIRTH")] = np.round(-ages * rules.DAYS_PER_YEAR).astype(int)

    dup_rows = rng.choice(n, int(n * duplicate_id_share), replace=False)
    donors = rng.choice(np.setdiff1d(np.arange(n), dup_rows), len(dup_rows), replace=True)
    out.iloc[dup_rows, out.columns.get_loc("SK_ID_CURR")] = out["SK_ID_CURR"].to_numpy()[donors]

    out[text_column] = out[text_column].astype(str)

    spec = {
        "seed": seed,
        "rows": n,
        "null_income_rows": len(null_rows),
        "under_18_rows": len(young_rows),
        "duplicated_id_rows": len(dup_rows),
        "text_column": text_column,
    }
    return out, spec
