"""Drift statistics per feature: PSI (effect size, drives alerts), KS and chi-square (evidence), JS distance.

PSI is the alert signal on purpose: with thousands of rows a p-value flags almost any difference, while PSI
measures how much the distribution moved. Missing values get their own bin, so a rise in nulls shows up as drift.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

N_BINS = 10
PSI_FLOOR = 1e-4
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25  # the usual industry reading of PSI
MISSING = "__missing__"
JITTER = 0.02  # relative, seeded; see _smooth


def severity(psi: float) -> str:
    return "major" if psi >= PSI_MAJOR else "moderate" if psi >= PSI_MODERATE else "none"


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e, a = np.clip(expected, PSI_FLOOR, None), np.clip(actual, PSI_FLOOR, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _smooth(values: pd.Series, seed: int) -> pd.Series:
    """Spread tied values by +-2% so a spike (61% of incomes sit on 8 round numbers) cannot straddle a decile edge:
    without it PSI jumps around (0.40 for a 10% shift, 0.19 for a 14% one). Seeded, so reruns match."""
    noise = np.random.default_rng(seed).uniform(-JITTER, JITTER, len(values))
    return values * (1 + noise)


def _numeric_shares(base: pd.Series, current: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    b, c = base.dropna().to_numpy(), current.dropna().to_numpy()
    if len(np.unique(b)) <= N_BINS:  # flags and small counts: one bin per value seen in the baseline, plus "other"
        values = np.unique(b)
        count = lambda x: np.append([(x == v).sum() for v in values], (~np.isin(x, values)).sum())
    else:  # decile bins from the baseline; the open ends catch anything outside its range
        edges = np.unique(np.quantile(b, np.linspace(0, 1, N_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        count = lambda x: np.histogram(x, bins=edges)[0]
    expected = np.append(count(b), base.isna().sum()) / len(base)
    actual = np.append(count(c), current.isna().sum()) / len(current)
    return expected, actual


def _category_counts(base: pd.Series, current: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    b, c = base.fillna(MISSING).astype(str), current.fillna(MISSING).astype(str)
    categories = sorted(set(b) | set(c))
    return (b.value_counts().reindex(categories, fill_value=0).to_numpy(),
            c.value_counts().reindex(categories, fill_value=0).to_numpy())


def numeric_drift(base: pd.Series, current: pd.Series) -> dict:
    if base.nunique() > N_BINS:  # flags and small counts are binned by value and need no smoothing
        smooth_base, smooth_current = _smooth(base, 1), _smooth(current, 2)
    else:
        smooth_base, smooth_current = base, current
    expected, actual = _numeric_shares(smooth_base, smooth_current)
    b, c = base.dropna(), current.dropna()  # KS uses the original values
    ks = stats.ks_2samp(b, c) if len(b) and len(c) else None
    return {"psi": _psi(expected, actual),
            "ks_stat": float(ks.statistic) if ks else np.nan, "ks_p": float(ks.pvalue) if ks else np.nan,
            "chi2_p": np.nan, "js": float(jensenshannon(expected, actual, base=2))}


def categorical_drift(base: pd.Series, current: pd.Series) -> dict:
    base_counts, current_counts = _category_counts(base, current)
    expected, actual = base_counts / base_counts.sum(), current_counts / current_counts.sum()
    table = np.vstack([base_counts, current_counts])
    table = table[:, table.sum(axis=0) > 0]
    chi2_p = float(stats.chi2_contingency(table)[1]) if table.shape[1] > 1 else np.nan
    return {"psi": _psi(expected, actual), "ks_stat": np.nan, "ks_p": np.nan, "chi2_p": chi2_p,
            "js": float(jensenshannon(expected, actual, base=2))}


def drift_table(base: pd.DataFrame, current: pd.DataFrame, kinds: dict[str, str], monitored: set[str]) -> pd.DataFrame:
    """One row per feature: PSI, KS (numeric), chi-square (categorical), JS distance, and whether it is monitored."""
    rows = []
    for feature, kind in kinds.items():
        measure = numeric_drift if kind == "numeric" else categorical_drift
        rows.append({"feature": feature, "kind": kind, "monitored": feature in monitored,
                     **measure(base[feature], current[feature])})
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> dict:
    """Headline numbers for one week and one baseline. Counts use the monitored features only."""
    monitored = table[table["monitored"] & (table["feature"] != "SCORE")]
    worst = monitored.loc[monitored["psi"].idxmax()]
    score = table[table["feature"] == "SCORE"]
    return {
        "n_monitored": len(monitored),
        "n_psi_moderate": int((monitored["psi"] >= PSI_MODERATE).sum()),
        "n_psi_major": int((monitored["psi"] >= PSI_MAJOR).sum()),
        "max_psi": round(float(worst["psi"]), 4), "max_psi_feature": worst["feature"],
        "mean_psi": round(float(monitored["psi"].mean()), 4),
        "n_all_psi_moderate": int((table[table["feature"] != "SCORE"]["psi"] >= PSI_MODERATE).sum()),
        "score_psi": round(float(score["psi"].iloc[0]), 4) if len(score) else np.nan,
        "score_ks": round(float(score["ks_stat"].iloc[0]), 4) if len(score) else np.nan,
    }
