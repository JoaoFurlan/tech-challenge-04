"""Which features the weekly report watches. Chosen by a rule on the Reference only, never from the injected drift.

Applied to the model's own input frame (raw columns plus the features the champion engineers).
Rule: the 20 numeric columns with the highest univariate AUC against the target (de-duplicated: a column is skipped
when it correlates above 0.9 with one already chosen), the four loan-size fundamentals every lender watches, the
champion's financial ratios (debt-to-income and friends), and the 5 categorical columns most associated with the
target (Cramer's V). Everything else is still measured (see `drift_table`) but kept out of the HTML report so it
stays readable.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from credit_drift.data import DAYS_EMPLOYED_SENTINEL, TARGET

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "monitored_features.json"
TOP_NUMERIC, TOP_CATEGORICAL, CORR_LIMIT = 20, 5, 0.9
LOAN_SIZE = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
RATIOS = ["ANNUITY_INCOME", "CREDIT_INCOME", "CREDIT_GOODS", "ANNUITY_CREDIT", "EMPLOYED_AGE"]
NOT_MONITORED = {"SK_ID_CURR", "CODE_GENDER", TARGET}  # the ID, and gender (not a model input; a fairness slice)


def _univariate_auc(train: pd.DataFrame, numeric: list[str]) -> pd.Series:
    scores = {}
    for col in numeric:
        values = train[col].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
        known = values.notna()
        if known.sum() > 1000 and values[known].nunique() > 1:
            auc = roc_auc_score(train.loc[known, TARGET], values[known])
            scores[col] = max(auc, 1 - auc)
    return pd.Series(scores).sort_values(ascending=False)


def _cramers_v(train: pd.DataFrame, col: str) -> float:
    table = pd.crosstab(train[col].fillna(MISSING_TOKEN), train[TARGET])
    chi2 = stats.chi2_contingency(table)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


MISSING_TOKEN = "__missing__"


def select_monitored(train: pd.DataFrame, kinds: dict[str, str]) -> dict:
    numeric = [c for c, k in kinds.items() if k == "numeric" and c not in NOT_MONITORED]
    categorical = [c for c, k in kinds.items() if k == "categorical" and c not in NOT_MONITORED]
    sample = train.sample(min(len(train), 60_000), random_state=0)

    auc = _univariate_auc(sample, numeric)
    candidates = list(auc.index[:80])
    corr = sample[candidates].replace(DAYS_EMPLOYED_SENTINEL, np.nan).corr().abs()
    chosen: list[str] = []
    for col in candidates:
        if len(chosen) == TOP_NUMERIC:
            break
        if all(not corr.loc[col, kept] > CORR_LIMIT for kept in chosen):
            chosen.append(col)

    cramers = pd.Series({c: _cramers_v(sample, c) for c in categorical}).sort_values(ascending=False)
    return {
        "rule": __doc__.strip().split("\n\n")[1].replace("\n", " "),
        "numeric_by_univariate_auc": {c: round(float(auc[c]), 4) for c in chosen},
        "loan_size": LOAN_SIZE,
        "financial_ratios": RATIOS,
        "categorical_by_cramers_v": {c: round(float(cramers[c]), 4) for c in cramers.index[:TOP_CATEGORICAL]},
    }


def monitored_columns(config: dict) -> list[str]:
    cols = [*config["numeric_by_univariate_auc"], *config["loan_size"], *config["financial_ratios"],
            *config["categorical_by_cramers_v"]]
    return list(dict.fromkeys(cols))


def save_config(config: dict, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1), encoding="utf-8")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    from credit_drift.split import REFERENCE_DIR
    from credit_drift.training.features import FeatureEngineer

    train = pd.read_parquet(REFERENCE_DIR / "train.parquet")
    frame = FeatureEngineer().fit(train).transform(train.drop(columns=TARGET)).assign(**{TARGET: train[TARGET]})
    kinds = {c: ("categorical" if frame[c].dtype == object else "numeric") for c in frame.columns if c != TARGET}
    config = select_monitored(frame, kinds)
    save_config(config)
    print(f"{len(monitored_columns(config))} monitored features -> {CONFIG_PATH}")
    print(json.dumps({k: v for k, v in config.items() if k != "rule"}, indent=1))
