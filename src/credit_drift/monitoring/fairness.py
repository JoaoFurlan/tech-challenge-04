"""A group-fairness check on the champion by gender (README, "Governança e LGPD").

  python -m credit_drift.monitoring.fairness          # writes docs/calibration/fairness.json

Gender is not a model input (training/features.py excludes it); it is kept in the data only so the check is possible. The
check is read on the Reference test part (labels known, never used for fitting). It reports, per group: share flagged as
high risk, recall and false-positive rate at the decision threshold, observed default rate, mean predicted probability
and AUC-PR. It does not prove fairness: other columns can proxy for gender, and the groups here are as the data records them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from credit_drift.data import TARGET
from credit_drift.split import REFERENCE_DIR

OUT = Path(__file__).resolve().parents[3] / "docs" / "calibration" / "fairness.json"
GROUPS = ("F", "M")


def group_table(y: np.ndarray, p: np.ndarray, group: np.ndarray, threshold: float) -> dict:
    flagged = p >= threshold
    out = {}
    for g in GROUPS:
        m = group == g
        yy, ff = y[m], flagged[m]
        out[g] = {"rows": int(m.sum()), "observed_default_rate": float(yy.mean()), "mean_predicted": float(p[m].mean()),
                  "flag_rate": float(ff.mean()), "recall": float(ff[yy == 1].mean()),
                  "false_positive_rate": float(ff[yy == 0].mean()), "auc_pr": float(average_precision_score(yy, p[m]))}
    a, b = out["F"], out["M"]
    out["gaps_M_minus_F"] = {k: round(b[k] - a[k], 4) for k in ("flag_rate", "recall", "false_positive_rate", "auc_pr")}
    out["flag_rate_ratio_M_over_F"] = round(b["flag_rate"] / a["flag_rate"], 3)
    return out


def main() -> None:
    from credit_drift.serving.registry import load_champion

    champion = load_champion()
    test = pd.read_parquet(REFERENCE_DIR / "test.parquet")
    keep = test["CODE_GENDER"].isin(GROUPS).to_numpy()  # a handful of rows carry another code; they are left out
    p = champion.model.predict_proba(test.drop(columns=TARGET))[:, 1]
    result = group_table(test[TARGET].to_numpy()[keep], p[keep], test["CODE_GENDER"].to_numpy()[keep], champion.threshold)
    result.update(model_version=champion.version, threshold=champion.threshold, rows_other_code=int((~keep).sum()))
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
