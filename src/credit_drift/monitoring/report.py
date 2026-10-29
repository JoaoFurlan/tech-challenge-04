"""Evidently HTML report for one week against one baseline: PSI report plus KS (numeric) / chi-square (categorical)."""
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

REFERENCE_SAMPLE = 20_000  # the baseline is subsampled for the HTML report only; the tables use all of it
# Evidently calls a dataset "drifted" when this share of columns drifted. Its default (0.5) can never fire for a few
# drifted inputs out of 32, so for the PSI preset the banner follows our alert rule: 2 or more monitored features at
# PSI >= 0.10 (2 of 32 = 6%). The KS / chi-square preset keeps the default: p-values flag 4 of 32 columns even in the
# clean control week, which is exactly why they are evidence and not the alert signal.
PSI_DRIFT_SHARE = 0.05


def build_report(current: pd.DataFrame, baseline: pd.DataFrame, kinds: dict[str, str], title: str, path: Path) -> Path:
    """`current` and `baseline` hold the monitored columns plus SCORE; `kinds` maps column -> numeric/categorical."""
    columns = [c for c in current.columns if c in kinds or c == "SCORE"]
    definition = DataDefinition(
        numerical_columns=[c for c in columns if c == "SCORE" or kinds[c] == "numeric"],
        categorical_columns=[c for c in columns if c != "SCORE" and kinds[c] == "categorical"])
    reference = baseline[columns].sample(min(len(baseline), REFERENCE_SAMPLE), random_state=0)
    report = Report([DataDriftPreset(method="psi", drift_share=PSI_DRIFT_SHARE),
                    DataDriftPreset(num_method="ks", cat_method="chisquare")])
    snapshot = report.run(Dataset.from_pandas(current[columns], data_definition=definition),
                          Dataset.from_pandas(reference, data_definition=definition), name=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return path
