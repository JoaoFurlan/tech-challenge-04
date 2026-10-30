"""The data a retrain may use: the Reference train part plus the production weeks whose labels have arrived.

Respecting the label delay is the point: a retrain run after week N's batch can only use weeks up to N-1 (labels arrive
a week late), and a week that never got labels (the blocked week 9) is simply absent. The last two labelled weeks are
the "recent" slice: split at random into a part that joins the training rows, a part that fits the calibrator, and a
part that is held out for the promotion gate, so no row serves two purposes (docs/decisions.md, #18).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_drift.data import ID_COL, TARGET
from credit_drift.split import PRODUCTION_DIR

RECENT_WEEKS = 2
FIT_SHARE, CALIBRATION_SHARE, GATE_SHARE = 0.30, 0.30, 0.40  # of the recent slice
HALF_LIFE_WEEKS = 2.0  # a labelled week counts half as much for every two weeks of age
PRODUCTION_WEIGHT_SHARE = 0.5  # production rows carry as much total weight as the Reference rows


def labelled_weeks(through_week: int) -> list[int]:
    folder = PRODUCTION_DIR / "weeks"
    return [w for w in range(1, through_week + 1)
            if (folder / f"week_{w:02d}_features.parquet").exists() and (folder / f"week_{w:02d}_labels.parquet").exists()]


def load_week(week: int) -> pd.DataFrame:
    folder = PRODUCTION_DIR / "weeks"
    features = pd.read_parquet(folder / f"week_{week:02d}_features.parquet")
    labels = pd.read_parquet(folder / f"week_{week:02d}_labels.parquet")
    return features.merge(labels, on=ID_COL, how="inner").assign(week=week)


@dataclass
class Window:
    through_week: int
    weeks: list[int]
    recent_weeks: list[int]
    fit: pd.DataFrame  # Reference train + earlier weeks + the fit part of the recent slice (with a `week` column)
    calibration: pd.DataFrame
    gate: pd.DataFrame
    weights: np.ndarray  # sample weight of every row in `fit`

    def describe(self) -> dict:
        return {"through_week": self.through_week, "weeks": self.weeks, "recent_weeks": self.recent_weeks,
                "rows": {"fit": len(self.fit), "calibration": len(self.calibration), "gate": len(self.gate)},
                "defaulters": {"calibration": int(self.calibration[TARGET].sum()), "gate": int(self.gate[TARGET].sum())}}


def sample_weights(weeks: pd.Series, reference_rows: int, latest_week: int) -> np.ndarray:
    """Reference rows weigh 1 each; production rows share the same total, split by recency (week 0 = Reference)."""
    weights = np.ones(len(weeks))
    production = weeks.to_numpy() > 0
    recency = 0.5 ** ((latest_week - weeks.to_numpy()[production]) / HALF_LIFE_WEEKS)
    weights[production] = recency / recency.sum() * reference_rows * PRODUCTION_WEIGHT_SHARE / (1 - PRODUCTION_WEIGHT_SHARE)
    return weights


def build_window(through_week: int, reference_train: pd.DataFrame, seed: int = 42) -> Window:
    weeks = labelled_weeks(through_week)
    if len(weeks) < RECENT_WEEKS + 1:
        raise SystemExit(f"only weeks {weeks} have labels through week {through_week}: too few to retrain")
    recent_weeks, earlier_weeks = weeks[-RECENT_WEEKS:], weeks[:-RECENT_WEEKS]
    recent = pd.concat([load_week(w) for w in recent_weeks], ignore_index=True)
    strata = recent[TARGET].astype(str) + recent["week"].astype(str)
    rest, gate = train_test_split(recent, test_size=GATE_SHARE, stratify=strata, random_state=seed)
    fit_part, calibration = train_test_split(
        rest, test_size=CALIBRATION_SHARE / (FIT_SHARE + CALIBRATION_SHARE),
        stratify=(rest[TARGET].astype(str) + rest["week"].astype(str)), random_state=seed)
    earlier = [load_week(w) for w in earlier_weeks]
    fit = pd.concat([reference_train.assign(week=0), *earlier, fit_part], ignore_index=True)
    return Window(through_week, weeks, recent_weeks, fit, calibration.reset_index(drop=True),
                  gate.reset_index(drop=True), sample_weights(fit["week"], len(reference_train), weeks[-1]))
