"""One entry point: Pandera first (fail fast), Great Expectations always (audit trail + warnings)."""
from dataclasses import dataclass

import pandas as pd

from credit_drift.contracts.gx_suite import validate_with_gx
from credit_drift.contracts.pandera_gate import (
    build_scoring_schema,
    validate_labels,
    validate_scoring_batch,
)
from credit_drift.contracts.profile import load_profile
from credit_drift.contracts.result import ContractResult


@dataclass
class BatchReport:
    batch_name: str
    pandera: ContractResult
    gx: ContractResult

    @property
    def passed(self) -> bool:
        return self.pandera.passed and self.gx.passed


def check_batch(df: pd.DataFrame, batch_name: str, *, with_labels: bool = False,
                expected_rows: tuple[int, int] | None = None, profile: dict | None = None) -> BatchReport:
    profile = profile or load_profile()
    pandera_result = validate_scoring_batch(df, build_scoring_schema(profile))
    if with_labels:
        pandera_result.blocking += validate_labels(df).blocking
    gx_result = validate_with_gx(
        df, batch_name, profile, suite_name="training_data" if with_labels else "scoring_batch",
        with_labels=with_labels, expected_rows=expected_rows)
    return BatchReport(batch_name, pandera_result, gx_result)
