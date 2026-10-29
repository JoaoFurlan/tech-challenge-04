"""Fast inline gate: the blocking rules only. Fails fast and reports counts, never values."""
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from credit_drift.contracts import rules
from credit_drift.contracts.result import ContractResult, Failure
from credit_drift.data import DAYS_EMPLOYED_SENTINEL, TARGET


def _numeric_dtype() -> pa.Check:
    return pa.Check(lambda s: pd.api.types.is_numeric_dtype(s), name="numeric_dtype")


def _text_dtype() -> pa.Check:
    return pa.Check(lambda s: s.dtype == object or pd.api.types.is_string_dtype(s), name="text_dtype")


def build_scoring_schema(profile: dict) -> pa.DataFrameSchema:
    """Schema of a scoring batch: every profiled column present with the right kind, plus the rules."""
    columns: dict[str, pa.Column] = {}
    for name, info in profile["columns"].items():
        checks = [_numeric_dtype() if info["kind"] == "numeric" else _text_dtype()]
        columns[name] = pa.Column(checks=checks, nullable=True, required=True)

    def with_checks(name: str, *extra: pa.Check, nullable: bool, unique: bool = False) -> None:
        base = columns[name].checks
        columns[name] = pa.Column(checks=[*base, *extra], nullable=nullable, unique=unique, required=True)

    with_checks("SK_ID_CURR", nullable=False, unique=True)
    with_checks("AMT_INCOME_TOTAL", pa.Check(lambda s: s > 0, name="income_positive"), nullable=False)
    with_checks(
        "DAYS_BIRTH",
        pa.Check(lambda s: s.between(rules.DAYS_BIRTH_MIN, rules.DAYS_BIRTH_MAX), name="age_18_to_100"),
        nullable=False,
    )
    with_checks(
        "DAYS_EMPLOYED",
        pa.Check(lambda s: (s <= 0) | (s == DAYS_EMPLOYED_SENTINEL), element_wise=False, name="days_employed_valid"),
        nullable=True,
    )
    with_checks("AMT_CREDIT", pa.Check(lambda s: s > 0, name="credit_positive"), nullable=False)
    with_checks("AMT_ANNUITY", pa.Check(lambda s: s >= 0, name="annuity_non_negative"), nullable=True)
    return pa.DataFrameSchema(columns, strict=False)


def validate_scoring_batch(df: pd.DataFrame, schema: pa.DataFrameSchema) -> ContractResult:
    result = ContractResult(tool="pandera")
    try:
        schema.validate(df.drop(columns=[TARGET], errors="ignore"), lazy=True)
    except SchemaErrors as err:
        cases = err.failure_cases
        for (column, check), group in cases.groupby(["column", "check"], dropna=False):
            count = len(group) if group["index"].notna().any() else None
            result.blocking.append(
                Failure(rule=str(check), column=None if pd.isna(column) else str(column),
                        detail=f"{check} failed on {column}", count=count)
            )
    return result


def validate_labels(df: pd.DataFrame) -> ContractResult:
    """Label contract, applied when labels mature: TARGET present and binary, IDs unique."""
    schema = pa.DataFrameSchema(
        {
            "SK_ID_CURR": pa.Column(nullable=False, unique=True),
            TARGET: pa.Column(checks=[pa.Check(lambda s: s.isin([0, 1]), name="target_binary")], nullable=False),
        },
        strict=False,
    )
    result = ContractResult(tool="pandera")
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as err:
        for (column, check), group in err.failure_cases.groupby(["column", "check"], dropna=False):
            result.blocking.append(
                Failure(rule=str(check), column=str(column), detail=f"{check} failed", count=len(group)))
    return result
