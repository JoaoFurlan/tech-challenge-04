"""Great Expectations suite: blocking rules (CRITICAL) plus warnings, with Data Docs as the audit trail."""
import contextlib
import shutil
from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
from great_expectations.core.run_identifier import RunIdentifier
from great_expectations.exceptions import DataContextError
from great_expectations.expectations.row_conditions import Column

from credit_drift.contracts import rules
from credit_drift.contracts.result import ContractResult, Failure
from credit_drift.data import DAYS_EMPLOYED_SENTINEL, TARGET

GX_ROOT = Path(__file__).resolve().parents[3]  # the file context lives in <repo>/gx (git-ignored)
DATA_DOCS_SITE = GX_ROOT / "gx" / "uncommitted" / "data_docs" / "local_site"
NUMERIC_TYPES = ["int64", "float64", "Int64", "Float64", "int32", "float32"]
TEXT_TYPES = ["object", "str", "string"]
# Never ship offending values into results or Data Docs: counts only (LGPD).
RESULT_FORMAT = {"result_format": "BASIC", "partial_unexpected_count": 0}
BLOCKING, WARNING = "critical", "warning"


def get_context():
    return gx.get_context(mode="file", project_root_dir=str(GX_ROOT))


def build_suite(context, name: str, profile: dict, *, with_labels: bool = False,
                expected_rows: tuple[int, int] | None = None):
    """(Re)create the named suite from the rules in `rules.py` and the training profile."""
    with contextlib.suppress(DataContextError):  # first run: nothing to delete
        context.suites.delete(name)
    suite = context.suites.add(gx.ExpectationSuite(name=name))
    add = suite.add_expectation
    cols = profile["columns"]

    # --- blocking: structure and types
    add(gxe.ExpectTableColumnsToMatchSet(column_set=list(cols), exact_match=False, severity=BLOCKING))
    for col, info in cols.items():
        types = NUMERIC_TYPES if info["kind"] == "numeric" else TEXT_TYPES
        add(gxe.ExpectColumnValuesToBeInTypeList(column=col, type_list=types, severity=BLOCKING))

    # --- blocking: the strict rules
    add(gxe.ExpectColumnValuesToNotBeNull(column="SK_ID_CURR", severity=BLOCKING))
    add(gxe.ExpectColumnValuesToBeUnique(column="SK_ID_CURR", severity=BLOCKING))
    add(gxe.ExpectColumnValuesToNotBeNull(column="AMT_INCOME_TOTAL", severity=BLOCKING))
    add(gxe.ExpectColumnValuesToBeBetween(column="AMT_INCOME_TOTAL", min_value=0, strict_min=True, severity=BLOCKING))
    add(gxe.ExpectColumnValuesToNotBeNull(column="DAYS_BIRTH", severity=BLOCKING))
    add(gxe.ExpectColumnValuesToBeBetween(
        column="DAYS_BIRTH", min_value=rules.DAYS_BIRTH_MIN, max_value=rules.DAYS_BIRTH_MAX, severity=BLOCKING))
    add(gxe.ExpectColumnValuesToBeBetween(
        column="DAYS_EMPLOYED", max_value=0, severity=BLOCKING,
        row_condition=Column("DAYS_EMPLOYED") != DAYS_EMPLOYED_SENTINEL))
    add(gxe.ExpectColumnValuesToNotBeNull(column="AMT_CREDIT", severity=BLOCKING))
    add(gxe.ExpectColumnValuesToBeBetween(column="AMT_CREDIT", min_value=0, strict_min=True, severity=BLOCKING))
    add(gxe.ExpectColumnValuesToBeBetween(column="AMT_ANNUITY", min_value=0, severity=BLOCKING))
    if with_labels:  # label contract: only once labels have matured
        add(gxe.ExpectColumnValuesToNotBeNull(column=TARGET, severity=BLOCKING))
        add(gxe.ExpectColumnValuesToBeInSet(column=TARGET, value_set=[0, 1], severity=BLOCKING))

    # --- warnings: logged and alerted, the batch is still scored
    add(gxe.ExpectColumnValuesToBeBetween(
        column="AMT_INCOME_TOTAL", max_value=rules.INCOME_CEILING, mostly=rules.INCOME_CEILING_MOSTLY, severity=WARNING))
    add(gxe.ExpectColumnValuesToBeInSet(column="CODE_GENDER", value_set=rules.GENDER_DOMAIN, severity=WARNING))
    for col, info in cols.items():
        if col not in rules.KEY_COLUMNS:
            mostly = max(0.0, 1 - (info["null_share"] + rules.NULL_RATE_TOLERANCE))
            add(gxe.ExpectColumnValuesToNotBeNull(column=col, mostly=mostly, severity=WARNING))
        if info["kind"] == "categorical" and col != "CODE_GENDER":
            add(gxe.ExpectColumnValuesToBeInSet(
                column=col, value_set=info["categories"], mostly=rules.UNSEEN_CATEGORY_MOSTLY, severity=WARNING))
    if expected_rows:
        add(gxe.ExpectTableRowCountToBeBetween(min_value=expected_rows[0], max_value=expected_rows[1], severity=WARNING))
    return suite


def validate_with_gx(df: pd.DataFrame, batch_name: str, profile: dict, *, suite_name: str = "scoring_batch",
                     with_labels: bool = False, expected_rows: tuple[int, int] | None = None) -> ContractResult:
    context = get_context()
    suite = build_suite(context, suite_name, profile, with_labels=with_labels, expected_rows=expected_rows)
    source = context.data_sources.add_or_update_pandas("pandas")
    try:
        asset = source.get_asset("batch")
    except LookupError:
        asset = source.add_dataframe_asset("batch")
    batch_definition = asset.add_batch_definition_whole_dataframe(f"whole_{suite_name}")
    with contextlib.suppress(DataContextError):
        context.validation_definitions.delete(suite_name)
    definition = context.validation_definitions.add(
        gx.ValidationDefinition(name=suite_name, data=batch_definition, suite=suite))
    raw = definition.run(
        batch_parameters={"dataframe": df}, result_format=RESULT_FORMAT, run_id=RunIdentifier(run_name=batch_name))

    result = ContractResult(tool="great_expectations")
    for item in raw.results:
        if item.success:
            continue
        config = item.expectation_config
        failure = Failure(
            rule=config.type, column=config.kwargs.get("column"), detail=config.type,
            count=item.result.get("unexpected_count"))
        if str(config.severity).split(".")[-1].lower() == BLOCKING:
            result.blocking.append(failure)
        else:
            result.warnings.append(failure)
    context.build_data_docs()
    return result


def export_data_docs(destination: Path) -> Path:
    """Copy the built Data Docs site to a committed location (docs/reports/...)."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(DATA_DOCS_SITE, destination)
    return destination
