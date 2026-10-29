"""Contract demo: training data and a clean batch pass, a corrupted batch is rejected by both tools.

Run: uv run python -m credit_drift.contracts.demo
Builds the Great Expectations Data Docs and copies them to docs/reports/data_docs (the audit trail).
"""
import shutil

import pandas as pd

from credit_drift.contracts.check import BatchReport, check_batch
from credit_drift.contracts.corruption import corrupt_batch
from credit_drift.contracts.gx_suite import GX_ROOT, export_data_docs
from credit_drift.contracts.rules import WEEKLY_BATCH_ROWS as WEEKLY_ROWS
from credit_drift.data import TARGET
from credit_drift.split import PRODUCTION_DIR, REFERENCE_DIR

DEMO_SEED = 9
DOCS_DIR = GX_ROOT / "docs" / "reports" / "data_docs"


def _summary(report: BatchReport) -> str:
    verdict = "PASSED" if report.passed else "REJECTED"
    lines = [f"{report.batch_name}: {verdict}"]
    for tool in (report.pandera, report.gx):
        rules = sorted({(f.column, f.count) for f in tool.blocking}, key=str)
        lines.append(f"  {tool.tool:<18} blocking={len(tool.blocking)} warnings={len(tool.warnings)}  {rules if rules else ''}")
    return "\n".join(lines)


def main() -> None:
    # Start from empty validation results so the exported Data Docs show exactly this demo.
    shutil.rmtree(GX_ROOT / "gx" / "uncommitted", ignore_errors=True)

    train = pd.read_parquet(REFERENCE_DIR / "train.parquet")
    pool = pd.read_parquet(PRODUCTION_DIR / "pool.parquet")
    clean = pool.sample(5_000, random_state=DEMO_SEED).reset_index(drop=True).drop(columns=TARGET)
    corrupted, spec = corrupt_batch(clean, seed=DEMO_SEED)

    reports = [
        check_batch(train, "1_reference_train", with_labels=True),
        check_batch(clean, "2_clean_weekly_batch", expected_rows=WEEKLY_ROWS),
        check_batch(corrupted, "3_corrupted_weekly_batch", expected_rows=WEEKLY_ROWS),
    ]
    for report in reports:
        print(_summary(report))
    print("injected:", spec)

    export_data_docs(DOCS_DIR)
    print(f"Data Docs exported to {DOCS_DIR}")
    assert reports[0].passed and reports[1].passed and not reports[2].passed


if __name__ == "__main__":
    main()
