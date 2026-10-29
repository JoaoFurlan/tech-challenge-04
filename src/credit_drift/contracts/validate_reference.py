"""Validate the three Reference parts (train, calibration, test) against the full contract, labels included.

  uv run python -m credit_drift.contracts.validate_reference

Exits with status 1 if any part is rejected, so an orchestrator stops the pipeline before any model is trained.
"""
import sys

import pandas as pd

from credit_drift.contracts.check import check_batch
from credit_drift.split import REFERENCE_DIR

PARTS = ("train", "calibration", "test")


def main() -> int:
    passed = True
    for part in PARTS:
        report = check_batch(pd.read_parquet(REFERENCE_DIR / f"{part}.parquet"), f"reference_{part}", with_labels=True)
        status = "PASSED" if report.passed else "REJECTED"
        print(f"reference_{part}: {status} | pandera blocking={len(report.pandera.blocking)} "
              f"gx blocking={len(report.gx.blocking)} warnings={len(report.gx.warnings)}", flush=True)
        for failure in report.pandera.blocking + report.gx.blocking:
            print(f"   {failure.column}: {failure.rule} ({failure.count} rows)")
        passed &= report.passed
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
