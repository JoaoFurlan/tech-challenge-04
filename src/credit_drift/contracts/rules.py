"""Thresholds shared by the Pandera gate and the Great Expectations suite (docs/decisions.md, #1)."""
from credit_drift.data import DAYS_EMPLOYED_SENTINEL  # noqa: F401  (re-exported for the rules)

AGE_MIN_YEARS, AGE_MAX_YEARS = 18, 100
DAYS_PER_YEAR = 365.25
INCOME_CEILING = 10_000_000  # warning only: 3 rows of the training data exceed it
INCOME_CEILING_MOSTLY = 0.999
GENDER_DOMAIN = ["F", "M", "XNA"]
NULL_RATE_TOLERANCE = 0.05  # warn when a column's null share rises > 5 points over the training profile
UNSEEN_CATEGORY_MOSTLY = 0.999
WEEKLY_BATCH_ROWS = (4_000, 6_000)

# DAYS_BIRTH is negative: age 18 -> -6574.5 days, age 100 -> -36525 days.
DAYS_BIRTH_MIN = -AGE_MAX_YEARS * DAYS_PER_YEAR
DAYS_BIRTH_MAX = -AGE_MIN_YEARS * DAYS_PER_YEAR

KEY_COLUMNS = ("SK_ID_CURR", "AMT_INCOME_TOTAL", "DAYS_BIRTH")  # not-null is a blocking rule
