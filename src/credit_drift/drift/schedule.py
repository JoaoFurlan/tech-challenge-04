"""The 13-week drift schedule (docs/decisions.md, #10). Level-based: each week is an absolute state of the world,
not a change on top of the previous week, so adding a week never alters earlier ones."""
from dataclasses import dataclass

SEED = 42
WEEK_ROWS = 5_000
N_WEEKS = 13
LABEL_MODEL_SEED = 7  # independent of the champion's seed on purpose

# Category-mix drift: log-weights at full strength that tilt a week's category shares (rows are resampled,
# so every row keeps its own label: pure covariate shift).
MIX_COLUMN = "NAME_INCOME_TYPE"
MIX_LOG_WEIGHTS = {"Working": 0.12, "Pensioner": -0.40}

# Concept-drift segment, defined on the ORIGINAL (pre-drift) features: age <= 45 and income <= the training median.
SEGMENT_MAX_AGE_YEARS = 45
EXT_REVERSAL_FACTOR = -0.5  # EXT_SOURCE_2's linear coefficient becomes -0.5 x itself: reversed and halved


@dataclass(frozen=True)
class WeekState:
    week: int
    income_mult: float = 1.0
    employment_mult: float = 1.0
    annuity_mult: float = 1.0
    mix_level: float = 0.0  # 0 = original mix, 1 = full category-mix shift
    concept_drift: bool = False
    corrupted: bool = False
    note: str = ""


_REGIME = {"income_mult": 0.86, "employment_mult": 0.85, "annuity_mult": 1.15, "mix_level": 1.0}

SCHEDULE: tuple[WeekState, ...] = (
    WeekState(1, note="control"),
    WeekState(2, note="control"),
    WeekState(3, income_mult=0.90, employment_mult=0.92, note="gradual covariate shift"),
    WeekState(4, income_mult=0.86, employment_mult=0.85, mix_level=0.5,
              note="gradual covariate shift; the category mix starts to move"),
    WeekState(5, **_REGIME, note="abrupt shift: annuity shock"),
    WeekState(6, **_REGIME, note="new normal"),
    WeekState(7, **_REGIME, note="new normal"),
    WeekState(8, **_REGIME, concept_drift=True, note="concept drift starts"),
    WeekState(9, **_REGIME, concept_drift=True, corrupted=True,
              note="corrupted batch, blocked by the contract; labels never arrive"),
    *(WeekState(w, **_REGIME, concept_drift=True, note="concept drift persists") for w in range(10, 14)),
)
