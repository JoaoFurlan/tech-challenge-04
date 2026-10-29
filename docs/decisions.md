# Decisions from the first exploration

Each decision points back to an observation in `docs/reports/01_eda.html`. All six were
approved by the repo owner.

| # | Decision | Status |
|---|---|---|
| 1 | Validation rules (blocking vs warning) | Approved |
| 2 | Null, sentinel and outlier treatment | Approved |
| 3 | Feature variants A / B / C | Approved |
| 4 | Variables to drift | Approved |
| 5 | Slices to monitor | Approved |
| 6 | Drift accumulation | Approved |

## 1. Validation rules

Two contracts, because labels arrive late in production.

**Scoring-batch contract** (no `TARGET`), applied to every weekly batch:

- *Blocking* (the batch is rejected):
  - expected columns present and numeric columns numeric;
  - `SK_ID_CURR` not null and unique;
  - `AMT_INCOME_TOTAL` not null and > 0;
  - age 18-100 from `DAYS_BIRTH` (observed 20.5-69.1);
  - `DAYS_EMPLOYED` <= 0 or exactly 365243 (no other positive value exists);
  - `AMT_CREDIT` > 0 and `AMT_ANNUITY` >= 0 where present.
- *Warning* (logged and alerted, batch still scored):
  - `AMT_INCOME_TOTAL` <= 10M with a 99.9% tolerance. Approved as a warning: 3 training rows
    exceed 10M (max 117M), so a blocking rule would reject our own clean data, and one suspicious
    row should not stop a whole batch;
  - `CODE_GENDER` in {F, M, XNA};
  - per-column null rate within +-5 points of the training profile;
  - unseen categories and batch size.

**Label contract**, applied when labels mature (one week later): IDs match rows already scored,
`TARGET` in {0, 1}, no duplicates.

Implementation: Pandera runs the blocking set inline (fail fast). Great Expectations holds the
full suite with `mostly` thresholds for the warnings and produces the Data Docs. The week-9
corruption trips four blocking rules (null incomes, ages under 18, duplicate IDs, a text column).

## 2. Null, sentinel and outlier treatment

- `DAYS_EMPLOYED == 365243` becomes null plus a flag. The same 55,374 rows have
  `ORGANIZATION_TYPE == 'XNA'` and no `OCCUPATION_TYPE`: one "not employed" group seen three ways.
- `OWN_CAR_AGE` stays null (`FLAG_OWN_CAR` already explains it).
- Add a "housing block missing" indicator (the 47 housing columns vanish together and the missing
  group defaults more: 9.2% vs 7.0%).
- Keep nulls in `EXT_SOURCE_1/3` and add indicators (missing is informative).
- Income: log scale and winsorise at the 99.9th percentile of the training split (about 900k).
- Gender is not a model input.
- All statistics (medians, clip values, category lists) are fitted on the training split only and
  packaged as a scikit-learn pipeline inside the MLflow model, so training, retraining and
  serving transform identically. Tree models take nulls natively; logistic regression gets median
  fill plus the indicators.

## 3. Feature variants

- **A (minimal):** raw columns, median fill, one-hot encoding; the sentinel is deliberately left
  untreated.
- **B (engineered):** A plus decision 2, ratios (annuity/income, credit/income, credit/price,
  employed/age), `EXT_SOURCE` mean/min/product, and the indicators.
- **C (selected):** B minus near-duplicate columns (|corr| > 0.95; 42 pairs among the housing
  columns) and low-importance features. Selection happens inside the training folds only.
- Gender is excluded from all variants (discriminatory in credit decisions). The README should
  note that proxies can still leak it, which is why the gender slice is monitored.
- Grid: 3 variants x 3 families, Optuna inside each cell. Data split into train, calibration and
  final test; every run logs the dataset hash to MLflow.

## 4. Variables to drift

- Income and employment (weeks 3-4) and annuity (week 5), as in `CLAUDE.md`, applied in place
  with labels unchanged. State this in the README as measurement/level drift.
- Approved: a small shift in the category mix (for example more "Working", fewer
  "Pensioner") in weeks 4-5, done by resampling rows so each row keeps its own label. It
  represents the scenario's "new customer profiles" and exercises Evidently's chi-square/JS tests.
  First thing to cut if time is short.
- Concept drift (weeks 8 and 10-13): `EXT_SOURCE_2` relationship reversed and halved for a
  segment chosen in step 9, not identical to any monitored slice, sized so the overall AUC-PR
  falls by about 0.04.
- Finding to state in the demo: **data drift moves PSI/KS and the mean predicted probability but
  barely moves AUC-PR or Brier** (measured on a throwaway model: AUC-PR change under 0.002,
  predicted rate 8.05% -> 8.42%), while concept drift does the opposite.
- Consequence: retrain #1 (v2) should give a calibration-level gain, and recalibration (candidate
  A) should roughly match a full retrain (candidate B). `CLAUDE.md` needs updating once this is
  approved.

## 5. Slices to monitor

- 14 slices: `EXT_SOURCE_1/2/3`, income, age and employment, each below/above the median, plus
  gender F/M (fairness check for the governance section).
- Cut points are fixed from the champion's training window, so slice definitions do not drift.
- Approved: slice metrics use a rolling two-week window with bootstrap confidence intervals,
  because `EXT_SOURCE_1` high has about 47 defaulters per week and `EXT_SOURCE_3` high about 85
  (rule of thumb: about 100). The overall AUC-PR stays on a one-week window.

## 6. Drift accumulation

Level-based, not compounding: each week has an absolute state of the world (for example "income
x0.86 from week 4"), applied to fresh, non-overlapping rows. Gradual drift is a ramp of levels and
persistence is the same level repeated. Each week uses its own seed (base seed + week), so
adding a week never changes earlier ones, and `drift_spec.json` stays a simple table.

## 7. Train / calibration / test split (step 4, approved)

- **Pools:** random, seeded (42), disjoint by `SK_ID_CURR`. Reference 207,511 rows; Production pool
  100,000 rows (13 weeks x 5,000 = 65,000 plus headroom for the resampled category-mix drift and
  the retraining windows). Random because the data has no timestamps; the Production pool is not
  stratified, so weekly default rates fluctuate like real ones.
- **Reference parts**, stratified on `TARGET`: train 70% (145,257 rows), calibration 10% (20,751),
  test 20% (41,503); default rate 8.1% in each.
- **No fixed validation set:** tuning and feature selection use 3-fold stratified cross-validation
  inside train (a 20k-row validation set has AUC-PR noise of about 0.005, which 9 x 30 Optuna
  trials would overfit).
- **Selection:** rank the 9 tuned combinations by cross-validated AUC-PR (Brier and the cost metric
  as tie-breakers), calibrate the winner on the calibration part, evaluate on test once. Test never
  chooses a model.
- **Production weeks** act as the time-ordered test set. Weeks 1-2 (control) should match the test
  score within its confidence interval. Features and labels are stored in separate files
  (`week_NN_features.parquet`, `week_NN_labels.parquet`), as in production where labels arrive late.
- **Baselines:** the champion's train part is the PSI baseline; its test score is the performance
  baseline monitoring compares against.
- **Code:** `src/credit_drift/split.py`; outputs in `data/reference/` and `data/production/`
  (DVC-tracked, pushed to S3).

## 8. Model selection and tuning (step 6, approved)

- **Families:** logistic regression, LightGBM, XGBoost (compared in stages, see the amendment below). Deep
  learning is out: on tabular data of this size boosted trees match or beat it and are easier to
  explain (LGPD art. 20, to be checked in the README). Random forest dropped; CatBoost is an
  optional fourth family if time allows.
- **Selection rule** (applied at every stage): (1) rank the candidates by cross-validated AUC-PR;
  (2) shortlist every candidate within 1 standard error of the best; (3) pick the shortlisted candidate with the lowest expected
  cost, `min_t [C_FN*pi*(1-TPR(t)) + C_FP*(1-pi)*FPR(t)]` on out-of-fold predictions;
  (4) Brier is a sanity gate, not a ranking term: the calibrated champion must beat the
  constant-prior predictor (pi*(1-pi) = 0.074).
- **Cost ratio:** C_FN = 5 x C_FP, an assumption (a real lender would use its own figures);
  rankings are also reported at 3:1 and 10:1.
- **Search spaces** (Optuna TPE, seed 42, median pruner after the first fold; objective = mean
  3-fold AUC-PR on train, early stopping on log-loss inside each fold):
  - logistic regression: `C` log-uniform 1e-4..10, L2, no class weights, standardised inputs; 12 trials.
  - LightGBM: learning rate (log 0.01-0.2), leaves 15-127, min child samples 20-200, feature
    fraction 0.5-1, bagging fraction 0.5-1, L1/L2 (log 1e-3..10); up to 1,000 trees; 25 trials.
  - XGBoost: learning rate (log 0.01-0.2), depth 3-8, min child weight (log 1-50), subsample 0.5-1,
    column sample 0.5-1, lambda (log 0.01-20), gamma 0-5; `hist`; up to 1,000 trees; 25 trials.
- No class weights anywhere: they distort the probabilities that calibration and Brier rely on.
- **MLflow:** see the amendment below. Runs log params and metrics explicitly instead of
  `autolog()`, which would store hundreds of per-fold model artifacts.
- **Implementation note:** preprocessing does not depend on hyperparameters, so it is fitted
  once per fold and cached; each trial only fits the estimator. Variant C's feature selection is
  part of that per-fold preprocessing, so it never sees validation rows.

### Amendment: staged experiments (requested by the repo owner, supersedes the 9-cell grid)

Instead of tuning all 3 variants x 3 families, the work runs as three MLflow experiments, each
feeding the next, so every stage is a table that can be sorted and filtered and the best result
moves on:

1. **`feature_engineering`:** variants A, B, C, each with a linear (logistic regression) and a tree
   (LightGBM) probe at fixed default settings. The best run (selection rule) decides the variant.
2. **`model_selection`:** the three families on that variant, default settings. Every family within
   1 standard error of the best moves to tuning (so statistically tied families are both tuned).
3. **`hyperparameter_tuning`:** Optuna per shortlisted family, one flat run per trial plus one
   summary run for the refit.
4. **`champion`:** rank the tuned families by the same rule, calibrate on the calibration part,
   evaluate on test once, register `credit_scoring@champion`.

Every run carries the params `model`, `feature_variant` and `trial` (0 outside tuning), the tag
`stage`, the dataset hash, and the same metrics (`cv_auc_pr_mean`, `cv_auc_pr_se`, `oof_brier`,
`oof_cost_3/5/10`, ...). Run names are readable (`B / lightgbm`, `lightgbm trial 07`).

Trade-off, stated plainly: a staged search is greedy (it assumes the best variant at default
settings stays best after tuning), and it is far cheaper than the grid. Carrying every
within-1-SE candidate forward limits that risk.

Cloud: only the champion's run, its metrics and the registered model go to the AWS MLflow, plus a
summary table of the stages; the full trial history stays local (screenshots/exports for the README).
