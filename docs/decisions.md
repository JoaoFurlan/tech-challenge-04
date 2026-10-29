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

## 9. Decision threshold (agreed in discussion; implemented with the serving app, step 11)

- AUC-PR and AUC-ROC are threshold-free; a threshold only matters when deciding (approve / reject).
- It is derived on the training side, from the lender's risk appetite: with calibrated probabilities
  and the 5:1 cost ratio, flag when p > C_FP / (C_FP + C_FN) = 1/6 (about 0.167); 3:1 gives 0.25,
  10:1 about 0.09. It is never tuned on the test set; test only reports the outcome at the frozen
  threshold (cost, flag rate, defaulters caught).
- Production applies the same frozen number. Drift changes what it does (flag rate, predicted vs
  observed default rate), which the dashboard shows. Retraining or recalibration recomputes it.
- The threshold is stored with the registered model as a versioned parameter next to the cost
  ratio, so the serving app returns the probability and the decision, and any change leaves a
  trace in the registry. The 5:1 ratio is an assumption a real lender would replace.

## 10. Drift simulator (step 9)

Code in `src/credit_drift/drift/`; run with `uv run python -m credit_drift.drift.simulate`. Output in
`data/production/weeks/` (DVC), ground truth in `data/production/drift_spec.json` (copy in
`docs/reports/drift_spec.json`, aggregates only). Features and labels are separate files
(`week_NN_features.parquet`, `week_NN_labels.parquet`) because labels arrive late in production.

- **Level-based schedule:** each week is an absolute state applied to fresh, non-overlapping rows.
  Weeks 1-2 control; 3: income x0.90, employment x0.92; 4: income x0.86, employment x0.85, category mix
  at half strength; 5-13: income x0.86, employment x0.85, annuity x1.15, full mix shift; concept drift
  from week 8; week 9 is the corrupted batch and its labels never arrive. Each week has its own seed
  (base + week); adding a week never changes earlier ones (tested).
- **Numeric drift** changes values in place with labels untouched (measurement/level drift); the
  365243 sentinel in `DAYS_EMPLOYED` is a code and is left alone.
- **Category-mix drift** (`NAME_INCOME_TYPE`): each category gets its exact target count, drawn from the
  unused rows, so rows keep their labels (pure covariate shift). Pensioners go from 18.2% to 14.8% (week
  4) and 12.0% (weeks 5-13), Working from 51.6% to about 58%. A first version used weighted draws from the
  shrinking pool and let the mix creep back (pensioners 15.9% by week 13); exact counts fixed that.
- **Concept drift** (weeks 8 and 10-13): for the segment *age <= 45 and income <= the training median
  (144,000)*, defined on the original features (about 26% of rows, 1,369 of 5,000 in week 8), labels are
  redrawn from a label model with EXT_SOURCE_2's linear coefficient changed from -0.46 to +0.23 (reversed
  and halved); the intercept is shifted so the segment keeps its model-implied default rate (concept
  drift, not label shift). Labels outside the segment stay real. Segment and size were chosen by
  experiment so overall AUC-PR falls by about 0.04 against the week-5 level.
- **Label model:** an independent LightGBM (different seed and settings), Platt-calibrated on the
  calibration part. Measured on the champion: redrawing all labels from it moves AUC-PR by +0.002
  (0.2526 vs 0.2509 real), so it adds no confound. Rejected: a logistic regression (redrawing alone
  cuts the champion's AUC-PR from 0.238 to as low as 0.162) and the champion's own probabilities
  (circular, flattering: 0.2589).
- **Measured effect on the champion** (whole pool, labels unchanged for covariate weeks): weeks 3-4
  (income, employment) leave AUC-PR at 0.251; the week-5 annuity shock lowers it to 0.238 (-0.013) and
  AUC-ROC from 0.766 to 0.756. **This corrects the earlier "< 0.002" estimate**, which came from a
  throwaway numeric-only model; the real champion uses annuity ratios, so a constant annuity shift moves
  rows across its splits. Mean predicted default rate rises from 0.082 to about 0.088 while the observed
  rate stays near 0.083.
- **Simulated weeks 8-13 on the champion:** AUC-PR averages 0.194 (weeks 1-7 about 0.26) and AUC-ROC
  falls from 0.76-0.78 to 0.72-0.74.
- **Noise to plan for:** one 5,000-row week has an AUC-PR confidence interval of about +-0.04, so a
  single week cannot reliably show a 0.04 drop (AUC-ROC is steadier). This is why slice metrics use a
  two-week window and why detector thresholds must be tuned offline on more than one simulation.

## 11. Drift detection (step 10)

Code in `src/credit_drift/monitoring/`; run with `uv run python -m credit_drift.monitoring.run_drift`.
Showcase reports (weeks 1, 5, 8) and the summary table are in `docs/reports/drift/`; the full set of 12
HTML reports (about 7.8 MB each) is regenerated locally into `reports/drift/` and not committed.

- **What is monitored:** the champion's own input frame (the raw columns plus the features it engineers),
  taken from its pipeline, so a new champion is monitored on what it actually consumes. Every column
  gets PSI, KS (numeric), chi-square (categorical) and JS distance in `data/monitoring/drift_features.parquet`,
  plus the model **SCORE** (its distribution against the held-out test scores, not training scores, which
  are overfit). The ID and gender are left out (gender is a fairness slice, not an input).
- **Monitored subset** (HTML report and alert counts), by a rule on the Reference only, never from the
  injected drift: the 20 numeric columns with the highest univariate AUC (de-duplicated at |corr| 0.9),
  the four loan-size fundamentals (income, credit, annuity, goods price), the champion's five financial
  ratios (debt-to-income and friends), and the 5 categoricals most associated with default (Cramer's V):
  32 features, in `config/monitored_features.json`.
- **PSI drives alerts, p-values are evidence.** With thousands of rows a p-value flags almost any
  difference, while PSI measures how far a distribution moved. Bands: < 0.10 none, 0.10-0.25 moderate,
  >= 0.25 major. These are provisional until step 14 freezes thresholds tuned on two simulations.
- **PSI smoothing:** 61% of incomes sit on 8 round numbers, so decile-based PSI jumped around (0.40 for a
  10% shift, 0.19 for a 14% one) when a spike straddled a bin edge. Continuous columns get a seeded +-2%
  jitter before binning (KS uses the original values); flags are binned by value. Missing values have
  their own bin. Evidently's own PSI in the HTML report uses its binning, so values can differ slightly.
- **Two baselines:** the original Reference (informational) and the champion's training window (alerts).
  For the first champion they are the same data, so one table is computed and reused; the second
  re-points to the new training window on every promotion.
- **A batch that fails the contract is not scored and gets no report** (week 9), as in the production loop.
- **Detection timeline on the simulated weeks (monitored features at PSI >= 0.10):**

  | Weeks | Result |
  |---|---|
  | 1-2 | 0 (maximum PSI 0.012, score PSI <= 0.004): the control weeks stay quiet |
  | 3 | 0: income 0.086, still building |
  | 4 | 2: income 0.156, annuity/income 0.10 |
  | 5-8, 10-13 | 3, two of them major: annuity/credit 0.80-0.94, annuity/income 0.27-0.34, income 0.15-0.18 |
  | 9 | blocked by the contract |

  Score drift is small (PSI 0.02, KS 0.06-0.07, mean 0.082 -> 0.088). The category mix is a good
  illustration of the tests' roles: `NAME_INCOME_TYPE` has PSI 0.031 (low) and JS 0.075 but a chi-square
  p-value near 1e-26. Weeks 8 and 10-13 look identical to weeks 5-7: concept drift moves no input, so only
  performance (with labels) can reveal it.
- **Evidently's "dataset drift" banner.** Evidently calls a dataset drifted when 50% of the columns drifted,
  a default that cannot fire for a few drifted inputs out of 32 (it said "not drifted" in every week while the
  per-column results showed drift). For the PSI preset the banner now follows our alert rule, `drift_share` 0.05
  (2 or more monitored features at PSI >= 0.10): control weeks "not detected", weeks 4-13 "detected". The KS /
  chi-square preset keeps the default 0.5 on purpose: p-values flag 4 of 32 columns even in the clean control
  week (12%) and 14 of 32 in weeks 5 and 8 (44%), so they cannot drive an alert.
- **Raw annuity alone would have missed the abrupt shock** (PSI 0.08, KS 0.13); the ratios catch it, which
  is why the champion's ratios are always monitored.

## 12. Model server (step 11)

Code in `src/credit_drift/serving/` (`registry.py` resolves the champion, `app.py` is the FastAPI app); image
`docker/Dockerfile.serving`; compose service `serving` on `127.0.0.1:8010` (8000 is taken by another local project).

- **Endpoints:** `POST /predict` (raw applicant records in, probability and `high_risk` flag out), `GET /metrics`
  (Prometheus), `POST /reload`, plus `/health` (alive), `/ready` (503 until a model is loaded) and `/model`.
- **The contract guards the door:** every request goes through the same Pandera rules as the weekly batches.
  A violation is a 422 listing column, rule and row count, never the offending values (tested). Extra fields
  are ignored; JSON nulls become NaN; a numeric column carrying strings such as "123" is rejected as malformed.
  Up to 10,000 records per request.
- **Registry, not files:** the server looks up `models:/credit_scoring@champion`, then loads that exact version
  (so the model and its metadata always match). `/reload` repeats this, which is how a promotion reaches a
  running server. A failed reload keeps serving the previous model (502, tested). `/reload` requires the
  `X-Reload-Token` header when `RELOAD_TOKEN` is set: the compose default is a dev value, any shared host must
  set a real secret. If the registry is down at startup the server stays alive but not ready, and `/reload`
  fixes it later.
- **Decision threshold (agreed in #9):** stored as tags `cost_ratio` and `decision_threshold` on the registered
  model version (`uv run python -m credit_drift.training.run tag` backfilled version 1; new champions get them
  at registration). The server reads them, so a promotion with another threshold changes decisions without a
  code change (tested). `high_risk` is a flag for review, not an automated rejection: the final decision should
  allow human review (to be detailed in the governance section).
- **Metrics:** `predict_requests_total{outcome}`, `predict_rows_total`, `predict_high_risk_total`,
  `predict_latency_seconds`, `prediction_score` (histogram of predicted probabilities, the push-free score drift
  signal), `model_info{name,version,alias}`, `model_decision_threshold`, `model_loaded_timestamp_seconds`,
  `model_reloads_total{outcome}`.
- **Measured through the container:** 500 rows in 178 ms and a whole 5,000-row week in 0.68 s; probabilities
  match the champion run locally to 5e-7 (the API rounds to 6 decimals).
- **Image:** `python:3.14-slim`, multi-stage, 827 MB. scipy, pandas, numpy and scikit-learn are about 300 MB of
  the venv; `mlflow-skinny` and its dependencies about 75 MB. Dependencies follow the champion's family: it is
  LightGBM, so `lightgbm` (plus `libgomp1`, about 0.3 MB of system library) is installed and xgboost is not;
  promoting an XGBoost champion means adding it. Exporting the champion to S3 as a plain file would drop
  `mlflow-skinny`, which links to the open question of how the cloud server reads the model.
- **MLflow server:** it rejects requests addressed to unknown host names as DNS rebinding, so
  `--allowed-hosts` (env `MLFLOW_ALLOWED_HOSTS`, default `localhost:*,127.0.0.1:*,mlflow:*`) now lists the
  compose name; the public name must be added on AWS. The mlflow service also has a healthcheck, which the
  serving service waits for.

