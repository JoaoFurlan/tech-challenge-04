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

Later sections (7 onward) record the build decisions; #16 holds the frozen detector thresholds.

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

## 13. Airflow and the training pipeline (step 8)

Services in `docker-compose.yml` under the profile `airflow` (the box sets `COMPOSE_PROFILES=airflow` in its `.env`,
so a plain `docker compose up` on the laptop stays light); image `docker/Dockerfile.airflow`; DAG `dags/training_pipeline.py`.

- **Airflow 3.3.2** (official `slim` image on Python 3.14), LocalExecutor, a dedicated Postgres for its metadata
  (separate from MLflow's), `api-server` (UI and API, `127.0.0.1:8080`), `scheduler` and `dag-processor`. Idle, the four
  services together use under 1 GB.
- **The project lives in its own virtualenv** (`/opt/credit-drift/.venv`, built from `uv.lock` with the training,
  validation and monitoring groups), so Airflow's pinned dependencies never fight LightGBM, Great Expectations, MLflow
  or Evidently. DAG tasks are thin `BashOperator`s that call `python -m credit_drift...` in that venv; the logic stays in
  `src/`. The image is 3.7 GB because of that environment.
- **Code is mounted, not baked:** `src`, `config`, `data`, `artifacts`, `reports`, `gx` and `dags` are bind mounts, so
  `git pull` on the box changes behaviour without rebuilding. The containers run as the host user
  (`AIRFLOW_UID=1000` in the box's `.env`), so files they write belong to that user, not to root.
- **Auth:** `SimpleAuthManager` with all users as admins, safe only because the UI listens on `127.0.0.1` and is reached
  through the SSH tunnel (the SSH key is the login). Secrets (`AIRFLOW_FERNET_KEY`, `AIRFLOW_JWT_SECRET`,
  `AIRFLOW_API_SECRET`, `AIRFLOW_DB_PASSWORD`) default to dev values; set real ones in the box's `.env` before sharing.
- **`training_pipeline`** (manual trigger, params `quick` and `trials`): `check_data` -> `contracts` (all three Reference
  parts against the full contract, labels included; stops the run if one is rejected) -> `feature_engineering` ->
  `model_selection` -> `hyperparameter_tuning` -> `champion` (registers the model and tags the decision threshold).
  `quick = true` is the smoke test: 10k rows, 3 trials, separate experiment and model names.
- **Verified on the box:** a quick run went green end to end in about 2.3 minutes (contracts alone 59 s). Quick runs
  reuse experiment names like `quick_feature_engineering`. MLflow refuses to reuse a *deleted* experiment's name, so
  `activate_experiment` (`training/tracking.py`) restores a deleted `quick_*` experiment before using it, which makes the
  smoke test repeatable after any cleanup. A deleted real experiment is never restored silently. This first failed for
  the repo owner on a second quick run, which had been left as a documented manual step; it is now automatic and tested.
- **Things that bit during setup:** the slim image has no Postgres drivers (`psycopg2-binary` and, for Airflow 3's async
  engine, `asyncpg`) and no `libgomp1` (LightGBM); each went into its own late Dockerfile layer so the 3 GB environment
  layer stays cached. Containers need `AIRFLOW_UID` or bind-mounted outputs end up root-owned.
- **Restart:** every service uses `restart: ${RESTART_POLICY:-no}`; the box sets `unless-stopped`, so the stack returns
  after a reboot, while the laptop does not auto-start it.
- **Not yet built:** the `production_monitoring` DAG (step 12) and the stretch `retraining_pipeline` DAG, which reuses
  this DAG's tasks.

## 14. The production monitoring loop (step 12)

Code in `src/credit_drift/monitoring/` (`weekly.py` has one subcommand per DAG task; `slices.py`, `performance.py`,
`alerts.py` hold the logic; `context.py` is the drift measurement shared with `run_drift.py`); DAG
`dags/production_monitoring.py`; thresholds in `config/monitoring_thresholds.json`; Pushgateway and Prometheus added to
the compose file under the profile `monitoring`.

- **One DAG run is one simulated week.** Scheduled daily from 2026-01-01 to 2026-01-13 with catch-up, one run at a time
  (January N = week N); the DAG is created paused, and unpausing runs all 13 weeks in order (about 65 s each, 14
  minutes in total). A `week` parameter reruns one week by hand.
- **Tasks:** `gate` (Pandera + Great Expectations) -> `choose_path` -> `score_batch` (through the model server over HTTP)
  -> `drift_report`; or `quarantine_batch` for a rejected batch. In parallel `label_performance` finishes week N-1.
  Then `evaluate_alerts` -> `log_mlflow` -> `push_metrics`. A rejected batch still ends the run and still alerts.
- **Steps talk through files** in `artifacts/monitoring/week_NN/` (gate, score, drift, performance, alerts,
  diagnosis), so any step can be rerun alone and a failed week can be inspected after the fact.
- **Label delay:** run N backfills week N-1: AUC-PR with a bootstrap interval, AUC-ROC, Brier, observed versus predicted
  rate, flag rate, cost at the decision threshold, and the 14 slices over a two-week window (fixed cut points from the
  champion's training window). The champion's own baseline (overall and per slice, on the untouched Reference test
  part) is computed once per version into `baseline_v<version>.json`. Comparison models (aliases `baseline`,
  `challenger`) are scored on the same week when they exist; today none do.
- **Alert rules** (all thresholds provisional until step 14): batch blocked (critical); any monitored feature at PSI
  >= 0.25 (high) or two at >= 0.10 (medium); score PSI >= 0.10 or KS >= 0.10; AUC-PR 0.04 below the champion's
  baseline (high); calibration gap >= 20% (medium); a degraded slice (medium).
- **Slice rule, corrected after the first real week:** a slice is flagged only when its drop exceeds the overall drop
  (floored at zero) by 0.03 **and** its bootstrap interval sits entirely below baseline-minus-overall. Without the
  floor, a control week where the whole population improved flagged three slices that merely improved less.
- **Diagnosis** (week over week, final once labels are in; a rejected batch is diagnosed `technical` at once): features
  changed = a monitored numeric feature's KS rose >= 0.045 or a categorical's JS >= 0.03; score changed = score KS rose
  >= 0.03 or its mean moved >= 1 point; performance dropped = AUC-PR >= 0.04 below baseline. Order: technical,
  covariate shift, concept drift, label shift.
- **Metrics:** pushed to the Pushgateway in two groups per week (`stage=batch` describes week N, `stage=labels`
  describes week N-1), so label-dependent series appear a week late by construction. Per-feature PSI and KS,
  score drift, AUC-PR/Brier/rates, per-slice AUC-PR against its baseline, alerts and the diagnosis. Prometheus scrapes
  the Pushgateway (`honor_labels`) and the model server's `/metrics` directly (push and pull both demonstrated).
  MLflow gets one run per week in the experiment `production_monitoring`.
- **Result of the first full run (13 weeks, 13 green runs):**

  | Weeks | What the loop said |
  |---|---|
  | 1-2 | no alerts (control weeks quiet) |
  | 3 | no alert (PSI 0.09) but diagnosed covariate shift (week-over-week KS rise) |
  | 4 | `data_drift_moderate`, first alert |
  | 5 | `data_drift_major` (annuity/credit PSI 0.80), covariate shift |
  | 6-7 | PSI alerts keep firing against the old baseline, diagnosis `none` (the new normal stopped changing) |
  | 8 | `performance_drop` once its labels arrived at run 9; diagnosed `concept_drift` |
  | 9 | `batch_blocked`, diagnosed `technical`; `score_batch` and `drift_report` skipped, batch quarantined |
  | 10-11 | `performance_drop` and `concept_drift` again |
  | 12 | AUC-PR 0.034 below baseline: **missed** by the 0.04 threshold (single-week noise is about +-0.04) |
  | 13 | its labels would arrive at run 14, beyond the 13-week horizon |

- **Findings to carry forward:**
  1. **The PSI alert repeats every week from week 5 to 13** because the baseline is the champion's training window until a
     promotion re-points it. That is correct but noisy; a real alerting layer would deduplicate or escalate "ongoing".
  2. **The slice scan ranks the right slices but the flag threshold is conservative.** By week 11 the top slices are
     `AMT_INCOME_TOTAL low` (drop 0.093) and `AGE low` (0.092), the two variables that define the injected segment, but
     they only just reach overall drop + 0.03, and nothing was flagged. In week 8 the window still contains healthy
     week 7. Step 14 tunes this on two seeds; the dashboard should rank slices by drop rather than rely on the flag.
  3. **One week of AUC-PR is too noisy for a hard 0.04 rule** (week 12 missed, 8/10/11 caught). AUC-ROC is steadier.


## 15. The Grafana dashboard (step 13)

Grafana 12.1 runs under the compose profile `monitoring`. Everything is provisioned as code, so a fresh box shows the
same dashboard: `docker/grafana/provisioning/` (Prometheus datasource, dashboard file provider) and
`docker/grafana/dashboards/credit-scoring.json`, which is **generated** by `docker/grafana/build_dashboard.py` (edit the
script, rerun it, commit both).

- **Weeks are labels, not time.** The simulated weeks are pushed in one afternoon, so a time axis would draw every
  week at the same instant. Panels use the `week` label as the x axis (instant queries, join by `week`, sort). Weeks
  are zero-padded in the query (`label_replace`) so that 10 sorts after 9.
- **Label delay is visible.** Panels that depend on labels (AUC-PR, Brier, observed rate, slice matrix, performance
  alerts) only have a week N once week N+1 has run; the dashboard shows that gap instead of hiding it. Week 9 has no
  labels by design and shows as a hole.
- **Slice matrix, not 14 lines.** Slices down, weeks across, cells coloured by AUC-PR drop against the champion's own
  baseline (0.03 yellow, 0.06 orange, 0.09 red). It ranks slices without relying on the `slice_flagged` rule, which is
  conservative (decision 14).
- **Static reference lines** (AUC-PR baseline 0.246, Brier 0.0745) are constants in the generator: rebuild the
  dashboard when the champion is promoted.
- **Pushgateway persists** its state in a volume (`--persistence.file`). The weekly metrics are pushed once, so a
  recreated container used to empty the dashboard. If it happens anyway, repush from the stored artifacts:
  `python -m credit_drift.monitoring.weekly push --week N` for each week and stage.
- **Anonymous viewer** is enabled because Grafana is only reachable through the SSH tunnel (127.0.0.1 on the box). The
  admin password defaults to `admin` for local use; set `GRAFANA_ADMIN_PASSWORD` before exposing it anywhere (AWS: the
  security group stays restricted to the operator's IP).
- Verified: every panel's query returns data through the Grafana API, and the rendered dashboard was checked visually.

## 16. Calibrating and freezing the detector thresholds (step 14)

Code: `monitoring/calibration.py` (simulates a seed in memory, scores it with the champion, keeps only summary numbers
in `docs/calibration/seed_<seed>.json`), `monitoring/replay.py` (reruns the alert rules and the diagnosis over those
signals and scores them against the injection), notebook `04_drift_evaluation` (HTML in `docs/reports/`), thresholds
in `config/monitoring_thresholds.json` (status `frozen`).

**Exception to real-world-first, approved by the repo owner.** Ground truth (the injection schedule) and extra
simulated worlds exist only here, offline, to calibrate thresholds. In production the equivalent is incident history,
backtests and alert review. Nothing on the dashboard or in the DAG uses ground truth, and the official seed (42) is
never used for calibration.

**Protocol.** Seeds 101 and 202 to tune, 303 and 404 to validate, looked at once after freezing. Disclosure: the first
tuning pass used 101 only; validating on 202 exposed a failure (a control week and most concept weeks wrong), so 202
became a tuning seed and 303/404 were generated as the clean validation.

| Threshold | Before | Frozen | Why |
|---|---|---|---|
| Features at PSI >= 0.10 needed for a data alert | 2 | **1** | control weeks peak at 0.01-0.015 on four seeds; weeks 3-4 move one variable at a time and were alerted late |
| Performance drop | one week's AUC-PR >= 0.04 below the baseline | **mean of the last 3 labelled weeks + 1.645 standard errors still below the baseline** | a week has ~400 defaulters (AUC-PR +-0.04), the injected drop is 0.03-0.05; the single-week rule false-alarmed on a control week and missed most concept weeks on seed 202 |
| PSI bands, score, slices, diagnosis | provisional | unchanged | a sweep on seed 101 found no gain without false alarms |

**Validation (seeds 303 and 404, thresholds frozen):** no false alarm in the control weeks, none in the performance or
data rules; first data alert in week 3-4; concept-drift weeks caught 5/5 and 4/5; diagnosis 13/13 and 12/13. The
provisional thresholds scored 3/5 and 4/5 on the same weeks.

**What still fails, and is accepted.**
- Week 8 can go unflagged: the 3-week window still holds healthy weeks, so the first concept week is caught one or
  two weeks late. That is the price of not alarming on noise.
- The slice flag is weak: with the bootstrap interval required it almost never fires on healthy weeks (one false in four
  simulations) but catches only about 2 of 5 concept weeks; without it, it false-alarms on several healthy weeks (14
  slices, ~47 defaulters in some). It stays conservative; the dashboard's slice matrix ranks the drop instead.
- The PSI alert still repeats from the first drifted week until a retrain re-points the baseline.
- All four seeds share one simulator (same drift types, same segment). The thresholds are validated on new draws, not on
  drift of an unknown kind.

Changing a frozen threshold now needs a new decision here, not a quiet edit.

## 17. The reference run with frozen thresholds (step 15)

`production_monitoring` ran 13/13 green on the box from a clean slate (artifacts, MLflow runs and Pushgateway wiped),
official seed 42, thresholds frozen in decision #16, no retraining. Every Grafana panel returns data.

| Week | Injected | What the loop said |
|---|---|---|
| 1-2 | control | no alerts, diagnosis none |
| 3 | mild covariate shift | no alert (PSI just under 0.10), diagnosis covariate shift |
| 4 | covariate shift, mix starts to move | `data_drift_moderate`, covariate shift |
| 5 | annuity shock | `data_drift_major`, covariate shift |
| 6-7 | new normal | PSI alert repeats (baseline not re-pointed yet), diagnosis none |
| 8 | concept drift starts | diagnosis **none**: the 3-week AUC-PR window (0.232) had not fallen below the baseline yet (the accepted delay, #16) |
| 9 | corrupted batch | `batch_blocked` (critical), batch quarantined, diagnosis technical |
| 10-12 | concept drift persists | diagnosis concept drift; `performance_drop` raised for weeks 10, 11 and 12 |
| 13 | concept drift persists | PSI alert only: its labels would arrive in a week 14 that does not run |

Findings: the frozen rule fixed the old week-12 miss, and the only concept week not caught is week 8, as agreed. A single
week's AUC-PR still swings a lot (week 1 scored 0.303, week 2 0.226, both control weeks), which is why the rule averages
three. The PSI alert keeps firing through week 13 until a retrain re-points the baseline (step 16). This run is the
reference for the video and for the comparison with the AWS rerun.

## 18. Retraining, the promotion gate and what the loop did (step 16)

Code in `src/credit_drift/retraining/`: `window.py` (the data a retrain may use), `candidates.py`, `gate.py`, `policy.py`
(the trigger), `run.py` (`build` and `promote` steps), `analysis.py` (notebook helpers). DAG `retraining_pipeline`
(`decide -> build_candidates -> promote`); `production_monitoring` ends with `check_retraining`, which triggers it and
waits, so the next simulated week scores with whatever model is champion by then. Notebooks `02_retrain_v2` and
`03_retrain_v3` (HTML in `docs/reports/`) only read what the pipeline logged. Approved by the repo owner: label delay of
one week, and a paired-bootstrap gate instead of a fixed margin.

**Data window.** Reference train part plus the production weeks whose labels have arrived (a run after week N's batch uses
weeks up to N-1; the blocked week 9 has none). The last two labelled weeks are the "recent" slice, split at random into
30% that joins the training rows, 30% that fits the calibrator and 40% held out for the gate (about 4,000 rows and 320
defaulters), so no row serves two purposes. Production rows carry as much total weight as the Reference rows, spread by
recency (half-life two weeks). These shares were fixed before seeing results.

**Candidates.** A: the champion's own pipeline, recalibrated on the calibration slice. B: the champion's recipe (same
steps and settings, cloned) refitted on the window with the weights, then recalibrated. Isotonic or Platt is chosen by
cross-fitted Brier. A or B goes forward by cross-fitted expected cost on the calibration slice, never the gate slice. Both
are registered with tags (candidate, calibration, labels_through_week, window_weeks, gate result); the chosen one gets the
`challenger` alias.

**Gate.** Paired bootstrap on the gate slice (1,000 draws). Promote only if (1) the expected-cost gain (C_FN = 5 x C_FP,
at each model's own threshold) has its 95% interval entirely above zero, (2) the AUC-PR difference is not significantly
negative, and (3) Brier beats the constant-prior predictor. A fixed margin would be arbitrary: one week of AUC-PR moves
by +-0.04 and recalibration never moves it.

**Promotion.** Moves `champion`, keeps the first champion as `baseline` (the approved comparison exception) and the one
before as `previous`, drops `challenger`, and POSTs `/reload` to the server. A promoted model's drift baseline re-points:
its window is equal parts Reference and the production weeks it trained on (tag `window_weeks`; candidate A keeps its base
model's window because only the calibrator changed). The performance baseline is per model version, as before.

**Trigger.** Retrain on the first major data-drift alert; afterwards on a performance-drop alert about a week the champion
has not seen. After any attempt, wait for three new labelled weeks before trying again (my first version retried every
week on nearly the same labels and was fixed before the reference run).

**What the reference run did (seed 42, frozen thresholds).**

| Run | Labels through | Trigger | Candidates (versions) | Gate on the chosen one | Outcome |
|---|---|---|---|---|---|
| week 5 | 4 | `data_drift_major` | A = v8, B = v9 (chosen) | cost gain -0.021, 95% interval [-0.038, -0.006]; AUC-PR -0.014 | failed, v9 stays `challenger` |
| week 11 | 10 | `performance_drop` | A = v10, B = v11 (chosen) | cost gain +0.004, [-0.012, +0.019]; AUC-PR +0.004 | failed, v11 stays `challenger` |

v1 served all 13 weeks, so the promotion path (aliases, reload, baseline re-point) did not run in the real loop; it is
covered by unit tests with a stubbed registry and server.

**Findings.**
- Retrain 1 (covariate shift): recalibration is neutral; refitting on four weeks is worse than v1 over weeks 6-13
  (AUC-PR -0.006, 95% interval -0.011 to -0.002; cost +0.007). The gate was right. Retraining is not always the answer.
- Retrain 2 (concept drift): the retrained candidate is statistically the same as v1 on weeks 11-13 (AUC-PR +0.0015,
  interval includes zero); inside the injected segment AUC-PR goes from 0.178 to 0.184, the expected direction but within
  noise. Recalibration does not move AUC-PR, as predicted. With two labelled drifted weeks, 70% of them held out for
  calibration and the gate, there is not enough evidence to switch. Recovering needs more labelled weeks of the new regime.
- This differs from the plan's hope that v3 would recover the loss. Letting the gate decide, rather than tuning the shares
  until a candidate passed, is the point; a design with a larger fit share would be a new decision, not a tweak.
- The static reference lines on the dashboard stay valid because the champion never changed.
