# Build plan

Ordered implementation plan for the Tech Challenge 4 project. Design decisions live in
`CLAUDE.md`; this file is the order of work and who does what.

- **[R]** = required by the rubric. **[extra]** = beyond the rubric (stretch).
- **Claude** = built in a Claude Code session. **You** = needs the repo owner (accounts,
  downloads, credentials, approvals, recording). Claude will ask when it reaches a **You** item.
- Steps 1-15 are the full rubric deliverable. Do not start 16-19 until step 15 works.
- Everything is built and proven **locally first** (staging). AWS is a redeploy of the same
  Docker Compose stack, with the same git commit, DVC data, frozen threshold config and
  champion model. The AWS run is a real rerun, not pasted numbers; it should reproduce the
  local run (same seeds), and the local run is the reference to diff against. Never tune
  thresholds or code on AWS: fix locally, then rerun.

## Phase 1 - Foundation and data

### 1. Repo skeleton [R]
- [x] **You:** install Docker Desktop and Python 3.14 (uv installs it) (git is already set up; the repo is
      `JoaoFurlan/tech-challenge-04`, `origin` already points to it).
- [x] **Claude:** folders (`src/`, `notebooks/`, `dags/`, `config/`, `docker/`, `tests/`),
      Python environment and lockfile, base Docker Compose, DVC init, `.gitignore`.
- **Why:** everything hangs off it, and the rubric grades the semantic commit history. All
  code lands in `src/`; notebooks and DAGs import from it.

### 2. Pick the dataset and download it [R]
- [x] **You:** confirm Home Credit Default Risk (or choose another) - open item.
- [x] **You:** download `application_train.csv` (Kaggle account, accept the competition rules,
      then the `kaggle` CLI or manual download) into `data/raw/`. Claude cannot do this for you.
- [x] **Claude:** DVC-track the raw file (local remote), write a download/README note.
- **Why:** it blocks the EDA, the pool split and the model work.

### 3. EDA notebook `01_eda`
- [ ] **Claude:** class balance, missingness mechanism (MCAR/MAR/MNAR), ranges, duplicates,
      correlations, leakage checks. Output: documented decisions.
- [ ] **You:** review the decisions (contract rules, which 2+ variables to drift, which
      slices to monitor) before they are frozen.
- **Why:** justifies the contract rules, imputation, feature variants, drifted variables and
  slices. Feeds steps 5, 7 and 9.

### 4. Split Reference and Production pools [R]
- [ ] **Claude:** seeded, disjoint split; Reference (training) and Production pool; DVC-tracked.
- **Why:** the rubric requires separation; the fixed seed makes it reproducible. The Reference
  trains the model; the Production pool becomes the 13 weekly batches.

## Phase 2 - Contracts and first model

### 5. Data contracts [R]
- [ ] **Claude:** Pandera schema (inline gate) and Great Expectations suite (Data Docs HTML),
      at least 3 strict rules, plus a corrupted batch that both reject; tests.
- **Why:** "garbage does not enter the model" (25% of the grade). Gates the training data now
  and every weekly batch later (blocks week 9).

### 6. Local MLflow + training code [R]
- [ ] **Claude:** MLflow + Postgres in Compose; training functions in `src/` for feature
      variants A/B/C x Logistic Regression / XGBoost / LightGBM; Optuna inside each
      combination, trials as nested runs; dataset hash via `mlflow.log_input()`.
- [ ] **You:** agree the Optuna search spaces and the AUC-PR / Brier / cost weighting
      (open items) when Claude proposes them.
- **Why:** fair, reproducible comparison with lineage. Feeds step 7.

### 7. Choose and register the champion [R]
- [ ] **Claude:** rank candidates, calibrate the winner on held-out data, register `champion` v1.
- **Why:** one agreed model is the only thing production uses. Serving loads this alias and
  every later comparison starts from it.

### 8. `training_pipeline` DAG [extra]
- [ ] **Claude:** local Airflow wrapping steps 5-7 (ingest -> contracts -> experiments -> register).
- **Why:** reproducible, visible training; the retraining DAG reuses its tasks.

## Phase 3 - Drift and the monitoring loop

### 9. Drift simulator [R]
- [ ] **You:** decide whether drift accumulates across weeks or is independent per week
      (open item; must be settled before writing the simulator).
- [ ] **Claude:** cut the Production pool into 13 weekly samples, apply per-week transforms,
      write `week_NN.parquet` and `drift_spec.json`; seeded, DVC-tracked.
- **Why:** creates the degradation and the ground truth. Feeds the monitoring DAG and the
  evaluation notebook. Independent of step 11, so they can be built in parallel.

### 10. Drift detection [R]
- [ ] **Claude:** Evidently report per week; PSI and KS against both baselines (original
      Reference, and the champion's training window).
- **Why:** the statistical evidence for the rubric (25%). Feeds the dashboard and alerts.

### 11. Model server [R, via observability]
- [ ] **Claude:** containerised FastAPI with `/predict`, `/metrics`, `/reload`, loading
      `models:/<name>@champion`.
- **Why:** mirrors real serving and gives Prometheus something to scrape. Called over HTTP
  by the monitoring DAG.

### 12. `production_monitoring` DAG [R]
- [ ] **Claude:** per simulated week: gate batch -> score -> Evidently report -> backfill week
      N-1 performance (label delay) -> score comparison models -> log to MLflow -> push
      metrics -> alert.
- **Why:** the heart of the "sustentacao e confiabilidade" layer.

### 13. Prometheus, Pushgateway and Grafana [R]
- [ ] **Claude:** add to Compose; dashboard: drift severity (two PSI lines), alerts, AUC-PR,
      Brier, predicted vs observed default rate, slice chart, diagnosis table, pipeline
      health, latency. Provisioned as code so it redeploys unchanged.
- **Why:** the one-screen story for the video (20%).

## Phase 4 - Calibrate, then the full local run

### 14. Calibrate detector thresholds
- [ ] **Claude:** in `04_drift_evaluation`, tune thresholds on one simulation, validate on a
      second seed, then freeze them in a versioned config.
- [ ] **You:** review and approve the frozen thresholds.
- **Why:** evidence-based thresholds without tuning on the official run.

### 15. Full local run without retraining [R]
- [ ] **Claude:** run all 13 weeks end to end; check the dashboard, the contract block in
      week 9, and detectors vs `drift_spec.json`. Save this run as the reference.
- [ ] **You:** look at the dashboard and confirm the story reads clearly.
- **Why:** proves the required deliverable works before spending time on extras, in the
  cheap environment where fixes are fast.

## Phase 5 - Retraining [extra]

### 16. Retraining, promotion gate and analysis notebooks
- [ ] **You:** agree the promotion gate margin and label delay (open items).
- [ ] **Claude:** `retraining_pipeline` DAG; after the week-5 alert build v2 from labels
      available under the delay, register `challenger`, pass the gate, move the alias, call
      `/reload`; repeat for v3 after week 11. Notebooks `02_retrain_v2`, `03_retrain_v3`
      analyse why and whether it helped.
- **Why:** shows the lifecycle (alert -> retrain -> compare -> promote) and measures what
  helped. Updates the dashboard's champion, baseline window and comparison lines.
- Repeat step 15's check with retraining enabled; this becomes the local reference run.

## Phase 6 - Documentation and cloud

### 17. Causal analysis and governance [R]
- [ ] **Claude:** use the evaluation notebook for injected-vs-detected and counterfactual
      numbers; README in Portuguese with LGPD plan (PII handling, legal basis for credit
      decisions, retention, bias check from the gender slice); HTML exports.
- [ ] **You:** review the legal/governance text, since it carries your name.
- **Why:** governance is 15% of the grade and ties the technical work to compliance.

### 18. AWS runbook, setup and dry run [extra]
- [ ] **You:** create/confirm the AWS account; check the credits' expiry and service
      exclusions in the Billing Console; set a Budget alert; create an IAM user (not root)
      and a key pair; share your public IP for the security groups; install the AWS CLI and
      run `aws configure` yourself (credentials never go in the repo or the chat).
- [ ] **Claude:** spin-up/tear-down runbook (script or documented steps): S3 bucket
      (`mlflow-artifacts/`, `dvc-store/`), EC2 with the Compose stack, Elastic Beanstalk app,
      security groups restricted to your IP, champion sync (re-log script or
      `mlflow-export-import`). One dry run.
- [ ] **You:** run the commands that touch your AWS account (Claude prepares and explains
      them; use `! <command>` in the prompt to run them in the session).
- **Why:** shows a realistic deployment with bounded cost. Tests what local cannot: EB
  networking, security groups, S3, IAM, cross-machine scraping.

### 19. Official run, evidence and teardown [extra]
- [ ] **You:** start the AWS stack the same day you record.
- [ ] **Claude:** run the weeks on AWS, diff against the local reference run, capture
      Grafana screenshots and reports.
- [ ] **You:** terminate everything the same day and confirm in the Billing Console.
- **Why:** clean reproducible evidence without costs left running.

### 20. STAR video and final cleanup [R]
- [ ] **You:** record the 5-minute video (Situation, Task, Action, Result). Claude can draft
      the script and the demo order.
- [ ] **Claude:** tidy commit history, final README pass, check the rubric list.
- **Why:** 15% of the grade, and where the detected drift and lessons learned show up.

## Open decisions to settle along the way

| Decision | Needed by |
|---|---|
| Final dataset | Step 2 |
| Optuna search spaces; AUC-PR / Brier / cost weighting | Step 6-7 |
| Whether drift accumulates across weeks | Step 9 |
| Notebook tooling (nbstripout/jupytext vs executed notebooks) | Step 3 |
| Promotion gate margin, label delay, diagnosis thresholds | Step 14-16 |
| EB reads EC2 MLflow vs champion exported to S3 | Step 18 |
