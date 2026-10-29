# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Tech Challenge — Fase 4 (Machine Learning Engineering, Pos Tech). Theme: **Observabilidade
Contínua, Detecção de Drift, Qualidade de Dados e Governança em Produção**.

Scenario: a credit-scoring model for a fintech is suspected of degrading silently after
economic shifts (inflation, new customer profiles). The project builds the "sustentação e
confiabilidade" layer around that model: data contracts, drift simulation/detection,
observability, and LGPD governance documentation — including a causal analysis, not just
detection.

Full requirements: `MLET - Tech Challenge - Fase 4.pdf` (repo root). Course background for
every technique used here is summarized in `docs/course-notes/*.md` (curated from the
original lecture PDFs, which are gitignored — see `docs/course-notes/*.pdf`).

The official rubric only requires Pandera-or-GX, Evidently, and a basic dashboard/report.
This project deliberately goes further — full MLflow experimentation, Airflow orchestration,
Prometheus+Grafana, DVC, and an AWS deployment — to simulate a realistic production setup,
not just satisfy the minimum.

## Architecture decisions

### Dataset
- Public tabular credit-risk dataset, binary classification, ≥5,000 rows (per challenge spec).
- Candidate: Home Credit Default Risk (`application_train.csv`, ~307k rows) — flat table only,
  no joins to `bureau.csv` / `previous_application.csv` / etc. Feature engineering depth is
  kept intentionally modest so project effort stays on MLOps, not competition-grade feature
  engineering. **Confirmed**: Home Credit, `application_train.csv` (first exploration in
  `notebooks/01_eda.ipynb`; decisions that followed are in `docs/decisions.md`).
- Split: a **Reference pool** (training) and a held-out, disjoint **Production pool**, further
  cut into ~13 synthetic "weeks" with progressively injected drift (see below). There is no
  real timestamp in the data — "time" here is entirely synthetic, by design (mirrors the
  pattern used in the course's own hands-on exercises).

### Model experimentation (Etapa 1, extended)
- Model families: Logistic Regression (baseline), XGBoost, LightGBM, optionally Random Forest.
- **Staged MLflow experiments**, each feeding the next (details in `docs/decisions.md`, #8):
  `feature_engineering` (variants A raw/minimal, B engineered, C selected, each with a linear and
  a tree probe) → `model_selection` (logistic regression, LightGBM, XGBoost on the best variant)
  → `hyperparameter_tuning` (**Optuna**, only on the shortlisted families; one flat run per
  trial) → `champion`. Runs carry the params `model`, `feature_variant`, `trial` and the tag
  `stage`, so they can be sorted and filtered. Runs log explicitly, not via `autolog()`.
- Selection rule at every stage: rank by cross-validated AUC-PR, shortlist everything within one
  standard error of the best, pick the lowest expected cost (`C_FN` = 5 × `C_FP`, the formula from
  the Data Drift mitigation material). Brier is a sanity gate: the calibrated champion must beat
  the constant-prior predictor.
- Winner is registered in the **MLflow Model Registry** (alias `champion`) and is the only
  model used downstream (validation gates, scoring, serving).

### Data validation (Etapa 1)
Both tools, different jobs — not redundant:
- **Pandera** — fast inline schema gate at ingestion (`DataFrameModel`, `@pa.check_types`).
  The "fail fast" guardrail inside the pipeline.
- **Great Expectations** — richer Expectation Suite, auto-generated **Data Docs HTML**. Doubles
  as the audit-trail artifact for the Governance section and as one of the HTML
  reports/dashboards the repo deliverable requires.
- At least 3 strict rules required by the rubric (age bounds, no nulls on income, no
  duplicates, etc.) — implemented in both tools for the demo. Two severities (see
  `docs/decisions.md`): **blocking** rules reject the whole batch (schema/types, unique IDs, income
  not null and > 0, age 18–100, valid `DAYS_EMPLOYED`, positive amounts); **warning** rules are
  logged and alerted but the batch is still scored (income ≤ 10M with 99.9% tolerance — the training
  data itself has 3 rows above it —, gender domain, per-column null rates, unseen categories).
  Two contracts: the scoring batch (no `TARGET`) and the label contract applied when labels mature.
- Demo of contract failure: one deliberately corrupted batch (nulls, negative ages, duplicate
  IDs) rejected by both tools.

### Drift simulation & detection (Etapa 2)
The rubric requires intentionally altering **at least two important variables** (data drift)
and covering Data and Concept Drift over simulated time; it does not require retraining. The
schedule is kept small so the 5-minute video can explain it, and so the later weeks double as
a controlled test of the retraining stretch phase (see below). Drift is injected into the
disjoint Production pool, one episode per synthetic week:

| Week(s) | Drift type | Mechanism |
|---|---|---|
| 1–2 | none (control) | sanity check — detectors must NOT fire |
| 3–4 | gradual covariate shift | slow drift in income and employment duration (≥2 variables); weeks 4–5 also shift the category mix (e.g. income type) by resampling rows |
| 5 | abrupt covariate shift | sudden shock to debt-to-income ratio |
| 6–7 | new normal (covariate regime persists) | the week-5 regime stays in place; v2 (trained on labels through week 4 under a one-week label delay) has seen the income shift but not the week-5 payment shock, so it only partly covers this regime |
| 8 | concept drift | invert/dampen a feature↔target relationship for a subset — a challenger has *not* seen this |
| 9 | technical/schema drift | corrupted batch — contract blocks ingestion inside the production loop too, not just the one-off Etapa 1 demo |
| 10–13 | concept drift persists | fresh rows each week; unseen by v1/v2, but v3 (trained after week 11 on labels through week 10) has seen it |

**Simulation mechanics.** Time is synthetic: the Production pool is shuffled once with a fixed
seed and cut into 13 non-overlapping weekly samples of ~5,000 rows, and a transform function
changes each week's rows. Each week is saved as `week_NN.parquet` plus a `drift_spec.json`
listing every transform (the ground truth, used only offline). Column names follow Home Credit
and may change with the final dataset. Starting values, to tune until the detectors behave
sensibly:
- Weeks 1–2: no transform.
- Week 3: `AMT_INCOME_TOTAL` × 0.90, employment duration × 0.92. Week 4: income × 0.86,
  employment duration × 0.85.
- Week 5: `AMT_ANNUITY` × 1.15 on top of income × 0.86, so annuity ÷ income jumps. Weeks 6–7: the
  week-5 transform again on fresh rows.
- Weeks 4–5 also: a small shift in the category mix (more "Working", fewer "Pensioner") by
  resampling rows, so each row keeps its own label (pure covariate shift, exercises χ²/JS tests;
  first thing to cut if time is short). The numeric transforms change values in place with labels
  unchanged (measurement/level drift) — state this assumption in the README.
- Week 8: features as in week 5. For a segment, labels are redrawn from a logistic model fit on
  the Reference with the `EXT_SOURCE_2` coefficient reversed and halved and the intercept
  adjusted to keep the overall default rate (so it is concept drift, not label shift). Weeks
  10–13: the same on fresh rows.
- Week 9: a copy of the week-9 sample with ~12% null incomes, ages under 18, ~3% duplicated IDs
  and one numeric column turned into text.

- Concept drift is mostly invisible to PSI/KS (the features did not move again); the signal is
  the performance drop once labels arrive (AUC-PR, Brier). Say so explicitly in the demo.
- Tool: **Evidently AI** (`DataDriftPreset` + explicit KS/PSI/χ²/JS tests).
- One HTML report generated per week (Evidently, against the current champion's training window).
- **Two drift baselines**: PSI against the original Reference (informational — how far the world
  has moved overall) and PSI against the **current champion's training window** (drives the
  alerts). The alert baseline re-points on every promotion (v1 → v2 → v3), because retraining
  changes the model, not the customers: against the fixed Reference, PSI stays high forever
  after week 5 and keeps alerting on a regime the champion already knows. Neither PSI sees
  concept drift; that only shows in performance.
- Optional extras, only after the core works: label/prior shift (resample toward a higher
  default rate) and subtle multivariate drift (reverse the correlation between two features,
  marginals untouched — needs MMD or a correlation check, since Evidently's preset alone
  won't catch it).
- **Level-based, not compounding**: each week has an absolute state of the world applied to fresh,
  non-overlapping rows; gradual drift is a ramp of levels. Each week has its own seed (base +
  week), so adding a week never changes earlier ones.
- **Finding to state in the demo** (measured on a throwaway model in the exploration): covariate
  drift moves PSI/KS and the mean predicted probability (8.05% → 8.42%) but barely moves AUC-PR
  or Brier (< 0.002); concept drift does the opposite. The week-8 segment is chosen in step 9, is
  not identical to any monitored slice, and is sized so overall AUC-PR falls ≈ 0.04.

### Retraining & recalibration (stretch phase)
Built only after validation, drift detection and observability work end to end. Not graded by
the rubric; it supports the "lessons learned" part of the video.
- **Trigger**: the week-5 drift alert. A `retraining_pipeline` DAG (reusing the training tasks
  with a different data window) builds two candidates from labeled weeks only, respecting an
  explicit **label delay** (one week assumed: retraining right after the week-5 alert only has
  labels through week 4, so v2 only partly covers the week-5 shock; the alternative is waiting a
  week): **A** = current champion + recalibration, **B** = retrained model, then recalibrated on
  a held-out recent slice that was not used for fitting.
- **Recalibration** options: prior-shift correction of probabilities (π_train → π_new), Platt
  scaling or isotonic regression on recent labels, and a cost-optimal threshold
  (`t* = C_FP(1−π) / [C_FP(1−π) + C_FN·π]`). Recalibration changes Brier/threshold, never AUC-PR.
- **Challenger flow**: the best candidate is registered as `challenger` (v2); `champion` does not
  move until a gate passes (beats v1 by an agreed margin on the held-out slice). Promotion =
  moving the `champion` alias, then calling the model server's `/reload`. No DAG changes.
- **Measuring the effect**: v1 and v2 are scored on the **same** batches every later week, so the
  v2 − v1 gap isolates the effect of the intervention regardless of drift type. Put a
  bootstrap confidence interval on that gap. Never evaluate on weeks the challenger trained on.
  This is an **approved exception** to real-world-first: in prod the old model would be retired
  after promotion, but here v1 keeps scoring offline (loaded from the registry inside the
  monitoring DAG; the serving app stays champion-only) purely as the comparison baseline.
- **Expected outcome of retrain #1 (a finding, not a failure)**: covariate drift barely moves
  ranking, so v2 gives a calibration-level gain in weeks 6–7 at most (candidate A, recalibration,
  should roughly match candidate B, a full retrain), and does not help in weeks 8–11 (concept
  drift it never saw). Retraining is not always the answer; recalibration is the cheap fix.
- **Retrain #2 for concept drift**: with a one-week label delay, labels through week 10 only
  become available after week 11 (week 8 labels are the first containing the drift; week 9 is a
  blocked batch with none). Retrain on them → **v3**, recalibrate, and score v1, v2, v2
  recalibrated and v3 on the same weeks 12–13. Expected: recalibration leaves AUC-PR flat (the
  ranking itself is wrong), v3 recovers it, mostly in the affected segment. This is the reason
  the timeline is 13 weeks. Promote v3 only through the same gate.
- **Interim mitigations while labels mature** (document in the README, not necessarily built):
  tighter approval thresholds or manual review for the affected segment, down-weighting or
  dropping the affected feature, falling back to a simpler model, sample-weighting recent data
  in the next retrain, and shortening label delay with early-delinquency proxies.

### Diagnosing the cause of a drift
The kind of drift is read from which signals moved **this week versus last week**, not versus
the Reference (which stays "drifted" forever after week 5):

| Features changed? | Score distribution changed? | Performance dropped (labels)? | Likely cause |
|---|---|---|---|
| Yes | Yes | Maybe | Covariate shift |
| No | Yes (mean score moves) | Calibration off | Label (prior) shift |
| No | No | Yes | Concept drift |
| Contract fails | — | — | Technical / schema issue |

- Starting thresholds for the automated rule (tune later), all week over week: KS rises
  ≥ 0.045, the champion's AUC-PR drop versus its own validation baseline grows ≥ 0.04, mean
  score moves ≥ 1 point. Decision order: covariate → concept → label.
- **Slice analysis**: an automated scan of AUC-PR drop across 14 slices (`EXT_SOURCE_1/2/3`,
  income, age and employment, each below/above the median, plus gender F/M) ranks the worst ones.
  Cut points are fixed from the champion's training window, and slice metrics use a **rolling
  two-week window** with bootstrap CIs (`EXT_SOURCE_1` high has ≈47 defaulters per week);
  the segment where drift was injected is not hand-picked. Concept drift concentrates in a
  slice; covariate shift spreads across all. Guard against noise: minimum slice size, flag only
  drops clearly above the overall drop (e.g. ≥ 0.03), ideally with bootstrap CIs, since with
  many slices some look bad by chance. The gender slice doubles as a bias check for the LGPD
  documentation. A scan only finds slices it includes; tree-based error analysis can search for
  them automatically (optional).
- **Slice chart design (Grafana)**: one time-series panel with one line per slice, for the
  **current champion** against that model's own validation baseline (so the baseline resets on
  every promotion), grey unless a slice exceeded the overall drop by ≥ 0.03, with the overall
  drop dashed. v1 stays only as the offline comparison baseline. Slices need enough rows and
  defaulters (rule of thumb ≈ 1,000 rows and ≈ 100 defaulters, else pool several weeks).
- **Label delay on the dashboard**: every label-dependent panel (AUC-PR, Brier, observed default
  rate, slice chart, performance alerts, the performance column of the diagnosis) shows week N
  only after one week. Features, score distribution, pipeline health and latency do not wait.
- **Importance reweighting**: reweight the Reference to match production features; if the
  model's performance on it matches production, covariate shift explains the drop, and the gap
  left over is concept drift.
- **Ground truth**: `drift_spec.json` records what was injected, so the README's causal analysis
  can compare "what we injected" with "what the detectors attributed".
- **Grafana panels** beyond PSI/KS, alerts, AUC-PR and Brier: predicted vs observed default rate
  (separates label shift), the slice chart above (localizes concept drift without knowing the segment in advance), and
  an automated per-week diagnosis table. Checking the diagnosis against `drift_spec.json` happens offline
  (evaluation notebook, README), not on the dashboard, because prod has no ground truth.

### Notebooks (analysis, not production)
Jupyter notebooks live in `notebooks/` and are for **exploration and analysis**; nothing in
production depends on them. Real logic lives in importable code (`src/`), notebooks call it, and
anything that must run unattended is an Airflow task.
- `01_eda.ipynb` (before Etapa 1): an observation-only first look — class balance, missingness
  and its mechanism, ranges, duplicates, relationships with the target, leakage checks — ending
  in facts and open questions. Decisions taken from it (contract rules, null treatment, feature
  variants, drift variables, slices, drift accumulation) live in `docs/decisions.md`.
- Retraining analysis notebooks, one per retrain (`02_retrain_v2.ipynb`, `03_retrain_v3.ipynb`):
  investigate *why* and *whether it helped* — the drift diagnosis behind the trigger, candidate
  comparison, reliability diagrams before/after recalibration, per-slice AUC-PR, bootstrap CIs
  on the v2 − v1 and v3 − v1 gaps. The retraining itself (data window, training, registration)
  runs through `retraining_pipeline` and is logged to MLflow; notebooks do not train or promote
  models by hand. Prototyping a candidate in a notebook is fine if it calls the same `src/`
  functions and logs to MLflow.
- `04_drift_evaluation.ipynb` (offline): compares the diagnosis and detectors with
  `drift_spec.json` (precision/recall per feature, false positives in weeks 1–2) — the only
  place ground truth is used.
- Reproducibility: fixed seeds, data read from the DVC-tracked files (no data committed), no raw
  customer rows in outputs (LGPD), final notebooks committed executed and exported to HTML as
  evaluator-facing reports. Narrative markdown in Portuguese (evaluator-facing), code and
  comments in English.

### Orchestration — Airflow
Two core DAGs, plus one stretch DAG (`retraining_pipeline`, see above):
- `training_pipeline` (manual trigger, run once): ingest reference data → Pandera gate → GX
  suite (+ Data Docs) → DVC-track the reference split → model experimentation (Optuna+MLflow)
  → register `champion` in the Model Registry.
- `production_monitoring` (scheduled, each run = one simulated week N): pull next week-bucket →
  Pandera+GX gate (fails on week 9 by design) → score via the deployed model (HTTP) → Evidently
  report vs. the current champion's training window, plus PSI vs. the original Reference →
  **label backfill**: compute AUC-PR, Brier, observed default rate and the slice metrics for
  week N−1, whose labels have now matured (one-week delay) → score the offline comparison
  models (v1 baseline, any challenger in shadow) from the registry → log metrics to MLflow
  (nested run) → push metrics to Prometheus via Pushgateway → alert if thresholds are crossed.
- Self-hosted via Docker Compose (LocalExecutor is sufficient at this scale) — not MWAA. MWAA
  is cost-feasible too, but its 20–30 min environment create/update time and extra
  VPC/S3-for-DAGs/IAM setup aren't worth it for a recording-window-only deployment.

### Experiment tracking & model registry — MLflow
- Tracking server + Model Registry, backed by a **containerized Postgres** (not RDS — skipped
  deliberately to keep the AWS footprint simple for a short-lived deployment).
- `mlflow.log_input()` ties each model run to a dataset hash — this is the lineage evidence
  referenced in the Governance docs.
- Training and experimentation run **locally** (local MLflow). Only the champion model and its
  registry entry are synced to the AWS MLflow/S3 (re-log + register via script, or
  `mlflow-export-import`); the Optuna trial history stays local and is exported as a
  comparison table/screenshots for the README. A later challenger v2 follows the same path.
- Serving: the model server resolves `models:/<name>@champion` from the registry at startup and
  on `/reload`. Open choice: read the EC2 MLflow (needs an EB→EC2 security-group rule) or load a
  champion exported to S3 (fewer moving parts).

### Data versioning — DVC
Scoped narrowly to **data versioning only** — not pipeline orchestration (`dvc.yaml`) and not
experiment tracking (`dvc exp`), since Airflow and MLflow already own those jobs respectively.
Tracks: the raw downloaded dataset, the Reference split, and the 13 generated weekly
Production batches (so the exact perturbed data behind each simulated week is a fixed, hashed,
reproducible artifact — not silently regenerated differently on a later run).
- Remote: local disk during development; S3 (same bucket as MLflow artifacts, different
  prefix: `dvc-store/` vs `mlflow-artifacts/`) for the cloud deployment.

### Observability — Prometheus + Grafana (self-hosted, no external SaaS)
Two deliberately different collection patterns, both demonstrated:
- **Push**: Airflow's `production_monitoring` DAG runs short-lived batch tasks — pushes
  metrics to a **Pushgateway**, which Prometheus scrapes.
- **Pull**: the Elastic Beanstalk serving app is long-running — Prometheus scrapes its
  `/metrics` endpoint directly.
- Grafana is the **headline dashboard** for the video: time-series views across all 13 simulated weeks
  of drift severity (PSI vs. the Reference and vs. the champion's window), alerts, AUC-PR,
  Brier, predicted vs. observed default rate, the slice chart, the automated diagnosis table,
  per-task pipeline health, and inference traffic/latency — the "whole story at a glance"
  artifact (a mock of it exists as a draft artifact). Evidently/GX/MLflow/Airflow are where you go for the backing detail.

### Where to look at results (reference table)
| Question | Where |
|---|---|
| Did this batch pass validation? What failed? | GX Data Docs (HTML) |
| Quick schema violation list | Pandera `SchemaErrors` / Airflow task logs |
| Which features drifted this week, by how much? | Evidently HTML report (per week) |
| Compare models/features/hyperparameters | MLflow UI — Experiments |
| Which model is live, trained on what data? | MLflow Model Registry |
| What does the data look like; why these rules, features and drifted variables? | EDA notebook |
| Why did we retrain, and did it help? | Retraining analysis notebooks (HTML export) + MLflow comparison |
| Do the detectors match what we injected? | Drift evaluation notebook (offline, uses `drift_spec.json`) |
| Did the DAG run succeed? Which task failed? | Airflow UI |
| The whole story arc across 13 weeks | **Grafana** |
| Live inference traffic right now | Grafana (scraping the EB app) |

### Cloud — AWS (temporary, recording-window only)
- **One EC2 instance**: Airflow + MLflow + Prometheus + Grafana as a Docker Compose stack
  (consolidated — not split across instances).
- **Elastic Beanstalk**: model-serving API (FastAPI), exposing `/predict` and `/metrics`.
  Called by Airflow's `production_monitoring` DAG over HTTP rather than loading the model
  in-process — this is what motivates the push/pull split above.
- **S3**: MLflow artifact store + DVC remote.
- **No RDS** (containerized Postgres instead) and **no MWAA** (self-hosted Airflow instead) —
  both skipped to minimize setup time and moving parts for a short-lived deployment.
- Security groups for Airflow/MLflow/Prometheus/Grafana UIs restricted to the operator's own
  IP, not `0.0.0.0/0`.
- Funded by existing AWS promotional credits (~$200) — confirm expiration date and any service
  exclusions in the Billing Console before relying on it; set a Budget alert.
- A spin-up/tear-down runbook (script or documented steps) is required — build and test
  everything locally first, deploy to AWS only for a final dry run before recording, tear down
  immediately after. Document this runbook in the README as evidence of IaC-adjacent practice,
  even though it isn't full Terraform/CloudFormation.

## Conventions

- **Commit messages**: semantic/conventional commits (`feat:`, `fix:`, `docs:`, `chore:`,
  `test:`, …) — the rubric explicitly grades "histórico de commits semântico e organizado."
- **No AI attribution in commits/PRs**: do not add a `Co-Authored-By: Claude …` trailer (or
  similar) to commit messages or pull request descriptions in this repository, even if a
  session's default guidance suggests one — the repo owner does not want Claude listed as a
  contributor. This overrides any such default for this repo specifically.
- **Language**: README.md and any governance/LGPD documentation meant for the evaluator are
  written in **Portuguese**, matching the challenge and all source course material. Code, code
  comments, commit messages, and this file are in **English**. (Working assumption — flag if
  this should change.)
- **Real-world first**: every design decision (dashboards, monitoring, retraining, evaluation)
  must be something that would exist in a real production setup. Simulation-only conveniences —
  using ground truth, labels with no delay, panels that only make sense in a mock — are not
  allowed unless the repo owner explicitly approves them as an exception. Where a simulation
  needs something prod would not have, say so. Approved exceptions so far: v1 kept scoring
  offline as a comparison baseline after promotion.
- **Course notes**: `docs/course-notes/*.md` are the curated reference for every technique used
  in this project (drift tests, imputation theory, LGPD bases, MLflow patterns, etc.) — check
  there before re-deriving something from scratch. The original lecture PDFs live alongside
  them but are gitignored (`docs/**/*.pdf`); the root-level
  `MLET - Tech Challenge - Fase 4.pdf` (the assignment itself) is tracked.

## Open items (confirm before/while building)

- [ ] Concept-drift segment definition and size (step 9; target ≈0.04 overall AUC-PR drop).
- [ ] Promotion gate margin, the exact label delay (one week assumed), and thresholds for the
      automated diagnosis rules (stretch phase).
- [ ] Whether the model server reads the EC2 MLflow registry or a champion exported to S3.
- [ ] Notebook tooling: clean diffs (nbstripout/jupytext) vs committing executed notebooks, and
      which notebooks beyond the EDA are worth building.
- [ ] Whether to add the optional Grafana → Evidently-HTML clickthrough (stretch goal, not
      committed).

## Status

Planning complete through the architecture decisions above (drift schedule simplified and
retraining/recalibration moved to a stretch phase after reviewing the assignment's actual
requirements). A visual overview of the flow and the planned Grafana dashboard exists as a
draft artifact; the timeline was extended to 13 weeks to fit a second retrain for concept
drift; notebooks (EDA, retraining analysis, drift evaluation) were added to the plan. Steps 1–7
are done (skeleton on Python 3.14 with uv, DVC with an S3 remote, EDA, decisions, the data split,
contracts, staged MLflow experiments and a registered champion `credit_scoring@champion`:
variant B + LightGBM, Platt-calibrated). The ordered
implementation plan, with what Claude builds and what the repo owner must do, is in
`docs/build-plan.md` (tick items off as they finish). Everything is proven locally first, then
rerun on AWS with the same commit, data, frozen thresholds and champion; never tune on AWS.
Next: step 8 (`training_pipeline` DAG) and step 9 (drift simulator).
