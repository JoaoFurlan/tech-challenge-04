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
  engineering. **Final dataset choice is still open — confirm before building Etapa 1.**
- Split: a **Reference pool** (training) and a held-out, disjoint **Production pool**, further
  cut into ~10 synthetic "weeks" with progressively injected drift (see below). There is no
  real timestamp in the data — "time" here is entirely synthetic, by design (mirrors the
  pattern used in the course's own hands-on exercises).

### Model experimentation (Etapa 1, extended)
- Model families: Logistic Regression (baseline), XGBoost, LightGBM, optionally Random Forest.
- Feature engineering variants (outer loop, *not* searched): (A) raw/minimal, (B) engineered
  (ratios, interaction terms, imputation chosen per the actual MCAR/MAR/MNAR mechanism
  observed in the data), (C) feature-selected.
- Hyperparameter tuning: **Optuna**, scoped *only* to tuning within a fixed
  (feature variant × model family) combination — it does not choose the model family or the
  feature variant, those are deliberate experimental conditions we control explicitly. Each
  trial = one nested MLflow run via `autolog()`.
- Selection criteria for the champion: AUC-PR (imbalance-aware), Brier score (calibration
  matters directly for credit decisions), and a cost-weighted metric (the `C_FN`/`C_FP`/`π`
  formulation from the Data Drift mitigation material).
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
  duplicates, etc.) — implemented in both tools for the demo.
- Demo of contract failure: one deliberately corrupted batch (nulls, negative ages, duplicate
  IDs) rejected by both tools.

### Drift simulation & detection (Etapa 2)
Full taxonomy, one type injected per synthetic week, against the disjoint Production pool:

| Week(s) | Drift type | Mechanism |
|---|---|---|
| 1–2 | none (control) | sanity check — detectors must NOT fire |
| 3–4 | gradual covariate shift | slow drift in income / employment duration |
| 5 | abrupt covariate shift | sudden shock to debt-to-income ratio |
| 6 | prior probability / label shift | resample toward higher default rate |
| 7 | concept drift | invert/dampen a feature↔target relationship for a subset |
| 8 | multivariate/subtle drift | reverse correlation between two features, marginals untouched — invisible to PSI/KS, needs MMD or correlation tracking |
| 9 | technical/schema drift | corrupted batch — contract blocks ingestion inside the production loop too, not just the one-off Etapa 1 demo |
| 10+ | new regime stabilizes | PSI plateaus — "new normal," not runaway drift |

- Tool: **Evidently AI** (`DataDriftPreset` + explicit KS/PSI/χ²/JS tests); MMD or a custom
  correlation check for the week-8 multivariate case, since Evidently's preset alone won't
  catch it.
- One HTML report generated per week.

### Orchestration — Airflow
Two DAGs:
- `training_pipeline` (manual trigger, run once): ingest reference data → Pandera gate → GX
  suite (+ Data Docs) → DVC-track the reference split → model experimentation (Optuna+MLflow)
  → register `champion` in the Model Registry.
- `production_monitoring` (scheduled, each run = one simulated week): pull next week-bucket →
  Pandera+GX gate (fails on week 9 by design) → score via the deployed model → Evidently
  report vs. reference → log drift metrics to MLflow (nested run) → push metrics to
  Prometheus via Pushgateway → alert if thresholds are crossed.
- Self-hosted via Docker Compose (LocalExecutor is sufficient at this scale) — not MWAA. MWAA
  is cost-feasible too, but its 20–30 min environment create/update time and extra
  VPC/S3-for-DAGs/IAM setup aren't worth it for a recording-window-only deployment.

### Experiment tracking & model registry — MLflow
- Tracking server + Model Registry, backed by a **containerized Postgres** (not RDS — skipped
  deliberately to keep the AWS footprint simple for a short-lived deployment).
- `mlflow.log_input()` ties each model run to a dataset hash — this is the lineage evidence
  referenced in the Governance docs.

### Data versioning — DVC
Scoped narrowly to **data versioning only** — not pipeline orchestration (`dvc.yaml`) and not
experiment tracking (`dvc exp`), since Airflow and MLflow already own those jobs respectively.
Tracks: the raw downloaded dataset, the Reference split, and the 10 generated weekly
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
- Grafana is the **headline dashboard** for the video: one time-series view of drift severity,
  pipeline health, and alert counts across all 10 simulated weeks — the "whole story at a
  glance" artifact. Evidently/GX/MLflow/Airflow are where you go for the backing detail.

### Where to look at results (reference table)
| Question | Where |
|---|---|
| Did this batch pass validation? What failed? | GX Data Docs (HTML) |
| Quick schema violation list | Pandera `SchemaErrors` / Airflow task logs |
| Which features drifted this week, by how much? | Evidently HTML report (per week) |
| Compare models/features/hyperparameters | MLflow UI — Experiments |
| Which model is live, trained on what data? | MLflow Model Registry |
| Did the DAG run succeed? Which task failed? | Airflow UI |
| The whole story arc across 10 weeks | **Grafana** |
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
- **Course notes**: `docs/course-notes/*.md` are the curated reference for every technique used
  in this project (drift tests, imputation theory, LGPD bases, MLflow patterns, etc.) — check
  there before re-deriving something from scratch. The original lecture PDFs live alongside
  them but are gitignored (`docs/**/*.pdf`); the root-level
  `MLET - Tech Challenge - Fase 4.pdf` (the assignment itself) is tracked.

## Open items (confirm before/while building)

- [ ] Final dataset choice + which columns/feature engineering steps (pending review).
- [ ] Exact Optuna search spaces per model family.
- [ ] Exact weighting between AUC-PR / Brier / cost-weighted metric for champion selection.
- [ ] Whether to add the optional Grafana → Evidently-HTML clickthrough (stretch goal, not
      committed).

## Status

Planning complete through the architecture decisions above. No implementation has started yet.
