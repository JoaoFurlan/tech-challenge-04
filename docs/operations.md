# Operations: where things run

The stack (Postgres, MLflow, the model server, and later Airflow, Prometheus, Grafana) runs on the Linux box
`pricing-server` (Ubuntu 24.04, 6 cores / 12 threads, 15 GB RAM). The Windows laptop is the workstation: code
editing, git, the browser and short analysis scripts. Reason: the laptop has about 2 GB of free memory, which
killed a training run and is too little for Airflow.

| What | Where |
|---|---|
| Code | git. Edit on the laptop, `git push`, then on the box `git pull` |
| Stack (`docker compose`) | `~/tech-challenge-04` on the box, SSH alias `pricing-server` |
| MLflow state (runs, registry, artifacts) | Docker volumes `credit-drift_pgdata` and `credit-drift_mlartifacts` on the box |
| Data (`data/reference`, `data/production`) | DVC remote on S3; the box got a copy with `scp` (no AWS keys on the box) |
| The old local stack | stopped, volumes kept on the laptop as a backup (`docker compose down` would keep them too; never `down -v`) |

## Daily use

```bash
# open the UIs and the server on the laptop (nothing is exposed on the LAN)
ssh -N -L 5000:localhost:5000 -L 8010:localhost:8010 -L 8080:localhost:8080 pricing-server
# MLflow http://localhost:5000, model server :8010, Airflow http://localhost:8080

# put new code on the box, and rebuild what changed
scripts/sync-box.sh                                  # sends the commits GitHub or the box lack, as a git bundle
ssh pricing-server 'cd ~/tech-challenge-04 && docker compose up -d --build'
# (a plain `git push` then `git pull` on the box also works, but the global pre-push hook asks for a terminal)

# status, logs
ssh pricing-server 'cd ~/tech-challenge-04 && docker compose ps && docker compose logs --tail 30 serving'
```

With the tunnel open, laptop scripts that talk to `http://127.0.0.1:5000` (MLflow) or `:8010` (the server) reach the
box transparently, so `uv run python -m credit_drift.training.run ...` keeps working unchanged. Ports 5000 and 8010
must be free on the laptop, i.e. its own compose stack must be stopped.

## Airflow

The Airflow services run only where the profile is on: the box's `.env` holds

```
AIRFLOW_UID=1000                  # the box user's uid, so mounted outputs are not root-owned
COMPOSE_PROFILES=airflow
RESTART_POLICY=unless-stopped     # the stack comes back after a reboot
```

and `artifacts/`, `reports/` and `gx/` must exist before the first `up` (otherwise Docker creates them as root).
Trigger a run from the UI (DAG `training_pipeline`, "Trigger with config": `{"quick": true}` for a smoke test) or:

```bash
ssh pricing-server 'cd ~/tech-challenge-04 && docker compose exec -T airflow-scheduler airflow dags trigger training_pipeline --conf "{\"quick\": true}"'
```

## The monitoring loop

The box's `.env` also needs the `monitoring` profile: `COMPOSE_PROFILES=airflow,monitoring`. That adds the Pushgateway
(`localhost:9091` on the box) and Prometheus (`localhost:9090`); add `-L 9090:localhost:9090` to the tunnel to open it.

`production_monitoring` is created paused. Unpausing it (UI toggle, or `airflow dags unpause production_monitoring`)
runs the 13 simulated weeks in order, about 14 minutes. To rerun one week, trigger it with `{"week": 8}`. To start from
a clean slate (what I did before the first full run):

```bash
ssh pricing-server 'cd ~/tech-challenge-04 && rm -rf artifacts/monitoring && docker compose restart pushgateway'
# and delete the runs of the MLflow experiment `production_monitoring` (the DAG logs one run per week)
```

Outputs: `artifacts/monitoring/week_NN/*.json` and `baseline_v<version>.json` on the box, `reports/drift/week_NN_*.html`,
the MLflow experiment `production_monitoring`, and the Prometheus series (`drift_*`, `model_*`, `slice_*`, `alert_firing`,
`diagnosis`).

## Grafana

Part of the `monitoring` profile. Open `http://localhost:3000` through the tunnel (add `-L 3000:localhost:3000`); the
dashboard "Credit scoring: drift and reliability" is the home page. It is generated code: after editing
`docker/grafana/build_dashboard.py` run it (`python docker/grafana/build_dashboard.py`), commit the JSON, sync the box
and Grafana picks the file up within about 30 seconds. If the Pushgateway is ever recreated without its volume, repush
the weeks with `weekly push --week N` (see decision 15).

## Calibration (offline)

`python -m credit_drift.monitoring.calibration --seed N` (needs `MLFLOW_TRACKING_URI`, reads the champion from the
registry) simulates seed N in memory and writes `docs/calibration/seed_N.json`, about 2 minutes on the laptop. The
official seed 42 is refused. `credit_drift.monitoring.replay` reruns the rules over those files; the thresholds
are frozen (decision #16), so run it to evaluate a change, not to retune on the official run.

## Heavy runs

Run them on the box so the laptop does not run out of memory:

```bash
ssh pricing-server 'cd ~/tech-challenge-04 && export PATH=$HOME/.local/bin:$PATH && uv sync && \
  MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python -m credit_drift.training.run all'
```

(on the box MLflow is local, so no tunnel is involved; add `nohup ... &` for runs that outlive the SSH session).

## Backing up the MLflow state

```bash
ssh pricing-server 'cd ~/tech-challenge-04 && docker compose exec -T postgres pg_dump -U mlops --clean --if-exists mlflow' > mlflow_dump.sql
ssh pricing-server 'docker run --rm -v credit-drift_mlartifacts:/data alpine tar czf - -C /data .' > mlartifacts.tgz
```

Restore: start `postgres`, `psql < mlflow_dump.sql`, create the volume and `tar xzf` into it, then `docker compose up -d`
(this is exactly how the state was moved from the laptop to the box). The dump is about 0.7 MB and the artifacts 1.2 MB.

## Security notes

- Login is by SSH key; passwords and keys never go in the repo or in chat.
- The compose services publish ports on `127.0.0.1` only, so they are reachable from the box itself or through the
  SSH tunnel, not from the LAN. Postgres publishes nothing.
- `RELOAD_TOKEN` defaults to a dev value in `docker-compose.yml`; set a real one in a `.env` file on the box before
  exposing anything beyond the tunnel.
- The user `pricing` is in the `docker` group, which is effectively root on that machine.

## Retraining and starting a run from scratch

`retraining_pipeline` is created paused: unpause it together with `production_monitoring`, or the loop's `retrain` task
waits for a run that never starts. The loop triggers it on its own; to run it by hand give `run_week` (the simulated week
whose labels the retrain may use up to run_week - 1). `build` registers two model versions per run in the registry.

Clean slate before a full rerun (what I did before the reference run), in this order:

```bash
# pause both DAGs, delete their run history (the DAG files stay; Airflow re-registers them paused)
airflow dags pause production_monitoring; airflow dags pause retraining_pipeline
airflow dags delete -y production_monitoring; airflow dags delete -y retraining_pipeline
rm -rf artifacts/monitoring reports/drift/*                  # on the box
# wipe the Pushgateway: remove its container and the pushgatewaydata volume, then up -d pushgateway
# MLflow: delete the runs of `production_monitoring` and `retraining`, delete registered versions above 1
# and the aliases challenger/previous/baseline, leaving only `champion` on version 1
```
Then unpause both DAGs.
