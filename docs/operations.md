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
ssh -N -L 5000:localhost:5000 -L 8010:localhost:8010 pricing-server     # MLflow on :5000, server on :8010

# put new code on the box, and rebuild what changed
git push
ssh pricing-server 'cd ~/tech-challenge-04 && git pull && docker compose up -d --build'

# status, logs
ssh pricing-server 'cd ~/tech-challenge-04 && docker compose ps && docker compose logs --tail 30 serving'
```

With the tunnel open, laptop scripts that talk to `http://127.0.0.1:5000` (MLflow) or `:8010` (the server) reach the
box transparently, so `uv run python -m credit_drift.training.run ...` keeps working unchanged. Ports 5000 and 8010
must be free on the laptop, i.e. its own compose stack must be stopped.

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
