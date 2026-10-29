"""Model server: /predict, /metrics, /reload (plus /health, /ready, /model).

It scores raw applicant records with the champion from the registry. Every request goes through the same
Pandera contract as the weekly batches, so a malformed payload is rejected with the broken rules (never the
values). `/reload` re-resolves the `champion` alias, which is how a promotion reaches the running server.
"""
import asyncio
import hmac
import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from credit_drift.contracts.pandera_gate import build_scoring_schema, validate_scoring_batch
from credit_drift.contracts.profile import load_profile
from credit_drift.serving.registry import LoadedModel, load_champion

log = logging.getLogger("credit_drift.serving")
MAX_RECORDS = 10_000
SCORE_BUCKETS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_RECORDS)


def _is_number(value: Any) -> bool:
    return value is None or (isinstance(value, int | float) and not isinstance(value, bool))


def _to_frame(records: list[dict[str, Any]], numeric_columns: list[str]) -> pd.DataFrame:
    """JSON has no NaN: a null becomes one. A numeric column that holds anything but numbers or nulls is left
    untouched so the contract rejects it (a string like "123" is a malformed payload, not a number)."""
    frame = pd.DataFrame(records)
    for column in numeric_columns:
        if column in frame and frame[column].dtype == object and frame[column].map(_is_number).all():
            frame[column] = pd.to_numeric(frame[column])
    return frame


def _build_metrics(registry: CollectorRegistry) -> SimpleNamespace:
    return SimpleNamespace(
        requests=Counter("predict_requests_total", "Prediction requests by outcome", ["outcome"], registry=registry),
        rows=Counter("predict_rows_total", "Applicants scored", registry=registry),
        flagged=Counter("predict_high_risk_total", "Applicants at or above the decision threshold", registry=registry),
        latency=Histogram("predict_latency_seconds", "Time to validate and score one request",
                          buckets=LATENCY_BUCKETS, registry=registry),
        score=Histogram("prediction_score", "Distribution of predicted default probabilities",
                        buckets=SCORE_BUCKETS, registry=registry),
        reloads=Counter("model_reloads_total", "Reload attempts by outcome", ["outcome"], registry=registry),
        info=Gauge("model_info", "Currently served model", ["name", "version", "alias"], registry=registry),
        threshold=Gauge("model_decision_threshold", "Probability at or above which an applicant is flagged",
                        registry=registry),
        loaded_at=Gauge("model_loaded_timestamp_seconds", "When the served model was loaded", registry=registry),
    )


def create_app(loader: Callable[[], LoadedModel] | None = None, reload_token: str | None = None,
               startup_attempts: int = 10) -> FastAPI:
    loader = loader or (lambda: load_champion(os.environ.get("MODEL_NAME", "credit_scoring"),
                                              os.environ.get("MODEL_ALIAS", "champion")))
    reload_token = reload_token if reload_token is not None else os.environ.get("RELOAD_TOKEN")
    profile = load_profile()
    schema = build_scoring_schema(profile)
    columns = list(profile["columns"])
    numeric_columns = [c for c, info in profile["columns"].items() if info["kind"] == "numeric"]
    registry = CollectorRegistry()
    metrics = _build_metrics(registry)
    state = SimpleNamespace(model=None)

    def publish(loaded: LoadedModel) -> None:
        state.model = loaded
        metrics.info.clear()
        metrics.info.labels(loaded.name, loaded.version, loaded.alias).set(1)
        metrics.threshold.set(loaded.threshold)
        metrics.loaded_at.set(loaded.loaded_at)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        for attempt in range(1, startup_attempts + 1):  # the registry may still be starting
            try:
                publish(await asyncio.to_thread(loader))  # blocking I/O off the event loop
                log.info("serving %s v%s", state.model.name, state.model.version)
                break
            except Exception as error:  # noqa: BLE001 - keep trying, stay "not ready" meanwhile
                log.warning("model not loaded (attempt %d/%d): %s", attempt, startup_attempts, error)
                await asyncio.sleep(min(3 * attempt, 15) if attempt < startup_attempts else 0)
        yield

    app = FastAPI(title="Credit scoring model server", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response) -> dict:
        if state.model is None:
            response.status_code = 503
            return {"status": "no model loaded"}
        return {"status": "ready", "version": state.model.version}

    @app.get("/model")
    def model_info() -> dict:
        loaded = state.model
        if loaded is None:
            raise HTTPException(503, "no model loaded")
        return {"name": loaded.name, "version": loaded.version, "alias": loaded.alias, "run_id": loaded.run_id,
                "decision_threshold": loaded.threshold, "cost_ratio": loaded.cost_ratio, "loaded_at": loaded.loaded_at}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict:
        loaded = state.model
        if loaded is None:
            metrics.requests.labels("no_model").inc()
            raise HTTPException(503, "no model loaded")
        started = time.perf_counter()
        frame = _to_frame(request.records, numeric_columns)
        gate = validate_scoring_batch(frame, schema)
        if not gate.passed:
            metrics.requests.labels("rejected").inc()
            violations = [{"column": f.column, "rule": f.rule, "rows": f.count} for f in gate.blocking]
            raise HTTPException(422, {"error": "contract violation", "violations": violations})
        probability = loaded.model.predict_proba(frame[columns])[:, 1]
        high_risk = probability >= loaded.threshold
        for p in probability:
            metrics.score.observe(float(p))
        metrics.rows.inc(len(probability))
        metrics.flagged.inc(int(high_risk.sum()))
        metrics.requests.labels("ok").inc()
        metrics.latency.observe(time.perf_counter() - started)
        return {
            "model": {"name": loaded.name, "version": loaded.version},
            "decision_threshold": loaded.threshold,
            "predictions": [{"id": int(i), "probability": round(float(p), 6), "high_risk": bool(h)}
                            for i, p, h in zip(frame["SK_ID_CURR"], probability, high_risk, strict=True)],
        }

    @app.post("/reload")
    def reload(x_reload_token: str | None = Header(default=None)) -> dict:
        if reload_token and not hmac.compare_digest(x_reload_token or "", reload_token):
            raise HTTPException(401, "invalid reload token")
        previous = state.model.version if state.model else None
        try:
            publish(loader())
        except Exception as error:
            metrics.reloads.labels("failed").inc()
            log.error("reload failed, still serving v%s: %s", previous, error)
            raise HTTPException(502, f"reload failed, still serving version {previous}") from error
        metrics.reloads.labels("ok").inc()
        return {"previous_version": previous, "version": state.model.version, "changed": previous != state.model.version}

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
