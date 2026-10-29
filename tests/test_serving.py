"""Model server behaviour with a stub model: no MLflow, no Docker."""
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from credit_drift.serving.app import MAX_RECORDS, create_app
from credit_drift.serving.registry import LoadedModel, decision_threshold


class StubModel:
    """Probability falls with income, so tests can predict which applicants are flagged."""

    def predict_proba(self, X):
        p = np.clip(0.4 - X["AMT_INCOME_TOTAL"].to_numpy() / 1_000_000, 0.01, 0.99)
        return np.column_stack([1 - p, p])


def loaded(version="1", threshold=1 / 6):
    return LoadedModel(model=StubModel(), name="credit_scoring", version=version, alias="champion", run_id="run",
                       threshold=threshold, cost_ratio=5.0, loaded_at=time.time())


def serve(loader=None, **kwargs):
    return TestClient(create_app(loader or (lambda: loaded()), startup_attempts=1, **kwargs))


def records(batch, n=5):
    return batch.head(n).to_dict(orient="records")


def metric(client, line_start):
    text = client.get("/metrics").text
    return next(float(line.split()[-1]) for line in text.splitlines() if line.startswith(line_start))


def test_threshold_follows_the_cost_ratio():
    assert decision_threshold(5.0) == pytest.approx(1 / 6) and decision_threshold(3.0) == pytest.approx(0.25)


def test_predict_returns_probability_and_decision_per_applicant(batch):
    with serve() as client:
        response = client.post("/predict", json={"records": records(batch)})
    body = response.json()
    assert response.status_code == 200 and body["model"] == {"name": "credit_scoring", "version": "1"}
    assert [p["id"] for p in body["predictions"]] == list(batch["SK_ID_CURR"].head(5))
    for p in body["predictions"]:
        assert 0 < p["probability"] < 1 and p["high_risk"] == (p["probability"] >= body["decision_threshold"])


def test_nulls_are_allowed_where_the_contract_allows_them(batch):
    rows = records(batch, 2)
    rows[0]["AMT_ANNUITY"] = None   # nullable in the contract
    rows[0]["OWN_CAR_AGE"] = None   # a numeric column that is all null in this payload
    rows[1]["OWN_CAR_AGE"] = None
    with serve() as client:
        assert client.post("/predict", json={"records": rows}).status_code == 200


def test_extra_fields_are_ignored(batch):
    rows = records(batch, 2)
    for row in rows:
        row["TARGET"] = 1
        row["SOMETHING_NEW"] = "x"
    with serve() as client:
        assert client.post("/predict", json={"records": rows}).status_code == 200


@pytest.mark.parametrize(("column", "bad", "rule"), [
    ("AMT_INCOME_TOTAL", None, "not_nullable"),
    ("AMT_INCOME_TOTAL", -5, "income_positive"),
    ("DAYS_BIRTH", -3000, "age_18_to_100"),
    ("AMT_CREDIT", "987654321", "numeric_dtype"),  # a number sent as a string is a malformed payload
])
def test_contract_violations_are_rejected_without_echoing_values(batch, column, bad, rule):
    rows = records(batch, 3)
    rows[0][column] = bad
    with serve() as client:
        response = client.post("/predict", json={"records": rows})
    assert response.status_code == 422
    violations = response.json()["detail"]["violations"]
    assert any(v["column"] == column and rule in v["rule"] for v in violations)
    assert "987654321" not in response.text  # rule names and counts only, never the offending value


def test_duplicate_ids_and_missing_columns_are_rejected(batch):
    rows = records(batch, 3)
    rows[1]["SK_ID_CURR"] = rows[0]["SK_ID_CURR"]
    with serve() as client:
        assert client.post("/predict", json={"records": rows}).status_code == 422
        del rows[1]["SK_ID_CURR"], rows[0]["AMT_CREDIT"]
        assert client.post("/predict", json={"records": rows}).status_code == 422


def test_empty_and_oversized_requests_are_rejected():
    with serve() as client:
        assert client.post("/predict", json={"records": []}).status_code == 422
        assert client.post("/predict", json={"records": [{}] * (MAX_RECORDS + 1)}).status_code == 422


def test_metrics_count_requests_scores_and_the_served_model(batch):
    with serve() as client:
        client.post("/predict", json={"records": records(batch, 5)})
        bad = records(batch, 1)
        bad[0]["AMT_INCOME_TOTAL"] = -1
        client.post("/predict", json=({"records": bad}))
        assert metric(client, 'predict_requests_total{outcome="ok"}') == 1
        assert metric(client, 'predict_requests_total{outcome="rejected"}') == 1
        assert metric(client, "predict_rows_total") == 5
        assert metric(client, "prediction_score_count") == 5
        assert metric(client, "predict_latency_seconds_count") == 1
        assert metric(client, "model_decision_threshold") == pytest.approx(1 / 6)
        assert 'model_info{alias="champion",name="credit_scoring",version="1"} 1.0' in client.get("/metrics").text


def test_health_ready_and_model_endpoints():
    with serve() as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200
        info = client.get("/model").json()
        assert info["version"] == "1" and info["decision_threshold"] == pytest.approx(1 / 6)


def test_not_ready_while_no_model_could_be_loaded(batch):
    def failing():
        raise RuntimeError("registry down")

    with serve(failing) as client:
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200  # alive, just not ready
        assert client.post("/predict", json={"records": records(batch, 1)}).status_code == 503


def test_reload_picks_up_a_promoted_model_and_needs_the_token():
    versions = iter(["1", "2"])
    with serve(lambda: loaded(next(versions)), reload_token="secret") as client:
        assert client.post("/reload").status_code == 401
        assert client.post("/reload", headers={"X-Reload-Token": "wrong"}).status_code == 401
        result = client.post("/reload", headers={"X-Reload-Token": "secret"}).json()
        assert result == {"previous_version": "1", "version": "2", "changed": True}
        assert client.get("/model").json()["version"] == "2"
        text = client.get("/metrics").text
        assert 'version="2"' in text and 'version="1"' not in text  # the old model's series is gone
        assert metric(client, 'model_reloads_total{outcome="ok"}') == 1


def test_a_failed_reload_keeps_serving_the_previous_model():
    calls = []

    def loader():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("registry down")
        return loaded("1")

    with serve(loader) as client:
        assert client.post("/reload").status_code == 502
        assert client.get("/model").json()["version"] == "1"
        assert metric(client, 'model_reloads_total{outcome="failed"}') == 1


def test_a_promoted_model_changes_the_decision_threshold(batch):
    models = iter([loaded("1", threshold=0.9), loaded("2", threshold=0.001)])
    with serve(lambda: next(models)) as client:
        before = client.post("/predict", json={"records": records(batch)}).json()
        client.post("/reload")
        after = client.post("/predict", json={"records": records(batch)}).json()
    assert not any(p["high_risk"] for p in before["predictions"])
    assert all(p["high_risk"] for p in after["predictions"])
