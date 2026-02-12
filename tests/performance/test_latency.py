"""Performance tests — basic latency benchmark.

Uses the FastAPI TestClient (in-process), so results reflect
application-level overhead, not network latency.
"""

import time
import statistics
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture(autouse=True)
def _patch_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
    with patch("src.api.main.model", mock_model):
        yield


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


@pytest.fixture()
def auth_token(client):
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin_password"},
    )
    return resp.json()["access_token"]


SAMPLE_TXN = {
    "transaction_id": "txn_perf",
    "amount": 100.0,
    "merchant_id": "m_1",
    "customer_id": "c_1",
    "timestamp": "2024-06-01T12:00:00Z",
    "payment_method": "debit_card",
}


@pytest.mark.performance
def test_predict_latency_under_200ms(client, auth_token):
    """Each in-process prediction should complete in < 200 ms."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        resp = client.post("/api/v1/predict", json=SAMPLE_TXN, headers=headers)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        latencies.append(elapsed_ms)

    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    assert p95 < 200, f"p95 latency {p95:.1f} ms exceeds 200 ms"


@pytest.mark.performance
def test_health_latency_under_50ms(client):
    """Health check should be very fast."""
    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        resp = client.get("/api/v1/health")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        latencies.append(elapsed_ms)

    avg = statistics.mean(latencies)
    assert avg < 50, f"average health latency {avg:.1f} ms exceeds 50 ms"
