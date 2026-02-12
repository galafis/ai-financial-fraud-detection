"""Integration tests for the FastAPI fraud-detection API.

Uses the FastAPI TestClient so no running server is needed.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Patch the model import before importing the app so module-level
# model loading doesn't fail when model files are absent.


@pytest.fixture(autouse=True)
def _patch_model():
    """Provide a fake model for every test in this module."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

    with patch("src.api.main.model", mock_model):
        yield mock_model


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


@pytest.fixture()
def auth_token(client):
    """Get a valid JWT token via the demo auth endpoint."""
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "admin", "password": "admin_password"},
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")
        assert "version" in body

    def test_health_reports_model_loaded(self, client):
        resp = client.get("/api/v1/health")
        # With the mock model in place the field should be True
        assert resp.json()["model_loaded"] is True


class TestAuthEndpoint:
    def test_login_success(self, client):
        resp = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "admin_password"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        resp = client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "wrong"},
        )
        assert resp.status_code == 401

    def test_login_unknown_user(self, client):
        resp = client.post(
            "/api/v1/auth/token",
            data={"username": "nobody", "password": "x"},
        )
        assert resp.status_code == 401


class TestPredictEndpoint:
    VALID_TXN = {
        "transaction_id": "txn_001",
        "amount": 250.00,
        "merchant_id": "m_100",
        "customer_id": "c_200",
        "timestamp": "2024-06-01T12:00:00Z",
        "payment_method": "credit_card",
    }

    def test_predict_requires_auth(self, client):
        resp = client.post("/api/v1/predict", json=self.VALID_TXN)
        assert resp.status_code == 401

    def test_predict_valid_transaction(self, client, auth_token):
        resp = client.post(
            "/api/v1/predict",
            json=self.VALID_TXN,
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["transaction_id"] == "txn_001"
        assert 0 <= body["fraud_probability"] <= 1
        assert isinstance(body["is_fraud"], bool)
        assert body["risk_level"] in ("low", "medium", "high")
        assert body["processing_time_ms"] >= 0

    def test_predict_rejects_negative_amount(self, client, auth_token):
        bad_txn = {**self.VALID_TXN, "amount": -10.0}
        resp = client.post(
            "/api/v1/predict",
            json=bad_txn,
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert resp.status_code == 422

    def test_predict_rejects_missing_fields(self, client, auth_token):
        resp = client.post(
            "/api/v1/predict",
            json={"transaction_id": "txn_bad"},
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert resp.status_code == 422
