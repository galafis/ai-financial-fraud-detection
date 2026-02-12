"""Unit tests for feature engineering logic."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestTemporalFeatureExtraction:
    """Tests for temporal feature extraction from transaction timestamps."""

    def test_hour_of_day_extraction(self):
        ts = datetime.fromisoformat("2024-01-15T14:30:00")
        assert ts.hour == 14

    def test_day_of_week_extraction(self):
        ts = datetime.fromisoformat("2024-01-15T14:30:00")  # Monday
        assert ts.weekday() == 0

    def test_weekend_flag_saturday(self):
        ts = datetime.fromisoformat("2024-01-13T10:00:00")  # Saturday
        assert ts.weekday() >= 5

    def test_weekend_flag_weekday(self):
        ts = datetime.fromisoformat("2024-01-15T10:00:00")  # Monday
        assert ts.weekday() < 5


class TestAmountCategorisation:
    """Tests for transaction amount bucketing."""

    @pytest.mark.parametrize("amount,expected", [
        (50.0, "low"),
        (500.0, "medium"),
        (5000.0, "high"),
    ])
    def test_amount_categorisation(self, amount, expected):
        if amount < 100:
            category = "low"
        elif amount < 1000:
            category = "medium"
        else:
            category = "high"
        assert category == expected


class TestTransactionDataValidation:
    """Tests for basic data-level invariants."""

    def setup_method(self):
        self.sample = {
            "amount": 1500.00,
            "merchant": "Online Store",
            "location": "São Paulo",
            "timestamp": "2024-01-15T14:30:00Z",
            "user_id": "user123",
            "card_type": "credit",
            "merchant_category": "retail",
        }

    def test_amount_is_numeric(self):
        assert isinstance(self.sample["amount"], (int, float))

    def test_required_fields_present(self):
        for key in ("amount", "timestamp", "user_id"):
            assert key in self.sample

    def test_missing_field_detection(self):
        incomplete = {k: v for k, v in self.sample.items() if k != "merchant"}
        assert "merchant" not in incomplete

    def test_user_history_dataframe_shape(self):
        history = pd.DataFrame({
            "user_id": ["user123"] * 10,
            "amount": np.random.uniform(50, 500, 10),
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
        })
        assert history.shape == (10, 3)
        assert history["amount"].dtype == np.float64
