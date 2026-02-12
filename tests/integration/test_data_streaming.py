"""Integration tests for Kafka streaming components.

These tests validate the TransactionConsumer class using mocks.
They do not require a running Kafka broker.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.data.streaming.kafka_consumer import TransactionConsumer, DataStreamingError


class TestTransactionConsumerInit:
    """Verify consumer initialisation and config merging."""

    def test_default_bootstrap_servers_from_env(self, monkeypatch):
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker1:9092,broker2:9092")
        consumer = TransactionConsumer(topic="txns")
        assert consumer.config["bootstrap_servers"] == ["broker1:9092", "broker2:9092"]

    def test_explicit_bootstrap_servers(self):
        consumer = TransactionConsumer(topic="txns", bootstrap_servers="custom:9092")
        assert consumer.config["bootstrap_servers"] == ["custom:9092"]

    def test_custom_consumer_group(self):
        consumer = TransactionConsumer(topic="txns", consumer_group="my-group")
        assert consumer.config["group_id"] == "my-group"

    def test_extra_kafka_config_merged(self):
        consumer = TransactionConsumer(topic="txns", fetch_min_bytes=1024)
        assert consumer.config["fetch_min_bytes"] == 1024


class TestJsonDeserialisation:
    """Verify the JSON deserialiser handles good and bad data."""

    def setup_method(self):
        self.consumer = TransactionConsumer(topic="txns")

    def test_valid_json(self):
        data = json.dumps({"id": "txn_001"}).encode("utf-8")
        assert self.consumer._deserialize_json(data) == {"id": "txn_001"}

    def test_none_input(self):
        assert self.consumer._deserialize_json(None) is None

    def test_invalid_json_returns_none(self):
        assert self.consumer._deserialize_json(b"not-json") is None

    def test_invalid_utf8_returns_none(self):
        assert self.consumer._deserialize_json(b"\xff\xfe") is None


class TestConsumerConnect:
    """Verify connect() wires up KafkaConsumer correctly."""

    @patch("src.data.streaming.kafka_consumer.KafkaConsumer")
    def test_connect_subscribes_to_topic(self, mock_kafka_cls):
        consumer = TransactionConsumer(topic="payments")
        consumer.connect()
        mock_kafka_cls.assert_called_once()
        mock_kafka_cls.return_value.subscribe.assert_called_once_with(["payments"])

    @patch("src.data.streaming.kafka_consumer.KafkaConsumer")
    def test_connect_failure_raises(self, mock_kafka_cls):
        from kafka.errors import KafkaError
        mock_kafka_cls.side_effect = KafkaError("connection refused")
        consumer = TransactionConsumer(topic="payments")
        with pytest.raises(DataStreamingError):
            consumer.connect()


class TestConsumerContextManager:
    """Verify __enter__ / __exit__ lifecycle."""

    @patch("src.data.streaming.kafka_consumer.KafkaConsumer")
    def test_context_manager_closes(self, mock_kafka_cls):
        with TransactionConsumer(topic="txns") as consumer:
            assert consumer.consumer is not None
        mock_kafka_cls.return_value.close.assert_called_once()
