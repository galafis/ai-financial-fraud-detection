# Streaming module for real-time data processing
# Module responsible for Kafka streaming and real-time transaction processing

from .kafka_consumer import TransactionConsumer

__all__ = [
    'TransactionConsumer',
]
