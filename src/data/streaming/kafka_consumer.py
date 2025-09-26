"""
Kafka Consumer for Real-time Transaction Processing
Consome real-time transactions from Kafka topics for fraud detection processing.
"""

import json
import logging
from typing import Dict, Iterator, Optional, Any
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from ...utils.exceptions import DataStreamingError
from ...config.settings import get_settings


class TransactionConsumer:
    """
    Kafka consumer for processing real-time financial transactions.
    
    Handles consumption of transaction messages from Kafka topics,
    with automatic deserialization and error handling.
    """
    
    def __init__(
        self,
        topic: str,
        consumer_group: str = 'fraud-detection-group',
        bootstrap_servers: Optional[str] = None,
        **kafka_config: Any
    ):
        """
        Initialize the transaction consumer.
        
        Args:
            topic: Kafka topic to consume from
            consumer_group: Consumer group ID
            bootstrap_servers: Kafka bootstrap servers
            **kafka_config: Additional Kafka configuration
        """
        self.topic = topic
        self.consumer_group = consumer_group
        self.settings = get_settings()
        
        # Set default bootstrap servers
        if bootstrap_servers is None:
            bootstrap_servers = self.settings.kafka_bootstrap_servers
            
        # Default Kafka configuration
        default_config = {
            'bootstrap_servers': bootstrap_servers.split(','),
            'group_id': consumer_group,
            'auto_offset_reset': 'latest',
            'enable_auto_commit': True,
            'value_deserializer': self._deserialize_json,
            'key_deserializer': self._deserialize_json,
            'consumer_timeout_ms': 1000,
            'max_poll_records': 100,
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 10000
        }
        
        # Merge with custom config
        default_config.update(kafka_config)
        
        self.consumer = None
        self.config = default_config
        self.logger = logging.getLogger(__name__)
        
    def _deserialize_json(self, data: bytes) -> Optional[Dict]:
        """
        Deserialize JSON data from Kafka message.
        
        Args:
            data: Raw bytes data
            
        Returns:
            Deserialized dictionary or None if invalid
        """
        if data is None:
            return None
            
        try:
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.error(f"Failed to deserialize message: {e}")
            return None
    
    def connect(self) -> None:
        """
        Establish connection to Kafka and subscribe to topic.
        
        Raises:
            DataStreamingError: If connection fails
        """
        try:
            self.consumer = KafkaConsumer(**self.config)
            self.consumer.subscribe([self.topic])
            self.logger.info(f"Connected to Kafka topic: {self.topic}")
            
        except KafkaError as e:
            raise DataStreamingError(f"Failed to connect to Kafka: {e}")
    
    def consume(self) -> Iterator[Dict[str, Any]]:
        """
        Consume messages from Kafka topic.
        
        Yields:
            Dictionary containing transaction data
            
        Raises:
            DataStreamingError: If consumption fails
        """
        if self.consumer is None:
            self.connect()
            
        try:
            self.logger.info(f"Starting to consume from topic: {self.topic}")
            
            for message in self.consumer:
                if message.value is None:
                    continue
                    
                # Add metadata to the transaction
                transaction = message.value.copy()
                transaction.update({
                    'kafka_partition': message.partition,
                    'kafka_offset': message.offset,
                    'kafka_timestamp': message.timestamp,
                    'consumed_at': datetime.utcnow().isoformat()
                })
                
                self.logger.debug(f"Consumed transaction: {transaction.get('transaction_id', 'unknown')}")
                yield transaction
                
        except KafkaError as e:
            self.logger.error(f"Error consuming messages: {e}")
            raise DataStreamingError(f"Failed to consume messages: {e}")
        except KeyboardInterrupt:
            self.logger.info("Consumer interrupted by user")
            self.close()
        except Exception as e:
            self.logger.error(f"Unexpected error in consumer: {e}")
            raise DataStreamingError(f"Unexpected consumption error: {e}")
    
    def consume_batch(self, max_records: int = 100, timeout_ms: int = 5000) -> list[Dict[str, Any]]:
        """
        Consume a batch of messages.
        
        Args:
            max_records: Maximum number of records to return
            timeout_ms: Timeout for polling
            
        Returns:
            List of transaction dictionaries
        """
        if self.consumer is None:
            self.connect()
            
        try:
            message_batch = self.consumer.poll(
                timeout_ms=timeout_ms, 
                max_records=max_records
            )
            
            transactions = []
            for topic_partition, messages in message_batch.items():
                for message in messages:
                    if message.value is None:
                        continue
                        
                    transaction = message.value.copy()
                    transaction.update({
                        'kafka_partition': message.partition,
                        'kafka_offset': message.offset,
                        'kafka_timestamp': message.timestamp,
                        'consumed_at': datetime.utcnow().isoformat()
                    })
                    transactions.append(transaction)
            
            self.logger.debug(f"Consumed batch of {len(transactions)} transactions")
            return transactions
            
        except KafkaError as e:
            self.logger.error(f"Error consuming batch: {e}")
            raise DataStreamingError(f"Failed to consume batch: {e}")
    
    def seek_to_beginning(self) -> None:
        """
        Seek to the beginning of the topic.
        """
        if self.consumer is None:
            self.connect()
            
        self.consumer.seek_to_beginning()
        self.logger.info("Seeked to beginning of topic")
    
    def commit_async(self) -> None:
        """
        Commit current offsets asynchronously.
        """
        if self.consumer:
            self.consumer.commit_async()
            self.logger.debug("Committed offsets asynchronously")
    
    def get_consumer_group_metadata(self) -> Dict[str, Any]:
        """
        Get consumer group metadata.
        
        Returns:
            Dictionary containing consumer group information
        """
        if self.consumer is None:
            return {}
            
        try:
            partitions = self.consumer.assignment()
            committed = self.consumer.committed(partitions)
            positions = {tp: self.consumer.position(tp) for tp in partitions}
            
            return {
                'group_id': self.consumer_group,
                'assigned_partitions': [str(tp) for tp in partitions],
                'committed_offsets': {str(tp): offset for tp, offset in committed.items()},
                'current_positions': {str(tp): pos for tp, pos in positions.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metadata: {e}")
            return {}
    
    def close(self) -> None:
        """
        Close the Kafka consumer connection.
        """
        if self.consumer:
            self.consumer.close()
            self.logger.info("Kafka consumer connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class HighThroughputConsumer(TransactionConsumer):
    """
    High-throughput consumer optimized for maximum performance.
    
    Uses optimized settings for high-volume transaction processing.
    """
    
    def __init__(self, topic: str, **kwargs):
        # High-performance configuration
        high_perf_config = {
            'max_poll_records': 500,
            'fetch_min_bytes': 50000,
            'fetch_max_wait_ms': 500,
            'max_partition_fetch_bytes': 1048576,  # 1MB
            'consumer_timeout_ms': 100,
            'enable_auto_commit': False  # Manual commit for better control
        }
        
        kwargs.update(high_perf_config)
        super().__init__(topic, **kwargs)
        
    def consume_high_throughput(self) -> Iterator[list[Dict[str, Any]]]:
        """
        Consume messages in high-throughput batches.
        
        Yields:
            Batches of transaction dictionaries
        """
        if self.consumer is None:
            self.connect()
            
        try:
            while True:
                batch = self.consume_batch(max_records=500, timeout_ms=100)
                if batch:
                    yield batch
                    # Manual commit after processing batch
                    self.consumer.commit()
                    
        except KeyboardInterrupt:
            self.logger.info("High-throughput consumer interrupted")
            self.close()
