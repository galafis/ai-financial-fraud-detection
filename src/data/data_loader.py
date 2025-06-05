"""
AI-Powered Financial Fraud Detection System
Data Loader Module

This module provides functionality to load and preprocess transaction data
from various sources (CSV, database, Kafka, etc.).

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional
import sqlite3
import json
import logging
from sqlalchemy import create_engine

# Local imports
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Data loader for transaction data from various sources.
    
    This class provides methods to load data from:
    1. CSV files
    2. SQL databases
    3. Kafka streams
    4. JSON files
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        database_url: Optional[str] = None,
        kafka_config: Optional[Dict] = None
    ):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing data files
            database_url: URL for database connection
            kafka_config: Configuration for Kafka connection
        """
        self.data_dir = data_dir
        self.database_url = database_url
        self.kafka_config = kafka_config
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info("Initialized DataLoader")
    
    def load_csv(
        self,
        file_path: str,
        date_columns: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            date_columns: List of columns to parse as dates
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from CSV file: {file_path}")
        
        # Resolve path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.data_dir, file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path, **kwargs)
        
        # Parse date columns
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    
    def load_from_database(
        self,
        query: str,
        database_url: Optional[str] = None,
        date_columns: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a SQL database.
        
        Args:
            query: SQL query to execute
            database_url: URL for database connection (overrides instance attribute)
            date_columns: List of columns to parse as dates
            **kwargs: Additional arguments to pass to pd.read_sql
            
        Returns:
            DataFrame with loaded data
        """
        # Use provided database_url or instance attribute
        db_url = database_url or self.database_url
        
        if not db_url:
            raise ValueError("Database URL not provided")
        
        logger.info(f"Loading data from database with query: {query[:100]}...")
        
        # Create engine
        engine = create_engine(db_url)
        
        # Load data
        df = pd.read_sql(query, engine, **kwargs)
        
        # Parse date columns
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        logger.info(f"Loaded {len(df)} rows from database")
        return df
    
    def load_from_sqlite(
        self,
        query: str,
        db_path: str,
        date_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Load data from a SQLite database.
        
        Args:
            query: SQL query to execute
            db_path: Path to the SQLite database file
            date_columns: List of columns to parse as dates
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from SQLite database: {db_path}")
        
        # Resolve path
        if not os.path.isabs(db_path):
            db_path = os.path.join(self.data_dir, db_path)
        
        # Check if file exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Load data
        df = pd.read_sql(query, conn)
        
        # Close connection
        conn.close()
        
        # Parse date columns
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        logger.info(f"Loaded {len(df)} rows from SQLite database")
        return df
    
    def load_from_json(
        self,
        file_path: str,
        date_columns: List[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            date_columns: List of columns to parse as dates
            **kwargs: Additional arguments to pass to pd.read_json
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from JSON file: {file_path}")
        
        # Resolve path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.data_dir, file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load data
        df = pd.read_json(file_path, **kwargs)
        
        # Parse date columns
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    
    def load_from_kafka(
        self,
        topic: str,
        timeout_ms: int = 10000,
        max_records: int = 1000,
        kafka_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load data from a Kafka topic.
        
        Args:
            topic: Kafka topic to consume from
            timeout_ms: Timeout in milliseconds
            max_records: Maximum number of records to consume
            kafka_config: Configuration for Kafka connection (overrides instance attribute)
            
        Returns:
            DataFrame with loaded data
        """
        # Use provided kafka_config or instance attribute
        config = kafka_config or self.kafka_config
        
        if not config:
            raise ValueError("Kafka configuration not provided")
        
        logger.info(f"Loading data from Kafka topic: {topic}")
        
        try:
            # Import Kafka library
            from confluent_kafka import Consumer
            
            # Create consumer
            consumer_config = {
                'bootstrap.servers': config.get('bootstrap.servers', 'localhost:9092'),
                'group.id': config.get('group.id', 'fraud_detection_group'),
                'auto.offset.reset': 'earliest'
            }
            
            consumer = Consumer(consumer_config)
            
            # Subscribe to topic
            consumer.subscribe([topic])
            
            # Consume messages
            messages = []
            start_time = datetime.now()
            
            while len(messages) < max_records:
                # Check timeout
                if (datetime.now() - start_time).total_seconds() * 1000 > timeout_ms:
                    logger.info(f"Timeout reached after consuming {len(messages)} messages")
                    break
                
                # Poll for message
                msg = consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                # Parse message
                try:
                    value = json.loads(msg.value().decode('utf-8'))
                    messages.append(value)
                except Exception as e:
                    logger.error(f"Error parsing message: {str(e)}")
            
            # Close consumer
            consumer.close()
            
            # Convert to DataFrame
            df = pd.DataFrame(messages)
            
            logger.info(f"Loaded {len(df)} records from Kafka topic {topic}")
            return df
        
        except ImportError:
            logger.error("confluent_kafka package not installed")
            raise ImportError("Please install confluent_kafka: pip install confluent-kafka")
        
        except Exception as e:
            logger.error(f"Error loading data from Kafka: {str(e)}")
            raise
    
    def generate_synthetic_data(
        self,
        n_samples: int = 10000,
        fraud_ratio: float = 0.01,
        start_date: datetime = None,
        end_date: datetime = None,
        n_customers: int = 1000,
        n_merchants: int = 100,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic transaction data for testing.
        
        Args:
            n_samples: Number of transactions to generate
            fraud_ratio: Ratio of fraudulent transactions
            start_date: Start date for transactions
            end_date: End date for transactions
            n_customers: Number of unique customers
            n_merchants: Number of unique merchants
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic transaction data
        """
        logger.info(f"Generating {n_samples} synthetic transactions")
        
        # Set random seed
        np.random.seed(random_state)
        
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime(2024, 1, 1)
        
        if end_date is None:
            end_date = datetime(2025, 6, 1)
        
        # Generate timestamps
        timestamps = [
            start_date + timedelta(
                seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
            )
            for _ in range(n_samples)
        ]
        
        # Generate customer IDs
        customer_ids = [f"CUST{i:04d}" for i in np.random.randint(1, n_customers + 1, n_samples)]
        
        # Generate merchant IDs
        merchant_ids = [f"MERCH{i:03d}" for i in np.random.randint(1, n_merchants + 1, n_samples)]
        
        # Generate transaction amounts
        amounts = np.random.exponential(scale=100, size=n_samples)
        
        # Generate payment methods
        payment_methods = np.random.choice(
            ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet'],
            size=n_samples,
            p=[0.5, 0.3, 0.1, 0.1]
        )
        
        # Generate fraud labels (imbalanced)
        fraud_labels = np.random.choice([0, 1], size=n_samples, p=[1 - fraud_ratio, fraud_ratio])
        
        # Create DataFrame
        data = pd.DataFrame({
            'transaction_id': [f"TX{i:06d}" for i in range(n_samples)],
            'timestamp': timestamps,
            'customer_id': customer_ids,
            'merchant_id': merchant_ids,
            'amount': amounts,
            'payment_method': payment_methods,
            'is_fraud': fraud_labels
        })
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Add more features for fraudulent transactions to make them detectable
        for i, row in data.iterrows():
            if row['is_fraud'] == 1:
                # Fraudulent transactions tend to have unusual amounts
                data.at[i, 'amount'] = np.random.choice([
                    np.random.uniform(1, 10),  # Very small amount
                    np.random.uniform(1000, 5000)  # Very large amount
                ])
                
                # Fraudulent transactions often occur at unusual times
                hour = np.random.randint(0, 24)
                if hour < 6:  # Night time (midnight to 6 AM)
                    data.at[i, 'timestamp'] = data.at[i, 'timestamp'].replace(hour=hour)
        
        logger.info(f"Generated {n_samples} synthetic transactions ({sum(fraud_labels)} fraudulent)")
        return data
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        file_path: str,
        **kwargs
    ) -> None:
        """
        Save data to a CSV file.
        
        Args:
            df: DataFrame to save
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to df.to_csv
        """
        logger.info(f"Saving {len(df)} rows to CSV file: {file_path}")
        
        # Resolve path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.data_dir, file_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save data
        df.to_csv(file_path, **kwargs)
        
        logger.info(f"Saved {len(df)} rows to {file_path}")
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        target_column: str = 'is_fraud',
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.
        
        Args:
            df: DataFrame to split
            target_column: Name of the target column
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            stratify: Whether to stratify by target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        logger.info(f"Splitting data into train and test sets (test_size={test_size})")
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Split data
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        logger.info(f"Split data: train={len(X_train)} samples, test={len(X_test)} samples")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader()
    
    # Generate synthetic data
    data = data_loader.generate_synthetic_data(n_samples=10000)
    
    # Save to CSV
    data_loader.save_to_csv(data, "synthetic_transactions.csv", index=False)
    
    # Load from CSV
    loaded_data = data_loader.load_csv(
        "synthetic_transactions.csv",
        date_columns=['timestamp']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = data_loader.split_train_test(loaded_data)
    
    print(f"Loaded data shape: {loaded_data.shape}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Fraud ratio in training: {y_train.mean():.4f}")
    print(f"Fraud ratio in test: {y_test.mean():.4f}")

