"""
AI-Powered Financial Fraud Detection System
API Configuration

This module contains configuration parameters for the FastAPI application.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import os
from typing import List

# API information
API_TITLE = "AI-Powered Financial Fraud Detection API"
API_DESCRIPTION = """
Enterprise-grade API for real-time financial fraud detection.

This API provides endpoints for:
- Transaction fraud analysis
- Model performance monitoring
- System health checks
"""
API_VERSION = "1.0.0"

# Authentication settings
SECRET_KEY = os.getenv("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# CORS settings
ALLOWED_ORIGINS: List[str] = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://fraud-detection-dashboard.example.com"
]

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", "models/ensemble")

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.getenv("LOG_FILE", "logs/api.log")
LOG_REQUESTS = True

# Rate limiting
RATE_LIMIT_ENABLED = True
RATE_LIMIT_REQUESTS = 100  # Number of requests
RATE_LIMIT_WINDOW = 60  # Time window in seconds

# Monitoring settings
PROMETHEUS_ENABLED = True
METRICS_ENDPOINT = "/api/v1/metrics"

# Kafka settings for real-time processing
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_TRANSACTIONS = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")
KAFKA_TOPIC_PREDICTIONS = os.getenv("KAFKA_TOPIC_PREDICTIONS", "fraud_predictions")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "fraud_detection_api")

# Redis settings for caching
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))  # Time in seconds

