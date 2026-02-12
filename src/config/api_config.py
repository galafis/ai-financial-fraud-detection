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
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
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

