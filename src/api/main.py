"""
FastAPI Application for AI-Powered Financial Fraud Detection System

This module implements the REST API for the fraud detection system,
providing endpoints for transaction analysis, model monitoring,
and system health checks.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator
import jwt
from jwt.exceptions import InvalidTokenError

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Local imports
from ..models.ensemble_model import FraudDetectionEnsemble
from ..utils.logger import get_logger
from ..config.api_config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALLOWED_ORIGINS,
    MODEL_PATH,
    LOG_REQUESTS,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW
)

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize Prometheus metrics
FRAUD_PREDICTIONS = Counter(
    'fraud_predictions_total',
    'Total number of fraud predictions',
    ['result']  # 'fraud' or 'legitimate'
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Latency of fraud predictions',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

MODEL_CONFIDENCE = Gauge(
    'model_confidence',
    'Average confidence score of the model',
)

API_REQUESTS = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status']
)

# Rate limiting
rate_limit_store = {}  # Simple in-memory store for rate limiting

# Load the fraud detection model
try:
    model = FraudDetectionEnsemble.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None


# Pydantic models for request/response validation
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


class Location(BaseModel):
    latitude: float
    longitude: float


class Transaction(BaseModel):
    transaction_id: str
    amount: float
    merchant_id: str
    customer_id: str
    timestamp: datetime
    payment_method: str
    card_last_four: Optional[str] = None
    ip_address: Optional[str] = None
    device_id: Optional[str] = None
    location: Optional[Location] = None
    additional_data: Optional[Dict[str, Any]] = None
    
    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    explanation: Optional[Dict[str, Any]] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    timestamp: datetime


class MetricsResponse(BaseModel):
    model_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any]
    timestamp: datetime


# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hashed password."""
    # In a real system, use proper password hashing (e.g., bcrypt)
    # This is a simplified example
    return plain_password == hashed_password


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    # In a real system, fetch from database
    # This is a simplified example with hardcoded users
    users_db = {
        "admin": {
            "username": "admin",
            "full_name": "Admin User",
            "email": "admin@example.com",
            "hashed_password": "admin_password",
            "disabled": False
        },
        "api_user": {
            "username": "api_user",
            "full_name": "API User",
            "email": "api@example.com",
            "hashed_password": "api_password",
            "disabled": False
        }
    }
    
    if username in users_db:
        user_dict = users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Union[UserInDB, bool]:
    """Authenticate user with username and password."""
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    if RATE_LIMIT_ENABLED:
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean up old entries
        for ip in list(rate_limit_store.keys()):
            if current_time - rate_limit_store[ip]["timestamp"] > RATE_LIMIT_WINDOW:
                del rate_limit_store[ip]
        
        # Check rate limit
        if client_ip in rate_limit_store:
            if rate_limit_store[client_ip]["count"] >= RATE_LIMIT_REQUESTS:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."}
                )
            rate_limit_store[client_ip]["count"] += 1
        else:
            rate_limit_store[client_ip] = {
                "count": 1,
                "timestamp": current_time
            }
    
    # Continue with the request
    response = await call_next(request)
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    if LOG_REQUESTS:
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Log request details
        process_time = time.time() - start_time
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"Status: {response.status_code} "
            f"Duration: {process_time:.4f}s"
        )
        
        # Update Prometheus metrics
        API_REQUESTS.labels(
            endpoint=request.url.path,
            method=request.method,
            status=response.status_code
        ).inc()
        
        return response
    else:
        return await call_next(request)


# API endpoints
@app.post("/api/v1/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Endpoint to get JWT access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_fraud(
    transaction: Transaction,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """
    Predict fraud probability for a transaction.
    
    This endpoint analyzes a financial transaction and returns:
    - Fraud probability (0-1)
    - Binary fraud decision (true/false)
    - Risk level (low, medium, high)
    - Explanation of the prediction
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    start_time = time.time()
    
    try:
        # Convert transaction to DataFrame
        transaction_dict = transaction.dict()
        
        # Extract nested fields
        if transaction_dict.get("location"):
            transaction_dict["latitude"] = transaction_dict["location"]["latitude"]
            transaction_dict["longitude"] = transaction_dict["location"]["longitude"]
            del transaction_dict["location"]
        
        # Extract additional data if present
        additional_data = transaction_dict.get("additional_data", {})
        if additional_data:
            for key, value in additional_data.items():
                transaction_dict[f"additional_{key}"] = value
            del transaction_dict["additional_data"]
        
        # Convert timestamp to features
        transaction_dict["hour_of_day"] = transaction_dict["timestamp"].hour
        transaction_dict["day_of_week"] = transaction_dict["timestamp"].weekday()
        transaction_dict["is_weekend"] = 1 if transaction_dict["day_of_week"] >= 5 else 0
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_dict])
        
        # Remove non-numeric columns that the model doesn't use
        df = df.drop(columns=["transaction_id", "timestamp"])
        
        # One-hot encode categorical variables
        categorical_cols = ["payment_method", "merchant_id"]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Get prediction with details
        prediction_details = model.predict_with_details(df)
        fraud_probability = prediction_details["probabilities"][0]
        is_fraud = prediction_details["predictions"][0] == 1
        
        # Determine risk level
        if fraud_probability < 0.3:
            risk_level = "low"
        elif fraud_probability < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update metrics in background
        background_tasks.add_task(
            update_metrics,
            is_fraud=is_fraud,
            fraud_probability=fraud_probability,
            processing_time=processing_time_ms / 1000  # Convert to seconds
        )
        
        # Log prediction
        logger.info(
            f"Prediction: transaction_id={transaction.transaction_id}, "
            f"fraud_probability={fraud_probability:.4f}, "
            f"is_fraud={is_fraud}, risk_level={risk_level}"
        )
        
        return {
            "transaction_id": transaction.transaction_id,
            "fraud_probability": float(fraud_probability),
            "is_fraud": bool(is_fraud),
            "risk_level": risk_level,
            "explanation": prediction_details.get("explanation", {}),
            "processing_time_ms": processing_time_ms
        }
    
    except Exception as e:
        logger.error(f"Error processing transaction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing transaction: {str(e)}"
        )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and model.
    """
    return {
        "status": "ok" if model is not None else "degraded",
        "version": API_VERSION,
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow()
    }


@app.get("/api/v1/metrics")
async def get_metrics(current_user: User = Depends(get_current_active_user)):
    """
    Get system and model metrics.
    
    Returns Prometheus metrics in the standard format.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/api/v1/model/metrics", response_model=MetricsResponse)
async def get_model_metrics(current_user: User = Depends(get_current_active_user)):
    """
    Get detailed model metrics.
    
    Returns metrics about the model performance and system usage.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    # In a real system, these would be fetched from a monitoring system
    model_metrics = {
        "auc_roc": 0.98,
        "precision": 0.95,
        "recall": 0.91,
        "f1_score": 0.93,
        "false_positive_rate": 0.005,
        "false_negative_rate": 0.09,
        "average_confidence": 0.87
    }
    
    system_metrics = {
        "requests_per_minute": 120,
        "average_latency_ms": 45,
        "p95_latency_ms": 78,
        "p99_latency_ms": 95,
        "error_rate": 0.001
    }
    
    return {
        "model_metrics": model_metrics,
        "system_metrics": system_metrics,
        "timestamp": datetime.utcnow()
    }


def update_metrics(is_fraud: bool, fraud_probability: float, processing_time: float):
    """Update Prometheus metrics."""
    # Update fraud predictions counter
    result = "fraud" if is_fraud else "legitimate"
    FRAUD_PREDICTIONS.labels(result=result).inc()
    
    # Update prediction latency histogram
    PREDICTION_LATENCY.observe(processing_time)
    
    # Update model confidence gauge
    MODEL_CONFIDENCE.set(fraud_probability if is_fraud else 1 - fraud_probability)


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting Fraud Detection API")
    
    # Check if model is loaded
    if model is None:
        logger.warning("Model not loaded. Service will run in degraded mode.")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down Fraud Detection API")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

