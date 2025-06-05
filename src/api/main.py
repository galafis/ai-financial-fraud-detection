"""
FastAPI Application for AI-Powered Financial Fraud Detection System

High-performance REST API for real-time fraud detection with comprehensive
monitoring, authentication, and explainability features.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import asyncio
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import redis
import json
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn

# Import our fraud detection model
from src.models.ensemble_model import FraudDetectionEnsemble
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.monitoring.metrics_collector import MetricsCollector

# Configure logging
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Financial Fraud Detection API",
    description="Enterprise-grade fraud detection system with real-time processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()

# Global variables
fraud_model: Optional[FraudDetectionEnsemble] = None
redis_client: Optional[redis.Redis] = None
metrics_collector: Optional[MetricsCollector] = None

# Prometheus metrics
REQUEST_COUNT = Counter('fraud_api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('fraud_api_request_duration_seconds', 'Request duration')
FRAUD_PREDICTIONS = Counter('fraud_predictions_total', 'Total fraud predictions', ['result'])
MODEL_LATENCY = Histogram('fraud_model_latency_seconds', 'Model prediction latency')
ACTIVE_CONNECTIONS = Gauge('fraud_api_active_connections', 'Active API connections')

# Pydantic models
class TransactionRequest(BaseModel):
    """Transaction data for fraud detection."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(..., min_length=3, max_length=3, description="Currency code")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    merchant_category: str = Field(..., description="Merchant category code")
    country_code: str = Field(..., min_length=2, max_length=2, description="Country code")
    payment_method: str = Field(..., description="Payment method")
    
    # Optional fields for enhanced detection
    customer_age: Optional[int] = Field(None, ge=18, le=120, description="Customer age")
    customer_income: Optional[float] = Field(None, ge=0, description="Customer income")
    transaction_hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of transaction")
    is_weekend: Optional[bool] = Field(None, description="Is weekend transaction")
    days_since_last_transaction: Optional[int] = Field(None, ge=0, description="Days since last transaction")
    
    @validator('currency')
    def validate_currency(cls, v):
        """Validate currency code."""
        if not v.isupper():
            raise ValueError('Currency code must be uppercase')
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp is not in the future."""
        if v > datetime.utcnow():
            raise ValueError('Transaction timestamp cannot be in the future')
        return v

class FraudPredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    is_fraud: bool = Field(..., description="Binary fraud classification")
    risk_level: str = Field(..., description="Risk level (LOW, MEDIUM, HIGH, CRITICAL)")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    # Explainability
    top_risk_factors: List[Dict[str, Any]] = Field(..., description="Top contributing risk factors")
    
class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    
    transactions: List[TransactionRequest] = Field(..., max_items=1000, description="List of transactions")
    include_explanations: bool = Field(True, description="Include SHAP explanations")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    
    predictions: List[FraudPredictionResponse]
    total_processed: int
    processing_time_ms: float
    batch_id: str

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: datetime
    version: str
    model_loaded: bool
    redis_connected: bool
    uptime_seconds: float

class ModelMetrics(BaseModel):
    """Model performance metrics."""
    
    total_predictions: int
    fraud_rate: float
    average_latency_ms: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_updated: datetime

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application components."""
    global fraud_model, redis_client, metrics_collector
    
    logger.info("Starting AI Fraud Detection API...")
    
    try:
        # Load configuration
        settings = get_settings()
        
        # Initialize Redis connection
        redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Redis connection established")
        
        # Load fraud detection model
        model_path = settings.model_path
        fraud_model = FraudDetectionEnsemble.load_model(model_path)
        logger.info(f"Fraud detection model loaded from {model_path}")
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector(redis_client)
        logger.info("Metrics collector initialized")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AI Fraud Detection API...")
    
    if redis_client:
        redis_client.close()
    
    logger.info("API shutdown completed")

# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect request metrics."""
    start_time = time.time()
    
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path
        ).inc()
        REQUEST_DURATION.observe(duration)
        
        return response
    
    finally:
        ACTIVE_CONNECTIONS.dec()

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API token."""
    # In production, implement proper JWT validation
    if credentials.credentials != "your-api-token":
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Feature engineering function
def engineer_features(transaction: TransactionRequest) -> pd.DataFrame:
    """Engineer features from transaction data."""
    
    # Base features
    features = {
        'amount': transaction.amount,
        'amount_log': np.log1p(transaction.amount),
        'hour': transaction.timestamp.hour,
        'day_of_week': transaction.timestamp.weekday(),
        'is_weekend': transaction.timestamp.weekday() >= 5,
        'month': transaction.timestamp.month,
    }
    
    # Categorical encoding (simplified)
    merchant_categories = ['grocery', 'gas', 'restaurant', 'retail', 'online', 'other']
    for cat in merchant_categories:
        features[f'merchant_{cat}'] = 1 if transaction.merchant_category.lower() == cat else 0
    
    payment_methods = ['credit', 'debit', 'cash', 'mobile', 'other']
    for method in payment_methods:
        features[f'payment_{method}'] = 1 if transaction.payment_method.lower() == method else 0
    
    # Customer features
    if transaction.customer_age:
        features['customer_age'] = transaction.customer_age
        features['customer_age_group'] = transaction.customer_age // 10
    else:
        features['customer_age'] = 35  # Default
        features['customer_age_group'] = 3
    
    if transaction.customer_income:
        features['customer_income'] = transaction.customer_income
        features['customer_income_log'] = np.log1p(transaction.customer_income)
    else:
        features['customer_income'] = 50000  # Default
        features['customer_income_log'] = np.log1p(50000)
    
    # Historical features (would come from Redis in production)
    features.update({
        'customer_transaction_count_1d': 5,
        'customer_transaction_count_7d': 25,
        'customer_avg_amount_7d': transaction.amount * 0.8,
        'customer_std_amount_7d': transaction.amount * 0.3,
        'merchant_transaction_count_1d': 100,
        'merchant_fraud_rate_7d': 0.02,
        'time_since_last_transaction_hours': 24,
        'velocity_1h': 1,
        'velocity_24h': 5,
    })
    
    # Add more engineered features to reach ~50 features
    for i in range(len(features), 50):
        features[f'feature_{i}'] = np.random.normal(0, 1)
    
    return pd.DataFrame([features])

def calculate_risk_level(fraud_probability: float) -> str:
    """Calculate risk level based on fraud probability."""
    if fraud_probability >= 0.8:
        return "CRITICAL"
    elif fraud_probability >= 0.6:
        return "HIGH"
    elif fraud_probability >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "AI-Powered Financial Fraud Detection API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    start_time = getattr(app.state, 'start_time', time.time())
    uptime = time.time() - start_time
    
    # Check Redis connection
    redis_connected = False
    try:
        if redis_client:
            redis_client.ping()
            redis_connected = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if fraud_model and redis_connected else "degraded",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        model_loaded=fraud_model is not None,
        redis_connected=redis_connected,
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Predict fraud for a single transaction."""
    
    if not fraud_model:
        raise HTTPException(status_code=503, detail="Fraud detection model not available")
    
    start_time = time.time()
    
    try:
        # Engineer features
        features_df = engineer_features(transaction)
        
        # Make prediction
        with MODEL_LATENCY.time():
            fraud_probability = fraud_model.predict(features_df)[0]
            predictions, explanations = fraud_model.predict_with_explanation(features_df, explain_top_n=5)
        
        # Calculate derived metrics
        is_fraud = fraud_probability > 0.5
        risk_level = calculate_risk_level(fraud_probability)
        confidence_score = abs(fraud_probability - 0.5) * 2  # Distance from decision boundary
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Extract top risk factors
        top_risk_factors = []
        if explanations:
            top_risk_factors = explanations[0]['top_features']
        
        # Record metrics
        FRAUD_PREDICTIONS.labels(result='fraud' if is_fraud else 'legitimate').inc()
        
        # Store prediction in Redis for monitoring
        background_tasks.add_task(
            store_prediction_async,
            transaction.transaction_id,
            fraud_probability,
            is_fraud,
            processing_time_ms
        )
        
        response = FraudPredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            risk_level=risk_level,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            model_version="1.0.0",
            timestamp=datetime.utcnow(),
            top_risk_factors=top_risk_factors
        )
        
        logger.info(f"Prediction completed for transaction {transaction.transaction_id}: "
                   f"fraud_probability={fraud_probability:.4f}, processing_time={processing_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed for transaction {transaction.transaction_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Predict fraud for multiple transactions."""
    
    if not fraud_model:
        raise HTTPException(status_code=503, detail="Fraud detection model not available")
    
    start_time = time.time()
    batch_id = f"batch_{int(time.time())}"
    
    try:
        predictions = []
        
        for transaction in request.transactions:
            # Engineer features
            features_df = engineer_features(transaction)
            
            # Make prediction
            fraud_probability = fraud_model.predict(features_df)[0]
            
            # Get explanations if requested
            top_risk_factors = []
            if request.include_explanations:
                _, explanations = fraud_model.predict_with_explanation(features_df, explain_top_n=3)
                if explanations:
                    top_risk_factors = explanations[0]['top_features']
            
            # Calculate derived metrics
            is_fraud = fraud_probability > 0.5
            risk_level = calculate_risk_level(fraud_probability)
            confidence_score = abs(fraud_probability - 0.5) * 2
            
            prediction = FraudPredictionResponse(
                transaction_id=transaction.transaction_id,
                fraud_probability=fraud_probability,
                is_fraud=is_fraud,
                risk_level=risk_level,
                confidence_score=confidence_score,
                processing_time_ms=0,  # Will be calculated for batch
                model_version="1.0.0",
                timestamp=datetime.utcnow(),
                top_risk_factors=top_risk_factors
            )
            
            predictions.append(prediction)
            
            # Record metrics
            FRAUD_PREDICTIONS.labels(result='fraud' if is_fraud else 'legitimate').inc()
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Store batch results
        background_tasks.add_task(
            store_batch_results_async,
            batch_id,
            predictions,
            processing_time_ms
        )
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=processing_time_ms,
            batch_id=batch_id
        )
        
        logger.info(f"Batch prediction completed: {len(predictions)} transactions, "
                   f"processing_time={processing_time_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/metrics", response_model=ModelMetrics)
async def get_model_metrics(current_user: str = Depends(get_current_user)):
    """Get model performance metrics."""
    
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not available")
    
    try:
        metrics = await metrics_collector.get_model_metrics()
        return ModelMetrics(**metrics)
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus metrics."""
    return generate_latest()

# Background tasks
async def store_prediction_async(transaction_id: str, fraud_probability: float, 
                                is_fraud: bool, processing_time_ms: float):
    """Store prediction results asynchronously."""
    try:
        if redis_client:
            prediction_data = {
                'transaction_id': transaction_id,
                'fraud_probability': fraud_probability,
                'is_fraud': is_fraud,
                'processing_time_ms': processing_time_ms,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store with TTL of 7 days
            redis_client.setex(
                f"prediction:{transaction_id}",
                timedelta(days=7),
                json.dumps(prediction_data)
            )
            
            # Update metrics
            if metrics_collector:
                await metrics_collector.update_prediction_metrics(
                    fraud_probability, is_fraud, processing_time_ms
                )
                
    except Exception as e:
        logger.error(f"Failed to store prediction: {e}")

async def store_batch_results_async(batch_id: str, predictions: List[FraudPredictionResponse], 
                                   processing_time_ms: float):
    """Store batch results asynchronously."""
    try:
        if redis_client:
            batch_data = {
                'batch_id': batch_id,
                'total_predictions': len(predictions),
                'fraud_count': sum(1 for p in predictions if p.is_fraud),
                'processing_time_ms': processing_time_ms,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store with TTL of 7 days
            redis_client.setex(
                f"batch:{batch_id}",
                timedelta(days=7),
                json.dumps(batch_data)
            )
            
    except Exception as e:
        logger.error(f"Failed to store batch results: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    # Store start time
    app.state.start_time = time.time()
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )

