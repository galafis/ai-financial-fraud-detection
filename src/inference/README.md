# Inference Engine - AI Financial Fraud Detection

## 🎯 Overview

This directory contains the real-time inference system for fraud detection, providing fast and accurate predictions on transaction data.

## 📁 Structure

```
inference/
├── engines/             # Inference engine implementations
│   ├── __init__.py
│   ├── fraud_detector.py    # Main fraud detection engine
│   ├── ensemble_engine.py   # Ensemble model inference
│   └── streaming_engine.py  # Stream processing engine
├── pipelines/           # Inference pipelines
│   ├── __init__.py
│   ├── preprocessing.py     # Data preprocessing pipeline
│   ├── postprocessing.py    # Result post-processing
│   └── validation.py        # Input validation
├── cache/              # Caching mechanisms
│   ├── __init__.py
│   ├── redis_cache.py      # Redis-based caching
│   └── memory_cache.py     # In-memory caching
├── explainers/         # Model explainability
│   ├── __init__.py
│   ├── shap_explainer.py   # SHAP-based explanations
│   └── lime_explainer.py   # LIME explanations
└── __init__.py
```

## 🚀 Key Components

### Fraud Detection Engine
- **Real-time Processing**: <100ms response time
- **Ensemble Models**: Combines multiple ML models
- **Adaptive Thresholds**: Context-aware decision boundaries
- **Batch Processing**: Handle multiple transactions efficiently

### Preprocessing Pipeline
- **Input Validation**: Schema validation and data quality checks
- **Feature Engineering**: Real-time feature computation
- **Data Transformation**: Scaling, encoding, and normalization
- **Missing Value Handling**: Imputation strategies

### Explainability
- **SHAP Values**: Feature importance explanations
- **LIME**: Local interpretable model explanations
- **Feature Contribution**: Individual feature impact analysis

## 💻 Usage Example

```python
from src.inference.engines.fraud_detector import FraudDetector
from src.inference.explainers.shap_explainer import ShapExplainer

# Initialize fraud detector
detector = FraudDetector()
detector.load_models()

# Process transaction
transaction = {
    'amount': 1500.00,
    'merchant': 'Online Store',
    'location': 'São Paulo',
    'timestamp': '2024-01-15T14:30:00Z',
    'user_id': 'user123'
}

# Get prediction with explanation
result = detector.predict_with_explanation(transaction)

print(f"Fraud Score: {result['fraud_score']:.3f}")
print(f"Is Fraud: {result['is_fraud']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Top Risk Factors: {result['top_features']}")
```

## ⚡ Performance Optimization

### Caching Strategy
- **Model Caching**: Pre-loaded models in memory
- **Feature Caching**: Redis-based feature store
- **Result Caching**: Recent predictions cache

### Batch Processing
```python
from src.inference.engines.fraud_detector import FraudDetector

detector = FraudDetector()
transactions = [transaction1, transaction2, transaction3]  # List of transactions

# Batch prediction for efficiency
results = detector.predict_batch(transactions)
```

## 🔄 Real-time Streaming

### Kafka Integration
```python
from src.inference.engines.streaming_engine import StreamingEngine

# Initialize streaming engine
engine = StreamingEngine(
    input_topic='raw_transactions',
    output_topic='fraud_predictions'
)

# Start processing stream
engine.start_processing()
```

### Processing Pipeline
1. **Input**: Transaction from Kafka topic
2. **Validation**: Schema and data quality validation
3. **Feature Engineering**: Real-time feature computation
4. **Prediction**: Model inference with ensemble
5. **Explanation**: Generate SHAP values
6. **Output**: Publish results to output topic

## 📊 Monitoring

### Inference Metrics
- **Latency**: P50, P95, P99 response times
- **Throughput**: Transactions per second
- **Error Rate**: Failed predictions percentage
- **Model Accuracy**: Real-time accuracy tracking

### Health Checks
```python
from src.inference.engines.fraud_detector import FraudDetector

detector = FraudDetector()
health_status = detector.health_check()

print(f"Status: {health_status['status']}")
print(f"Model Load: {health_status['models_loaded']}")
print(f"Cache Status: {health_status['cache_healthy']}")
```

## 🛡️ Security Features

### Data Protection
- **Input Sanitization**: Prevent injection attacks
- **Rate Limiting**: Prevent abuse
- **Audit Logging**: Track all inference requests
- **Encryption**: Secure data transmission

### Model Security
- **Model Versioning**: Track model changes
- **Access Control**: Authorized model access
- **Model Validation**: Integrity checks

## 🎯 Decision Thresholds

### Context-Aware Thresholds
```python
THRESHOLDS = {
    'high_value': {'amount': 5000, 'threshold': 0.3},
    'international': {'threshold': 0.4},
    'night_time': {'time_range': '22:00-06:00', 'threshold': 0.5},
    'weekend': {'threshold': 0.45},
    'default': {'threshold': 0.6}
}
```

## 📈 A/B Testing

### Model Comparison
```python
from src.inference.engines.ab_testing import ABTestEngine

# Compare model versions
ab_engine = ABTestEngine()
result = ab_engine.compare_models(
    model_a='fraud_model_v1.0',
    model_b='fraud_model_v1.1',
    traffic_split=0.1  # 10% to model B
)
```

## 🔧 Configuration

### Environment Variables
```bash
# Model settings
MODEL_VERSION=v1.2.0
MODEL_CACHE_SIZE=1000
INFERENCE_TIMEOUT=100

# Cache settings
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL=3600

# Performance settings
BATCH_SIZE=32
WORKER_THREADS=4
MAX_CONCURRENT_REQUESTS=1000
```
