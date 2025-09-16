# Feature Engineering - AI Financial Fraud Detection

## 🎯 Overview

This directory contains feature engineering modules for extracting, transforming, and managing features for fraud detection models.

## 📁 Structure

```
features/
├── extractors/          # Feature extraction modules
│   ├── __init__.py
│   ├── behavioral.py    # Behavioral features
│   ├── temporal.py      # Time-based features
│   └── geographic.py    # Location-based features
├── transformers/        # Feature transformation pipelines
│   ├── __init__.py
│   ├── scaler.py        # Feature scaling
│   └── encoder.py       # Categorical encoding
├── store/              # Feature store implementation
│   ├── __init__.py
│   └── redis_store.py   # Redis-based feature store
├── validation/         # Feature quality validation
│   ├── __init__.py
│   └── validators.py    # Feature validation rules
└── __init__.py
```

## 🔧 Key Components

### Feature Extractors
- **Behavioral**: User transaction patterns, spending habits
- **Temporal**: Time-based patterns, seasonality
- **Geographic**: Location-based features, velocity calculations

### Feature Transformers
- **Scalers**: StandardScaler, RobustScaler for numerical features
- **Encoders**: One-hot, target encoding for categorical features

### Feature Store
- Redis-based caching for real-time feature serving
- Feature versioning and lineage tracking

## 🚀 Usage Example

```python
from src.features.extractors.behavioral import BehavioralFeatures
from src.features.transformers.scaler import FeatureScaler

# Extract behavioral features
extractor = BehavioralFeatures()
features = extractor.extract_user_features(user_id='123', timeframe='30d')

# Scale features
scaler = FeatureScaler()
scaled_features = scaler.transform(features)
```

## 📊 Feature Categories

### Transactional Features
- Amount-based features (normalized amounts, amount percentiles)
- Frequency-based features (transactions per day/hour)
- Merchant-based features (merchant risk scores)

### Behavioral Features
- Historical spending patterns
- Device and channel preferences
- Transaction timing patterns

### Contextual Features
- Geographic distance from home
- Time since last transaction
- Account age and activity level

## ⚡ Real-time Feature Serving

Features are cached in Redis for low-latency serving:

```python
from src.features.store.redis_store import FeatureStore

store = FeatureStore()
# Get cached features
features = store.get_user_features(user_id='123')
```

## 🧪 Feature Validation

All features undergo quality validation:
- Data type validation
- Range checks
- Missing value detection
- Drift detection

```python
from src.features.validation.validators import FeatureValidator

validator = FeatureValidator()
validation_results = validator.validate(features)
```
