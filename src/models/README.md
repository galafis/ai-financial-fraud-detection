# Machine Learning Models - AI Financial Fraud Detection

## 🎯 Overview

This directory contains machine learning model implementations for fraud detection, including supervised learning algorithms, unsupervised anomaly detection, and ensemble methods.

## 📁 Structure

```
models/
├── supervised/              # Supervised learning models
│   ├── __init__.py
│   ├── random_forest.py     # Random Forest classifier
│   ├── xgboost_model.py     # XGBoost implementation
│   ├── neural_network.py    # Deep learning models
│   └── logistic_regression.py
├── unsupervised/            # Unsupervised learning
│   ├── __init__.py
│   ├── isolation_forest.py  # Anomaly detection
│   ├── autoencoder.py       # Neural autoencoder
│   └── clustering.py        # Clustering algorithms
├── ensemble/                # Ensemble methods
│   ├── __init__.py
│   ├── voting_classifier.py # Voting ensemble
│   ├── stacking_model.py    # Stacking ensemble
│   └── fraud_ensemble.py    # Custom fraud ensemble
├── training/                # Training pipelines
│   ├── __init__.py
│   ├── trainer.py           # Main training orchestrator
│   ├── validator.py         # Model validation
│   └── hyperparams.py       # Hyperparameter tuning
├── evaluation/              # Model evaluation
│   ├── __init__.py
│   ├── metrics.py           # Custom metrics
│   └── reports.py           # Evaluation reports
└── __init__.py
```

## 🤖 Model Types

### Supervised Learning Models
- **Random Forest**: Robust ensemble of decision trees with high interpretability
- **XGBoost**: Gradient boosting with excellent performance on tabular data
- **Neural Networks**: Deep learning models for complex pattern recognition
- **Logistic Regression**: Linear model baseline with good interpretability

### Unsupervised Learning
- **Isolation Forest**: Tree-based anomaly detection for outlier identification
- **Autoencoders**: Neural networks for detecting anomalous patterns
- **Clustering**: K-means and DBSCAN for transaction pattern analysis

### Ensemble Methods
- **Voting Classifier**: Combines multiple models through majority voting
- **Stacking**: Meta-learning approach with multiple model layers
- **Custom Fraud Ensemble**: Specialized ensemble for fraud detection

## 🚀 Usage Examples

### Training a Single Model
```python
from src.models.supervised.xgboost_model import XGBoostClassifier
from src.models.training.trainer import ModelTrainer

# Initialize model
model = XGBoostClassifier()
trainer = ModelTrainer(model)

# Train with data
trainer.train(X_train, y_train, X_val, y_val)

# Save trained model
trainer.save_model('fraud_xgboost_v1.0.0')
```

### Using Ensemble Model
```python
from src.models.ensemble.fraud_ensemble import FraudEnsemble

# Load pre-trained ensemble
ensemble = FraudEnsemble()
ensemble.load_models(['random_forest_v1.0', 'xgboost_v1.0', 'neural_net_v1.0'])

# Make predictions
predictions = ensemble.predict(X_test)
proba = ensemble.predict_proba(X_test)
```

### Hyperparameter Tuning
```python
from src.models.training.hyperparams import HyperparameterTuner
from src.models.supervised.xgboost_model import XGBoostClassifier

tuner = HyperparameterTuner(
    model_class=XGBoostClassifier,
    param_grid={
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    }
)

best_model = tuner.tune(X_train, y_train, cv=5)
```

## 📊 Model Performance

### Evaluation Metrics
- **Precision**: Reduce false positive rate
- **Recall**: Capture actual fraud cases
- **F1-Score**: Balance between precision and recall
- **AUC-ROC**: Overall model discrimination ability
- **PR-AUC**: Area under precision-recall curve (important for imbalanced data)

### Model Comparison
```python
from src.models.evaluation.reports import ModelComparison

comparison = ModelComparison()
comparison.add_model('RandomForest', rf_predictions, rf_probabilities)
comparison.add_model('XGBoost', xgb_predictions, xgb_probabilities)
comparison.add_model('Ensemble', ensemble_predictions, ensemble_probabilities)

# Generate comprehensive report
report = comparison.generate_report(y_test)
print(report)
```

## 🔧 Model Configuration

### Configuration Files
Models are configured through YAML files in the `config/` directory:

```yaml
# config/xgboost_config.yaml
xgboost:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 6
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
  
  # Fraud-specific settings
  scale_pos_weight: 10  # Handle class imbalance
  eval_metric: 'auc'
  early_stopping_rounds: 10
```

### Loading Configuration
```python
from src.models.supervised.xgboost_model import XGBoostClassifier
from src.config.model_config import load_model_config

config = load_model_config('xgboost')
model = XGBoostClassifier(**config)
```

## 🎛️ Model Versioning

### Version Management
```python
from src.models.training.trainer import ModelTrainer

trainer = ModelTrainer(model)
trainer.train(X_train, y_train)

# Save with semantic versioning
trainer.save_model(
    name='fraud_xgboost',
    version='1.2.0',
    metadata={
        'accuracy': 0.95,
        'training_date': '2024-01-15',
        'features_used': feature_names,
        'performance_metrics': metrics
    }
)
```

### Model Registry
```python
from src.models.registry import ModelRegistry

registry = ModelRegistry()

# Register model
registry.register_model(
    name='fraud_ensemble_v2',
    model_path='models/fraud_ensemble_v2.pkl',
    metrics={'auc': 0.98, 'precision': 0.85},
    stage='production'
)

# Load production model
production_model = registry.load_model('fraud_ensemble', stage='production')
```

## 🧪 Model Testing

### Unit Tests
```bash
# Run model-specific tests
pytest tests/unit/models/test_xgboost.py -v

# Run all model tests
pytest tests/unit/models/ -v
```

### Integration Tests
```bash
# Test model training pipeline
pytest tests/integration/test_training_pipeline.py -v

# Test model serving
pytest tests/integration/test_model_serving.py -v
```

## 📈 Performance Monitoring

### Model Drift Detection
```python
from src.models.monitoring.drift_detector import DriftDetector

drift_detector = DriftDetector(reference_data=X_train)

# Check for data drift
drift_score = drift_detector.detect_drift(new_data=X_current)
if drift_score > 0.1:
    print("Data drift detected - consider model retraining")
```

### Performance Tracking
```python
from src.models.monitoring.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()
tracker.log_prediction(y_true, y_pred, model_version='v1.2.0')

# Get performance trends
trends = tracker.get_performance_trends(days=30)
```

## 🔄 Model Retraining

### Automated Retraining
```python
from src.models.training.retrainer import AutoRetrainer

retrainer = AutoRetrainer(
    model_config='xgboost_config.yaml',
    retrain_threshold=0.05,  # Retrain if performance drops by 5%
    schedule='weekly'
)

retrainer.start_monitoring()
```

### Manual Retraining
```python
from src.models.training.trainer import ModelTrainer

# Load existing model
trainer = ModelTrainer.load_checkpoint('fraud_model_v1.1.0')

# Continue training with new data
trainer.continue_training(X_new, y_new)

# Save updated model
trainer.save_model('fraud_model_v1.2.0')
```

## 🛠️ Development Guidelines

### Adding New Models
1. Create model class inheriting from `BaseModel`
2. Implement required methods: `fit()`, `predict()`, `predict_proba()`
3. Add configuration file
4. Write unit tests
5. Update documentation

### Code Standards
- Follow scikit-learn API conventions
- Include type hints and docstrings
- Implement proper error handling
- Add logging for debugging

### Testing Requirements
- Unit tests for all model classes
- Integration tests for training pipelines
- Performance benchmarks
- Edge case handling

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

## 🤝 Contributing

When adding new models:
1. Follow the established directory structure
2. Include comprehensive documentation
3. Add appropriate tests
4. Update performance benchmarks
5. Consider ensemble integration

For questions or issues, please open a GitHub issue or refer to the main project documentation.
