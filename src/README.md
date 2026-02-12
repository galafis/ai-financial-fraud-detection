# src/

Source code for the AI Financial Fraud Detection system.

## Modules

| Directory | Contents |
|---|---|
| `api/` | FastAPI application (`main.py`) — endpoints, auth, middleware |
| `config/` | Settings files (`api_config.py`, `model_config.py`) |
| `data/` | Data loading (`data_loader.py`), feature engineering (`feature_engineering.py`), Kafka streaming (`streaming/kafka_consumer.py`) |
| `models/` | Ensemble model (`ensemble_model.py`) with sub-directories for supervised, unsupervised, ensemble, and training |
| `monitoring/` | Model monitoring (`model_monitoring.py`) |
| `utils/` | Shared logger (`logger.py`) |
| `features/` | Placeholder module — no implementation yet |
| `inference/` | Placeholder module — no implementation yet |
