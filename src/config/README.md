# Config

Centralised configuration for the fraud detection system.

## Files

- `api_config.py` — API settings (title, auth, CORS, rate limiting, Kafka/Redis connection params). Values are read from environment variables with sensible defaults.
- `model_config.py` — ML model hyper-parameters, fraud thresholds, ensemble weights, and feature engineering parameters.
