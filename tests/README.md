# Tests

## Structure

```
tests/
├── unit/
│   ├── test_features.py        # Feature engineering & data validation tests
│   └── test_ensemble_model.py  # Ensemble model load/predict tests
├── integration/
│   ├── test_api.py             # FastAPI endpoint tests (uses TestClient)
│   └── test_data_streaming.py  # Kafka consumer tests (mocked)
└── performance/
    └── test_latency.py         # Latency benchmarks via TestClient
```

## Running

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (no running services needed — uses mocks/TestClient)
pytest tests/integration/ -v

# Performance benchmarks
pytest tests/performance/ -m performance -v

# Everything
pytest tests/ -v
```
