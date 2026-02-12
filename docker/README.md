# Docker

## Files

- `Dockerfile` — Production container image based on Python 3.9-slim. Runs the FastAPI app via Uvicorn on port 8000 as a non-root user. Health check hits `/api/v1/health`.

## Building

```bash
# From the repo root
docker build -f docker/Dockerfile -t fraud-detection-api .

# Or via docker-compose (from repo root)
docker-compose -f config/docker-compose.yml up --build
```

## Environment variables

See `src/config/api_config.py` for all supported variables. Key ones:

| Variable | Default |
|---|---|
| `SECRET_KEY` | `change-me-in-production` |
| `MODEL_PATH` | `models/ensemble` |
| `LOG_LEVEL` | `INFO` |
| `KAFKA_ENABLED` | `false` |
| `REDIS_ENABLED` | `false` |
