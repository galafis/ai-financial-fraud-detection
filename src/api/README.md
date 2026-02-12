# API

FastAPI application for the fraud detection system.

## Files

- `main.py` — Application entry point. Defines all endpoints (`/api/v1/predict`, `/api/v1/health`, `/api/v1/metrics`, `/api/v1/auth/token`), middleware (rate limiting, request logging), and Pydantic models.

## Running locally

```bash
uvicorn src.api.main:app --reload
```

Interactive docs are served at `/docs` (Swagger UI) when the app is running.
