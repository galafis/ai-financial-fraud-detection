# Performance Tests

Latency and throughput benchmarks for the fraud detection API.

## Files

- `test_latency.py` — Measures per-request latency using the FastAPI `TestClient` (in-process, no network). Checks that p95 prediction latency stays under 200 ms and average health-check latency stays under 50 ms.

## Running

```bash
pytest tests/performance/ -m performance -v
```
