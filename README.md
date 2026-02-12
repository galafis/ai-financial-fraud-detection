# AI Financial Fraud Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

Real-time fraud detection system built with FastAPI and scikit-learn, using ensemble ML methods (Random Forest, XGBoost, neural networks, Isolation Forest) to score financial transactions. Includes Kafka-based streaming, Prometheus metrics, and Docker/Kubernetes deployment configs.

## Quick Start

### API Usage

```bash
# Start the services
docker-compose -f config/docker-compose.yml up -d

# Get an auth token (demo credentials)
TOKEN=$(curl -s -X POST "http://localhost:8000/api/v1/auth/token" \
  -d "username=admin&password=admin_password" | jq -r .access_token)

# Score a transaction
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_12345",
    "amount": 1500.00,
    "merchant_id": "merchant_001",
    "customer_id": "cust_789",
    "timestamp": "2024-01-15T10:30:00Z",
    "payment_method": "credit_card"
  }'
```

**Python example:**
```python
import requests

# Authenticate
token_resp = requests.post(
    "http://localhost:8000/api/v1/auth/token",
    data={"username": "admin", "password": "admin_password"}
)
token = token_resp.json()["access_token"]

# Score a transaction
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "transaction_id": "txn_12345",
        "amount": 1500.00,
        "merchant_id": "merchant_001",
        "customer_id": "cust_789",
        "timestamp": "2024-01-15T10:30:00Z",
        "payment_method": "credit_card"
    }
)

result = response.json()
print(f"Fraud probability: {result['fraud_probability']}")
print(f"Is fraud: {result['is_fraud']}")
print(f"Risk level: {result['risk_level']}")
```

**Sample response:**
```json
{
  "transaction_id": "txn_12345",
  "fraud_probability": 0.23,
  "is_fraud": false,
  "risk_level": "low",
  "explanation": null,
  "processing_time_ms": 45.2
}
```

## Models

- **Supervised**: Random Forest, XGBoost, feedforward neural network
- **Unsupervised**: Isolation Forest, autoencoder-based anomaly detection
- **Ensemble**: weighted combination with a stacking meta-learner
- **Explainability**: SHAP values (when available) for per-prediction feature attribution

## Project Structure

```
src/
├── api/            # FastAPI app (main.py)
├── config/         # Centralised settings (api_config.py, model_config.py)
├── data/           # Data loading, feature engineering, Kafka streaming
├── models/         # Ensemble model, supervised/unsupervised sub-modules
├── monitoring/     # Model monitoring utilities
└── utils/          # Logger and shared helpers
config/
├── docker-compose.yml
└── requirements.txt
docker/
└── Dockerfile
k8s/
├── deployment.yaml
└── service.yaml
tests/
├── unit/
├── integration/
└── performance/
```

## Setup

**Prerequisites:** Python 3.9+, Docker (optional), Kafka (optional, for streaming).

```bash
# Install dependencies
pip install -r config/requirements.txt

# Run the API locally
uvicorn src.api.main:app --reload

# Or use Docker
docker-compose -f config/docker-compose.yml up --build
```

## Configuration

Settings are managed through environment variables with sensible defaults in `src/config/api_config.py` and `src/config/model_config.py`.

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `change-me-in-production` | JWT signing key |
| `MODEL_PATH` | `models/ensemble` | Path to saved model artifacts |
| `KAFKA_ENABLED` | `false` | Enable Kafka streaming |
| `REDIS_ENABLED` | `false` | Enable Redis caching |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Monitoring

The API exposes Prometheus-compatible metrics at `/api/v1/metrics` (requires auth) covering prediction counts, latency histograms, and model confidence. A health check endpoint is available at `/api/v1/health` (no auth required).

## Testing

```bash
# Run tests that don't require external services
pytest tests/unit/

# Integration tests (need a running API)
pytest tests/integration/

# Performance benchmarks
pytest tests/performance/ -m performance
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. In short: fork, branch, add tests, open a PR.

## License

MIT — see [LICENSE](LICENSE).

---

# Sistema de Detecção de Fraudes Financeiras com IA

[![Licença: MIT](https://img.shields.io/badge/licenca-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

Sistema de detecção de fraudes em tempo real construído com FastAPI e scikit-learn, usando métodos de ensemble (Random Forest, XGBoost, redes neurais, Isolation Forest) para pontuar transações financeiras. Inclui streaming via Kafka, métricas Prometheus e configurações de deploy com Docker/Kubernetes.

## Início Rápido

```bash
# Iniciar os serviços
docker-compose -f config/docker-compose.yml up -d

# Obter token de autenticação (credenciais de demonstração)
TOKEN=$(curl -s -X POST "http://localhost:8000/api/v1/auth/token" \
  -d "username=admin&password=admin_password" | jq -r .access_token)

# Analisar uma transação
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_12345",
    "amount": 1500.00,
    "merchant_id": "merchant_001",
    "customer_id": "cust_789",
    "timestamp": "2024-01-15T10:30:00Z",
    "payment_method": "credit_card"
  }'
```

## Modelos

- **Supervisionados**: Random Forest, XGBoost, rede neural feedforward
- **Não supervisionados**: Isolation Forest, detecção de anomalias com autoencoder
- **Ensemble**: combinação ponderada com meta-aprendiz por stacking
- **Explicabilidade**: valores SHAP para atribuição de features por predição

## Configuração

Variáveis de ambiente controlam o comportamento do sistema. Veja `src/config/api_config.py` e `src/config/model_config.py` para valores padrão.

## Testes

```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/ -m performance
```

## Como Contribuir

Veja [CONTRIBUTING.md](CONTRIBUTING.md). Em resumo: fork, branch, adicione testes, abra um PR.

## Licença

MIT — veja [LICENSE](LICENSE).
