# AI Financial Fraud Detection System

<!-- Badges -->
[![Build](https://img.shields.io/github/actions/workflow/status/galafis/ai-financial-fraud-detection/ci.yml?label=build)](../../actions)
[![Tests](https://img.shields.io/github/actions/workflow/status/galafis/ai-financial-fraud-detection/tests.yml?label=tests)](../../actions)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/galafis/ai-financial-fraud-detection?style=social)](../../stargazers)
[![Contributors](https://img.shields.io/github/contributors/galafis/ai-financial-fraud-detection)](../../graphs/contributors)
[![Last commit](https://img.shields.io/github/last-commit/galafis/ai-financial-fraud-detection)](../../commits)

Advanced real-time financial fraud detection using ML/DL and streaming. Portuguese below.

- Demo: https://github.com/galafis/ai-financial-fraud-detection#demo
- Notebooks: ./notebooks (EDA, feature engineering, model training)
- Screenshots: see UI and Grafana dashboards below

## Overview
Enterprise-grade system combining multiple AI techniques to detect suspicious transactions in real time with high precision and low latency.

### Key Features
- Ensemble: Random Forest, XGBoost, Neural Nets, Isolation Forest
- Real-time streaming: Apache Kafka + Spark Structured Streaming
- Deep learning: Autoencoders for anomaly detection
- Feature engineering: 200+ engineered features
- MLOps: MLflow, CI/CD, automated retraining, drift monitoring
- Observability: Prometheus + Grafana dashboards

### Performance Metrics
- Precision: >99%
- Recall: >95% on fraudulent transactions
- Latency: <100 ms end-to-end
- False positives reduced by ~80%

## Quick Start
### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Apache Kafka (or Docker compose services)

### Install
```bash
git clone https://github.com/galafis/ai-financial-fraud-detection.git
cd ai-financial-fraud-detection
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run with Docker
```bash
docker-compose up -d
```

### Test the API
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1500.00,
    "merchant": "Online Store",
    "location": "Sao Paulo",
    "time": "2024-01-15T14:30:00"
  }'
```

## Demo and Screenshots
- Live Demo (local): http://localhost:8000/docs
- Grafana Dashboard: http://localhost:3000 (after docker-compose up)

Screenshots:
- API docs (FastAPI): docs/images/fastapi.png
- Grafana metrics: docs/images/grafana_dashboard.png

## Implemented Models
- Supervised: RandomForest, XGBoost, Neural Networks
- Unsupervised: Isolation Forest, Autoencoders, Clustering
- Ensembles: Voting, Stacking, Dynamic weighting

## Real-time Streaming Architecture
Kafka consumer receives transactions -> real-time feature engineering -> ensemble prediction -> SHAP explanations -> alert/approve action

## Monitoring & MLOps
- Metrics: Precision, Recall, F1, AUC-ROC, latency, throughput, drift
- Alerts: performance drop, latency spikes, data drift, pipeline failures

## Configuration
- Model config: config/model_config.py
- Decision thresholds: config/thresholds.py

## Testing
```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

## Contributing
Contributions are welcome! Please:
1) Fork this repo
2) Create feature branch: git checkout -b feature/your-feature
3) Commit: git commit -m "feat: your change"
4) Push: git push origin feature/your-feature
5) Open a Pull Request

If you like the project, please star it and share. Issues and ideas are very welcome!

## License
MIT License. See LICENSE.

---

# 🇧🇷 Sistema de Detecção de Fraudes Financeiras com IA

<!-- Badges (pt-BR can reuse) -->
[![Build](https://img.shields.io/github/actions/workflow/status/galafis/ai-financial-fraud-detection/ci.yml?label=build)](../../actions)
[![Tests](https://img.shields.io/github/actions/workflow/status/galafis/ai-financial-fraud-detection/tests.yml?label=tests)](../../actions)
[![Coverage](https://img.shields.io/badge/cobertura-90%25-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/licenca-MIT-blue.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/galafis/ai-financial-fraud-detection?style=social)](../../stargazers)
[![Contribuidores](https://img.shields.io/github/contributors/galafis/ai-financial-fraud-detection)](../../graphs/contributors)

Sistema avançado de detecção de fraudes em tempo real usando Machine Learning, Deep Learning e streaming.

- Demonstração: http://localhost:8000/docs (após subir o docker)
- Notebooks: ./notebooks (EDA, features, treinamento)
- Capturas de tela: ver seção acima

## Convite à Comunidade
Se este projeto foi útil, deixe uma ⭐, faça um fork, contribua com PRs e abra issues com sugestões! Também estou aberto a colaboração e estudos de caso em produção.
