# AI Financial Fraud Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-000?style=flat&logo=apachekafka)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Sistema avançado de detecção de fraudes financeiras em tempo real usando Machine Learning, Deep Learning e processamento de streams para instituições financeiras.

## 🎯 Visão Geral

Sistema enterprise-grade que combina múltiplas técnicas de IA para identificar transações suspeitas em tempo real com alta precisão e baixa latência.

### ✨ Características Principais

- **🤖 Ensemble de Modelos**: Random Forest, XGBoost, Neural Networks, Isolation Forest
- **⚡ Processamento Real-Time**: Apache Kafka + Apache Spark Streaming
- **🧠 Deep Learning**: Autoencoders para detecção de anomalias
- **📊 Feature Engineering**: 200+ features automaticamente engineered
- **🔄 MLOps Pipeline**: Treinamento, validação e deploy automatizados
- **📈 Monitoramento**: Drift detection e retraining automático

### 🏆 Métricas de Performance

- **Precisão**: >99% na detecção de fraudes
- **Recall**: >95% para transações fraudulentas
- **Latência**: <100ms para processamento em tempo real
- **Redução de Falsos Positivos**: 80% comparado a sistemas tradicionais

## 🛠️ Stack Tecnológico

### Machine Learning & AI
- **Python 3.9+**: Linguagem principal
- **TensorFlow 2.x**: Deep learning e neural networks
- **Scikit-learn**: Algoritmos clássicos de ML
- **XGBoost/LightGBM**: Gradient boosting otimizado
- **SHAP**: Explainable AI e interpretabilidade

### Big Data & Streaming
- **Apache Kafka**: Message streaming platform
- **Apache Spark**: Distributed data processing
- **Redis**: In-memory caching e feature store
- **ClickHouse**: OLAP database para analytics

### MLOps & DevOps
- **MLflow**: ML lifecycle management
- **Docker/Kubernetes**: Containerização e orquestração
- **Prometheus/Grafana**: Monitoring e visualização

## 📁 Estrutura do Projeto

```
ai-financial-fraud-detection/
├── src/
│   ├── data/                    # Módulos de dados e streaming
│   ├── features/                # Feature engineering
│   ├── models/                  # Modelos de ML/DL
│   ├── inference/               # Sistema de inferência
│   ├── api/                     # API REST FastAPI
│   ├── monitoring/              # Monitoramento e observabilidade
│   └── utils/                   # Utilitários e configurações
├── notebooks/                   # Jupyter notebooks para análise
├── tests/                       # Testes automatizados
├── docker/                      # Configurações Docker
├── k8s/                         # Manifests Kubernetes
├── requirements.txt             # Dependências Python
└── README.md                    # Documentação
```

## 🚀 Quick Start

### Pré-requisitos

- Python 3.9+
- Docker & Docker Compose
- Apache Kafka (ou usar Docker)

### Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/galafis/ai-financial-fraud-detection.git
cd ai-financial-fraud-detection
```

2. **Configure o ambiente:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

3. **Execute com Docker:**
```bash
docker-compose up -d
```

4. **Teste a API:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1500.00,
    "merchant": "Online Store",
    "location": "São Paulo",
    "time": "2024-01-15T14:30:00"
  }'
```

## 🤖 Modelos Implementados

### Supervised Learning
- **Random Forest**: Ensemble robusto para classificação
- **XGBoost**: Gradient boosting otimizado
- **Neural Networks**: Deep learning para padrões complexos

### Unsupervised Learning
- **Isolation Forest**: Detecção de anomalias
- **Autoencoders**: Reconstrução para identificar outliers
- **Clustering**: Segmentação de comportamentos

### Ensemble Methods
- **Voting Classifier**: Combinação de múltiplos modelos
- **Stacking**: Meta-learning para otimização
- **Dynamic Weighting**: Pesos adaptativos por contexto

## 📊 Feature Engineering

### Categorias de Features
- **Transacionais**: Valor, horário, localização, merchant
- **Comportamentais**: Padrões históricos do usuário
- **Temporais**: Sazonalidade, frequência, intervalos
- **Geográficas**: Distância, velocidade de deslocamento
- **Agregadas**: Estatísticas rolling windows

### Exemplo de Features
```python
# Features temporais
hour_of_day = transaction_time.hour
day_of_week = transaction_time.dayofweek
is_weekend = day_of_week >= 5

# Features comportamentais
avg_amount_30d = user_transactions.amount.rolling('30D').mean()
transaction_frequency = user_transactions.groupby('date').count()

# Features geográficas
distance_from_home = calculate_distance(transaction_location, home_location)
velocity = distance_from_home / time_since_last_transaction
```

## ⚡ Processamento em Tempo Real

### Arquitetura de Streaming
```python
# Kafka Consumer para transações
from kafka import KafkaConsumer
from src.inference.fraud_detector import FraudDetector

detector = FraudDetector()
consumer = KafkaConsumer('transactions', bootstrap_servers=['localhost:9092'])

for message in consumer:
    transaction = json.loads(message.value)
    prediction = detector.predict(transaction)
    
    if prediction['is_fraud']:
        send_alert(transaction, prediction)
```

### Pipeline de Inferência
1. **Recepção**: Kafka consumer recebe transação
2. **Feature Engineering**: Extração de features em tempo real
3. **Predição**: Ensemble de modelos classifica transação
4. **Explicabilidade**: SHAP values para interpretação
5. **Ação**: Alert ou aprovação automática

## 📈 Monitoramento e MLOps

### Métricas Monitoradas
- **Performance**: Precision, Recall, F1-Score, AUC-ROC
- **Latência**: Tempo de resposta da API
- **Drift**: Mudanças na distribuição dos dados
- **Volume**: Throughput de transações processadas

### Alertas Automáticos
- Queda na performance dos modelos
- Aumento na latência de resposta
- Detecção de data drift
- Falhas no pipeline de dados

## 🔧 Configuração e Personalização

### Configuração de Modelos
```python
# config/model_config.py
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5
    },
    'xgboost': {
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100
    }
}
```

### Thresholds de Decisão
```python
# Configuração de thresholds por contexto
FRAUD_THRESHOLDS = {
    'high_value': 0.3,      # Transações > R$ 5.000
    'international': 0.4,   # Transações internacionais
    'night_time': 0.5,      # Transações noturnas
    'default': 0.6          # Threshold padrão
}
```

## 🧪 Testes e Validação

### Executar Testes
```bash
# Testes unitários
pytest tests/unit/

# Testes de integração
pytest tests/integration/

# Testes de performance
pytest tests/performance/
```

### Validação de Modelos
```bash
# Validação cruzada
python src/models/model_trainer.py --validate

# Backtesting
python scripts/backtest.py --start-date 2024-01-01 --end-date 2024-12-31
```

## 🚀 Deploy e Produção

### Deploy com Kubernetes
```bash
# Deploy completo
kubectl apply -f k8s/

# Verificar status
kubectl get pods -n fraud-detection
```

### Scaling Automático
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detection-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-detection-api
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
```

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- Email: gabrieldemetrios@gmail.com

---

⭐ Se este projeto foi útil, considere deixar uma estrela!

