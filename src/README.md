# Source Code Directory - AI Financial Fraud Detection
*Diretório de Código Fonte - Detecção de Fraudes Financeiras com IA*

## 🏗️ Architecture Overview | Visão Geral da Arquitetura

This directory contains the core source code for the AI Financial Fraud Detection system, organized into specialized modules for maintainability and scalability.

*Este diretório contém o código-fonte principal do sistema de Detecção de Fraudes Financeiras com IA, organizado em módulos especializados para manutenibilidade e escalabilidade.*

---

## 📁 Directory Structure | Estrutura de Diretórios

```
src/
├── api/                 # REST API and web services
├── config/              # Configuration management
├── data/                # Data processing and streaming
├── features/            # Feature engineering modules
├── models/              # Machine learning models
├── monitoring/          # System monitoring and observability
└── utils/               # Common utilities and helpers
```

---

## 🔧 Module Descriptions | Descrições dos Módulos

### 📡 `/api` - REST API and Web Services
*Serviços Web e API REST*

**Purpose | Propósito:**
- FastAPI-based REST API for fraud detection endpoints
- Request/response handling and validation
- Authentication and authorization mechanisms

*API REST baseada em FastAPI para endpoints de detecção de fraudes*
*Manipulação e validação de requisições/respostas*
*Mecanismos de autenticação e autorização*

**Key Components | Componentes Principais:**
- `main.py` - FastAPI application entry point
- `routers/` - API route definitions
- `middleware/` - Custom middleware components
- `schemas/` - Pydantic models for request/response

**Usage Example | Exemplo de Uso:**
```python
from src.api.main import app
from src.api.schemas.transaction import TransactionRequest

# Start API server
# Iniciar servidor da API
uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### ⚙️ `/config` - Configuration Management
*Gerenciamento de Configurações*

**Purpose | Propósito:**
- Centralized configuration management
- Environment-specific settings
- Model parameters and thresholds

*Gerenciamento centralizado de configurações*
*Configurações específicas do ambiente*
*Parâmetros de modelos e thresholds*

**Key Components | Componentes Principais:**
- `settings.py` - Application settings
- `model_config.py` - ML model configurations
- `database.py` - Database connection settings
- `logging.py` - Logging configuration

**Usage Example | Exemplo de Uso:**
```python
from src.config.settings import get_settings

settings = get_settings()
print(f"API URL: {settings.api_url}")
print(f"Database: {settings.database_url}")
```

---

### 💾 `/data` - Data Processing and Streaming
*Processamento de Dados e Streaming*

**Purpose | Propósito:**
- Data ingestion from various sources
- Real-time streaming with Kafka
- Data preprocessing and transformation

*Ingestão de dados de várias fontes*
*Streaming em tempo real com Kafka*
*Pré-processamento e transformação de dados*

**Key Components | Componentes Principais:**
- `connectors/` - Database and API connectors
- `streaming/` - Kafka producers and consumers
- `preprocessing/` - Data cleaning and transformation
- `validation/` - Data quality checks

**Usage Example | Exemplo de Uso:**
```python
from src.data.streaming.kafka_consumer import TransactionConsumer

consumer = TransactionConsumer('transactions')
for transaction in consumer.consume():
    processed_data = preprocess_transaction(transaction)
```

---

### 🛠️ `/features` - Feature Engineering
*Engenharia de Features*

**Purpose | Propósito:**
- Feature extraction and engineering
- Statistical and behavioral feature generation
- Feature store management

*Extração e engenharia de features*
*Geração de features estatísticas e comportamentais*
*Gerenciamento de feature store*

**Key Components | Componentes Principais:**
- `extractors/` - Feature extraction modules
- `transformers/` - Feature transformation pipelines
- `store/` - Feature store implementations
- `validation/` - Feature quality validation

**Usage Example | Exemplo de Uso:**
```python
from src.features.extractors.behavioral import BehavioralFeatures

extractor = BehavioralFeatures()
features = extractor.extract_user_features(user_id, timeframe='30d')
```

---

### 🤖 `/models` - Machine Learning Models
*Modelos de Machine Learning*

**Purpose | Propósito:**
- ML model implementations and training
- Model versioning and management
- Ensemble methods and meta-learning

*Implementações e treinamento de modelos de ML*
*Versionamento e gerenciamento de modelos*
*Métodos de ensemble e meta-learning*

**Key Components | Componentes Principais:**
- `supervised/` - Classification models (RF, XGBoost, NN)
- `unsupervised/` - Anomaly detection (Isolation Forest, Autoencoders)
- `ensemble/` - Model combination strategies
- `training/` - Training pipelines and validation

**Usage Example | Exemplo de Uso:**
```python
from src.models.ensemble.fraud_ensemble import FraudEnsemble

ensemble = FraudEnsemble()
ensemble.load_models()
prediction = ensemble.predict(features)
```

---

### 📊 `/monitoring` - System Monitoring and Observability
*Monitoramento do Sistema e Observabilidade*

**Purpose | Propósito:**
- System health monitoring
- Model performance tracking
- Drift detection and alerting

*Monitoramento da saúde do sistema*
*Acompanhamento da performance dos modelos*
*Detecção de drift e alertas*

**Key Components | Componentes Principais:**
- `metrics/` - Performance and business metrics
- `alerts/` - Alert management and notifications
- `drift/` - Data and model drift detection
- `dashboards/` - Monitoring dashboard components

**Usage Example | Exemplo de Uso:**
```python
from src.monitoring.metrics.model_metrics import ModelMetrics

metrics = ModelMetrics()
metrics.log_prediction(y_true, y_pred, features)
metrics.check_drift_alert()
```

---

### 🔨 `/utils` - Common Utilities and Helpers
*Utilitários Comuns e Auxiliares*

**Purpose | Propósito:**
- Common utility functions
- Helper classes and decorators
- Shared constants and enums

*Funções utilitárias comuns*
*Classes auxiliares e decorators*
*Constantes e enums compartilhados*

**Key Components | Componentes Principais:**
- `decorators.py` - Common decorators
- `exceptions.py` - Custom exception classes
- `constants.py` - System constants
- `helpers.py` - General helper functions

**Usage Example | Exemplo de Uso:**
```python
from src.utils.decorators import timing, retry
from src.utils.helpers import normalize_amount

@timing
@retry(max_attempts=3)
def process_transaction(amount):
    normalized = normalize_amount(amount)
    return normalized
```

---

## 🚀 Getting Started | Começando

### Prerequisites | Pré-requisitos
- Python 3.9+
- Docker and Docker Compose
- Apache Kafka
- PostgreSQL/MongoDB

### Installation | Instalação

1. **Install dependencies | Instalar dependências:**
```bash
pip install -r requirements.txt
```

2. **Set up environment | Configurar ambiente:**
```bash
cp .env.example .env
# Edit .env with your configurations
# Edite .env com suas configurações
```

3. **Initialize database | Inicializar banco de dados:**
```bash
python src/data/migrations/init_db.py
```

4. **Start services | Iniciar serviços:**
```bash
docker-compose up -d kafka redis postgres
```

---

## 📋 Usage Examples | Exemplos de Uso

### Training a Model | Treinando um Modelo
```python
from src.models.training.trainer import ModelTrainer
from src.data.preprocessing.pipeline import DataPipeline

# Prepare data | Preparar dados
pipeline = DataPipeline()
train_data, val_data = pipeline.prepare_training_data()

# Train model | Treinar modelo
trainer = ModelTrainer(model_type='xgboost')
model = trainer.train(train_data, val_data)
trainer.save_model(model, version='v1.0.0')
```

### Real-time Prediction | Predição em Tempo Real
```python
from src.api.main import app
from src.models.ensemble.fraud_ensemble import FraudEnsemble

# Load ensemble | Carregar ensemble
detector = FraudEnsemble()
detector.load_latest_models()

# Make prediction | Fazer predição
transaction = {...}  # Transaction data | Dados da transação
result = detector.predict(transaction)
```

---

## 🏗️ Best Practices | Boas Práticas

### Code Organization | Organização do Código
- Keep modules focused and cohesive
- Use clear naming conventions
- Implement proper error handling
- Add comprehensive logging

*Mantenha módulos focados e coesos*
*Use convenções de nomenclatura claras*
*implemente tratamento adequado de erros*
*Adicione logging abrangente*

### Testing | Testes
- Write unit tests for all modules
- Include integration tests for APIs
- Implement performance tests
- Use test fixtures and mocks appropriately

*Escreva testes unitários para todos os módulos*
*Inclua testes de integração para APIs*
*Implemente testes de performance*
*Use fixtures e mocks apropriadamente*

### Documentation | Documentação
- Document all public APIs
- Include usage examples
- Maintain architectural decision records
- Keep README files updated

*Documente todas as APIs públicas*
*Inclua exemplos de uso*
*Mantenha registros de decisões arquiteturais*
*Mantenha arquivos README atualizados*

---

## 🤝 Contributing | Contribuindo

### Development Workflow | Fluxo de Desenvolvimento

1. **Fork the repository | Fork o repositório**
2. **Create feature branch | Criar branch de feature:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes and add tests | Fazer mudanças e adicionar testes**
4. **Run quality checks | Executar verificações de qualidade:**
   ```bash
   black src/
   isort src/
   flake8 src/
   pytest tests/
   ```
5. **Submit pull request | Submeter pull request**

### Code Standards | Padrões de Código
- Follow PEP 8 style guide
- Use type hints
- Maximum line length: 88 characters
- Use meaningful variable and function names

*Siga o guia de estilo PEP 8*
*Use type hints*
*Comprimento máximo de linha: 88 caracteres*
*Use nomes de variáveis e funções significativos*

---

## 📚 Documentation and Tools | Documentação e Ferramentas

### Official Documentation | Documentação Oficial
- [Python Documentation](https://docs.python.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)

### Development Tools | Ferramentas de Desenvolvimento
- [Black Code Formatter](https://black.readthedocs.io/)
- [isort Import Sorter](https://pycqa.github.io/isort/)
- [Flake8 Linter](https://flake8.pycqa.org/)
- [pytest Testing Framework](https://docs.pytest.org/)

### Monitoring and Observability | Monitoramento e Observabilidade
- [Prometheus Monitoring](https://prometheus.io/)
- [Grafana Dashboards](https://grafana.com/)
- [MLflow Model Tracking](https://mlflow.org/)
- [SHAP Explainability](https://shap.readthedocs.io/)

---

## 📞 Support | Suporte

For questions and support, please:
*Para questões e suporte, por favor:*

- Open an issue in the GitHub repository
- Check existing documentation
- Review code examples and tests

*Abra uma issue no repositório GitHub*
*Verifique a documentação existente*
*Revise exemplos de código e testes*

---

## 📄 License | Licença

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

*Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](../LICENSE) para detalhes.*
