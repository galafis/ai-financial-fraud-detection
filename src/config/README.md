# Configuration Management - AI Financial Fraud Detection

Este diretório contém os módulos de configuração centralizados do sistema de detecção de fraudes financeiras, proporcionando gerenciamento unificado de parâmetros, thresholds e configurações específicas do ambiente.

## 📁 Estrutura de Arquivos

### 📄 Arquivos Principais

- **`api_config.py`** - Configurações da API REST e endpoints
- **`model_config.py`** - Parâmetros dos modelos de ML e thresholds de decisão
- **`database_config.py`** - Configurações de conexão com bancos de dados (se implementado)
- **`logging_config.py`** - Configuração de logs e monitoramento (se implementado)
- **`kafka_config.py`** - Configurações de streaming e messaging (se implementado)

## 🎯 Objetivo

Centralizar todas as configurações do sistema para:
- **Facilitar manutenção** - Configurações em local único
- **Gerenciar ambientes** - Dev, staging, production
- **Versionamento** - Controle de mudanças em configurações
- **Segurança** - Separação de configurações sensíveis

## ⚙️ Configurações Disponíveis

### 🔧 Configurações de API (api_config.py)
```python
# Exemplo de configurações de API
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'workers': 4,
    'timeout': 30,
    'max_request_size': '16MB',
    'cors_origins': ['*'],
    'api_version': 'v1'
}

# Rate limiting
RATE_LIMIT_CONFIG = {
    'requests_per_minute': 100,
    'burst_size': 20
}
```

### 🤖 Configurações de Modelos (model_config.py)
```python
# Configurações de modelos de ML
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    'xgboost': {
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100,
        'subsample': 0.8
    },
    'isolation_forest': {
        'contamination': 0.1,
        'random_state': 42
    }
}

# Thresholds de detecção
FRAUD_THRESHOLDS = {
    'high_value': 0.3,      # Transações > R$ 5.000
    'international': 0.4,   # Transações internacionais
    'night_time': 0.5,      # Transações noturnas (22h-6h)
    'weekend': 0.45,        # Transações em finais de semana
    'default': 0.6          # Threshold padrão
}

# Configurações de ensemble
ENSEMBLE_CONFIG = {
    'voting_method': 'soft',
    'weights': {
        'random_forest': 0.3,
        'xgboost': 0.4,
        'isolation_forest': 0.3
    },
    'meta_learner': 'logistic_regression'
}
```

### 🗄️ Configurações de Base de Dados
```python
# Exemplo de configurações de banco
DATABASE_CONFIG = {
    'primary': {
        'host': 'localhost',
        'port': 5432,
        'database': 'fraud_detection',
        'pool_size': 20,
        'max_overflow': 30
    },
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'decode_responses': True
    }
}
```

## 🔄 Uso das Configurações

### Carregando Configurações
```python
from src.config.api_config import API_CONFIG, RATE_LIMIT_CONFIG
from src.config.model_config import MODEL_CONFIG, FRAUD_THRESHOLDS

# Usar configurações de API
app = FastAPI(
    debug=API_CONFIG['debug'],
    version=API_CONFIG['api_version']
)

# Usar configurações de modelo
rf_params = MODEL_CONFIG['random_forest']
rf_model = RandomForestClassifier(**rf_params)
```

### Configurações por Ambiente
```python
import os
from typing import Dict, Any

def get_config(env: str = None) -> Dict[str, Any]:
    """Retorna configurações baseadas no ambiente."""
    env = env or os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        return PRODUCTION_CONFIG
    elif env == 'staging':
        return STAGING_CONFIG
    else:
        return DEVELOPMENT_CONFIG
```

## 🛡️ Boas Práticas de Configuração

### 1. Variáveis de Ambiente
```python
# Use variáveis de ambiente para dados sensíveis
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/fraud_db')
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
KAFKA_BROKERS = os.getenv('KAFKA_BROKERS', 'localhost:9092').split(',')
```

### 2. Validação de Configurações
```python
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    api_host: str = '0.0.0.0'
    api_port: int = 8000
    database_url: str
    
    @validator('api_port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    class Config:
        env_file = '.env'
```

### 3. Configurações Imutáveis
```python
from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class ModelThresholds:
    high_value: float = 0.3
    international: float = 0.4
    night_time: float = 0.5
    default: float = 0.6
```

## 📊 Monitoramento de Configurações

### Logging de Mudanças
```python
import logging
from functools import wraps

def log_config_change(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Config change: {func.__name__} called with {kwargs}")
        return func(*args, **kwargs)
    return wrapper

@log_config_change
def update_threshold(model: str, threshold: float):
    FRAUD_THRESHOLDS[model] = threshold
```

### Health Checks
```python
def validate_config_health() -> Dict[str, bool]:
    """Valida se todas as configurações estão corretas."""
    checks = {
        'database_reachable': test_database_connection(),
        'kafka_available': test_kafka_connection(),
        'models_loaded': verify_model_configs(),
        'thresholds_valid': validate_thresholds()
    }
    return checks
```

## 🔧 Configuração de Desenvolvimento

### Arquivo .env
```bash
# Configurações de desenvolvimento
ENVIRONMENT=development
DEBUG=true
API_HOST=localhost
API_PORT=8000
DATABASE_URL=postgresql://user:pass@localhost/fraud_dev
KAFKA_BROKERS=localhost:9092
REDIS_URL=redis://localhost:6379/0
```

### Docker Compose Override
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  api:
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    volumes:
      - .env:/app/.env
```

## 📈 Exemplo de Uso Completo

```python
#!/usr/bin/env python3
"""Exemplo de uso das configurações do sistema."""

from src.config.api_config import API_CONFIG
from src.config.model_config import MODEL_CONFIG, FRAUD_THRESHOLDS
from src.models.ensemble.fraud_ensemble import FraudEnsemble
from fastapi import FastAPI

# Inicializar aplicação com configurações
app = FastAPI(
    title="Fraud Detection API",
    version=API_CONFIG['api_version'],
    debug=API_CONFIG['debug']
)

# Inicializar ensemble com configurações
ensemble = FraudEnsemble(
    models_config=MODEL_CONFIG,
    thresholds=FRAUD_THRESHOLDS
)

# Endpoint que usa configurações
@app.post("/predict")
async def predict_fraud(transaction: dict):
    # Aplicar threshold baseado no tipo de transação
    threshold = get_threshold_for_transaction(transaction)
    
    # Fazer predição
    prediction = ensemble.predict(
        transaction, 
        threshold=threshold
    )
    
    return {
        "is_fraud": prediction['is_fraud'],
        "confidence": prediction['confidence'],
        "threshold_used": threshold
    }

def get_threshold_for_transaction(transaction: dict) -> float:
    """Seleciona threshold baseado nas características da transação."""
    if transaction.get('amount', 0) > 5000:
        return FRAUD_THRESHOLDS['high_value']
    elif transaction.get('is_international', False):
        return FRAUD_THRESHOLDS['international']
    elif is_night_time(transaction.get('timestamp')):
        return FRAUD_THRESHOLDS['night_time']
    else:
        return FRAUD_THRESHOLDS['default']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        workers=API_CONFIG['workers']
    )
```

## 📚 Referências e Recursos

- [Pydantic Settings](https://pydantic-docs.helpmanual.io/usage/settings/) - Gerenciamento de configurações
- [Python Decouple](https://github.com/henriquebastos/python-decouple) - Separação de configurações
- [Hydra](https://hydra.cc/) - Framework de configuração complexa
- [FastAPI Settings](https://fastapi.tiangolo.com/advanced/settings/) - Configurações em FastAPI

---

## 💡 Dicas de Implementação

1. **Use Enums** para configurações com valores fixos
2. **Valide configurações** na inicialização da aplicação
3. **Mantenha configurações sensíveis** em variáveis de ambiente
4. **Documente mudanças** em configurações críticas
5. **Use configurações imutáveis** sempre que possível
