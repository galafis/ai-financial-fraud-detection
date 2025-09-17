# 🧪 Tests - Sistema de Detecção de Fraudes Financeiras

Documentação completa da suíte de testes automatizados do projeto AI Financial Fraud Detection, incluindo organização, execução, boas práticas e critérios de qualidade.

## 📁 Estrutura de Organização

```
tests/
├── unit/                    # Testes unitários
│   ├── README.md           # Documentação específica dos testes unitários
│   ├── test_models/        # Testes dos modelos de ML
│   ├── test_features/      # Testes de feature engineering
│   ├── test_data/          # Testes de processamento de dados
│   └── test_utils/         # Testes de utilitários
├── integration/            # Testes de integração
│   ├── README.md           # Documentação específica dos testes de integração
│   ├── test_api/           # Testes da API REST
│   ├── test_pipeline/      # Testes do pipeline completo
│   └── test_streaming/     # Testes de processamento em tempo real
├── performance/            # Testes de performance
│   ├── README.md           # Documentação específica dos testes de performance
│   ├── test_latency/       # Testes de latência
│   ├── test_throughput/    # Testes de throughput
│   └── test_load/          # Testes de carga
├── fixtures/               # Dados de teste e fixtures
│   ├── sample_data/        # Dados de exemplo para testes
│   └── mocks/              # Mocks e stubs
├── conftest.py             # Configurações pytest compartilhadas
└── README.md               # Esta documentação
```

## 🎯 Tipos de Testes

### 1. Testes Unitários (`unit/`)
- **Objetivo**: Testar componentes individuais isoladamente
- **Escopo**: Funções, classes e métodos específicos
- **Cobertura**: ≥90% para módulos críticos, ≥80% geral
- **Execução**: Rápida (<1s por teste)

### 2. Testes de Integração (`integration/`)
- **Objetivo**: Testar interação entre componentes
- **Escopo**: APIs, pipelines de dados, conexões externas
- **Cobertura**: Todos os endpoints e fluxos principais
- **Execução**: Moderada (1-10s por teste)

### 3. Testes de Performance (`performance/`)
- **Objetivo**: Validar requisitos não-funcionais
- **Escopo**: Latência, throughput, uso de memória
- **Métricas**: <100ms latência, >1000 transações/s
- **Execução**: Lenta (10s-5min por teste)

## 🚀 Execução dos Testes

### Comandos Básicos

```bash
# Instalar dependências de teste
pip install -r requirements-test.txt

# Executar todos os testes
pytest

# Executar testes específicos por categoria
pytest tests/unit/                    # Somente unitários
pytest tests/integration/            # Somente integração
pytest tests/performance/            # Somente performance

# Executar testes com cobertura
pytest --cov=src --cov-report=html --cov-report=term

# Executar testes em paralelo
pytest -n auto

# Executar testes específicos
pytest tests/unit/test_models/test_fraud_detector.py
pytest -k "test_fraud_detection"     # Por nome
pytest -m "slow"                     # Por marcador
```

### Configurações de Execução

```bash
# Execução verbosa com detalhes
pytest -v --tb=short

# Execução com relatório JUnit (CI/CD)
pytest --junitxml=reports/junit.xml

# Execução com falhas primeiro (fail-fast)
pytest -x

# Execução com re-execução de falhas
pytest --lf  # last-failed
pytest --ff  # failed-first
```

## 📊 Critérios de Cobertura

### Metas de Cobertura por Módulo

| Módulo | Cobertura Mínima | Cobertura Meta |
|--------|------------------|----------------|
| `src/models/` | 90% | 95% |
| `src/features/` | 85% | 90% |
| `src/inference/` | 90% | 95% |
| `src/api/` | 80% | 85% |
| `src/data/` | 75% | 80% |
| `src/utils/` | 70% | 75% |

### Métricas de Qualidade

```bash
# Relatório completo de cobertura
pytest --cov=src --cov-report=html

# Verificar cobertura com falha se abaixo do limite
pytest --cov=src --cov-fail-under=80

# Excluir arquivos de configuração da cobertura
pytest --cov=src --cov-report=term --cov-config=.coveragerc
```

## 🏗️ Boas Práticas para Desenvolvimento

### 1. Nomenclatura de Testes

```python
# ❌ Evitar
def test1():
def test_function():

# ✅ Recomendado
def test_fraud_detector_returns_high_score_for_suspicious_transaction():
def test_feature_engineering_handles_missing_values_correctly():
def test_api_returns_400_when_invalid_transaction_data():
```

### 2. Estrutura de Testes (AAA Pattern)

```python
def test_fraud_detection_model_prediction():
    # Arrange - Preparar dados e dependências
    model = FraudDetectionModel()
    transaction = {
        "amount": 1500.0,
        "merchant": "Online Store",
        "location": "São Paulo"
    }
    
    # Act - Executar ação a ser testada
    prediction = model.predict(transaction)
    
    # Assert - Verificar resultados
    assert prediction["fraud_probability"] > 0.5
    assert "risk_factors" in prediction
```

### 3. Uso de Fixtures

```python
# conftest.py
@pytest.fixture
def sample_transaction():
    return {
        "amount": 100.0,
        "merchant": "Test Store",
        "timestamp": "2024-01-15T10:00:00"
    }

@pytest.fixture
def trained_model():
    model = FraudDetectionModel()
    model.load_model("models/test_model.pkl")
    return model
```

### 4. Marcadores de Teste

```python
import pytest

@pytest.mark.unit
def test_feature_extraction():
    pass

@pytest.mark.integration
def test_api_endpoint():
    pass

@pytest.mark.slow
def test_model_training():
    pass

@pytest.mark.parametrize("amount,expected", [
    (100.0, "low_risk"),
    (5000.0, "high_risk"),
])
def test_risk_classification(amount, expected):
    pass
```

## 🔧 Configuração do Ambiente de Testes

### 1. Arquivo pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
python_classes = Test*
markers =
    unit: Testes unitários rápidos
    integration: Testes de integração
    performance: Testes de performance
    slow: Testes que demoram mais de 10s
addopts = 
    --strict-markers
    --strict-config
    --verbose
```

### 2. Variáveis de Ambiente

```bash
# .env.test
DATABASE_URL=sqlite:///test.db
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
REDIS_URL=redis://localhost:6379/1
LOG_LEVEL=DEBUG
```

### 3. Dependências de Teste

```bash
# requirements-test.txt
pytest==7.4.0
pytest-cov==4.1.0
pytest-xdist==3.3.1
pytest-mock==3.11.1
pytest-asyncio==0.21.1
pytest-benchmark==4.0.0
factory-boy==3.3.0
faker==19.3.0
responses==0.23.3
```

## 📈 Estratégia de Expansão dos Testes

### 1. Adição de Novos Testes

1. **Identificar Funcionalidade**: Nova feature ou correção de bug
2. **Classificar Tipo**: Unit, Integration ou Performance
3. **Definir Cenários**: Happy path, edge cases, error cases
4. **Implementar Testes**: Seguindo padrões estabelecidos
5. **Validar Cobertura**: Verificar se atende critérios

### 2. Manutenção Contínua

```bash
# Script de verificação da qualidade dos testes
./scripts/test-quality-check.sh

# Linting de código de teste
flake8 tests/
black tests/
isort tests/

# Verificação de testes obsoletos
pytest --collect-only | grep -i "deprecated"
```

### 3. Automação CI/CD

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Tests
        run: |
          pytest --cov=src --cov-report=xml
          pytest tests/integration/ --maxfail=1
          pytest tests/performance/ --benchmark-json=benchmark.json
```

## 📚 Documentação Específica

Consulte os READMEs específicos de cada categoria para detalhes técnicos:

- [`unit/README.md`](unit/README.md) - Testes unitários específicos
- [`integration/README.md`](integration/README.md) - Testes de integração e API
- [`performance/README.md`](performance/README.md) - Benchmarks e testes de carga

## 🛠️ Ferramentas Auxiliares

### Debugging de Testes

```bash
# Executar com debugger
pytest --pdb

# Executar com profiling
pytest --profile

# Capturar prints durante execução
pytest -s
```

### Relatórios e Métricas

```bash
# Relatório HTML interativo
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Benchmark comparativo
pytest --benchmark-compare=0001_baseline.json
```

## 🚨 Troubleshooting Comum

### Problemas Frequentes

1. **Testes Lentos**: Use `-n auto` para paralelização
2. **Fixtures Não Encontradas**: Verificar `conftest.py`
3. **Import Errors**: Configurar `PYTHONPATH` corretamente
4. **Database Locks**: Usar transações isoladas para testes

### Contato e Suporte

Para dúvidas sobre testes ou contribuições:
- Abrir issue no repositório
- Consultar documentação em `docs/testing-guide.md`
- Revisar exemplos em cada subdiretório

---

**Nota**: Esta documentação deve ser atualizada sempre que novos padrões ou ferramentas de teste forem adotados no projeto.
