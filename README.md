# 🤖 Ai Financial Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Português](#português)

---

## English

### 🎯 Overview

**Ai Financial Fraud Detection** — AI-powered fraud detection system for financial transactions. Uses ensemble models, anomaly detection, and real-time scoring to identify fraudulent patterns.

Total source lines: **4,681** across **31** files in **2** languages.

### ✨ Key Features

- **Production-Ready Architecture**: Modular, well-documented, and following best practices
- **Comprehensive Implementation**: Complete solution with all core functionality
- **Clean Code**: Type-safe, well-tested, and maintainable codebase
- **Easy Deployment**: Docker support for quick setup and deployment

### 🚀 Quick Start

#### Prerequisites
- Python 3.12+
- Docker and Docker Compose (optional)

#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/galafis/ai-financial-fraud-detection.git
cd ai-financial-fraud-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### Running

```bash
python src/api/main.py
```

## 🐳 Docker

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Project Structure

```
ai-financial-fraud-detection/
├── config/
│   ├── docker-compose.yml
│   └── requirements.txt
├── docker/
│   └── README.md
├── docs/
│   ├── images/
│   └── architecture_diagram.md
├── k8s/
│   ├── README.md
│   ├── deployment.yaml
│   └── service.yaml
├── notebooks/
│   └── README.md
├── src/
│   ├── api/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── main.py
│   ├── config/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── api_config.py
│   │   └── model_config.py
│   ├── data/
│   │   ├── connectors/
│   │   ├── streaming/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── feature_engineering.py
│   ├── features/
│   │   ├── README.md
│   │   └── __init__.py
│   ├── inference/
│   │   ├── README.md
│   │   └── __init__.py
│   ├── models/
│   │   ├── ensemble/
│   │   ├── supervised/
│   │   ├── training/
│   │   ├── unsupervised/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── ensemble_model.py
│   ├── monitoring/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── model_monitoring.py
│   ├── utils/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── README.md
│   ├── __init__.py
│   └── backtest.py
├── tests/
│   ├── integration/
│   │   ├── test_api.py
│   │   └── test_data_streaming.py
│   ├── performance/
│   │   ├── README.md
│   │   └── test_latency.py
│   ├── unit/
│   │   ├── test_ensemble_model.py
│   │   └── test_features.py
│   └── README.md
├── CONTRIBUTING.md
└── README.md
```

### 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | 30 files |
| HTML | 1 files |

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

---

## Português

### 🎯 Visão Geral

**Ai Financial Fraud Detection** — AI-powered fraud detection system for financial transactions. Uses ensemble models, anomaly detection, and real-time scoring to identify fraudulent patterns.

Total de linhas de código: **4,681** em **31** arquivos em **2** linguagens.

### ✨ Funcionalidades Principais

- **Arquitetura Pronta para Produção**: Modular, bem documentada e seguindo boas práticas
- **Implementação Completa**: Solução completa com todas as funcionalidades principais
- **Código Limpo**: Type-safe, bem testado e manutenível
- **Fácil Implantação**: Suporte Docker para configuração e implantação rápidas

### 🚀 Início Rápido

#### Pré-requisitos
- Python 3.12+
- Docker e Docker Compose (opcional)

#### Instalação

1. **Clone the repository**
```bash
git clone https://github.com/galafis/ai-financial-fraud-detection.git
cd ai-financial-fraud-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

#### Execução

```bash
python src/api/main.py
```

### 🧪 Testes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Estrutura do Projeto

```
ai-financial-fraud-detection/
├── config/
│   ├── docker-compose.yml
│   └── requirements.txt
├── docker/
│   └── README.md
├── docs/
│   ├── images/
│   └── architecture_diagram.md
├── k8s/
│   ├── README.md
│   ├── deployment.yaml
│   └── service.yaml
├── notebooks/
│   └── README.md
├── src/
│   ├── api/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── main.py
│   ├── config/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── api_config.py
│   │   └── model_config.py
│   ├── data/
│   │   ├── connectors/
│   │   ├── streaming/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── feature_engineering.py
│   ├── features/
│   │   ├── README.md
│   │   └── __init__.py
│   ├── inference/
│   │   ├── README.md
│   │   └── __init__.py
│   ├── models/
│   │   ├── ensemble/
│   │   ├── supervised/
│   │   ├── training/
│   │   ├── unsupervised/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── ensemble_model.py
│   ├── monitoring/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── model_monitoring.py
│   ├── utils/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── README.md
│   ├── __init__.py
│   └── backtest.py
├── tests/
│   ├── integration/
│   │   ├── test_api.py
│   │   └── test_data_streaming.py
│   ├── performance/
│   │   ├── README.md
│   │   └── test_latency.py
│   ├── unit/
│   │   ├── test_ensemble_model.py
│   │   └── test_features.py
│   └── README.md
├── CONTRIBUTING.md
└── README.md
```

### 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| Python | 30 files |
| HTML | 1 files |

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)
