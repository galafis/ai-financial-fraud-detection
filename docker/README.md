# Docker - AI Financial Fraud Detection

Diretório contendo arquivos de containerização, configurações Docker e recursos para deploy do sistema de detecção de fraudes financeiras.

## 📁 Estrutura de Arquivos

```
docker/
├── README.md              # Este arquivo - documentação completa
├── Dockerfile             # Container principal de produção (FastAPI)
├── Dockerfile.exemplo     # Exemplo educativo com melhores práticas
└── .env.example          # Template de variáveis de ambiente (a criar)
```

**Arquivos relacionados na raiz:**
- `docker-compose.yml` - Orquestração de serviços para desenvolvimento
- `requirements.txt` - Dependências Python

## 🐳 Principais Arquivos Docker

### Dockerfile
Container otimizado para produção com as seguintes características:
- **Base**: Python 3.9-slim
- **Usuário**: Não-root para segurança
- **Porta**: 8000 (FastAPI/Uvicorn)
- **Healthcheck**: `/health` endpoint
- **Volumes**: `/app/logs`, `/app/models`, `/app/data`

### Dockerfile.exemplo
Arquivo didático demonstrando melhores práticas:
- Multi-stage build (redução de ~50-70% no tamanho)
- Documentação detalhada de cada etapa
- Configurações de segurança avançadas
- Exemplos de uso e comandos
- Metadados completos

### docker-compose.yml (raiz)
Orquestração para desenvolvimento local:
- **API**: Container principal da aplicação
- **Redis**: Cache e feature store
- **Kafka/Zookeeper**: Streaming (opcional)
- **Networks**: Rede isolada para comunicação

## ⚙️ Variáveis de Ambiente

### Padrões de Configuração

```bash
# Aplicação
ENV=development|staging|production
DEBUG=true|false
LOG_LEVEL=debug|info|warning|error

# API
APP_HOST=0.0.0.0
APP_PORT=8000

# Banco de Dados e Cache
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql://user:pass@host:5432/db

# Machine Learning
MODEL_PATH=/app/models
FEATURE_STORE_PATH=/app/features

# Kafka (Streaming)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_TRANSACTIONS=transactions
KAFKA_TOPIC_ALERTS=fraud_alerts

# Segurança
API_KEY_LENGTH=32
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256

# Monitoramento
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

### Arquivo .env.example (Recomendado)
```bash
# Criar arquivo .env.example na raiz com:

# =============================================================================
# AI FINANCIAL FRAUD DETECTION - ENVIRONMENT VARIABLES
# =============================================================================
# Copie este arquivo para .env e configure as variáveis conforme seu ambiente

# APPLICATION SETTINGS
ENV=development
DEBUG=true
LOG_LEVEL=info

# API CONFIGURATION
APP_HOST=0.0.0.0
APP_PORT=8000

# REDIS CACHE
REDIS_URL=redis://localhost:6379/0

# ML MODELS
MODEL_PATH=./models
RETRAIN_INTERVAL_HOURS=24

# KAFKA STREAMING (opcional)
# KAFKA_BOOTSTRAP_SERVERS=localhost:9092
# KAFKA_TOPIC_TRANSACTIONS=transactions

# SECURITY
JWT_SECRET_KEY=change-this-in-production
API_KEY_LENGTH=32
```

## 🚀 Build e Deploy

### Desenvolvimento Local

```bash
# 1. Build da imagem
docker build -f docker/Dockerfile -t fraud-detection:dev .

# 2. Executar com Docker Compose
docker-compose up -d

# 3. Verificar status
docker-compose ps

# 4. Logs
docker-compose logs -f api
```

### Produção

```bash
# 1. Build otimizado para produção
docker build -f docker/Dockerfile \
  --target runtime \
  --build-arg APP_ENV=production \
  -t fraud-detection:latest .

# 2. Executar com variáveis de produção
docker run -d \
  --name fraud-detection-prod \
  --restart unless-stopped \
  -p 8000:8000 \
  -e ENV=production \
  -e DEBUG=false \
  -e LOG_LEVEL=warning \
  -v /host/logs:/app/logs \
  -v /host/models:/app/models \
  fraud-detection:latest

# 3. Verificar saúde
curl -f http://localhost:8000/health
```

## 🔧 Organização de Serviços

### Arquitetura de Containers

```yaml
# Serviços principais do docker-compose.yml:

fraud-detection-api:     # API FastAPI principal
fraud-detection-redis:   # Cache e feature store  
fraud-detection-kafka:   # Message streaming (opcional)
fraud-detection-zookeeper: # Kafka coordinator (opcional)
```

### Rede e Comunicação
- **Network**: `fraud-detection-network` (bridge)
- **DNS**: Containers se comunicam por nome
- **Ports**: Apenas API exposta externamente (8000)

### Volumes Persistentes
```yaml
volumes:
  redis_data: # Persistência do Redis
  models: # Modelos ML treinados
  logs: # Logs da aplicação
```

## 🔄 Boas Práticas de Build/Deploy

### Build
1. **Layer Caching**: Requirements.txt copiado antes do código fonte
2. **Multi-stage**: Separação build/runtime (Dockerfile.exemplo)
3. **Usuário não-root**: Segurança por princípio
4. **Healthcheck**: Monitoramento automático da aplicação

### Deploy
1. **Rolling Updates**: Zero downtime deployment
2. **Resource Limits**: CPU/Memory constraints
3. **Restart Policies**: Auto-recovery em caso de falhas
4. **Secrets Management**: Variáveis sensíveis via secrets

### Segurança
1. **Minimal Base Images**: Python slim, Alpine quando apropriado
2. **CVE Scanning**: Verificação regular de vulnerabilidades
3. **Non-root User**: UID/GID específicos (1000:1000)
4. **Read-only Root**: Filesystem imutável quando possível

## 🛠️ Personalização e Configuração

### Desenvolvimento Local
```yaml
# Override para development
# docker-compose.override.yml
version: '3.8'
services:
  api:
    volumes:
      - ./src:/app/src  # Hot reload
      - ./notebooks:/app/notebooks
    environment:
      - DEBUG=true
      - LOG_LEVEL=debug
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Staging/Testing
```bash
# Build para staging
docker build \
  --build-arg APP_ENV=staging \
  --build-arg LOG_LEVEL=info \
  -t fraud-detection:staging \
  -f docker/Dockerfile .

# Deploy staging com recursos limitados
docker run -d \
  --memory 1g \
  --cpus 0.5 \
  -e ENV=staging \
  fraud-detection:staging
```

### Produção Distribuída
```bash
# Docker Swarm / Kubernetes
# Ver diretório k8s/ para manifests completos

# Build multi-arch para produção
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -t your-registry/fraud-detection:v1.0.0 \
  -f docker/Dockerfile .
```

## 📊 Monitoramento e Observabilidade

### Healthchecks
```bash
# Verificar saúde do container
docker exec fraud-detection-api curl -f http://localhost:8000/health

# Status completo
curl http://localhost:8000/health | jq
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "redis_connected": true,
  "uptime_seconds": 3600
}
```

### Logs Estruturados
```bash
# Logs em tempo real
docker-compose logs -f --tail=100

# Logs específicos do serviço
docker-compose logs api redis

# Análise de logs
docker-compose logs api | grep ERROR
```

### Métricas
```bash
# Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Docker stats
docker stats fraud-detection-api
```

## 🧪 Testing

### Container Testing
```bash
# Teste de smoke
docker run --rm fraud-detection:latest python -c "import src; print('OK')"

# Teste de integração
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Teste de carga
docker run --rm -it --network fraud-detection-network \
  alpine/curl -X POST http://api:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"amount": 1000, "merchant": "test"}'
```

### Validation
```bash
# Validar estrutura da imagem
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  wagoodman/dive fraud-detection:latest

# Security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image fraud-detection:latest
```

## 🚨 Troubleshooting

### Problemas Comuns
```bash
# Container não inicia
docker-compose logs api
docker-compose exec api sh  # Debug interativo

# Conectividade entre containers
docker-compose exec api ping redis
docker network ls
docker network inspect fraud-detection-network

# Performance issues
docker stats
docker-compose exec api top
```

### Debug Mode
```bash
# Executar em modo debug
docker-compose -f docker-compose.yml -f docker-compose.debug.yml up

# Shell interativo no container
docker-compose exec api bash

# Verificar variáveis de ambiente
docker-compose exec api env | grep -E "(API|MODEL|REDIS)"
```

## 📚 Recursos Adicionais

### Comandos Úteis
```bash
# Limpeza completa
docker-compose down -v --remove-orphans
docker system prune -a

# Backup de volumes
docker run --rm -v fraud-detection_models:/backup alpine tar czf - /backup

# Atualização zero-downtime
docker-compose pull
docker-compose up -d --no-deps api
```

### Links Úteis
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [FastAPI with Docker](https://fastapi.tiangolo.com/deployment/docker/)
- [Python Docker Guide](https://docs.python.org/3/using/docker.html)

---

**Importante**: Para ambientes de produção, sempre configure adequadamente as variáveis de ambiente, use secrets para dados sensíveis e implemente monitoramento robusto.
