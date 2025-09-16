# Performance Tests

Este diretório contém testes de performance para o sistema de detecção de fraudes financeiras com IA. Os testes são organizados por categoria e visam garantir que a aplicação mantenha níveis aceitáveis de performance sob diferentes condições de carga.

## Estrutura dos Testes

### 1. `test_load.py`
- **Propósito**: Testes de carga para verificar o comportamento do sistema sob alta demanda
- **Cenários**: Múltiplos usuários simultâneos, picos de tráfego, condições de stress
- **Ferramentas**: pytest-benchmark, requests, concurrent.futures

### 2. `test_latency.py`
- **Propósito**: Medição de tempo de resposta dos endpoints críticos
- **Métricas**: Tempo médio, percentis (P50, P95, P99), tempo máximo
- **Endpoints testados**: Endpoints de análise de fraude, autenticação, consultas

### 3. `test_throughput.py`
- **Propósito**: Avaliação da capacidade de processamento (requests por segundo)
- **Cenários**: Requisições simultâneas, diferentes payloads, análise de gargalos
- **Ferramentas recomendadas**: pytest-benchmark, Locust, Artillery

## Como Executar

### Pré-requisitos
```bash
pip install pytest pytest-benchmark requests locust artillery
```

### Executar todos os testes de performance
```bash
pytest tests/performance/ -v
```

### Executar teste específico com benchmark
```bash
pytest tests/performance/test_throughput.py --benchmark-only
```

### Executar com relatório de performance
```bash
pytest tests/performance/ --benchmark-sort=mean --benchmark-histogram
```

## Métricas e Critérios de Aceitação

### Latência
- **Endpoint de análise de fraude**: < 500ms (P95)
- **Endpoints de consulta**: < 200ms (P95)
- **Autenticação**: < 100ms (P95)

### Throughput
- **Análise de fraude**: > 100 requests/segundo
- **Consultas simples**: > 500 requests/segundo
- **Operações CRUD**: > 200 requests/segundo

### Carga
- **Usuários simultâneos**: Suportar até 1000 usuários
- **Degradação graceful**: Performance aceitável até 150% da capacidade
- **Recuperação**: Sistema deve se recuperar em < 30 segundos após pico

## Expansões Futuras

### 1. Testes de Stress Avançados
- Implementar cenários de falha de componentes
- Testes de recuperação automática
- Análise de memory leaks e resource exhaustion

### 2. Testes de Performance de Machine Learning
- Benchmark de modelos de IA para detecção de fraude
- Testes de latência de inferência
- Avaliação de throughput de batch processing

### 3. Monitoramento Contínuo
- Integração com ferramentas de APM (Application Performance Monitoring)
- Alertas automáticos para degradação de performance
- Dashboards em tempo real

### 4. Testes de Escalabilidade
- Testes em ambiente distribuído
- Avaliação de auto-scaling
- Performance com diferentes configurações de hardware

### 5. Testes de Performance de Banco de Dados
- Benchmark de queries complexas
- Testes de concorrência de transações
- Avaliação de performance de índices

## Ferramentas Recomendadas

### Para Desenvolvimento
- **pytest-benchmark**: Microbenchmarks e testes unitários de performance
- **locust**: Testes de carga distribuídos
- **artillery**: Testes de carga rápidos e flexíveis

### Para Produção
- **Grafana + Prometheus**: Monitoramento e alertas
- **New Relic/Datadog**: APM completo
- **JMeter**: Testes de carga enterprise

## Configuração de Ambiente

Para executar testes realistas, configure as seguintes variáveis de ambiente:

```bash
export API_BASE_URL="http://localhost:8000"
export TEST_DATABASE_URL="postgresql://test_user:test_pass@localhost/test_fraud_db"
export CONCURRENT_USERS=100
export TEST_DURATION=300  # segundos
```

## Relatórios e Análise

Os testes geram relatórios detalhados incluindo:
- Gráficos de latência ao longo do tempo
- Distribuição de tempos de resposta
- Análise de throughput por endpoint
- Identificação de gargalos e bottlenecks

## Contribuindo

Ao adicionar novos testes de performance:
1. Siga os padrões de nomenclatura existentes
2. Documente claramente os cenários testados
3. Inclua critérios de aceitação específicos
4. Adicione exemplos de uso na documentação
