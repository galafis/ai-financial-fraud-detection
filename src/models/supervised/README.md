# Modelos Supervisionados – AI Financial Fraud Detection

Este diretório contém implementações de algoritmos de machine learning supervisionado para classificação de fraudes financeiras.

## Algoritmos Implementados

### Random Forest
- **Arquivo**: `random_forest_model.py`
- **Propósito**: Modelo ensemble robusto baseado em múltiplas árvores de decisão
- **Vantagens**: Resistente a overfitting, interpretabilidade das features
- **Uso**: Modelo baseline para comparação de performance

### XGBoost
- **Arquivo**: `xgboost_model.py` 
- **Propósito**: Gradient boosting otimizado para alta performance
- **Vantagens**: Excelente performance, otimização de hiperparâmetros automática
- **Uso**: Modelo principal para detecção de fraudes

### Neural Networks
- **Arquivo**: `neural_network_model.py`
- **Propósito**: Redes neurais profundas para capturar padrões complexos
- **Vantagens**: Capacidade de aprender representações não-lineares
- **Uso**: Detecção de fraudes sofisticadas e padrões emergentes

### LightGBM
- **Arquivo**: `lightgbm_model.py`
- **Propósito**: Gradient boosting eficiente para grandes volumes de dados
- **Vantagens**: Velocidade de treinamento, menor uso de memória
- **Uso**: Processamento de datasets extensos

## Estrutura de Arquivos

```
supervised/
├── README.md                    # Este arquivo
├── base_model.py               # Classe base para todos os modelos supervisionados
├── random_forest_model.py      # Implementação Random Forest
├── xgboost_model.py           # Implementação XGBoost
├── lightgbm_model.py          # Implementação LightGBM
├── neural_network_model.py    # Implementação Redes Neurais
├── model_trainer.py           # Classe para treinamento unificado
└── config/
    ├── rf_config.yaml         # Configuração Random Forest
    ├── xgb_config.yaml        # Configuração XGBoost
    ├── lgb_config.yaml        # Configuração LightGBM
    └── nn_config.yaml         # Configuração Neural Networks
```

## Como Usar

### Treinamento Individual

```python
from src.models.supervised.xgboost_model import XGBoostFraudDetector
from src.data.preprocessing.pipeline import DataPipeline

# Preparar dados
pipeline = DataPipeline()
X_train, y_train, X_val, y_val = pipeline.prepare_training_data()

# Treinar modelo XGBoost
model = XGBoostFraudDetector()
model.train(X_train, y_train, X_val, y_val)

# Salvar modelo
model.save_model('models/xgboost_v1.0.pkl')
```

### Treinamento em Batch

```python
from src.models.supervised.model_trainer import SupervisedTrainer

# Treinar múltiplos modelos
trainer = SupervisedTrainer()
models = trainer.train_all_models(
    models=['rf', 'xgb', 'lgb', 'nn'],
    train_data=(X_train, y_train),
    val_data=(X_val, y_val)
)

# Comparar performances
trainer.compare_models(models)
```

## Métricas de Avaliação

Todos os modelos supervisionados são avaliados usando:

- **Precision**: Proporção de predições positivas corretas
- **Recall**: Proporção de fraudes reais identificadas
- **F1-Score**: Média harmônica entre precision e recall
- **AUC-ROC**: Área sob a curva ROC
- **AUC-PR**: Área sob a curva Precision-Recall
- **Confusion Matrix**: Matriz de confusão detalhada

## Configuração de Hiperparâmetros

Cada modelo possui arquivo de configuração YAML específico:

```yaml
# Exemplo: xgb_config.yaml
model_params:
  n_estimators: 1000
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42

training_params:
  early_stopping_rounds: 50
  eval_metric: 'auc'
  verbose: 100

hyperopt_params:
  max_evals: 100
  cv_folds: 5
```

## Boas Práticas

### Desenvolvimento
- Herde sempre da classe `BaseModel` para consistência
- Implemente validação cruzada nos métodos de treinamento
- Use configurações YAML para hiperparâmetros
- Documente alterações no desempenho dos modelos

### Produção
- Valide modelos em dados out-of-time antes do deploy
- Monitore drift de performance continuamente
- Mantenha versionamento rigoroso dos modelos
- Implemente fallback para modelos de backup

## Integração com Outras Camadas

- **Features**: Utiliza features do módulo `src.features`
- **Data**: Consome dados processados de `src.data`
- **Ensemble**: Integra-se com `src.models.ensemble` para combinação
- **Inference**: Modelos são servidos via `src.inference`
- **Monitoring**: Métricas enviadas para `src.monitoring`

## Expansão Futura

Para adicionar novos algoritmos supervisionados:

1. Crie classe herdando de `BaseModel`
2. Implemente métodos obrigatórios (`train`, `predict`, `evaluate`)
3. Adicione arquivo de configuração YAML
4. Atualize `model_trainer.py` para incluir novo modelo
5. Documente no README e adicione testes
6. Execute benchmarks comparativos

## Suporte e Contribuição

Para dúvidas ou sugestões sobre modelos supervisionados:
- Consulte documentação da classe `BaseModel`
- Revise exemplos nos arquivos de teste
- Abra issue no repositório principal
- Consulte logs de treinamento em `logs/supervised/`
