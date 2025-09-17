# Engenharia de Features – AI Financial Fraud Detection

Diretório dedicado à extração, transformação, seleção e validação de features utilizadas no sistema de detecção de fraudes.

## Estrutura e principais arquivos
- `extractors/`: Funções para extração de features comportamentais, temporais e geográficas.
- `transformers/`: Pipelines para normalização, codificação e transformação dos dados.
- `store/`: Integração com feature store (ex: Redis) para caching e versionamento de features.
- `validation/`: Módulos de validação de qualidade dos dados de entrada.

## Orientações
- Ao criar novos tipos de features, documente no README e siga a modularização existente.
- Implemente testes unitários para todos os novos componentes.
- Utilize docstrings e typing para clareza em funções e classes.

Expanda este módulo conforme novas fontes, técnicas ou requisitos emergentes.
