# API REST – AI Financial Fraud Detection

Este diretório contém a implementação da API REST (FastAPI) responsável por expor endpoints para detecção de fraudes, consultas de resultados e integração com serviços externos.

## Principais arquivos
- `main.py`: Entrada principal da API FastAPI
- `routes/`: Implementações dos endpoints REST
- `schemas/`: Schemas Pydantic para validação de dados
- `dependencies/`: Middlewares e dependências (ex: autenticação, CORS)

## Instruções rápidas
- Para rodar em modo desenvolvimento: `uvicorn src.api.main:app --reload`
- Os endpoints REST estão documentados em `/docs` (Swagger) ao rodar a API
- Expanda os módulos conforme novas funcionalidades e integrações forem necessárias.

Para dúvidas ou sugestões, consulte o README principal ou abra uma issue no repositório.
