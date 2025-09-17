# Utilitários e Configurações – AI Financial Fraud Detection

Diretório de funções utilitárias, módulos de apoio e arquivos de configuração aplicáveis a vários componentes do sistema.

## Estrutura e principais arquivos
- `config.py`: Parâmetros globais de configuração do sistema
- `helpers.py`: Funções utilitárias e reuso geral
- `logger.py`: Configuração centralizada de logging
- `constants.py`: Definições de constantes usadas em múltiplos módulos

## Orientações
- Documente cada nova função ou módulo utilitário que for criado, com exemplos de uso quando possível.
- Prefira modularizar o código reusável nesta pasta para garantir manutenção e DRY.
- Mantenha variáveis e parâmetros sensíveis em arquivos .env fora do versionamento.

Consulte este diretório sempre que houver necessidade de reaproveitamento de funções ou parâmetro global.
