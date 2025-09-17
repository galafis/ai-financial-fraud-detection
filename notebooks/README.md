# Notebooks - AI Financial Fraud Detection

## 📋 Finalidade

Este diretório contém notebooks Jupyter destinados a:

- **Análises Exploratórias**: Investigação inicial dos dados, identificação de padrões, outliers e características dos datasets
- **Demonstrações**: Apresentação de funcionalidades do sistema, exemplos de uso dos modelos e visualizações
- **Experimentação**: Testes de algoritmos, comparação de modelos, validação de hipóteses e prototipagem

## 📁 Organização e Estrutura

### Convenções de Nomeação

Utilize o seguinte padrão para nomear os notebooks:

```
[número]_[categoria]_[descrição_breve]_[versão].ipynb
```

**Exemplos:**
- `01_eda_transaction_data_analysis_v1.ipynb`
- `02_model_fraud_detection_comparison_v2.ipynb`
- `03_demo_real_time_prediction_v1.ipynb`
- `04_exp_feature_engineering_tests_v3.ipynb`

### Categorias Recomendadas

- `eda`: Exploratory Data Analysis (Análise Exploratória)
- `model`: Modelagem e treinamento
- `demo`: Demonstrações e apresentações
- `exp`: Experimentação e testes
- `eval`: Avaliação e validação de modelos
- `viz`: Visualizações especiais

### Estrutura de Diretórios

```
notebooks/
├── README.md
├── exploratory/          # Análises exploratórias
├── modeling/            # Notebooks de modelagem
├── demonstrations/      # Demos e apresentações
├── experiments/         # Experimentações
└── archived/           # Notebooks antigos/descontinuados
```

## 🔒 Segurança e Dados Sensíveis

### ❌ NÃO SALVAR:

- Dados reais de clientes ou transações
- Credenciais de acesso (APIs, bancos de dados)
- Informações pessoais identificáveis (PII)
- Chaves de criptografia ou tokens de autenticação
- Dados proprietários ou confidenciais

### ✅ UTILIZAR:

- Dados sintéticos ou anonimizados
- Variáveis de ambiente para credenciais
- Datasets públicos para demonstrações
- Amostras pequenas e não identificáveis
- Placeholders para dados sensíveis

### Exemplo de Boas Práticas:

```python
# ❌ Evitar
api_key = "sk-1234567890abcdef"
df = pd.read_csv("real_customer_data.csv")

# ✅ Recomendado
api_key = os.getenv('API_KEY')
df = pd.read_csv("synthetic_fraud_dataset.csv")
```

## 📝 Boas Práticas para Notebooks

### Estrutura Padrão do Notebook

1. **Cabeçalho com Metadados**
```markdown
# Título do Notebook

**Autor:** Nome do Desenvolvedor  
**Data:** DD/MM/YYYY  
**Versão:** v1.0  
**Objetivo:** Breve descrição do propósito  
**Datasets:** Lista dos dados utilizados  
```

2. **Seções Organizadas**
```markdown
## 1. Configuração e Imports
## 2. Carregamento de Dados
## 3. Análise Exploratória
## 4. Processamento/Modelagem
## 5. Resultados e Visualizações
## 6. Conclusões
## 7. Próximos Passos
```

### Markdown e Documentação

#### Headers Hierárquicos
```markdown
# H1: Título Principal
## H2: Seções Principais
### H3: Subseções
#### H4: Detalhes Específicos
```

#### Formatação de Código
```markdown
- Use `código inline` para variáveis e funções
- Use blocos de código para snippets:

```python
def detect_fraud(transaction):
    return model.predict(transaction)
```
```

#### Listas e Organização
```markdown
- **Negrito** para termos importantes
- *Itálico* para ênfase
- > Blockquotes para observações importantes
- 📊 Emojis para categorizar seções
```

### Comentários no Código

#### Comentários Descritivos
```python
# Carregamento e preparação dos dados de transações financeiras
df = pd.read_csv('financial_transactions.csv')

# Removendo outliers usando IQR (Interquartile Range)
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df['amount'] < (Q1 - 1.5 * IQR)) | (df['amount'] > (Q3 + 1.5 * IQR)))]
```

#### Comentários de Insight
```python
# INSIGHT: 85% das transações fraudulentas ocorrem entre 22h-6h
# Isso sugere que o horário é um feature importante para o modelo
nighttime_fraud_rate = df.groupby('hour')['is_fraud'].mean()
```

#### TODO e FIXME
```python
# TODO: Implementar validação cruzada k-fold
# FIXME: Resolver problema de vazamento de dados (data leakage)
# NOTE: Considerar feature engineering adicional
```

### Padronização de Código

#### Imports Organizados
```python
# Bibliotecas padrão
import os
import sys
from datetime import datetime

# Bibliotecas de dados
import pandas as pd
import numpy as np

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Configurações
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
```

#### Configurações Globais
```python
# Configurações de reproducibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configurações de visualização
FIGSIZE = (12, 8)
DPI = 100

# Parâmetros do projeto
DATA_PATH = '../data/'
MODEL_PATH = '../models/'
OUTPUT_PATH = '../output/'
```

#### Funções Auxiliares
```python
def load_and_validate_data(filepath):
    """Carrega e valida dados de transações.
    
    Args:
        filepath (str): Caminho para o arquivo de dados
        
    Returns:
        pd.DataFrame: DataFrame validado
        
    Raises:
        FileNotFoundError: Quando arquivo não existe
        ValueError: Quando dados são inválidos
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validações básicas
    required_columns = ['transaction_id', 'amount', 'timestamp', 'is_fraud']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")
    
    return df
```

## 🔄 Versionamento

### Controle de Versão
- Incremente a versão para mudanças significativas
- Mantenha registro de mudanças no cabeçalho
- Archive versões antigas quando apropriado

### Exemplo de Versionamento
```markdown
## Histórico de Versões
- **v1.0** (15/09/2024): Versão inicial com análise básica
- **v1.1** (20/09/2024): Adicionadas visualizações interativas
- **v2.0** (25/09/2024): Refatoração completa com novos modelos
```

## 🧹 Limpeza e Manutenção

### Antes de Commitar
- [ ] Limpar outputs desnecessários
- [ ] Verificar se não há dados sensíveis
- [ ] Validar que o código executa do início ao fim
- [ ] Revisar comentários e documentação
- [ ] Confirmar estrutura de markdown

### Comando para Limpeza
```bash
# Limpar todos os outputs dos notebooks
jupyter nbconvert --clear-output --inplace *.ipynb
```

## 📚 Recursos Adicionais

- [Jupyter Notebook Best Practices](https://jupyter.readthedocs.io/en/latest/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Python PEP 8 Style Guide](https://pep8.org/)
- [Pandas Style Guide](https://pandas.pydata.org/docs/development/contributing.html#code-standards)

---

**Lembre-se:** Notebooks bem documentados e organizados facilitam a colaboração, reprodutibilidade e manutenção do projeto. Sempre priorize clareza e segurança!
