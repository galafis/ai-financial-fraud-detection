# Dockerfile para AI Financial Fraud Detection System
# Propósito: Container minimalista para API FastAPI de detecção de fraudes
# Base: Python 3.9 slim para otimizar tamanho da imagem

# Usar imagem oficial Python 3.9 slim como base
FROM python:3.9-slim

# Definir variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Criar usuário não-root para segurança
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Instalar dependencias do sistema necessarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivo de requirements primeiro (para aproveitar cache do Docker)
COPY config/requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fonte da aplicação
COPY src/ ./src/

# Criar diretórios necessários
RUN mkdir -p /app/models /app/logs /app/data

# Alterar proprietário dos arquivos para o usuário não-root
RUN chown -R appuser:appuser /app

# Mudar para usuário não-root
USER appuser

# Expor porta da aplicação
EXPOSE $PORT

# Health check para monitoramento
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# ENTRYPOINT para executar aplicação FastAPI
# Usar uvicorn como servidor ASGI para produção
ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
