# Single Agent Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY services/single_agent_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy common library
COPY services/common /app/services/common

# Copy service code
COPY services/single_agent_service /app/services/single_agent_service

# Create logs directory
RUN mkdir -p /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${HTTP_PORT:-8005}/health || exit 1

EXPOSE 8005

# Environment defaults
ENV SERVICE_HOST=0.0.0.0 \
    HTTP_PORT=8005 \
    LOG_LEVEL=INFO \
    LOG_TO_CONSOLE=true \
    LOG_TO_FILE=true \
    LOG_DIR=/app/logs \
    NACOS_ENABLED=true \
    NACOS_SERVER_ADDRESSES=nacos:8848 \
    NACOS_NAMESPACE=public \
    NACOS_GROUP=DEFAULT_GROUP \
    LLM_BASE_URL=http://ollama:11434 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

CMD ["python", "-m", "services.single_agent_service"]
