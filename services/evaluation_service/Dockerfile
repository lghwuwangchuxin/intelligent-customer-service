# Evaluation Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY services/evaluation_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy common library
COPY services/common /app/services/common

# Copy RAG service (for shared components)
COPY services/rag_service /app/services/rag_service

# Copy service code
COPY services/evaluation_service /app/services/evaluation_service

# Create logs directory
RUN mkdir -p /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${HTTP_PORT:-8003}/health || exit 1

EXPOSE 8003

# Environment defaults
ENV SERVICE_HOST=0.0.0.0 \
    HTTP_PORT=8003 \
    LOG_LEVEL=INFO \
    LOG_TO_CONSOLE=true \
    LOG_TO_FILE=true \
    LOG_DIR=/app/logs \
    NACOS_ENABLED=true \
    NACOS_SERVER_ADDRESSES=nacos:8848 \
    NACOS_NAMESPACE=public \
    NACOS_GROUP=DEFAULT_GROUP \
    REDIS_URL=redis://redis:6379 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

CMD ["python", "-m", "services.evaluation_service"]
