# Dockerfile for Travel Assistant Agent
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    "a2a-sdk[http-server]>=0.3.0" \
    uvicorn>=0.24.0 \
    httpx>=0.24.0

# Copy common module
COPY agents/common /app/common

# Copy agent module
COPY agents/travel_assistant /app/agent

EXPOSE 9001

CMD ["python", "-m", "agent"]
