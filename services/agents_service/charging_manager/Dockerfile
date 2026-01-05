# Dockerfile for Charging Manager Agent
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "a2a-sdk[http-server]>=0.3.0" \
    uvicorn>=0.24.0 \
    httpx>=0.24.0

COPY agents/common /app/common
COPY agents/charging_manager /app/agent

EXPOSE 9002

CMD ["python", "-m", "agent"]
