# Dockerfile for Data Analyst Agent
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
COPY agents/data_analyst /app/agent

EXPOSE 9005

CMD ["python", "-m", "agent"]
