# 智能客服系统 (Intelligent Customer Service)

基于 LlamaIndex + LangGraph 的企业级智能客服系统，提供高质量的 RAG 检索和智能对话能力。

## 目录

- [功能说明](#功能说明)
- [技术框架](#技术框架)
- [系统架构](#系统架构)
- [部署环境](#部署环境)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [RAG 模块](#rag-模块)
- [Agent 系统](#agent-系统)
- [核心模块](#核心模块)
- [API 文档](#api-文档)
- [调试说明](#调试说明)
- [测试](#测试)
- [开发指南](#开发指南)

---

## 功能说明

### 核心功能

| 功能模块 | 功能描述 | 技术实现 |
|---------|---------|---------|
| **智能对话** | 多轮对话、上下文理解、流式输出 | LangGraph + ReAct Agent |
| **知识库检索** | 向量检索、关键词匹配、混合检索 | LlamaIndex + Milvus |
| **文档处理** | PDF/Word/Excel/Markdown 解析、智能分块 | 语义分块 + 重叠窗口 |
| **多模型支持** | Ollama/OpenAI/Claude/DeepSeek/通义千问 | 统一 LLM 管理器 |
| **高性能缓存** | Redis 分布式缓存、Embedding 缓存、搜索结果缓存 | Redis + 内存双层缓存 |
| **可观测性** | 全链路追踪、性能监控、日志分析 | Langfuse 集成 |
| **工具调用** | 知识库查询、网页搜索、代码执行 | MCP 工具协议 |

### Embedding 高性能优化

| 功能 | 描述 | 性能提升 |
|------|------|---------|
| **直接 API 调用** | 绕过 LangChain 开销，直接调用 Ollama `/api/embed` | 减少 ~30% 延迟 |
| **HTTP 连接池** | 复用 TCP 连接，避免重复握手 | 减少 ~50% 连接开销 |
| **批量处理** | 50 文档/批次，20 并发请求 | 65+ docs/sec 吞吐量 |
| **线程池并行** | ThreadPoolExecutor 同步方法并行化 | 75+ docs/sec |
| **指数退避重试** | 自动重试失败请求，带随机抖动 | 提升稳定性 |
| **Redis 分布式锁** | 防止并发重复计算相同文本 | 避免资源浪费 |
| **预热机制** | 启动时加载模型到内存 | 消除冷启动延迟 |
| **缓存加速** | Redis 缓存已计算的 Embedding | 19x 缓存命中加速 |

### RAG 增强检索

| 功能 | 描述 | 配置项 |
|------|------|--------|
| **HyDE** | 假设文档嵌入，提升语义理解 | `RAG_ENABLE_HYDE` |
| **Query Expansion** | 查询扩展，多角度检索 | `RAG_ENABLE_QUERY_EXPANSION` |
| **Hybrid Retrieval** | 向量+BM25 混合检索 | `RAG_ENABLE_HYBRID` |
| **Jina Reranker** | 神经网络重排序 | `RAG_ENABLE_RERANK` |
| **Semantic Dedup** | 语义去重 | `RAG_ENABLE_DEDUP` |
| **MMR Diversity** | 结果多样性过滤 | 内置 |

### Agent 能力

| 功能 | 描述 |
|------|------|
| **LangGraph 状态机** | 复杂对话流程管理，支持条件分支 |
| **ReAct 推理** | Think → Act → Observe 循环推理 |
| **任务规划** | 复杂问题分解为子任务 |
| **并行工具执行** | 同时调用多个工具提升效率 |
| **对话记忆** | 多轮对话上下文管理与摘要 |
| **错误恢复** | 智能重试、降级策略、指数退避 |

### 前端功能

| 功能 | 描述 |
|------|------|
| **对话界面** | 流式响应、Markdown 渲染、代码高亮 |
| **知识库管理** | 文档上传、预览、删除 |
| **模型配置** | 动态切换模型、参数调整 |
| **会话管理** | 多会话支持、历史记录 |

---

## 技术框架

### 技术栈总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        技术栈架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   前端 (Frontend)                                               │
│   ├── React 18 + TypeScript                                     │
│   ├── Vite 构建工具                                             │
│   ├── TailwindCSS 样式框架                                       │
│   ├── React Markdown (Markdown 渲染)                            │
│   └── Axios (HTTP 客户端)                                       │
│                                                                  │
│   后端 (Backend)                                                 │
│   ├── Python 3.11+                                              │
│   ├── FastAPI (Web 框架)                                        │
│   ├── LlamaIndex 0.12+ (RAG 框架)                               │
│   ├── LangGraph/LangChain (Agent 框架)                          │
│   ├── Pydantic v2 (数据验证)                                    │
│   └── SQLAlchemy + aiosqlite (数据库)                           │
│                                                                  │
│   向量数据库 (Vector Store)                                      │
│   └── Milvus 2.5+ (分布式向量数据库)                             │
│                                                                  │
│   文档存储 (Document Store)                                       │
│   └── Elasticsearch 8.5+ (全文检索 + 元数据存储)                 │
│                                                                  │
│   大语言模型 (LLM)                                               │
│   ├── Ollama (本地部署: Qwen3, Llama3, DeepSeek)                │
│   ├── OpenAI (GPT-4o, GPT-4-turbo)                              │
│   ├── Anthropic Claude (Claude 3.5 Sonnet)                      │
│   ├── DeepSeek (DeepSeek-Chat, DeepSeek-Coder)                  │
│   ├── 阿里云通义千问 (Qwen-Turbo, Qwen-Max)                      │
│   ├── 百度文心一言 (ERNIE-4.0)                                   │
│   └── 其他: 火山引擎豆包、智谱 AI、月之暗面 Kimi                  │
│                                                                  │
│   嵌入模型 (Embedding)                                           │
│   ├── nomic-embed-text (Ollama)                                 │
│   ├── text-embedding-3-small (OpenAI)                           │
│   └── jina-embeddings-v2 (Jina AI)                              │
│                                                                  │
│   可观测性 (Observability)                                       │
│   └── Langfuse (追踪、监控、评估)                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心依赖版本

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.11+ | 运行环境 |
| FastAPI | 0.115+ | Web API 框架 |
| LlamaIndex | 0.12+ | RAG 检索框架 |
| LangGraph | 0.2+ | Agent 状态机 |
| LangChain | 0.3+ | Agent 工具集成 |
| Milvus | 2.5+ | 向量数据库 |
| Elasticsearch | 8.5+ | 全文检索 + 元数据存储 |
| Ollama | 0.12+ | 本地 LLM 服务 |
| Langfuse | 2.0+ | 可观测性平台 |

### 关键第三方库

```python
# RAG 相关
llama-index-core              # RAG 核心框架
llama-index-vector-stores-milvus  # Milvus 向量存储
llama-index-embeddings-ollama     # Ollama 嵌入模型
llama-index-llms-ollama           # Ollama LLM 集成

# Agent 相关
langgraph                     # 状态图 Agent
langchain-core                # Agent 核心组件
langchain-community           # 社区工具集成

# 检索增强
rank-bm25                     # BM25 关键词检索
sentence-transformers         # 神经网络重排序 (可选)
jieba                         # 中文分词

# 混合存储
elasticsearch                 # ES 全文检索 + 元数据存储

# Web 框架
fastapi                       # 异步 Web 框架
uvicorn                       # ASGI 服务器
python-multipart              # 文件上传支持

# 可观测性
langfuse                      # 追踪与监控
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │  对话界面   │  │ 知识库管理  │  │  模型配置   │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │   API Routes    │    │  Upload Service │                     │
│  │  /api/chat      │    │  /api/upload    │                     │
│  │  /api/agent     │    │  /api/knowledge │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                               │
│           ▼                      ▼                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   LangGraph Agent                          │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │ │
│  │  │ Planning │→ │  Tools   │→ │ Execute  │→ │  Response  │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────────┘ │ │
│  │                      │                                     │ │
│  │         ┌────────────┼────────────┐                       │ │
│  │         ▼            ▼            ▼                       │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐               │ │
│  │  │ knowledge │ │web_search │ │   code    │               │ │
│  │  │  _search  │ │           │ │ executor  │               │ │
│  │  └───────────┘ └───────────┘ └───────────┘               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Enhanced RAG (LlamaIndex)                     │ │
│  │                                                             │ │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────┐       │ │
│  │  │   Query    │ → │   Hybrid   │ → │   Jina     │       │ │
│  │  │ Transform  │    │ Retrieval  │    │ Reranker   │       │ │
│  │  │ (HyDE+Exp) │    │ (Vec+BM25) │    │            │       │ │
│  │  └────────────┘    └────────────┘    └────────────┘       │ │
│  │                                              │              │ │
│  │                                              ▼              │ │
│  │                                      ┌────────────┐        │ │
│  │                                      │   Post-    │        │ │
│  │                                      │ Processor  │        │ │
│  │                                      │ (Dedup+MMR)│        │ │
│  │                                      └────────────┘        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│           ┌──────────────────┼──────────────────┐               │
│           ▼                  ▼                  ▼               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │    Milvus    │   │    Ollama    │   │   Langfuse   │        │
│  │ Vector Store │   │     LLM      │   │ Observability│        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 部署环境

### 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **CPU** | 4 核 | 8 核+ |
| **内存** | 8 GB | 16 GB+ |
| **磁盘** | 20 GB SSD | 100 GB SSD |
| **GPU** | 无 (CPU 推理) | NVIDIA 8GB+ (GPU 推理) |

### 软件依赖

| 软件 | 版本 | 必需 | 说明 |
|------|------|------|------|
| Python | 3.11+ | 是 | 运行环境 |
| Node.js | 18+ | 是 | 前端构建 |
| Docker | 24+ | 推荐 | 容器化部署 |
| Milvus | 2.5+ | 是 | 向量数据库 |
| Elasticsearch | 8.5+ | 推荐 | 混合存储 (可禁用) |
| Redis | 7+ | 推荐 | 缓存与分布式锁 (可降级为内存) |
| Ollama | 0.12+ | 推荐 | 本地 LLM |

### 开发环境部署

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/intelligent-customer-service.git
cd intelligent-customer-service

# 2. 后端部署
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 配置

# 3. 前端部署
cd ../frontend
npm install

# 4. 启动 Milvus (Docker)
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:v2.5.0 \
  milvus run standalone

# 5. 启动 Elasticsearch (Docker) - 可选，用于混合存储
docker run -d --name ics-elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.5.0

# 6. 启动 Ollama 并下载模型
ollama serve &
ollama pull qwen3:latest
ollama pull nomic-embed-text

# 6. 启动服务
cd backend && python run.py &      # 后端 http://localhost:8000
cd frontend && npm run dev &       # 前端 http://localhost:3000
```

### 生产环境部署

#### Docker Compose 部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=ollama
      - LLM_BASE_URL=http://ollama:11434
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
      - REDIS_URL=redis://redis:6379/0
      - REDIS_CACHE_ENABLED=true
      - ELASTICSEARCH_ENABLED=true
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
      - HYBRID_STORAGE_ENABLED=true
    depends_on:
      - milvus
      - elasticsearch
      - ollama
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend

  milvus:
    image: milvusdb/milvus:v2.5.0
    ports:
      - "19530:19530"
    volumes:
      - milvus_data:/var/lib/milvus
    command: milvus run standalone

  elasticsearch:
    image: elasticsearch:8.5.0
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    volumes:
      - es_data:/usr/share/elasticsearch/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  milvus_data:
  es_data:
  ollama_data:
  redis_data:
```

#### Kubernetes 部署

```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: customer-service-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: your-registry/customer-service-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLM_PROVIDER
          value: "ollama"
        - name: MILVUS_HOST
          value: "milvus-service"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### 环境变量配置

```bash
# ============ 应用配置 ============
APP_NAME=Intelligent Customer Service
APP_VERSION=1.0.0
DEBUG=false                    # 生产环境设为 false
HOST=0.0.0.0
PORT=8000

# ============ LLM 配置 ============
LLM_PROVIDER=ollama            # ollama, openai, claude, deepseek, aliyun
LLM_MODEL=qwen3:latest
LLM_BASE_URL=http://localhost:11434
LLM_API_KEY=                   # 使用云服务时需要
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048

# ============ 嵌入模型 ============
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_BATCH_SIZE=50             # 批量处理文档数量
EMBEDDING_MAX_CONCURRENT=20         # 最大并发请求数

# ============ Redis 缓存 ============
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=                      # Redis 密码 (可选)
REDIS_MAX_CONNECTIONS=10
REDIS_CACHE_ENABLED=true             # 启用 Redis 缓存
REDIS_CACHE_TTL=3600                 # 默认缓存 TTL (1小时)
REDIS_EMBEDDING_CACHE_TTL=86400      # Embedding 缓存 (24小时)
REDIS_SEARCH_CACHE_TTL=300           # 搜索缓存 (5分钟)

# ============ 向量数据库 ============
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=customer_service_kb

# ============ Elasticsearch 混合存储 ============
ELASTICSEARCH_ENABLED=true              # 启用 ES 混合存储
ELASTICSEARCH_HOSTS=http://localhost:9200
ELASTICSEARCH_USERNAME=                 # ES 用户名 (可选)
ELASTICSEARCH_PASSWORD=                 # ES 密码 (可选)
ELASTICSEARCH_CHUNK_INDEX=knowledge_base_chunks
ELASTICSEARCH_USE_SSL=false
ELASTICSEARCH_VERIFY_CERTS=true
ELASTICSEARCH_TIMEOUT=30
ELASTICSEARCH_MAX_RETRIES=3

# ============ 混合检索配置 ============
HYBRID_STORAGE_ENABLED=true             # 启用混合存储 (ES + Milvus)
HYBRID_MILVUS_WEIGHT=0.7                # 向量检索权重
HYBRID_ES_WEIGHT=0.3                    # BM25 检索权重
HYBRID_SEARCH_TOP_K=20                  # 混合检索召回数量

# ============ RAG 配置 ============
RAG_ENABLE_HYDE=true
RAG_ENABLE_QUERY_EXPANSION=true
RAG_QUERY_EXPANSION_NUM=3
RAG_ENABLE_HYBRID=true
RAG_VECTOR_WEIGHT=0.7
RAG_BM25_WEIGHT=0.3
RAG_RETRIEVAL_TOP_K=10
RAG_ENABLE_RERANK=true
RAG_RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
RAG_RERANK_TOP_N=5
RAG_RERANK_SCORE_THRESHOLD=0.3
RAG_ENABLE_DEDUP=true
RAG_DEDUP_THRESHOLD=0.95
RAG_FINAL_TOP_K=5

# ============ Agent 配置 ============
AGENT_ENABLED=true
AGENT_MAX_ITERATIONS=10

# ============ MCP 工具 ============
MCP_WEB_SEARCH_ENABLED=true
MCP_CODE_EXECUTION_ENABLED=false  # 安全考虑默认关闭

# ============ 可观测性 ============
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 快速开始

### 1. 环境准备

```bash
# 安装 Python 3.11+
python --version  # 确认版本

# 安装 Node.js 18+
node --version

# 安装 Docker (可选，用于 Milvus)
docker --version
```

### 2. 启动依赖服务

```bash
# 方式一: Docker 启动 Milvus
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.5.0 \
  milvus run standalone

# 方式二: Milvus Lite (开发测试)
pip install pymilvus[lite]  # 内嵌模式，无需 Docker

# 启动 Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull qwen3:latest
ollama pull nomic-embed-text
```

### 3. 后端服务

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py
# 服务启动于 http://localhost:8000
# API 文档: http://localhost:8000/docs
```

### 4. 前端服务

```bash
cd frontend
npm install
npm run dev
# 服务启动于 http://localhost:3000
```

### 5. 验证部署

```bash
# 检查后端健康状态
curl http://localhost:8000/api/system/health

# 检查 Ollama 模型
curl http://localhost:8000/api/config/ollama/models

# 测试对话 API
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "use_rag": true}'
```

---

## 服务启动与停止

### 依赖服务管理

```bash
# ==================== Milvus ====================

# 启动 Milvus
docker start milvus
# 或首次启动
docker run -d --name milvus \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.5.0 \
  milvus run standalone

# 停止 Milvus
docker stop milvus

# 重启 Milvus
docker restart milvus

# 查看 Milvus 状态
docker ps | grep milvus
docker logs -f milvus

# 删除 Milvus 容器 (数据会保留在 volume 中)
docker rm milvus

# ==================== Elasticsearch ====================

# 首次启动 Elasticsearch
docker run -d --name ics-elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.5.0

# 启动 Elasticsearch
docker start ics-elasticsearch

# 停止 Elasticsearch
docker stop ics-elasticsearch

# 重启 Elasticsearch
docker restart ics-elasticsearch

# 查看 Elasticsearch 状态
docker ps | grep elasticsearch
docker logs -f ics-elasticsearch

# 验证 ES 连接
curl http://localhost:9200

# 删除 Elasticsearch 容器
docker rm ics-elasticsearch

# ==================== Ollama ====================

# 启动 Ollama (前台运行)
ollama serve

# 启动 Ollama (后台运行)
ollama serve &

# 停止 Ollama
pkill ollama
# 或
killall ollama

# 查看 Ollama 进程
ps aux | grep ollama

# 查看已下载的模型
ollama list

# 下载模型
ollama pull qwen3:latest
ollama pull nomic-embed-text

# 删除模型
ollama rm qwen3:latest
```

### 后端服务管理

```bash
# ==================== 开发模式 ====================

# 进入后端目录
cd backend

# 激活虚拟环境
source .venv/bin/activate      # Linux/Mac
# 或
.venv\Scripts\activate         # Windows

# 启动后端 (开发模式，支持热重载)
python run.py
# 服务地址: http://localhost:8000
# API 文档: http://localhost:8000/docs

# 停止后端 (Ctrl+C 或)
# 方式1: 查找并杀死进程
lsof -ti:8000 | xargs kill -9

# 方式2: 使用 pkill
pkill -f "python run.py"

# 方式3: 查找 uvicorn 进程
ps aux | grep uvicorn
kill -9 <PID>

# ==================== 生产模式 ====================

# 使用 gunicorn 启动 (多进程)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# 使用 uvicorn 启动 (单进程)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 后台运行 (nohup)
nohup python run.py > logs/backend.log 2>&1 &

# 查看后台日志
tail -f logs/backend.log

# ==================== 环境管理 ====================

# 退出虚拟环境
deactivate

# 重新安装依赖
pip install -r requirements.txt --upgrade
```

### 前端服务管理

```bash
# ==================== 开发模式 ====================

# 进入前端目录
cd frontend

# 安装依赖 (首次或更新后)
npm install

# 启动前端 (开发模式，支持热重载)
npm run dev
# 服务地址: http://localhost:3000

# 停止前端 (Ctrl+C 或)
# 方式1: 查找并杀死进程
lsof -ti:3000 | xargs kill -9

# 方式2: 使用 pkill
pkill -f "vite"

# ==================== 生产模式 ====================

# 构建生产版本
npm run build

# 预览构建结果
npm run preview

# 使用 serve 部署静态文件
npx serve -s dist -l 3000

# 使用 nginx 部署 (将 dist 目录复制到 nginx html 目录)
cp -r dist/* /usr/share/nginx/html/
```

### 一键启动脚本

```bash
#!/bin/bash
# start-all.sh - 一键启动所有服务

echo "Starting Milvus..."
docker start milvus || docker run -d --name milvus \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.5.0 milvus run standalone

echo "Starting Elasticsearch..."
docker start ics-elasticsearch || docker run -d --name ics-elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.5.0

echo "Starting Redis..."
docker start redis || docker run -d --name redis \
  -p 6379:6379 redis:7-alpine

echo "Starting Ollama..."
ollama serve &
sleep 3

echo "Starting Backend..."
cd backend
source .venv/bin/activate
nohup python run.py > logs/backend.log 2>&1 &
sleep 5

echo "Starting Frontend..."
cd ../frontend
nohup npm run dev > logs/frontend.log 2>&1 &

echo "All services started!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
```

```bash
#!/bin/bash
# stop-all.sh - 一键停止所有服务

echo "Stopping Frontend..."
pkill -f "vite" 2>/dev/null

echo "Stopping Backend..."
lsof -ti:8000 | xargs kill -9 2>/dev/null

echo "Stopping Ollama..."
pkill ollama 2>/dev/null

echo "Stopping Elasticsearch..."
docker stop ics-elasticsearch 2>/dev/null

echo "Stopping Milvus..."
docker stop milvus 2>/dev/null

echo "All services stopped!"
```

### 服务状态检查

```bash
# 检查所有服务状态
echo "=== Milvus ===" && docker ps | grep milvus
echo "=== Elasticsearch ===" && curl -s http://localhost:9200 | head -1
echo "=== Ollama ===" && curl -s http://localhost:11434/api/tags | head -1
echo "=== Backend ===" && curl -s http://localhost:8000/api/system/health
echo "=== Frontend ===" && curl -s -o /dev/null -w "%{http_code}" http://localhost:3000

# 检查端口占用
lsof -i:8000    # 后端
lsof -i:3000    # 前端
lsof -i:19530   # Milvus
lsof -i:9200    # Elasticsearch
lsof -i:11434   # Ollama
```

---

## 混合存储架构 (ES + Milvus)

### 架构概述

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           文档索引流程                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  原始文档 ──→ DocumentProcessor ──→ ES存储(文本+元数据)                   │
│     │              │                      │                             │
│     │              ↓                      ↓                             │
│     │         文本分块              chunk_id + 元数据                    │
│     │              │                      │                             │
│     │              ↓                      ↓                             │
│     │      EmbeddingService ──→ Milvus存储(向量 + chunk_id)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG检索流程                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  用户提问 ──→ 向量化 ──→ Milvus语义检索 ──→ Top K chunk_ids              │
│                              │                    │                     │
│                              ↓                    ↓                     │
│                    (可选) ES BM25检索      ES获取完整文本+元数据          │
│                              │                    │                     │
│                              └────→ RRF融合排序 ←────┘                   │
│                                        │                                │
│                                        ↓                                │
│                              LLM生成最终答案                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 组件职责

| 组件 | 核心角色 | 存储内容 | 优势 |
|------|---------|---------|------|
| **Elasticsearch** | 文档管家 + 元数据过滤器 | 原始/解析文本、丰富元数据、关键词索引 | 全文检索、复杂条件过滤、聚合分析 |
| **Milvus** | 向量搜索引擎 | 文本嵌入向量、ES文档关联ID | 大规模高维向量检索、低延迟、高召回率 |

### 核心模块

```
app/core/
├── elasticsearch_manager.py  # ES 客户端管理器
├── hybrid_store_manager.py   # 混合存储协调器
└── ...

app/rag/
├── hybrid_es_retriever.py    # ES+Milvus 混合检索器
└── ...
```

### 配置说明

```bash
# 启用混合存储
ELASTICSEARCH_ENABLED=true
HYBRID_STORAGE_ENABLED=true

# ES 连接配置
ELASTICSEARCH_HOSTS=http://localhost:9200
ELASTICSEARCH_CHUNK_INDEX=knowledge_base_chunks

# 混合检索权重
HYBRID_MILVUS_WEIGHT=0.7     # 向量检索权重 (语义相似度)
HYBRID_ES_WEIGHT=0.3         # BM25 检索权重 (关键词匹配)
```

### RRF 融合算法

混合检索使用 Reciprocal Rank Fusion (RRF) 算法融合向量检索和 BM25 检索结果:

```
RRF_score(d) = Σ weight_i / (k + rank_i(d))
```

- `k=60`: 平滑参数，防止高排名文档权重过大
- `weight`: 可配置的检索方式权重

### API 端点

| 端点 | 描述 |
|------|------|
| `POST /api/knowledge/search-advanced` | 高级混合搜索 (支持元数据过滤) |
| `GET /api/knowledge/storage/health` | 存储健康检查 (ES + Milvus) |
| `GET /api/knowledge/storage/stats` | 存储统计信息 |
| `GET /api/knowledge/document/{doc_id}` | 获取文档详情 |
| `DELETE /api/knowledge/document/{doc_id}` | 删除文档 (同步删除 ES 和 Milvus) |

### 降级策略

当 ES 不可用时，系统自动降级为纯 Milvus 向量检索:

```
ES 可用 + Milvus 可用 → 混合检索 (RRF 融合)
     ↓ ES 故障
纯 Milvus → 向量语义检索
```

---

## RAG 模块

### 目录结构

```
app/rag/
├── __init__.py           # 模块入口 (懒加载)
├── query_engine.py       # 核心查询引擎 (5阶段管道)
├── query_transform.py    # 查询转换 (HyDE, Expansion)
├── hybrid_retriever.py   # 混合检索 (Vector + BM25 + RRF)
├── reranker.py          # Jina 重排序 (CrossEncoder)
├── postprocessor.py     # 后处理 (语义去重, MMR)
└── index_manager.py     # 索引管理 (Milvus 连接)
```

### 处理流程

```
用户问题: "如何申请退款？"
         │
         ▼
┌─────────────────────────────────────┐
│  1. Query Transform                 │
│  ├─ HyDE: 生成假设答案文档          │
│  └─ Expansion: 生成 3 个扩展查询    │
│     - "退款申请流程是什么"          │
│     - "退款需要什么条件"            │
│     - "如何操作退款"                │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  2. Hybrid Retrieval                │
│  ├─ 向量检索: 语义相似度匹配        │
│  ├─ BM25 检索: 关键词精确匹配       │
│  └─ RRF 融合: 合并排序结果          │
│     检索 10 个候选文档              │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  3. Jina Reranker                   │
│  ├─ CrossEncoder 重新评分           │
│  └─ 按相关性排序，过滤低分文档      │
│     保留 5 个高质量文档             │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  4. Post-processing                 │
│  ├─ 语义去重: 移除相似度>95%的文档  │
│  └─ MMR: 增加结果多样性             │
│     最终 5 个多样化文档             │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  5. Response Generation             │
│  └─ LLM 基于上下文生成回答          │
└─────────────────────────────────────┘
```

### 核心算法

#### RRF (Reciprocal Rank Fusion)

```
RRF_score(d) = Σ weight_i / (k + rank_i(d))
```

- `k=60`: 平滑参数
- `weight`: 向量权重 0.7, BM25 权重 0.3

#### MMR (Maximal Marginal Relevance)

```
MMR(Di) = λ × Rel(Di, Q) - (1-λ) × max(Sim(Di, Dj))
```

- `λ=0.5`: 相关性与多样性平衡

---

## Agent 系统

### LangGraph 状态机

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  START  │ ──▶ │ ANALYZE │ ──▶ │ EXECUTE │ ──▶ │GENERATE │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                     │               │
                     ▼               ▼
              ┌─────────────┐  ┌─────────────┐
              │ Tool Select │  │ Tool Result │
              └─────────────┘  └─────────────┘
```

### ReAct 推理循环

```
Think: 分析问题，决定是否需要工具
  ↓
Act: 调用工具 (knowledge_search, web_search, etc.)
  ↓
Observe: 观察工具返回结果
  ↓
(重复直到获得足够信息)
  ↓
Final Answer: 生成最终回答
```

### MCP 工具

| 工具 | 描述 | 配置 |
|------|------|------|
| `knowledge_search` | 知识库检索 | 默认启用 |
| `knowledge_add_text` | 添加知识 | 默认启用 |
| `web_search` | 网页搜索 (DuckDuckGo) | `MCP_WEB_SEARCH_ENABLED` |
| `code_executor` | Python 代码执行 | `MCP_CODE_EXECUTION_ENABLED` |

---

## 核心模块

### Embedding 管理器

高性能 Embedding 模块，位于 `app/core/embeddings.py`:

```python
from app.core.embeddings import get_embedding_manager

manager = get_embedding_manager()

# 预热模型 (启动时调用)
await manager.warmup()

# 单文本 Embedding
embedding = await manager.aembed_query("用户问题")

# 批量异步 Embedding (推荐)
embeddings = await manager.aembed_documents_batch(
    texts=["文档1", "文档2", ...],
    batch_size=50,         # 每批处理数量
    max_concurrent=20,     # 最大并发请求
    show_progress=True,    # 显示进度条
    use_cache=True,        # 使用 Redis 缓存
)

# 线程池并行 Embedding (同步方法)
embeddings = manager.embed_documents_threaded(
    texts=["文档1", "文档2", ...],
    batch_size=10,
)
```

#### 架构特性

```
┌────────────────────────────────────────────────────────────┐
│                    EmbeddingManager                         │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ HTTP Client │  │   Redis     │  │  ThreadPoolExecutor │ │
│  │ (httpx)     │  │   Cache     │  │   (10 workers)      │ │
│  │             │  │             │  │                     │ │
│  │ - 连接池    │  │ - Embedding │  │ - 同步并行         │ │
│  │ - Keep-alive│  │   缓存      │  │ - 批量处理         │ │
│  │ - 重试机制  │  │ - 分布式锁  │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Retry Mechanism                        │ │
│  │  - 指数退避 (0.5s → 1s → 2s → 4s ...)                  │ │
│  │  - 随机抖动 (防止惊群效应)                              │ │
│  │  - 最大重试 3 次                                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Distributed Lock                       │ │
│  │  - Redis SET NX (防止并发重复计算)                      │ │
│  │  - 自动过期 (10秒超时)                                  │ │
│  │  - 内存锁降级 (Redis 不可用时)                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### 缓存管理器

Redis 双层缓存模块，位于 `app/core/cache.py`:

```python
from app.core.cache import get_cache_manager

cache = await get_cache_manager()

# 缓存 Embedding
await cache.set_embedding("文本内容", [0.1, 0.2, ...])
embedding = await cache.get_embedding("文本内容")

# 缓存搜索结果
await cache.set_search_results("查询", results, ttl=300)
results = await cache.get_search_results("查询")

# 清除缓存
await cache.clear_embeddings()
await cache.clear_search_results()
```

#### 缓存策略

| 缓存类型 | 键格式 | TTL | 说明 |
|---------|--------|-----|------|
| **Embedding** | `emb:{hash}` | 24h | 文本 SHA256 哈希作为键 |
| **搜索结果** | `search:{hash}` | 5m | 查询文本哈希 |
| **LLM 响应** | `llm:{hash}` | 1h | 请求哈希 |

#### 降级策略

```
Redis 可用 → 使用 Redis 缓存
     ↓ 失败
内存缓存 → 使用 LRU 内存缓存 (容量限制)
     ↓ 容量满
无缓存 → 直接调用 API
```

---

## API 文档

### 对话接口

```http
POST /api/chat/message
Content-Type: application/json

{
  "message": "如何申请退款？",
  "conversation_id": "optional-session-id",
  "use_rag": true,
  "stream": false
}
```

**响应**:
```json
{
  "response": "根据知识库信息，退款流程如下...",
  "sources": [
    {"content": "...", "source": "退款政策.pdf", "score": 0.85}
  ],
  "conversation_id": "abc123"
}
```

### Agent 对话

```http
POST /api/agent/chat
Content-Type: application/json

{
  "message": "帮我搜索一下公司的退款政策",
  "conversation_id": "optional",
  "stream": false
}
```

**响应**:
```json
{
  "response": "根据搜索结果...",
  "thoughts": [...],
  "tool_calls": [
    {"tool": "knowledge_search", "args": {"query": "退款政策"}, "result": "..."}
  ]
}
```

### 知识库搜索

```http
POST /api/knowledge/search
Content-Type: application/json

{
  "query": "退款政策",
  "top_k": 5
}
```

### 文件上传

```http
POST /api/knowledge/upload
Content-Type: multipart/form-data

file: <文件>
```

**支持格式**: PDF, DOCX, TXT, MD, HTML

### 模型管理

```http
# 获取所有可用模型
GET /api/config/models/all

# Ollama 健康检查
GET /api/config/ollama/health

# 获取 Ollama 模型列表
GET /api/config/ollama/models

# 切换模型
POST /api/config/switch
{
  "provider": "ollama",
  "model": "qwen3:latest"
}
```

### 系统状态

```http
# 健康检查
GET /api/system/health

# 系统信息
GET /api/system/info
```

---

## 调试说明

### 日志配置

```python
# config/settings.py
LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

日志位置: 控制台输出 + `logs/app.log`

### 常用调试命令

```bash
# 查看后端日志
tail -f logs/app.log

# 查看 Docker 容器日志
docker logs -f milvus

# 测试 Milvus 连接
python -c "
from pymilvus import connections, utility
connections.connect(host='localhost', port=19530)
print('Collections:', utility.list_collections())
"

# 测试 Ollama 连接
curl http://localhost:11434/api/tags

# 测试嵌入模型
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "测试"}'
```

### 常见问题排查

#### 1. Milvus 连接失败

```bash
# 检查 Milvus 状态
docker ps | grep milvus
docker logs milvus

# 重启 Milvus
docker restart milvus

# 验证连接
curl http://localhost:9091/healthz
```

#### 2. 向量字段不匹配

**错误**: `fieldName(embedding) not found`

**解决**: 确保 MilvusVectorStore 配置正确的字段名:

```python
MilvusVectorStore(
    embedding_field="vector",  # 匹配实际 schema
    text_key="text",
)
```

#### 3. Reranker 加载失败

**警告**: `sentence-transformers not available`

**说明**: CrossEncoder 依赖是可选的，系统会自动降级为基于分数的简单排序。

**安装 (可选)**:
```bash
pip install sentence-transformers
```

#### 4. LLM 响应超时

**配置调整**:
```bash
LLM_TIMEOUT=120  # 增加超时时间
```

**禁用耗时功能**:
```bash
RAG_ENABLE_HYDE=false  # 禁用 HyDE 加速响应
RAG_ENABLE_QUERY_EXPANSION=false
```

#### 5. 内存不足

**优化配置**:
```bash
RAG_RETRIEVAL_TOP_K=5   # 减少检索数量
RAG_FINAL_TOP_K=3
```

### Langfuse 追踪

1. 登录 [Langfuse](https://cloud.langfuse.com)
2. 创建项目获取 API Keys
3. 配置环境变量:
   ```bash
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-xxx
   LANGFUSE_SECRET_KEY=sk-lf-xxx
   ```
4. 查看追踪: Langfuse Dashboard → Traces

### 性能分析

```python
# 启用详细计时日志
import logging
logging.getLogger("app.rag").setLevel(logging.DEBUG)

# 查看各阶段耗时
# [EnhancedRAG] Step 1: Query Transform - 1.2s
# [EnhancedRAG] Step 2: Hybrid Retrieval - 0.5s
# [EnhancedRAG] Step 3: Reranking - 0.3s
# [EnhancedRAG] Step 4: Post-processing - 0.1s
# [EnhancedRAG] Step 5: Response Generation - 2.5s
```

---

## 测试

### 运行测试

```bash
cd backend

# 运行所有测试
python -m pytest tests/ -v

# 运行 Embedding 优化测试
python tests/test_embedding_optimization.py

# 运行特定测试模块
python -m pytest tests/test_embedding_optimization.py -v
```

### Embedding 优化测试

```bash
# 运行 Embedding 优化测试套件
python tests/test_embedding_optimization.py

# 测试输出示例:
# ============================================================
# EMBEDDING OPTIMIZATION TEST SUITE
# ============================================================
#
# TEST 1: Basic Single Text Embedding - PASS
# TEST 2: Batch Async Embedding - PASS (65.8 docs/sec)
# TEST 3: ThreadPoolExecutor Parallel Embedding - PASS (75.2 docs/sec)
# TEST 4: Retry Mechanism - PASS
# TEST 5: Redis Distributed Lock - PASS
# TEST 6: Cache Integration - PASS (19.9x speedup)
# TEST 7: Concurrent Embedding Requests - PASS (86.3 docs/sec)
# TEST 8: Error Handling - PASS
#
# TEST SUMMARY: 8/8 passed
```

### 测试模块

| 测试文件 | 描述 |
|---------|------|
| `test_embedding_optimization.py` | Embedding 性能优化测试 (批量、并发、缓存、重试) |

---

## 开发指南

### 项目结构

项目采用 **领域驱动设计 (DDD)** 架构，清晰分离关注点：

```
intelligent-customer-service/
├── backend/
│   ├── app/
│   │   ├── domain/           # 领域层 (Domain Layer)
│   │   │   ├── base/         # 基础抽象
│   │   │   │   ├── interfaces.py    # 端口接口 (ILLMService, IRAGService, etc.)
│   │   │   │   ├── entities.py      # 领域实体 (Message, Document, ToolCall, etc.)
│   │   │   │   └── lifecycle.py     # 服务生命周期管理
│   │   │   ├── chat/         # 对话领域 (扩展预留)
│   │   │   ├── knowledge/    # 知识库领域 (扩展预留)
│   │   │   └── agent/        # Agent 领域 (扩展预留)
│   │   │
│   │   ├── infrastructure/   # 基础设施层 (Infrastructure Layer)
│   │   │   └── factory/      # 服务工厂
│   │   │       ├── service_factory.py   # 工厂模式 - 服务创建
│   │   │       └── service_registry.py  # 服务注册表 - DI 容器
│   │   │
│   │   ├── api/              # API 层 (Application Layer)
│   │   │   ├── schemas/      # 请求/响应模型
│   │   │   │   ├── chat.py
│   │   │   │   ├── knowledge.py
│   │   │   │   ├── agent.py
│   │   │   │   ├── system.py
│   │   │   │   └── config.py
│   │   │   ├── endpoints/    # 领域端点处理器
│   │   │   │   ├── chat.py       # 对话 & RAG 端点
│   │   │   │   ├── knowledge.py  # 知识库管理端点
│   │   │   │   ├── agent.py      # Agent 端点
│   │   │   │   ├── mcp.py        # MCP 工具端点
│   │   │   │   ├── system.py     # 系统信息端点
│   │   │   │   └── config.py     # 模型配置端点
│   │   │   └── routes.py     # 路由汇总 (向后兼容)
│   │   │
│   │   ├── agent/            # LangGraph Agent
│   │   │   ├── langgraph_agent.py
│   │   │   ├── react_agent.py
│   │   │   ├── memory.py
│   │   │   ├── error_recovery.py
│   │   │   └── prompts.py
│   │   │
│   │   ├── core/             # 核心组件
│   │   │   ├── embeddings.py      # 高性能 Embedding (批量/并发/缓存)
│   │   │   ├── cache.py           # Redis 缓存管理器
│   │   │   ├── llm_manager.py     # LLM 统一管理
│   │   │   ├── vector_store.py    # Milvus 向量存储
│   │   │   ├── document_processor.py  # 文档处理
│   │   │   └── langfuse_service.py    # 可观测性服务
│   │   │
│   │   ├── mcp/              # MCP 工具
│   │   │   ├── registry.py
│   │   │   └── tools/
│   │   │       ├── base.py
│   │   │       ├── knowledge.py
│   │   │       ├── web_search.py
│   │   │       └── code_executor.py
│   │   │
│   │   ├── rag/              # RAG 模块 (LlamaIndex)
│   │   │   ├── query_engine.py
│   │   │   ├── query_transform.py
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── reranker.py
│   │   │   ├── postprocessor.py
│   │   │   └── index_manager.py
│   │   │
│   │   ├── services/         # 业务服务
│   │   │   ├── rag_service.py
│   │   │   └── upload_service.py
│   │   │
│   │   └── main.py           # 应用入口
│   │
│   ├── config/
│   │   └── settings.py       # 配置管理
│   │
│   ├── tests/                # 测试目录
│   │   ├── test_ddd_architecture.py     # DDD 架构测试
│   │   ├── test_api_endpoints.py        # API 端点测试
│   │   ├── test_embedding_optimization.py  # Embedding 优化测试
│   │   ├── test_rag_flow.py             # RAG 流程测试
│   │   └── test_agent_flow.py           # Agent 流程测试
│   │
│   ├── data/                 # 数据目录
│   │   └── knowledge_base/
│   │
│   ├── requirements.txt
│   ├── run.py
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── components/       # React 组件
│   │   ├── pages/            # 页面
│   │   ├── services/         # API 服务
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

### DDD 架构说明

```
┌────────────────────────────────────────────────────────────────────┐
│                         API Layer (api/)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  chat    │  │knowledge │  │  agent   │  │  config  │           │
│  │ endpoints│  │ endpoints│  │ endpoints│  │ endpoints│           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │             │             │             │                   │
│       └─────────────┴──────┬──────┴─────────────┘                   │
│                            │                                         │
├────────────────────────────┼────────────────────────────────────────┤
│               Infrastructure Layer (infrastructure/)                 │
│                            │                                         │
│  ┌─────────────────────────┴─────────────────────────┐              │
│  │              ServiceRegistry (DI Container)        │              │
│  │  ┌─────────────┐  ┌─────────────┐                 │              │
│  │  │   Factory   │  │  Lifecycle  │                 │              │
│  │  │   Pattern   │  │  Management │                 │              │
│  │  └─────────────┘  └─────────────┘                 │              │
│  └───────────────────────────────────────────────────┘              │
│                            │                                         │
├────────────────────────────┼────────────────────────────────────────┤
│                  Domain Layer (domain/)                              │
│                            │                                         │
│  ┌─────────────────────────┴─────────────────────────┐              │
│  │                   Interfaces (Ports)               │              │
│  │  ILLMService  IRAGService  IVectorStore  ...      │              │
│  └───────────────────────────────────────────────────┘              │
│                                                                      │
│  ┌───────────────────────────────────────────────────┐              │
│  │                   Entities                         │              │
│  │  Message  Document  ToolCall  AgentThought  ...   │              │
│  └───────────────────────────────────────────────────┘              │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

**设计模式应用**:

| 模式 | 实现位置 | 用途 |
|------|---------|------|
| **Factory** | `ServiceFactory` | 封装服务创建逻辑 |
| **Service Locator** | `ServiceRegistry` | 集中管理服务实例 |
| **Ports & Adapters** | `domain/base/interfaces.py` | 定义服务抽象接口 |
| **Lifecycle** | `ServiceLifecycle` | 统一服务初始化/关闭 |
| **Domain Entities** | `domain/base/entities.py` | 富领域对象 |

### 添加新工具

```python
# app/mcp/tools/my_tool.py
from app.mcp.tools.base import BaseMCPTool, ToolParameter

class MyTool(BaseMCPTool):
    name = "my_tool"
    description = "工具描述"
    parameters = [
        ToolParameter(
            name="param1",
            type="string",
            description="参数描述",
            required=True,
        ),
    ]

    async def execute(self, param1: str) -> dict:
        # 实现逻辑
        return {"result": "..."}

# 注册工具 (app/mcp/registry.py)
from app.mcp.tools.my_tool import MyTool
registry.register(MyTool())
```

### 扩展 RAG 组件

```python
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

class MyPostprocessor(BaseNodePostprocessor):
    # 声明 Pydantic 字段
    threshold: float = 0.5

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        # 自定义后处理逻辑
        return [n for n in nodes if n.score >= self.threshold]
```

### 代码规范

- 类型注解: 所有公共方法需要类型注解
- 日志: 使用 `logger.info/debug/error`
- 异常处理: 关键路径需要 try-catch
- 文档字符串: 公共类和方法需要 docstring

---

## 性能优化建议

| 场景 | 推荐配置 |
|------|---------|
| **快速响应** | 禁用 HyDE, 降低 TOP_K=5 |
| **高准确率** | 启用全部功能, TOP_K=15 |
| **低资源** | 禁用 Reranker, 使用简单去重 |
| **大规模部署** | 多副本 + GPU 推理 |

---

## 常见问题

### Q: Reranker 加载失败？
A: sentence-transformers 可选，系统会自动降级为简单评分排序。

### Q: 检索结果不准确？
A: 尝试调整 `RAG_VECTOR_WEIGHT` 和 `RAG_BM25_WEIGHT` 的比例。

### Q: 响应速度慢？
A: 可以禁用 `RAG_ENABLE_HYDE` 或减少 `RAG_RETRIEVAL_TOP_K`。

### Q: 向量维度不匹配？
A: 确保 embedding 模型与 Milvus collection 的维度一致。

### Q: Agent 工具调用失败？
A: 检查工具是否正确注册，查看 `/api/system/info` 确认可用工具列表。

### Q: Embedding 生成速度慢？
A: 可以尝试以下优化:
```bash
# 增大批量处理大小
EMBEDDING_BATCH_SIZE=50

# 增加并发数
EMBEDDING_MAX_CONCURRENT=20

# 确保 Redis 缓存启用
REDIS_CACHE_ENABLED=true
```

### Q: Redis 连接失败？
A: 系统会自动降级到内存缓存模式，不影响正常使用。如需启用 Redis:
```bash
# 启动 Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 配置 Redis URL
REDIS_URL=redis://localhost:6379/0
```

### Q: 并发请求 Embedding 出现重复计算？
A: 确保 Redis 分布式锁已启用:
```bash
REDIS_CACHE_ENABLED=true  # 启用 Redis，锁机制自动生效
```

### Q: Elasticsearch 连接失败？
A: 检查 ES 服务状态和配置:
```bash
# 检查 ES 容器
docker ps | grep elasticsearch
docker logs ics-elasticsearch

# 验证 ES 连接
curl http://localhost:9200

# 启动 ES
docker start ics-elasticsearch

# 禁用 ES (降级为纯向量检索)
ELASTICSEARCH_ENABLED=false
```

### Q: 混合检索效果不理想？
A: 调整向量和 BM25 的权重比例:
```bash
# 偏向语义相似度 (适合语义理解场景)
HYBRID_MILVUS_WEIGHT=0.8
HYBRID_ES_WEIGHT=0.2

# 偏向关键词匹配 (适合专业术语场景)
HYBRID_MILVUS_WEIGHT=0.5
HYBRID_ES_WEIGHT=0.5
```

### Q: ES 索引创建失败 (IK 分词器)?
A: 系统会自动降级为标准分词器，不影响基本功能。如需 IK 分词器:
```bash
# 使用包含 IK 插件的 ES 镜像
docker run -d --name ics-elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  elasticsearch-ik:8.5.0
```

---

## License

MIT License

---

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

- GitHub Issues: [提交问题](https://github.com/your-repo/issues)
- Email: support@example.com
