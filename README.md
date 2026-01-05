# 智能客服系统

基于微服务架构的企业级智能客服平台，集成 RAG 检索、多智能体协作、LangGraph 推理引擎。

## 目录

- [系统架构](#系统架构)
  - [整体架构](#整体架构)
  - [微服务架构](#微服务架构)
  - [服务调用关系](#服务调用关系)
- [技术组件](#技术组件)
  - [核心技术栈](#核心技术栈)
  - [中间件服务](#中间件服务)
- [部署文档](#部署文档)
  - [环境要求](#环境要求)
  - [快速启动](#快速启动)
  - [中间件部署](#中间件部署)
  - [微服务部署](#微服务部署)
  - [智能体部署](#智能体部署)
- [业务流程](#业务流程)
  - [单智能体对话流程](#单智能体对话流程)
  - [多智能体协作流程](#多智能体协作流程)
  - [RAG 检索流程](#rag-检索流程)
- [服务详情](#服务详情)
- [API 接口](#api-接口)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

---

## 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Frontend (React + TypeScript)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │  对话界面   │ │ 知识库管理  │ │  记忆面板   │ │ 多智能体面板│ │ 服务监控   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ HTTP/WebSocket
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         API Gateway (Port 8000)                                   │
│  • 统一入口 • 路由转发 • 负载均衡 • 认证鉴权 • 限流熔断                          │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │ HTTP (Service Discovery via Nacos)
         ┌────────────┬────────────┬───┴───┬────────────┬────────────┐
         ▼            ▼            ▼       ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ MCP Service │ │ RAG Service │ │Memory Svc   │ │Single Agent │ │Multi Agent  │
│  Port 8001  │ │  Port 8002  │ │  Port 8010  │ │  Port 8005  │ │  Port 8006  │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │               │               │
       └───────────────┴───────────────┴───────┬───────┴───────────────┘
                                               │
         ┌────────────┬────────────┬───────────┼───────────┬────────────┐
         ▼            ▼            ▼           ▼           ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Evaluation  │ │ Monitoring  │ │ LLM Manager │ │    Nacos    │ │   Agents    │
│  Port 8003  │ │  Port 8004  │ │  Port 8007  │ │  Port 8848  │ │ 9001-9008   │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
                                               │
┌──────────────────────────────────────────────┴───────────────────────────────────┐
│                              Infrastructure Layer                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Redis   │  │  Milvus  │  │   ES     │  │ Postgres │  │Prometheus│            │
│  │  :6379   │  │  :19530  │  │  :9200   │  │  :5432   │  │  :9090   │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 微服务架构

系统采用微服务架构，共包含 **9 个核心服务** 和 **8 个专业智能体**：

| 服务层 | 服务名称 | 端口 | 职责描述 |
|--------|----------|------|----------|
| **网关层** | API Gateway | 8000 | 统一入口、路由转发、认证鉴权 |
| **核心服务** | MCP Service | 8001 | MCP 工具管理与执行 |
| | RAG Service | 8002 | 知识检索与文档处理 |
| | Evaluation Service | 8003 | RAG 质量评估 (RAGAS) |
| | Monitoring Service | 8004 | 链路追踪与性能监控 |
| | Single Agent Service | 8005 | 单智能体对话服务 |
| | Multi Agent Service | 8006 | 多智能体协调服务 |
| | LLM Manager Service | 8007 | LLM 模型管理与配置 |
| | Memory Service | 8010 | 对话记忆与用户偏好 |
| **专业智能体** | Travel Assistant | 9101 | 出行助手 - 充电站搜索 |
| | Charging Manager | 9002 | 充电管家 - 充电状态监控 |
| | Billing Advisor | 9003 | 费用顾问 - 账单查询 |
| | Emergency Support | 9004 | 故障急救 - 问题诊断 |
| | Data Analyst | 9005 | 数据分析师 - 能效报告 |
| | Maintenance Expert | 9006 | 运维专家 - 设备健康 |
| | Energy Advisor | 9007 | 能源顾问 - 成本优化 |
| | Scheduling Advisor | 9008 | 调度顾问 - 负荷预测 |

### 服务调用关系

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                用户请求流向                                      │
└────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   Frontend   │
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │ API Gateway  │◄─────── Nacos (服务发现)
                              └──────┬───────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │Single Agent │          │Memory Service│         │Multi Agent  │
    │   Service   │          │             │          │   Service   │
    └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
           │                        │                        │
           │  ┌─────────────────────┤                        │
           │  │                     │                        │
           ▼  ▼                     ▼                        ▼
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │ RAG Service │◄─────────│ LLM Manager │─────────►│  A2A Agents │
    └──────┬──────┘          └─────────────┘          │  (8 agents) │
           │                                          └──────┬──────┘
           │  ┌───────────────────────────────────────────────┘
           ▼  ▼
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │ MCP Service │          │ Evaluation  │          │ Monitoring  │
    │   (Tools)   │          │   Service   │          │   Service   │
    └─────────────┘          └─────────────┘          └─────────────┘


┌────────────────────────────────────────────────────────────────────────────────┐
│                              服务依赖关系                                        │
└────────────────────────────────────────────────────────────────────────────────┘

  API Gateway
      │
      ├──► Single Agent Service ──┬──► MCP Service
      │                           ├──► RAG Service ──► Milvus + ES
      │                           ├──► Memory Service ──► Redis
      │                           └──► LLM Manager ──► Ollama/OpenAI/DeepSeek
      │
      ├──► Multi Agent Service ───┬──► A2A Agents (9101-9008)
      │                           └──► LLM Manager
      │
      ├──► RAG Service ───────────┬──► Milvus (向量检索)
      │                           ├──► Elasticsearch (BM25 检索)
      │                           └──► LLM Manager (Embedding)
      │
      ├──► Memory Service ────────┬──► Redis (短期记忆)
      │                           └──► PostgreSQL (长期记忆)
      │
      ├──► Evaluation Service ────┬──► RAG Service
      │                           └──► LLM Manager
      │
      └──► Monitoring Service ────┬──► Prometheus (指标)
                                  └──► Langfuse (链路追踪)
```

---

## 技术组件

### 核心技术栈

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **后端框架** | FastAPI | 0.100+ | 异步 API 服务 |
| | Pydantic | 2.0+ | 数据验证 |
| **智能体** | LangGraph | 0.2+ | 多步推理与状态管理 |
| | LangChain | 0.3+ | LLM 抽象与工具链 |
| **RAG 引擎** | LlamaIndex | 0.11+ | 检索增强生成 |
| **向量数据库** | Milvus | 2.3+ | 向量存储与检索 |
| **搜索引擎** | Elasticsearch | 8.5+ | BM25 检索与全文搜索 |
| **缓存** | Redis | 7+ | 会话缓存与 Embedding 缓存 |
| **数据库** | PostgreSQL | 15+ | 持久化存储 |
| **服务发现** | Nacos | 2.3+ | 服务注册与配置中心 |
| **监控** | Prometheus + Grafana | - | 指标采集与可视化 |
| | Langfuse | - | LLM 链路追踪 |
| **前端** | React + TypeScript | 18+ | 用户界面 |
| | Vite | 5+ | 构建工具 |
| | TailwindCSS | 3+ | 样式框架 |

### 中间件服务

| 中间件 | 端口 | 用途 | 数据目录 |
|--------|------|------|----------|
| **Nacos** | 8848, 9848, 9849 | 服务发现与配置中心 | `/infra/nacos` |
| **Redis** | 6379 | 缓存、会话存储 | `/infra/redis` |
| **Milvus** | 19530, 9091 | 向量数据库 | `/infra/milvus` |
| **Elasticsearch** | 9200, 9300 | 全文检索 | `/infra/elasticsearch` |
| **PostgreSQL** | 5432 | 关系型数据库 | `/infra/postgres` |
| **Prometheus** | 9090 | 指标采集 | `/infra/prometheus` |
| **Grafana** | 3000 | 监控可视化 | `/infra/grafana` |

---

## 部署文档

### 环境要求

| 软件 | 版本 | 必需 | 说明 |
|------|------|------|------|
| Python | 3.12+ | 是 | 推荐 3.12，3.13 部分依赖不兼容 |
| Node.js | 18+ | 是 | 前端开发 |
| Docker | 24+ | 是 | 容器化部署 |
| Docker Compose | 2.0+ | 是 | 服务编排 |
| uv | 最新版 | 推荐 | Python 包管理器 |

### 快速启动

```bash
# 1. 克隆项目
git clone <repository-url>
cd intelligent-customer-service

# 2. 启动中间件 (基础设施)
docker-compose -f docker-compose.infra.yml up -d

# 3. 等待中间件就绪 (约 60 秒)
sleep 60

# 4. 启动微服务
docker-compose -f docker-compose.services.yml up -d

# 5. 启动智能体 (可选)
docker-compose -f docker-compose.agents.yml up -d

# 6. 启动前端
cd frontend && npm install && npm run dev
```

### 中间件部署

#### Docker Compose 方式 (推荐)

```bash
# 启动所有中间件
docker-compose -f docker-compose.infra.yml up -d

# 查看状态
docker-compose -f docker-compose.infra.yml ps

# 查看日志
docker-compose -f docker-compose.infra.yml logs -f

# 停止服务
docker-compose -f docker-compose.infra.yml down
```

#### 手动启动方式

```bash
# Nacos (服务发现)
docker run -d --name nacos \
  -p 8848:8848 -p 9848:9848 -p 9849:9849 \
  -e MODE=standalone \
  nacos/nacos-server:v2.3.0

# Redis (缓存)
docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine

# Milvus (向量数据库)
docker run -d --name milvus \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.3.3 milvus run standalone

# Elasticsearch (全文检索)
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.5.0

# PostgreSQL (数据库)
docker run -d --name postgres \
  -p 5432:5432 \
  -e POSTGRES_USER=ics \
  -e POSTGRES_PASSWORD=ics_password \
  -e POSTGRES_DB=ics_db \
  postgres:15-alpine
```

#### 健康检查

```bash
# Nacos
curl http://localhost:8848/nacos/v1/console/health/readiness

# Redis
redis-cli ping

# Milvus
curl http://localhost:9091/healthz

# Elasticsearch
curl http://localhost:9200/_cluster/health

# PostgreSQL
pg_isready -h localhost -p 5432
```

### 微服务部署

#### Docker Compose 方式 (推荐)

```bash
# 确保中间件已启动
docker-compose -f docker-compose.infra.yml ps

# 启动所有微服务
docker-compose -f docker-compose.services.yml up -d

# 查看服务状态
docker-compose -f docker-compose.services.yml ps

# 查看日志
docker-compose -f docker-compose.services.yml logs -f api-gateway
docker-compose -f docker-compose.services.yml logs -f single-agent-service
```

#### 本地开发方式

```bash
# 配置环境变量
export NACOS_SERVER_ADDRESSES=localhost:8848
export LLM_PROVIDER=ollama
export LLM_BASE_URL=http://localhost:11434
export LLM_MODEL=qwen2.5:7b

# 启动各服务 (不同终端)
cd services/api_gateway && python run.py
cd services/mcp_service && python run.py
cd services/rag_service && python run.py
cd services/memory_service && python run.py
cd services/single_agent_service && python run.py
cd services/multi_agent_service && python run.py
cd services/llm_manager_service && python run.py
cd services/evaluation_service && python run.py
cd services/monitoring_service && python run.py

# 或使用启动脚本
./services/run_service.sh all
```

#### 服务配置

每个服务目录下有 `.env.example` 文件，复制并配置：

```bash
# 服务通用配置
SERVICE_NAME=api-gateway
SERVICE_HOST=0.0.0.0
HTTP_PORT=8000

# Nacos 配置
NACOS_SERVER_ADDRESSES=localhost:8848
NACOS_NAMESPACE=public
NACOS_GROUP=DEFAULT_GROUP
NACOS_ENABLED=true

# LLM 配置
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=

# 数据库配置
REDIS_URL=redis://localhost:6379
ELASTICSEARCH_URL=http://localhost:9200
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 智能体部署

#### Docker Compose 方式

```bash
# 启动所有智能体
docker-compose -f docker-compose.agents.yml up -d

# 启动特定智能体
docker-compose -f docker-compose.agents.yml up -d travel-assistant billing-advisor

# 查看智能体状态
docker-compose -f docker-compose.agents.yml ps

# 验证智能体 Agent Card
curl http://localhost:9101/.well-known/agent.json
curl http://localhost:9002/.well-known/agent.json
```

#### 本地启动方式

```bash
cd services/agents_service

# 安装依赖
pip install -r requirements.txt

# 启动所有智能体
./start_all.sh

# 启动单个智能体
python run_agent.py travel_assistant
python run_agent.py billing_advisor --port 9003
```

---

## 业务流程

### 单智能体对话流程

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                            单智能体对话流程                                      │
└────────────────────────────────────────────────────────────────────────────────┘

用户输入
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ API Gateway │────►│Single Agent │────►│Memory Service│
│             │     │   Service   │     │  (上下文)    │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐       ┌─────────┐
    │  意图   │       │  快速   │       │  工具   │
    │  识别   │       │  响应   │       │  调用   │
    └────┬────┘       └────┬────┘       └────┬────┘
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐       直接返回        ┌─────────────┐
    │ 需要RAG │       问候语          │ MCP Service │
    └────┬────┘                       │  执行工具   │
         │                            └──────┬──────┘
         ▼                                   │
    ┌─────────────┐                          │
    │ RAG Service │◄─────────────────────────┘
    │ 知识检索    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐     ┌─────────────┐
    │ LLM Manager │────►│  生成回复   │
    │ 调用模型    │     │  流式输出   │
    └─────────────┘     └─────────────┘
```

**ReAct 推理循环：**

```
┌─────────────────────────────────────────────────────────────────┐
│  Think (思考)  →  Act (行动)  →  Observe (观察)  →  Repeat     │
│                                                                 │
│  最多 5 次迭代，每次迭代包含：                                   │
│  1. 分析用户意图和当前状态                                       │
│  2. 决定是否调用工具                                            │
│  3. 执行工具并观察结果                                          │
│  4. 决定继续推理或生成最终回复                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 多智能体协作流程

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           多智能体协作流程                                       │
└────────────────────────────────────────────────────────────────────────────────┘

                              用户请求
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Multi Agent    │
                        │    Service      │
                        └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
               ┌────────┐  ┌────────┐  ┌────────┐
               │  AUTO  │  │PARALLEL│  │SEQUENTIAL│
               │ 自动   │  │ 并行   │  │ 串行    │
               └───┬────┘  └───┬────┘  └────┬────┘
                   │           │            │
                   ▼           │            │
          ┌─────────────┐      │            │
          │ 意图识别    │      │            │
          │ 关键词匹配  │      │            │
          └──────┬──────┘      │            │
                 │             │            │
                 ▼             ▼            ▼
          ┌─────────────────────────────────────────────────────┐
          │                 A2A 智能体池                         │
          │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐           │
          │  │出行   │ │充电   │ │费用   │ │故障   │ ... (8个) │
          │  │助手   │ │管家   │ │顾问   │ │急救   │           │
          │  └───────┘ └───────┘ └───────┘ └───────┘           │
          └─────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  结果聚合/编排  │
                        └────────┬────────┘
                                 │
                                 ▼
                           最终响应
```

**路由模式：**

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| **AUTO** | 根据关键词自动路由到单个智能体 | 用户问题明确属于某个领域 |
| **PARALLEL** | 同时调用多个智能体 | 问题涉及多个领域，需要综合信息 |
| **SEQUENTIAL** | 按顺序链式调用智能体 | 问题需要多步处理，后续依赖前序结果 |

### RAG 检索流程

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                             RAG 检索流程                                        │
└────────────────────────────────────────────────────────────────────────────────┘

用户查询
    │
    ▼
┌─────────────────┐
│  Query Transform │  ← HyDE (假设文档嵌入) / Query Expansion (可选)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Hybrid Retrieval                            │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │   Milvus (向量检索)      │  │  Elasticsearch (BM25)       │   │
│  │   语义相似度匹配         │  │  关键词匹配                  │   │
│  │   Weight: 0.7            │  │  Weight: 0.3                │   │
│  └───────────┬─────────────┘  └───────────────┬─────────────┘   │
│              │                                │                  │
│              └──────────────┬─────────────────┘                  │
│                             │                                    │
│                             ▼                                    │
│                    ┌─────────────────┐                          │
│                    │  RRF 融合算法   │                          │
│                    │  score = Σ w/(60+rank)                     │
│                    └─────────────────┘                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Jina Reranker  │  ← 神经网络重排序
                    │  Top-N 筛选     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Post-process   │  ← 去重 (>95% 相似度) + MMR
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  LLM 生成回复   │  ← 基于检索结果
                    └─────────────────┘
```

**RAG 配置选项：**

| 功能 | 环境变量 | 默认值 | 说明 |
|------|----------|--------|------|
| HyDE | `RAG_ENABLE_HYDE` | false | 假设文档嵌入 (耗时) |
| Query Expansion | `RAG_ENABLE_QUERY_EXPANSION` | false | 查询扩展 |
| Hybrid Retrieval | `RAG_ENABLE_HYBRID` | true | 混合检索 |
| Reranker | `RAG_ENABLE_RERANK` | true | 神经网络重排序 |
| Semantic Dedup | `RAG_ENABLE_DEDUP` | true | 语义去重 |

---

## 服务详情

### API Gateway (端口 8000)

统一请求入口，负责：
- 请求路由与转发
- 服务发现 (通过 Nacos)
- 负载均衡
- 认证鉴权
- 限流熔断

```bash
# 健康检查
curl http://localhost:8000/health

# 系统信息
curl http://localhost:8000/api/system/info
```

### RAG Service (端口 8002)

知识检索与文档处理：
- 文档上传与分块 (PDF/Word/Excel/Markdown)
- 向量化存储 (Milvus)
- 混合检索 (Vector + BM25)
- 重排序与后处理

```bash
# 知识搜索
curl -X POST http://localhost:8002/search \
  -H "Content-Type: application/json" \
  -d '{"query": "退款政策", "top_k": 5}'

# 文档上传
curl -X POST http://localhost:8002/upload \
  -F "file=@document.pdf"
```

### Memory Service (端口 8010)

对话记忆管理：
- 短期记忆 (Redis) - 会话上下文
- 长期记忆 (PostgreSQL) - 用户偏好、历史摘要
- 记忆检索与更新

```bash
# 创建会话
curl -X POST http://localhost:8010/conversations

# 获取记忆
curl http://localhost:8010/conversations/{id}/messages
```

### Single Agent Service (端口 8005)

单智能体对话服务：
- ReAct 推理引擎
- 工具调用 (MCP)
- 流式响应
- 上下文管理

```bash
# 对话请求
curl -X POST http://localhost:8005/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "conversation_id": "xxx"}'
```

### Multi Agent Service (端口 8006)

多智能体协调服务：
- A2A 协议通信
- 智能路由
- 并行/串行调度
- 结果聚合

```bash
# A2A 对话
curl -X POST http://localhost:8006/a2a/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "附近充电站", "mode": "auto"}'

# 并行调用
curl -X POST http://localhost:8006/a2a/chat/parallel \
  -H "Content-Type: application/json" \
  -d '{"message": "问题", "agents": ["agent1", "agent2"]}'
```

### LLM Manager Service (端口 8007)

LLM 模型管理：
- 多模型支持 (Ollama/OpenAI/Claude/DeepSeek/通义千问)
- 模型切换
- 配置管理
- Embedding 服务

```bash
# 获取可用模型
curl http://localhost:8007/models

# 切换模型
curl -X POST http://localhost:8007/config \
  -H "Content-Type: application/json" \
  -d '{"provider": "ollama", "model": "qwen2.5:7b"}'
```

---

## API 接口

### 对话接口

```http
# 单智能体对话
POST /api/chat/message
{
  "message": "如何申请退款？",
  "conversation_id": "optional",
  "use_rag": true,
  "stream": true
}

# 多智能体对话
POST /api/a2a/chat
{
  "message": "附近有什么充电站？",
  "mode": "auto",
  "conversation_id": "optional"
}

# 并行调用
POST /api/a2a/chat/parallel
{
  "message": "综合问题",
  "agents": ["agent1", "agent2"]
}

# 串行调用
POST /api/a2a/chat/sequential
{
  "message": "需要多步处理的问题",
  "agents": ["agent1", "agent2", "agent3"]
}
```

### 知识库接口

```http
# 搜索
POST /api/knowledge/search
{"query": "退款政策", "top_k": 5}

# 上传文档
POST /api/knowledge/upload
Content-Type: multipart/form-data
file: <文件>

# 高级搜索
POST /api/knowledge/search-advanced
{"query": "退款政策", "filters": {...}}
```

### 记忆接口

```http
# 创建会话
POST /api/memory/conversations

# 获取会话列表
GET /api/memory/conversations

# 获取会话消息
GET /api/memory/conversations/{id}/messages

# 添加消息
POST /api/memory/conversations/{id}/messages
{"role": "user", "content": "消息内容"}

# 获取用户偏好
GET /api/memory/users/{user_id}/preferences

# 更新用户偏好
PUT /api/memory/users/{user_id}/preferences
{"key": "value"}
```

### 系统接口

```http
# 健康检查
GET /api/system/health

# 系统信息
GET /api/system/info

# 服务状态
GET /api/services/status

# 模型配置
GET /api/config/providers
POST /api/config/switch
{"provider": "ollama", "model": "qwen2.5:7b"}
```

---

## 开发指南

### 项目结构

```
intelligent-customer-service/
├── frontend/                     # 前端 (React + TypeScript)
│   ├── src/
│   │   ├── components/           # UI 组件
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── KnowledgePanel.tsx
│   │   │   ├── MemoryPanel.tsx
│   │   │   ├── MultiAgentPanel.tsx
│   │   │   ├── ServiceStatusDashboard.tsx
│   │   │   └── MonitoringDashboard.tsx
│   │   ├── hooks/                # 自定义 Hooks
│   │   │   ├── useAgentChat.ts
│   │   │   └── useMultiAgentChat.ts
│   │   ├── services/             # API 服务
│   │   │   └── api.ts
│   │   └── App.tsx
│   └── package.json
│
├── services/                     # 微服务
│   ├── api_gateway/              # API 网关 (Port 8000)
│   │   ├── app.py
│   │   ├── routes/
│   │   └── run.py
│   │
│   ├── mcp_service/              # MCP 工具服务 (Port 8001)
│   │   ├── server.py
│   │   └── tools/
│   │
│   ├── rag_service/              # RAG 服务 (Port 8002)
│   │   ├── server.py
│   │   ├── pipeline/
│   │   │   ├── hybrid_retriever.py
│   │   │   ├── reranker.py
│   │   │   └── query_engine.py
│   │   └── service.py
│   │
│   ├── evaluation_service/       # 评估服务 (Port 8003)
│   ├── monitoring_service/       # 监控服务 (Port 8004)
│   ├── single_agent_service/     # 单智能体服务 (Port 8005)
│   ├── multi_agent_service/      # 多智能体服务 (Port 8006)
│   ├── llm_manager_service/      # LLM 管理服务 (Port 8007)
│   ├── memory_service/           # 记忆服务 (Port 8010)
│   │   ├── memory.py
│   │   ├── server.py
│   │   └── service.py
│   │
│   ├── agents_service/           # A2A 智能体
│   │   ├── common/               # 共享代码
│   │   ├── travel_assistant/     # 出行助手 (Port 9101)
│   │   ├── charging_manager/     # 充电管家 (Port 9002)
│   │   ├── billing_advisor/      # 费用顾问 (Port 9003)
│   │   ├── emergency_support/    # 故障急救 (Port 9004)
│   │   ├── data_analyst/         # 数据分析师 (Port 9005)
│   │   ├── maintenance_expert/   # 运维专家 (Port 9006)
│   │   ├── energy_advisor/       # 能源顾问 (Port 9007)
│   │   └── scheduling_advisor/   # 调度顾问 (Port 9008)
│   │
│   └── common/                   # 公共模块
│       ├── config.py             # 配置管理
│       ├── nacos_client.py       # Nacos 客户端
│       ├── http_client.py        # HTTP 客户端
│       └── logging.py            # 日志配置
│
├── docker-compose.infra.yml      # 中间件部署
├── docker-compose.services.yml   # 微服务部署
├── docker-compose.agents.yml     # 智能体部署
└── README.md
```

### 添加新服务

1. 创建服务目录：

```bash
mkdir -p services/new_service
cd services/new_service
```

2. 创建服务代码：

```python
# services/new_service/server.py
from fastapi import FastAPI
from services.common.fastapi_nacos import NacosMiddleware

app = FastAPI(title="New Service")

# 添加 Nacos 中间件
app.add_middleware(NacosMiddleware, service_name="new-service")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/endpoint")
async def endpoint():
    return {"result": "..."}
```

3. 配置环境变量：

```bash
# .env
SERVICE_NAME=new-service
HTTP_PORT=8xxx
NACOS_SERVER_ADDRESSES=localhost:8848
```

4. 添加到 docker-compose：

```yaml
new-service:
  build:
    context: .
    dockerfile: services/new_service/Dockerfile
  ports:
    - "8xxx:8xxx"
  environment:
    SERVICE_NAME: new-service
```

### 添加新智能体

1. 创建智能体目录：

```bash
mkdir -p services/agents_service/new_agent
```

2. 实现执行器：

```python
# services/agents_service/new_agent/executor.py
from services.agents_service.common.base_executor import BaseExecutor

class NewAgentExecutor(BaseExecutor):
    def __init__(self):
        super().__init__(
            name="new_agent",
            description="新智能体描述"
        )

    async def execute(self, message: str, context: dict) -> str:
        # 实现业务逻辑
        return "响应结果"
```

3. 注册到 A2A 系统。

---

## 常见问题

### 服务无法启动

```bash
# 检查 Nacos 连接
curl http://localhost:8848/nacos/v1/console/health/readiness

# 检查端口占用
lsof -i :8000

# 查看服务日志
docker-compose -f docker-compose.services.yml logs -f api-gateway
```

### 中间件连接失败

```bash
# 检查中间件状态
docker-compose -f docker-compose.infra.yml ps

# 重启中间件
docker-compose -f docker-compose.infra.yml restart

# 检查网络连接
docker network ls | grep ics
```

### RAG 检索不准确

```bash
# 调整检索权重
RAG_VECTOR_WEIGHT=0.7
RAG_BM25_WEIGHT=0.3

# 启用重排序
RAG_ENABLE_RERANK=true

# 调整返回数量
RAG_RETRIEVAL_TOP_K=10
RAG_FINAL_TOP_K=5
```

### 智能体通信失败

```bash
# 检查智能体状态
curl http://localhost:9101/.well-known/agent.json

# 检查 A2A 健康状态
curl http://localhost:8006/a2a/health

# 查看智能体日志
docker-compose -f docker-compose.agents.yml logs -f travel-assistant
```

### 性能优化

```bash
# 启用 Redis 缓存
REDIS_CACHE_ENABLED=true

# 增加并发数
EMBEDDING_MAX_CONCURRENT=20

# 减少检索数量
RAG_RETRIEVAL_TOP_K=5

# 禁用耗时功能
RAG_ENABLE_HYDE=false
RAG_ENABLE_QUERY_EXPANSION=false
```

---

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
