# 智能客服系统

基于 LlamaIndex + LangGraph 的企业级智能客服系统，提供高质量的 RAG 检索和智能对话能力。

## 目录

- [功能概述](#功能概述)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [系统架构](#系统架构)
- [API 接口](#api-接口)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

---

## 功能概述

### 核心能力

| 模块 | 功能 | 技术实现 |
|------|------|---------|
| **智能对话** | 多轮对话、上下文理解、流式输出 | LangGraph + ReAct Agent |
| **知识检索** | 向量检索 + BM25 混合检索、Jina 重排序 | LlamaIndex + Milvus + ES |
| **文档处理** | PDF/Word/Excel/Markdown 智能分块 | 语义分块 + 重叠窗口 |
| **多模型** | Ollama/OpenAI/Claude/DeepSeek/通义千问 | 统一 LLM 管理器 |
| **高性能** | Redis 缓存、批量 Embedding、连接池 | 65+ docs/sec 吞吐量 |
| **可观测** | 全链路追踪、性能监控 | Langfuse 集成 |

### Agent 能力

- **ReAct 推理**: Think → Act → Observe 循环，最多 5 次迭代
- **快速响应**: 问候语/闲聊类问题秒回，跳过工具调用
- **工具调用**: 知识库查询、网页搜索、代码执行
- **对话记忆**: 上下文管理、自动摘要、历史持久化

### RAG 增强

| 功能 | 说明 | 配置项 |
|------|------|--------|
| HyDE | 假设文档嵌入 (默认关闭，较耗时) | `RAG_ENABLE_HYDE` |
| Query Expansion | 查询扩展 (默认关闭) | `RAG_ENABLE_QUERY_EXPANSION` |
| Hybrid Retrieval | 向量 + BM25 混合检索 | `RAG_ENABLE_HYBRID` |
| Jina Reranker | 神经网络重排序 | `RAG_ENABLE_RERANK` |
| Semantic Dedup | 语义去重 (>95% 相似度) | `RAG_ENABLE_DEDUP` |

---

## 快速开始

### 环境要求

| 软件 | 版本 | 必需 |
|------|------|------|
| Python | 3.11+ | ✅ |
| Node.js | 18+ | ✅ |
| Docker | 24+ | 推荐 |
| Milvus | 2.5+ | ✅ |
| Elasticsearch | 8.5+ | 可选 |
| Redis | 7+ | 可选 |
| Ollama | 0.12+ | 推荐 |

### 1. 启动依赖服务

```bash
# Milvus (向量数据库)
docker run -d --name milvus \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:v2.5.0 milvus run standalone

# Elasticsearch (可选，用于混合检索)
docker run -d --name ics-elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.5.0

# Redis (可选，用于缓存)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Ollama (本地 LLM)
ollama serve &
ollama pull qwen3:latest
ollama pull nomic-embed-text
```

### 2. 启动后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # 编辑配置
python run.py              # http://localhost:8000
```

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev                # http://localhost:3000
```

### 4. 验证部署

```bash
# 健康检查
curl http://localhost:8000/api/system/health

# 测试对话
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "use_rag": true}'
```

### 服务管理

```bash
# 停止服务
docker stop milvus ics-elasticsearch redis
pkill ollama
lsof -ti:8000 | xargs kill -9  # 后端
pkill -f "vite"                 # 前端

# 启动服务
docker start milvus ics-elasticsearch redis
ollama serve &
```

---

## 配置说明

主要配置文件: `backend/.env`

### 核心配置

```bash
# ============ LLM 配置 ============
LLM_PROVIDER=ollama              # ollama, openai, claude, deepseek, aliyun
LLM_MODEL=qwen3:latest
LLM_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048

# ============ Embedding ============
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BATCH_SIZE=50          # 批量处理数量
EMBEDDING_MAX_CONCURRENT=20      # 最大并发

# ============ 向量数据库 ============
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=customer_service_kb
```

### RAG 配置

```bash
# 查询转换 (较耗时，按需开启)
RAG_ENABLE_HYDE=false
RAG_ENABLE_QUERY_EXPANSION=false

# 混合检索
RAG_ENABLE_HYBRID=true
RAG_VECTOR_WEIGHT=0.7            # 向量权重
RAG_BM25_WEIGHT=0.3              # BM25 权重
RAG_RETRIEVAL_TOP_K=10

# 重排序
RAG_ENABLE_RERANK=true
RAG_RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
RAG_RERANK_TOP_N=5
RAG_RERANK_SCORE_THRESHOLD=0.3

# 后处理
RAG_ENABLE_DEDUP=true
RAG_DEDUP_THRESHOLD=0.95
RAG_FINAL_TOP_K=5
```

### Agent 配置

```bash
AGENT_ENABLED=true
AGENT_MAX_ITERATIONS=5           # 最大推理次数
AGENT_MEMORY_MAX_MESSAGES=20     # 记忆消息数
AGENT_MEMORY_PERSIST_PATH=./data/conversations  # 对话持久化路径
AGENT_ENABLE_PLANNING=false      # 任务规划 (较耗时)
AGENT_MAX_TOOL_CONCURRENCY=5     # 工具并发数
```

### 可选服务

```bash
# Elasticsearch 混合存储
ELASTICSEARCH_ENABLED=true
ELASTICSEARCH_HOSTS=http://localhost:9200
HYBRID_STORAGE_ENABLED=true
HYBRID_MILVUS_WEIGHT=0.7
HYBRID_ES_WEIGHT=0.3

# Redis 缓存
REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_ENABLED=true
REDIS_EMBEDDING_CACHE_TTL=86400  # 24h

# Langfuse 可观测性
LANGFUSE_ENABLED=true
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│     对话界面  │  知识库管理  │  模型配置  │  会话管理         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  LangGraph Agent                        │ │
│  │   Planning → Tool Selection → Execute → Response       │ │
│  │                      │                                  │ │
│  │        ┌─────────────┼─────────────┐                   │ │
│  │        ▼             ▼             ▼                   │ │
│  │  knowledge_search  web_search  code_executor           │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Enhanced RAG (LlamaIndex)                  │ │
│  │                                                         │ │
│  │  Query Transform → Hybrid Retrieval → Rerank → Dedup   │ │
│  │   (HyDE/Expand)    (Vector+BM25)     (Jina)    (MMR)   │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
└──────────────────────────────┼───────────────────────────────┘
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │    Milvus    │   │Elasticsearch │   │    Redis     │
    │ Vector Store │   │  BM25 + Meta │   │    Cache     │
    └──────────────┘   └──────────────┘   └──────────────┘
```

### RAG 检索流程

```
用户问题 → Query Transform → Hybrid Retrieval → Reranker → Post-process → LLM
              (可选)          (Vec+BM25+RRF)    (Jina)     (Dedup+MMR)
```

**RRF 融合算法**: `score(d) = Σ weight_i / (60 + rank_i(d))`

### 混合存储 (ES + Milvus)

| 组件 | 职责 | 存储内容 |
|------|------|---------|
| Elasticsearch | 文档管理 + BM25 检索 | 原始文本、元数据、关键词索引 |
| Milvus | 向量语义检索 | 文本嵌入向量、chunk_id |

当 ES 不可用时，系统自动降级为纯 Milvus 向量检索。

---

## API 接口

### 对话

```http
POST /api/chat/message
{
  "message": "如何申请退款？",
  "conversation_id": "optional",
  "use_rag": true,
  "stream": false
}

# 响应
{
  "response": "根据知识库信息...",
  "sources": [{"content": "...", "source": "退款政策.pdf", "score": 0.85}],
  "conversation_id": "abc123"
}
```

### Agent 对话

```http
POST /api/agent/chat
{
  "message": "帮我搜索退款政策",
  "conversation_id": "optional"
}

# 响应
{
  "response": "根据搜索结果...",
  "thoughts": [...],
  "tool_calls": [{"tool": "knowledge_search", "args": {...}, "result": "..."}]
}
```

### 知识库

```http
# 搜索
POST /api/knowledge/search
{"query": "退款政策", "top_k": 5}

# 上传文档 (支持 PDF/DOCX/TXT/MD)
POST /api/knowledge/upload
Content-Type: multipart/form-data
file: <文件>

# 高级搜索 (混合检索)
POST /api/knowledge/search-advanced
{"query": "退款政策", "filters": {...}}
```

### 系统

```http
GET /api/system/health          # 健康检查
GET /api/system/info            # 系统信息
GET /api/config/ollama/models   # Ollama 模型列表
POST /api/config/switch         # 切换模型
{"provider": "ollama", "model": "qwen3:latest"}
```

---

## 开发指南

### 项目结构

```
backend/
├── app/
│   ├── domain/              # 领域层 (DDD)
│   │   └── base/
│   │       ├── interfaces.py    # 服务接口定义
│   │       ├── entities.py      # 领域实体
│   │       └── lifecycle.py     # 生命周期管理
│   │
│   ├── infrastructure/      # 基础设施层
│   │   └── factory/
│   │       ├── service_factory.py   # 服务工厂
│   │       └── service_registry.py  # DI 容器
│   │
│   ├── api/                 # API 层
│   │   ├── schemas/         # 请求/响应模型
│   │   ├── endpoints/       # 端点处理器
│   │   └── routes.py
│   │
│   ├── agent/               # LangGraph Agent
│   │   ├── react_agent.py       # ReAct 实现
│   │   ├── memory.py            # 对话记忆
│   │   └── prompts.py           # 提示词模板
│   │
│   ├── rag/                 # RAG 模块 (LlamaIndex)
│   │   ├── query_engine.py      # 查询引擎
│   │   ├── hybrid_retriever.py  # 混合检索
│   │   ├── reranker.py          # 重排序
│   │   └── postprocessor.py     # 后处理
│   │
│   ├── core/                # 核心组件
│   │   ├── embeddings.py        # 高性能 Embedding
│   │   ├── cache.py             # Redis 缓存
│   │   ├── llm_manager.py       # LLM 管理
│   │   └── vector_store.py      # Milvus 存储
│   │
│   ├── mcp/                 # MCP 工具
│   │   └── tools/
│   │
│   └── main.py
│
├── config/settings.py
├── tests/
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── package.json
```

### 添加新工具

```python
# app/mcp/tools/my_tool.py
from app.mcp.tools.base import BaseMCPTool, ToolParameter

class MyTool(BaseMCPTool):
    name = "my_tool"
    description = "工具描述"
    parameters = [
        ToolParameter(name="query", type="string", required=True),
    ]

    async def execute(self, query: str) -> dict:
        return {"result": "..."}

# 注册: app/mcp/registry.py
from app.mcp.tools.my_tool import MyTool
registry.register(MyTool())
```

### 运行测试

```bash
cd backend
python -m pytest tests/ -v
python tests/test_embedding_optimization.py  # Embedding 性能测试
```

---

## 常见问题

### 连接问题

**Milvus 连接失败**
```bash
docker ps | grep milvus
docker restart milvus
curl http://localhost:9091/healthz
```

**Elasticsearch 连接失败**
```bash
docker start ics-elasticsearch
curl http://localhost:9200
# 或禁用 ES: ELASTICSEARCH_ENABLED=false
```

### 性能问题

**响应速度慢**
```bash
# 禁用耗时功能
RAG_ENABLE_HYDE=false
RAG_ENABLE_QUERY_EXPANSION=false
AGENT_ENABLE_PLANNING=false

# 减少检索数量
RAG_RETRIEVAL_TOP_K=5
RAG_FINAL_TOP_K=3
```

**Embedding 生成慢**
```bash
EMBEDDING_BATCH_SIZE=50
EMBEDDING_MAX_CONCURRENT=20
REDIS_CACHE_ENABLED=true  # 启用缓存
```

### 检索问题

**检索结果不准确**
```bash
# 调整权重 (偏向语义)
HYBRID_MILVUS_WEIGHT=0.8
HYBRID_ES_WEIGHT=0.2

# 或偏向关键词
HYBRID_MILVUS_WEIGHT=0.5
HYBRID_ES_WEIGHT=0.5
```

**向量维度不匹配**
- 确保 embedding 模型与 Milvus collection 维度一致
- nomic-embed-text 维度: 768

### 其他

**Reranker 加载失败**: sentence-transformers 可选，系统会自动降级

**Agent 工具调用失败**: 查看 `/api/system/info` 确认可用工具列表

**Redis 连接失败**: 系统自动降级到内存缓存

---

## Docker Compose 部署

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - LLM_BASE_URL=http://ollama:11434
      - MILVUS_HOST=milvus
      - REDIS_URL=redis://redis:6379/0
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on: [milvus, elasticsearch, ollama, redis]

  frontend:
    build: ./frontend
    ports: ["3000:80"]

  milvus:
    image: milvusdb/milvus:v2.5.0
    ports: ["19530:19530"]
    command: milvus run standalone

  elasticsearch:
    image: elasticsearch:8.5.0
    ports: ["9200:9200"]
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
```

```bash
docker-compose up -d
```

---

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
