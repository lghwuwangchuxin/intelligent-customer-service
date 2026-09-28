# 智能客服平台

企业级 RAG + 智能体客服系统。Python 3.12 / LangChain 1.x / LangGraph 1.x / pgvector / MCP 2026-07-28。

---

## 核心命题

**不是「选对框架」，而是：假设依赖的一切都会在 2 年内死亡或不兼容，
设计一个换掉它们时爆炸半径可预测的结构。**

依据不是假想——过去 12 个月已实际发生：MCP 协议重构（握手整个移除）、
LangChain 移除 Agent 体系、LlamaIndex 类墓碑化、AutoGen 分裂停更、
One API / NebulaGraph / MetaGPT 停更。

变化速率分层，是全部设计的出发点：

| 层 | 半衰期 | 对策 |
|---|---|---|
| 实现层（SDK、框架 API） | 3~6 月 | 关进单一边界 |
| 协议层（MCP、OpenAI 兼容、OTel） | 1~2 年 | 可以深绑 |
| 原理与数据模型 | 5 年+ | 沉淀在 `core/` |

---

## 六层单向依赖

```
L5  api · worker · mcp_servers      只做协议转换
L4  agent                           编排与运行态（engine · patterns · schema）
L3  rag │ tools │ memory │ guard    ← 四者平级，互不 import
L2  model                           模型访问唯一收敛点（registry · gateway）
L1  infra                           pg · redis · queue
L0  core                            types · context · errors · config · telemetry · spi
```

🔴 **L3 平级独立是最易被破坏也最值钱的约束**——`rag` 一旦 import `tools`，
检索就再也无法单独评测。

```
src/ics/
├── core/          L0  全局词汇表。不依赖任何层，可被所有层引用
│   ├── types.py       Content/Part 多模态模型 · Chunk · Usage · LLMResponse
│   ├── context.py     RequestContext（租户 + 预算，🔴 显式传递，禁用 ContextVar）
│   ├── errors.py      retryable 是一等公民；响应体只出错误码与本地化文案
│   ├── config.py      frozen Settings，启动即校验
│   ├── telemetry.py   OTel GenAI 语义约定 + Prometheus
│   └── spi.py         Protocol 而非 ABC，框架类型不渗透进核心模型
├── infra/         L1  pg（池级注册向量类型）· cache（幂等/缓存故障态度相反）· queue（有界）
├── model/         L2  registry（别名表）· gateway · embedding · rerank
├── rag/           L3-a ingest/ index/ retrieve/ generate/
├── tools/         L3-b registry · mcp_client（🔴 全仓库唯一可 import mcp SDK 的模块）
├── memory/        L3-c conversation（会话索引）· longterm（跨会话）
├── guard/         L3-d budget · authz（写操作→待确认单）· pii（中文场景规则）
├── agent/         L4  schema · loader · engine · patterns/（🔴 唯一可 import langchain）
├── api/           L5  wiring（组装根）· routes/
├── worker/        L5  recall_probe
└── mcp_servers/   L5  对外暴露 MCP

prompts/           提示词是文件，不进代码
agents/            Agent 是数据，加 Agent = 加一个 yaml
evals/             与 src/ 平级 —— 它是产品资产，不是测试代码
```

---

## 六条设计原则

| | 原则 | 判据 |
|---|---|---|
| P1 | **Agent 是数据不是代码** | 代码里不存在任何具体 Agent；新增 = 插一行配置 |
| P2 | **上下文显式传递，fail-closed** | 🔴 禁用 ContextVar（异步/线程池边界静默丢失 → 跨租户泄漏） |
| P3 | **易变实现关在单一边界** | MCP 只在 `tools/mcp_client.py`；模型只在 `model/`；编排框架只在 `agent/` |
| P4 | **能力层平级独立** | 保住可单独评测 |
| P5 | **写操作只产出待确认操作单** | 无绕过路径 |
| P6 | **先收敛调用点，后抽象接口** | 🔴 一个 SPI 只有一个实现 = 净损失 |

---

## 扩展点矩阵

「可扩展」必须可度量。下面每一行的答案都必须是**改配置或起进程，不发版**：

| 要加什么 | 怎么做 | 需要发版？ |
|---|---|---|
| 加一个模型 | 改 `ICS_MODEL_*` 环境变量 | 否 |
| 加一个 Agent | 加 `agents/xxx.yaml` | 否 |
| 改提示词 | 加 `prompts/xxx.v2.md`，改 `prompt_ref` | 否 |
| 加一个知识库 | 调用摄取接口传新 `kb_id` | 否 |
| 加一个租户 | 首次写入自动建分区 | 否 |
| 加一个工具 | 在 `api/tools_builtin.py` 注册 | 是（仅此一项） |

## 变更面与爆炸半径

| 换掉什么 | 影响 | 成本 |
|---|---|---|
| MCP 协议版本 | `tools/mcp_client.py` | 1 个文件 |
| 编排框架 | `agent/engine.py` + `patterns/` | 2~4 周 |
| 模型厂商 | `model/registry.py` + 环境变量 | 1 天 |
| 向量库 | `rag/index/store.py` | 1 周 |
| 可观测后端 | 改 OTLP endpoint | ≈ 0 |
| **嵌入模型** | **全量重建索引** | **最贵** |

这张表可以 grep 验证真伪，不是口头承诺：

```bash
uv run lint-imports                                      # 四条契约
grep -rn '^from mcp\|^import mcp' src/ | grep -v mcp_client   # 应只剩 mcp_servers
```

---

## 三层稳定性

🔴 **LLM 应用的失效大多不是崩溃，而是静默劣化。** 所以除传统可用性外必须加两层：

| 层 | 手段 | 最易漏的告警（都不报错） |
|---|---|---|
| 可用性稳定 | `/health` 不查依赖、`/ready` 查、熔断、重试 | — |
| **质量稳定** | evals 进 CI、发布门槛、召回率基线定时比对 | **召回率基线下降** |
| **成本稳定** | 预算熔断、缓存命中率监控 | **降级触发次数**、**缓存命中率突降** |

---

## 快速开始

```bash
cp .env.example .env
docker compose up -d
docker compose exec ollama ollama pull bge-m3    # 首次必须，约 1.2GB

curl localhost:8000/health
curl localhost:8000/ready
```

本地开发：

```bash
uv sync --extra dev
docker compose up -d postgres redis ollama
uv run ics-api
```

冒烟（🔴 所有业务请求都必须带 `X-Tenant-Id`，否则 400）：

```bash
curl -X POST localhost:8000/knowledge/documents \
  -H 'Content-Type: application/json' -H 'X-Tenant-Id: demo' \
  -d '{"doc_id":"faq-1","text":"充电订单的退款一般在 3 个工作日内原路返回。"}'

curl -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' -H 'X-Tenant-Id: demo' \
  -d '{"message":"退款多久到账？"}'
```

---

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat` | 对话。不指定 `agent_id` 时自动分派 |
| POST | `/chat/stream` | SSE 流式 |
| GET | `/chat/agents` | Agent 列表 |
| GET/DELETE | `/chat/conversations[/{id}]` | 会话列表 / 详情 / 删除 |
| POST | `/knowledge/retrieve` | 混合检索 |
| POST | `/knowledge/documents[/upload]` | 摄取（按内容指纹增量去重） |
| GET | `/knowledge/stats` | 文档量与最近召回率 |
| GET/POST | `/pending[/{id}]` | 待确认写操作 |
| GET | `/memories` | 长期记忆 |
| GET | `/system/info` · `/metrics` · `/health` · `/ready` | 运维 |

---

## 前端

React 19 + Vite 8 + Tailwind 4 + TypeScript 5.9，不用 axios
（SSE over POST 必须靠 fetch + ReadableStream，axios 在浏览器里做不了流式）。

```
frontend/src/
├── lib/      http（租户头注入 · 错误归一 · SSE 分片重组）· errors · tenant
│             reasoning（拆 <think> 思维链）· nodes（图节点名 → 可读标签）
├── api/      types（后端响应形状镜像）· chat · knowledge · ops
├── hooks/    useChat（流式 + 可中断）· useIdentity · useAsync
└── components/  chat/ · knowledge/ · ops/ · ui/
```

三个必须知道的约定：

| | 说明 |
|---|---|
| **租户** | 🔴 每个请求都带 `X-Tenant-Id`，后端 fail-closed。没设租户时前端先渲染设置页，不发注定 400 的请求 |
| **路径** | 前端统一打 `/api/*`，由 vite 代理与 nginx 剥掉前缀。后端接口在根路径，带前缀直连是 404 |
| **SSE** | nginx 必须 `proxy_buffering off` + `chunked_transfer_encoding off`，否则 token 被攒着一次性吐出 |

```bash
cd frontend
npm install
npm run dev        # 3000，代理到 localhost:8000
npm test           # 14 个用例
npm run build
```

---

## 开发

```bash
uv run ruff check src tests evals
uv run lint-imports              # 四条架构契约
uv run pytest -q                 # 108 个用例
uv run python evals/run_eval.py --tenant demo --dataset evals/datasets/golden_v1.jsonl
```

### 测试守的是「静默失败」

不报错、只是悄悄变坏的地方才值得写测试：

| 文件 | 守的失败模式 |
|---|---|
| `test_core.py` | fail-open 租户、推理/缓存 token 混算、错误码漏翻译、跨线程租户串味 |
| `test_model.py` | 别名静默重定向、推理模型返回空串、弃用端点、静默截断、维度漂移 |
| `test_rag.py` | 加权求和让量纲支配排序、中文不分词 BM25 全 0、间接提示注入、二进制当文本入库 |
| `test_guard_agent.py` | 无步数上限、写操作绕过闸门、中文 PII 漏检、Agent 硬编码进代码 |
| `test_architecture.py` | 分层违规、死配置、死依赖、提示词内联、ContextVar 复活 |

前端测试守的是流式边界：租户头注入、错误契约还原、取消传播、
**SSE 事件被网络分片切断后能否拼回**、**思维链标签未闭合时答案不能漏出**。

---

## 运维要点

### 召回率巡检（必做）

🔴 HNSW 索引退化**完全无声**：不报错、不变慢，只是开始漏结果。

```bash
uv run ics-recall-probe --tenant demo     # 建议挂每日定时任务，非 0 退出即告警
```

处理顺序：① 调大 `ICS_RETRIEVAL__HNSW_EF_SEARCH` ② 确认迭代扫描已开
③ `REINDEX INDEX CONCURRENTLY` **再** `VACUUM`（HNSW 的 VACUUM 很慢，顺序别反）。

### 换嵌入模型

🔴 **本领域最危险的变更**。维度与语义空间都会变：

1. 改 `ICS_MODEL_EMBED_MAIN` 与 `model/registry.py` 的 `dimensions` / `max_input_tokens`
2. 改 `deploy/postgres/init.sql` 的 `vector(N)`
3. **重建全部向量** —— 只改配置不重建，检索断崖下跌且不报错

### 批量导入

顺序不能反：`COPY ... WITH (FORMAT BINARY)` → 调大 `maintenance_work_mem` → `CREATE INDEX CONCURRENTLY`。

---

## 演进阶段

当前停在**阶段一（单体 + 独立 MCP Server）**，这对 90% 的项目是正确的。
后续每一步都应由具体信号触发，而不是提前拆：

| 阶段 | 触发信号 |
|---|---|
| 拆 worker | 摄取任务阻塞在线请求 |
| 拆推理服务 | 🔴 进程内加载模型 = 副本数 × 模型内存 |
| 拆 MCP | 工具需要独立发布节奏 |
| 定义态平台化 | 多团队并行维护 Agent |
