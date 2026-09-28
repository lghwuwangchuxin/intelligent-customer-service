"""组装根（composition root）。

所有对象在**这一个地方**装配。各层自己不 new 依赖，因此：
  - 替换实现只改这里
  - 测试可以直接注入假对象，不需要 monkeypatch
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agent.engine import AgentEngine
from ..agent.loader import AgentStore
from ..agent.prompts import PromptLibrary
from ..agent.schema import AgentConfig
from ..core.config import settings
from ..core.telemetry import get_logger
from ..guard.pending import PgPendingStore
from ..memory.conversation import ConversationIndex
from ..memory.longterm import LongTermMemory
from ..model.embedding import OllamaEmbedder
from ..model.gateway import ModelGateway
from ..model.registry import get_registry
from ..rag.index.store import PgVectorStore
from ..rag.ingest.pipeline import IngestPipeline
from ..rag.retrieve.pipeline import RetrievalPipeline
from ..rag.retrieve.sparse import PgLexicalIndex
from ..tools.registry import ToolRegistry

logger = get_logger(__name__)


@dataclass
class Container:
    pool: Any
    llm: ModelGateway
    embedder: OllamaEmbedder
    store: PgVectorStore
    lexical: PgLexicalIndex
    retrieval: RetrievalPipeline
    ingest: IngestPipeline
    agents: AgentStore
    prompts: PromptLibrary
    engine: AgentEngine
    tools: ToolRegistry
    conversations: ConversationIndex
    memories: LongTermMemory
    pending: PgPendingStore
    checkpointer: Any
    durable_memory: bool


async def build_container() -> tuple[Container, Any]:
    """返回 (容器, 需要在关闭时 aclose 的 AsyncExitStack)。"""
    from contextlib import AsyncExitStack

    from ..infra.pg import open_pool

    stack = AsyncExitStack()
    pool = await open_pool()

    llm = ModelGateway()
    embedder = OllamaEmbedder()
    store = PgVectorStore(pool)
    lexical = PgLexicalIndex(pool)

    reranker = None   # 需要时在此注入 CrossEncoderReranker()（要 local-models extra）
    retrieval = RetrievalPipeline(store, embedder, lexical, reranker)
    ingest = IngestPipeline(store, embedder, pool)

    # 会话记忆
    dsn = settings.database_url.get_secret_value()
    durable = True
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.store.postgres.aio import AsyncPostgresStore

        checkpointer = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(dsn))
        await checkpointer.setup()
        lg_store = await stack.enter_async_context(AsyncPostgresStore.from_conn_string(dsn))
        await lg_store.setup()
    except Exception as exc:
        # 降级必须留痕：重启即丢、多副本不共享，线上表现是"用户偶尔失忆"
        logger.warning("PostgreSQL 会话记忆不可用，降级为进程内（重启丢失）: %s", exc)
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        checkpointer, lg_store, durable = InMemorySaver(), InMemoryStore(), False

    agents = AgentStore()
    prompts = PromptLibrary()
    tools = ToolRegistry()

    from .tools_builtin import register_builtin_tools

    register_builtin_tools(tools, retrieval)

    def model_factory(cfg: AgentConfig) -> Any:
        """别名 → 具体聊天模型。这里是**唯一**把别名解析成厂商模型的地方。

        🔴 两套模型串格式在这里交汇，必须显式转换：
           - 注册表用 LiteLLM 约定 ``provider/model``（如 ollama/qwen3）
           - ``init_chat_model`` 不认这个斜杠格式，必须把 provider 拆出来单传，
             否则报 "Unable to infer model provider"。
           保留注册表的单一格式，在这个边界上适配。
        """
        from langchain.chat_models import init_chat_model

        spec = get_registry().resolve(cfg.model)
        provider, _, model_name = spec.provider_model.partition("/")
        if not model_name:      # 没有前缀时让框架自己推断
            provider, model_name = None, spec.provider_model

        kw: dict[str, Any] = {
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if provider:
            kw["model_provider"] = provider
        if settings.llm_api_base:
            kw["base_url"] = settings.llm_api_base
        if key := settings.llm_api_key.get_secret_value():
            kw["api_key"] = key
        return init_chat_model(model_name, **kw)

    def tool_factory(cfg: AgentConfig) -> list[Any]:
        from .tools_builtin import as_langchain_tools

        return as_langchain_tools(tools, cfg.tools)

    engine = AgentEngine(model_factory, tool_factory, prompts, checkpointer, lg_store)

    container = Container(
        pool=pool, llm=llm, embedder=embedder, store=store, lexical=lexical,
        retrieval=retrieval, ingest=ingest, agents=agents, prompts=prompts,
        engine=engine, tools=tools,
        conversations=ConversationIndex(pool), memories=LongTermMemory(lg_store),
        pending=PgPendingStore(pool), checkpointer=checkpointer, durable_memory=durable,
    )
    logger.info("组装完成（会话记忆持久化=%s）", durable)
    return container, stack
