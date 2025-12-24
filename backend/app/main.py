"""
Main FastAPI application for Intelligent Customer Service.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from app.api.routes import (
    chat_router, knowledge_router, system_router, agent_router, mcp_router, config_router,
    async_init_services,
)
from app.models.database import DatabaseManager
from app.services.knowledge_base_service import initialize_knowledge_base, get_knowledge_base_service
from app.core.embeddings import get_embedding_manager, close_http_client

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Database manager
db_manager = DatabaseManager(settings.DATABASE_URL)


async def init_knowledge_base():
    """Initialize knowledge base with RAG service."""
    try:
        # Import here to avoid circular imports
        from app.api.routes import get_services

        # Get services (this initializes them if needed)
        services = get_services()
        rag_service = services.get('rag')

        if rag_service:
            result = await initialize_knowledge_base(rag_service)
            if result.get('success'):
                logger.info(
                    f"[Startup] Knowledge base initialized - "
                    f"status: {result.get('status')}, "
                    f"files: {result.get('files_indexed', 0)}"
                )
            else:
                logger.warning(
                    f"[Startup] Knowledge base initialization warning: {result.get('error', 'unknown')}"
                )
            return result
        else:
            logger.warning("[Startup] RAG service not available for knowledge base initialization")
            return {'success': False, 'error': 'RAG service not available'}
    except Exception as e:
        logger.error(f"[Startup] Knowledge base initialization error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def init_knowledge_base_background():
    """Initialize knowledge base in background (non-blocking)."""
    import asyncio
    # Small delay to allow server to start first
    await asyncio.sleep(2)
    logger.info("[Background] Starting knowledge base initialization...")

    result = await init_knowledge_base()

    if result.get('success'):
        kb_stats = get_knowledge_base_service().get_stats()
        logger.info(
            f"[Background] Knowledge base ready - "
            f"files: {kb_stats.get('total_files', 0)}, "
            f"size: {kb_stats.get('total_size_mb', 0)} MB"
        )
    else:
        logger.warning(f"[Background] Knowledge base initialization: {result.get('status', 'unknown')}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}, Model: {settings.LLM_MODEL}")
    logger.info(f"Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    logger.info(f"Agent Mode: {'Enabled' if settings.AGENT_ENABLED else 'Disabled'}")
    if settings.AGENT_ENABLED:
        logger.info(f"Agent Max Iterations: {settings.AGENT_MAX_ITERATIONS}")

    # Initialize database
    await db_manager.init_db()
    logger.info("Database initialized")

    # Asynchronously initialize all services (including embedding model warmup)
    # This ensures embedding models are loaded before handling requests
    try:
        logger.info("[Startup] Initializing services asynchronously...")
        await async_init_services()
        logger.info("[Startup] All services initialized successfully")
    except Exception as e:
        logger.warning(f"[Startup] Services initialization warning (non-critical): {e}")

    # Initialize knowledge base (auto-index documents)
    # Note: Disabled by default for faster startup. Use /api/knowledge/reindex to manually trigger.
    if settings.AUTO_INDEX_KNOWLEDGE_BASE:
        import asyncio
        logger.info("[Startup] Scheduling knowledge base initialization...")
        asyncio.create_task(init_knowledge_base_background())
    else:
        logger.info("[Startup] Knowledge base auto-indexing disabled. Use /api/knowledge/reindex to index manually.")

    yield

    # Shutdown
    logger.info("Shutting down application")

    # Shutdown embedding manager (thread pool)
    try:
        embedding_manager = get_embedding_manager()
        embedding_manager.shutdown()
        logger.info("[Shutdown] Embedding thread pool closed")
    except Exception as e:
        logger.warning(f"[Shutdown] Error closing embedding thread pool: {e}")

    # Close HTTP client for embeddings
    await close_http_client()
    logger.info("[Shutdown] HTTP client closed")

    # Close embedding lock (Redis connection)
    try:
        from app.core.embeddings import _embedding_lock
        if _embedding_lock:
            await _embedding_lock.close()
            logger.info("[Shutdown] Embedding lock closed")
    except Exception as e:
        logger.warning(f"[Shutdown] Error closing embedding lock: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    智能客服系统 API

    ## 功能特性

    * **智能对话** - 基于大语言模型的智能客服对话
    * **知识库问答** - RAG 技术实现的知识库检索问答
    * **文档管理** - 支持多种格式文档上传和索引
    * **流式响应** - 支持实时流式输出
    * **Agent 模式** - ReAct 推理模式，支持工具调用
    * **MCP 工具** - 知识库搜索、网页搜索、代码执行等工具

    ## 技术栈

    * FastAPI + LangChain + Milvus
    * 支持 Ollama / OpenAI / DeepSeek / Claude 等多种 LLM
    * MCP 协议工具集成
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
origins = settings.CORS_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(system_router)
app.include_router(agent_router)
app.include_router(mcp_router)
app.include_router(config_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/system/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
