"""
Configuration settings for the intelligent customer service system.
Based on patterns from Langchain and Dify projects.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "Intelligent Customer Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # LLM Configuration
    LLM_PROVIDER: str = "ollama"  # ollama, openai, deepseek, claude
    LLM_MODEL: str = "qwen2.5:7b"
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_API_KEY: Optional[str] = None
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048

    # Claude API Configuration
    CLAUDE_API_KEY: Optional[str] = None
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_MAX_TOKENS: int = 4096
    CLAUDE_EXTENDED_THINKING: bool = False
    CLAUDE_THINKING_BUDGET: int = 10000  # tokens for extended thinking

    # MCP Configuration
    MCP_WEB_SEARCH_ENABLED: bool = True
    MCP_CODE_EXECUTION_ENABLED: bool = False
    MCP_FILE_SYSTEM_ENABLED: bool = True
    MCP_CODE_EXECUTION_TIMEOUT: int = 30

    # Agent Configuration
    AGENT_ENABLED: bool = True
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_MEMORY_MAX_MESSAGES: int = 50
    AGENT_MEMORY_SUMMARY_THRESHOLD: int = 20
    AGENT_MEMORY_PERSIST_PATH: str = "./data/conversations"  # Path to persist conversation history

    # LangGraph Agent (Advanced)
    AGENT_ENABLE_PLANNING: bool = True
    AGENT_ENABLE_PARALLEL_TOOLS: bool = True
    AGENT_TOOL_TIMEOUT: float = 30.0
    AGENT_MAX_TOOL_CONCURRENCY: int = 5

    # Embedding Configuration
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_BATCH_SIZE: int = 50  # Number of documents per embedding batch (increased for better throughput)
    EMBEDDING_MAX_CONCURRENT: int = 20  # Maximum concurrent embedding requests (increased for parallelism)

    # Milvus Vector Store
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "customer_service_kb"
    MILVUS_DATABASE: str = "default"

    # Elasticsearch Configuration
    ELASTICSEARCH_ENABLED: bool = True  # Enable ES for hybrid storage
    ELASTICSEARCH_HOSTS: str = "http://localhost:9200"
    ELASTICSEARCH_USERNAME: Optional[str] = None
    ELASTICSEARCH_PASSWORD: Optional[str] = None
    ELASTICSEARCH_INDEX_PREFIX: str = "knowledge_base"
    ELASTICSEARCH_CHUNK_INDEX: str = "knowledge_base_chunks"
    ELASTICSEARCH_USE_SSL: bool = False
    ELASTICSEARCH_VERIFY_CERTS: bool = True
    ELASTICSEARCH_CA_CERTS: Optional[str] = None
    ELASTICSEARCH_TIMEOUT: int = 30
    ELASTICSEARCH_MAX_RETRIES: int = 3

    # Hybrid Storage Settings (ES + Milvus)
    HYBRID_STORAGE_ENABLED: bool = True  # Enable ES+Milvus hybrid storage
    HYBRID_ES_WEIGHT: float = 0.3  # ES BM25 weight in hybrid search
    HYBRID_MILVUS_WEIGHT: float = 0.7  # Milvus vector weight in hybrid search
    HYBRID_SEARCH_TOP_K: int = 20  # Top K for initial retrieval before fusion

    # RAG Configuration (Basic)
    RAG_CHUNK_SIZE: int = 512  # Optimized chunk size for Chinese text
    RAG_CHUNK_OVERLAP: int = 100  # Increased overlap for better context
    RAG_TOP_K: int = 5

    # RAG Query Transformation
    # 注意: HyDE 和 Query Expansion 会显著增加响应时间（每次额外 30-60 秒）
    # 如果 LLM 响应较慢，建议禁用以提升用户体验
    RAG_ENABLE_HYDE: bool = False  # HyDE (Hypothetical Document Embeddings) - 禁用以加速
    RAG_ENABLE_QUERY_EXPANSION: bool = False  # Query Expansion - 禁用以加速
    RAG_QUERY_EXPANSION_NUM: int = 3  # Number of expanded queries

    # RAG Hybrid Retrieval - Optimized for better recall
    RAG_ENABLE_HYBRID: bool = True  # Enable hybrid retrieval (Vector + BM25)
    RAG_VECTOR_WEIGHT: float = 0.6  # Vector weight (lowered for better hybrid balance)
    RAG_BM25_WEIGHT: float = 0.4  # BM25 weight (increased for keyword matching)
    RAG_RETRIEVAL_TOP_K: int = 20  # Increased for better recall before reranking

    # RAG Reranking (Jina Reranker)
    RAG_ENABLE_RERANK: bool = True  # Enable reranking
    RAG_RERANK_MODEL: str = "jinaai/jina-reranker-v2-base-multilingual"
    RAG_RERANK_TOP_N: int = 8  # More results after reranking
    RAG_RERANK_SCORE_THRESHOLD: float = 0.15  # Lower threshold to keep more results

    # RAG Post-processing
    RAG_ENABLE_DEDUP: bool = True  # Enable semantic deduplication
    RAG_DEDUP_THRESHOLD: float = 0.92  # Slightly lower to allow more diverse results
    RAG_FINAL_TOP_K: int = 5  # Final number of results

    # RAG Document Processing
    RAG_USE_SEMANTIC_CHUNKING: bool = False  # Use semantic chunking (disabled for large docs)
    RAG_SEMANTIC_CHUNK_BUFFER_SIZE: int = 1  # Buffer size for semantic chunking

    # RAG Evaluation & Quality
    RAG_ENABLE_EVALUATION: bool = True  # Enable RAG quality evaluation
    RAG_EVALUATION_LOG_LEVEL: str = "INFO"  # Evaluation logging level
    RAG_MIN_RELEVANCE_SCORE: float = 0.1  # Minimum relevance score to consider

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/customer_service.db"

    # Redis Cache Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 10
    REDIS_SOCKET_TIMEOUT: float = 5.0
    REDIS_CACHE_ENABLED: bool = True  # Enable/disable Redis caching
    REDIS_CACHE_TTL: int = 3600  # Default TTL in seconds (1 hour)
    REDIS_EMBEDDING_CACHE_TTL: int = 86400  # Embedding cache TTL (24 hours)
    REDIS_SEARCH_CACHE_TTL: int = 300  # Search result cache TTL (5 minutes)

    # Knowledge Base
    KNOWLEDGE_BASE_PATH: str = "./data/knowledge_base"
    AUTO_INDEX_KNOWLEDGE_BASE: bool = True  # Auto-index knowledge base on startup

    # Langfuse Observability
    LANGFUSE_ENABLED: bool = True
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"  # or self-hosted URL

    # ============ Logging Configuration ============
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_DIR: str = "./logs"  # Directory for log files
    LOG_TO_CONSOLE: bool = True  # Output logs to console
    LOG_TO_FILE: bool = True  # Output logs to file
    LOG_JSON_FORMAT: bool = False  # Use JSON format for file logs
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # Max file size before rotation (10MB)
    LOG_BACKUP_COUNT: int = 30  # Number of backup files to keep
    LOG_RAG_SEPARATE: bool = True  # Separate log file for RAG pipeline
    LOG_AGENT_SEPARATE: bool = True  # Separate log file for Agent reasoning
    LOG_RAGAS_SEPARATE: bool = True  # Separate log file for RAG quality evaluation
    LOG_CLEANUP_DAYS: int = 30  # Days to keep old log files

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
