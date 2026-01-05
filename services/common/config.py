"""Service configuration management."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from functools import lru_cache


@dataclass
class ServiceConfig:
    """Configuration for a microservice."""

    # Service identity
    service_name: str
    service_version: str = "1.0.0"

    # Network configuration
    host: str = "0.0.0.0"
    http_port: int = 8000
    grpc_port: int = 50050

    # Nacos configuration (service discovery)
    nacos_server_addresses: str = "localhost:8848"
    nacos_namespace: str = "public"
    nacos_group: str = "DEFAULT_GROUP"
    nacos_username: str = ""
    nacos_password: str = ""
    nacos_enabled: bool = True

    # LLM configuration
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    llm_api_key: str = ""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # Database configuration
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""
    redis_url: str = "redis://localhost:6379"

    # Milvus configuration
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "knowledge_base"

    # Embedding configuration
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://localhost:11434"
    embedding_batch_size: int = 50

    # RAG configuration
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50
    rag_top_k: int = 5
    rag_enable_hyde: bool = False
    rag_enable_rerank: bool = True
    rag_rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    rag_vector_weight: float = 0.7
    rag_bm25_weight: float = 0.3

    # Monitoring configuration
    langfuse_enabled: bool = False
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""

    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "json"
    log_to_console: bool = True
    log_to_file: bool = True
    log_dir: str = "/app/logs"

    # gRPC configuration
    grpc_max_workers: int = 10
    grpc_max_message_length: int = 100 * 1024 * 1024  # 100MB

    # Health check configuration
    health_check_interval: int = 30
    health_check_timeout: int = 10

    # Extra configuration
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, service_name: str) -> "ServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name=service_name,
            service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
            host=os.getenv("SERVICE_HOST", "0.0.0.0"),
            http_port=int(os.getenv("HTTP_PORT", "8000")),
            grpc_port=int(os.getenv("GRPC_PORT", "50050")),
            nacos_server_addresses=os.getenv("NACOS_SERVER_ADDRESSES", "localhost:8848"),
            nacos_namespace=os.getenv("NACOS_NAMESPACE", "public"),
            nacos_group=os.getenv("NACOS_GROUP", "DEFAULT_GROUP"),
            nacos_username=os.getenv("NACOS_USERNAME", ""),
            nacos_password=os.getenv("NACOS_PASSWORD", ""),
            nacos_enabled=os.getenv("NACOS_ENABLED", "true").lower() == "true",
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
            llm_model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            elasticsearch_url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
            elasticsearch_username=os.getenv("ELASTICSEARCH_USERNAME", ""),
            elasticsearch_password=os.getenv("ELASTICSEARCH_PASSWORD", ""),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            milvus_host=os.getenv("MILVUS_HOST", "localhost"),
            milvus_port=int(os.getenv("MILVUS_PORT", "19530")),
            milvus_collection=os.getenv("MILVUS_COLLECTION", "knowledge_base"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "50")),
            rag_chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
            rag_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
            rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
            rag_enable_hyde=os.getenv("RAG_ENABLE_HYDE", "false").lower() == "true",
            rag_enable_rerank=os.getenv("RAG_ENABLE_RERANK", "true").lower() == "true",
            rag_rerank_model=os.getenv("RAG_RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual"),
            rag_vector_weight=float(os.getenv("RAG_VECTOR_WEIGHT", "0.7")),
            rag_bm25_weight=float(os.getenv("RAG_BM25_WEIGHT", "0.3")),
            langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
            langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
            log_to_console=os.getenv("LOG_TO_CONSOLE", "true").lower() == "true",
            log_to_file=os.getenv("LOG_TO_FILE", "true").lower() == "true",
            log_dir=os.getenv("LOG_DIR", "/app/logs"),
            grpc_max_workers=int(os.getenv("GRPC_MAX_WORKERS", "10")),
            grpc_max_message_length=int(
                os.getenv("GRPC_MAX_MESSAGE_LENGTH", str(100 * 1024 * 1024))
            ),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            health_check_timeout=int(os.getenv("HEALTH_CHECK_TIMEOUT", "10")),
        )

    def get_nacos_server_addresses(self) -> str:
        """Get Nacos server addresses."""
        return self.nacos_server_addresses

    def get_service_address(self) -> str:
        """Get service HTTP address."""
        return f"{self.host}:{self.http_port}"

    def get_grpc_address(self) -> str:
        """Get service gRPC address."""
        return f"{self.host}:{self.grpc_port}"


# Cache for service configs
_config_cache: Dict[str, ServiceConfig] = {}


def get_service_config(service_name: str) -> ServiceConfig:
    """Get or create service configuration."""
    if service_name not in _config_cache:
        _config_cache[service_name] = ServiceConfig.from_env(service_name)
    return _config_cache[service_name]


def clear_config_cache():
    """Clear configuration cache."""
    _config_cache.clear()
