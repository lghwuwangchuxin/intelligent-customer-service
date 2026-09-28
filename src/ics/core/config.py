"""配置。进程启动即校验；配置错就起不来，不是跑到第一个请求才炸。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalDefaults(BaseModel):
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_k: int = 60
    rerank_top_n: int = 5
    max_context_tokens: int = 4000
    # 🔴 pgvector 默认 ef_search=40，带过滤检索时返回数会远少于 top_k
    hnsw_ef_search: int = 100
    iterative_scan: Literal["off", "strict_order", "relaxed_order"] = "relaxed_order"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ICS_", env_file=".env", extra="ignore", frozen=True
    )

    env: Literal["local", "dev", "prod"] = "local"
    log_level: str = "INFO"
    log_json: bool = False

    # ← SecretStr：repr / 日志自动打码
    database_url: SecretStr = SecretStr("postgresql://ics:ics@localhost:5432/ics")
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    pg_pool_min: int = 4
    pg_pool_max: int = 20          # ← 一次请求可能持有 2 条（检索 + checkpointer）

    # 模型别名 → 真实模型在 model/registry.py；这里只配端点与密钥
    llm_api_base: str | None = None
    llm_api_key: SecretStr = SecretStr("")
    embed_api_base: str = "http://localhost:11434"
    embed_api_key: SecretStr = SecretStr("")   # OpenAI 兼容端点用；Ollama 留空

    max_concurrency_per_model: int = 16
    request_timeout_s: float = 90.0
    embed_batch_size: int = 64
    rerank_concurrency: int = 2    # ← 本地模型不是线程安全的，限并发

    default_budget_tokens: int = 50_000
    cache_ttl_s: int = 300
    embed_cache_ttl_s: int = 86_400

    prompts_dir: str = "prompts"
    agents_dir: str = "agents"

    # 可观测绑 OTel 而非某个厂商 SDK：Langfuse / Phoenix / Jaeger 都接 OTLP，
    # 换后端只改 endpoint，不改一行埋点代码。
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4318/v1/traces"
    otel_headers: SecretStr = SecretStr("")   # 形如 "Authorization=Bearer xxx" 

    retrieval: RetrievalDefaults = Field(default_factory=RetrievalDefaults)


settings = Settings()   # ← 进程启动即校验
