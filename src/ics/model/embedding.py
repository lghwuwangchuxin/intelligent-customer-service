"""嵌入。实现 core.spi.Embedder。

🔴 四个静默失败点集中在这里：
 1. Ollama 的 ``/api/embeddings`` 已弃用 → 用 ``/api/embed``，响应字段是复数
    ``embeddings``（二维数组）
 2. ``truncate`` 默认 **true**：超长输入被静默砍掉，质量下降但不报错
 3. 非对称模型查询侧要加指令前缀，漏加静默降低召回 → 做成 ``kind`` 参数强制表态
 4. 换模型 = 维度与语义空间都变，旧索引必须重建 → 每次返回前校验维度
"""
from __future__ import annotations

import asyncio
from typing import Literal

import httpx

from ..core.config import settings
from ..core.context import RequestContext
from ..core.errors import ContextTooLong, ModelUnavailable
from ..core.telemetry import get_logger, span
from .registry import ModelRegistry, get_registry

logger = get_logger(__name__)

# 非对称模型的查询侧指令前缀。bge-m3 不需要；bge-large-zh 系需要。
QUERY_INSTRUCTIONS: dict[str, str] = {
    "bge-large-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "bge-base-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
}


def estimate_tokens(text: str) -> int:
    """粗估：中文约 1 字 1 token，其余约 4 字符 1 token。只用于挡住明显超长。"""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk + (len(text) - cjk) // 4


class OllamaEmbedder:
    def __init__(self, alias: str = "embed-main", registry: ModelRegistry | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        spec = (registry or get_registry()).resolve(alias)
        self._model = spec.provider_model
        self.dimensions = spec.dimensions
        self.max_input_tokens = spec.max_input_tokens
        headers = {}
        if key := settings.embed_api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {key}"   # OpenAI 兼容端点用
        self._client = client or httpx.AsyncClient(
            base_url=settings.embed_api_base.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )

    async def embed(
        self,
        ctx: RequestContext,
        texts: list[str],
        kind: Literal["query", "document"] = "document",
    ) -> list[list[float]]:
        if not texts:
            return []
        prefix = QUERY_INSTRUCTIONS.get(self._model, "") if kind == "query" else ""
        prepared = [prefix + t for t in texts]

        for original, text in zip(texts, prepared, strict=True):
            n = estimate_tokens(text)
            if n > self.max_input_tokens:
                raise ContextTooLong(
                    f"约 {n} token 超过嵌入模型 {self._model} 上限 {self.max_input_tokens}；"
                    "应在入库前切分，而不是依赖模型截断",
                    preview=original[:80],
                )

        out: list[list[float]] = []
        with span("gen_ai.embed", ctx, **{"gen_ai.request.model": self._model}):
            for i in range(0, len(prepared), settings.embed_batch_size):
                out.extend(await self._call(prepared[i : i + settings.embed_batch_size]))

        for v in out:
            if len(v) != self.dimensions:
                raise ModelUnavailable(
                    f"嵌入维度不符：得到 {len(v)}，登记为 {self.dimensions}。"
                    "换嵌入模型后必须重建全部索引",
                    model=self._model,
                )
        return out

    async def embed_query_cached(self, ctx: RequestContext, text: str) -> list[float]:
        """带缓存的查询嵌入。

        同一问题反复问是客服场景常态，命中可整次省掉模型调用。
        缓存键含模型名与维度：换模型时旧缓存自动失效，不会拿旧空间的向量去检索。
        """
        from ..infra.cache import CacheKey, get_cache

        key = CacheKey.build("embed:query", ctx, self._model, self.dimensions, text)
        cache = get_cache()
        hit = await cache.get_json(key)
        if isinstance(hit, list) and len(hit) == self.dimensions:
            return [float(x) for x in hit]
        vec = (await self.embed(ctx, [text], kind="query"))[0]
        await cache.set_json(key, vec, ttl=settings.embed_cache_ttl_s)
        return vec

    async def _call(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.post(
                "/api/embed",   # 🔴 不是 /api/embeddings（已弃用）
                json={"model": self._model, "input": batch, "truncate": False},
            )
            resp.raise_for_status()
            data = resp.json().get("embeddings")   # 🔴 复数，二维数组
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"嵌入服务调用失败: {exc}", model=self._model) from exc

        if not isinstance(data, list) or len(data) != len(batch):
            raise ModelUnavailable(
                f"嵌入返回条数不符：请求 {len(batch)}，返回 "
                f"{len(data) if isinstance(data, list) else type(data).__name__}",
                model=self._model,
            )
        return data

    async def aclose(self) -> None:
        await self._client.aclose()


_embedder: OllamaEmbedder | None = None


def get_embedder() -> OllamaEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbedder()
    return _embedder
