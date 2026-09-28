"""混合检索管线：稠密 + 稀疏 → RRF → 重排。"""
from __future__ import annotations

import asyncio
import time

from ...core.config import settings
from ...core.context import RequestContext
from ...core.spi import Embedder, LexicalIndex, Reranker, VectorStore
from ...core.telemetry import get_logger, metrics
from ...core.types import Chunk, RetrievalResult
from .fusion import rrf

logger = get_logger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        lexical: LexicalIndex | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._lexical = lexical
        self._reranker = reranker

    async def retrieve(
        self, ctx: RequestContext, query: str, *, kb_id: str | None = None,
        top_n: int | None = None,
    ) -> RetrievalResult:
        cfg = settings.retrieval
        top_n = top_n or cfg.rerank_top_n
        started = time.monotonic()
        filters = {"kb_id": kb_id}

        # 走缓存版（若实现提供）：同一问题反复问可整次省掉嵌入调用
        cached = getattr(self._embedder, "embed_query_cached", None)
        if cached is not None:
            vector = await cached(ctx, query)
        else:
            vector = (await self._embedder.embed(ctx, [query], kind="query"))[0]

        tasks: list[asyncio.Future | asyncio.Task] = [
            asyncio.ensure_future(self._store.search(ctx, vector, cfg.dense_top_k, filters))
        ]
        if self._lexical is not None:
            tasks.append(
                asyncio.ensure_future(
                    self._lexical.search(ctx, query, cfg.sparse_top_k, filters)
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, asyncio.CancelledError):
                raise r   # 🔴 gather(return_exceptions=True) 会把取消当普通结果收走

        dense = results[0]
        if isinstance(dense, BaseException):
            raise dense   # 稠密路是主路，挂了就该失败

        sparse_ids: list[str] = []
        if len(results) > 1:
            if isinstance(results[1], BaseException):
                # 稀疏路是增强项：失败降级为纯向量，但必须留痕并打指标
                logger.warning("词法检索失败，本次降级为纯向量: %s", results[1])
                metrics().degraded.labels("retrieval", "lexical_failed").inc()
            else:
                sparse_ids = results[1]

        by_id: dict[str, Chunk] = {c.id: c for c in dense}
        missing = [i for i in sparse_ids if i not in by_id]
        if missing:
            by_id.update(await self._store.fetch(ctx, missing))

        rankings: list[list[str]] = [[c.id for c in dense]]
        if sparse_ids:
            rankings.append(sparse_ids)
        fused = rrf(rankings, k=cfg.rrf_k, top_n=max(top_n * 4, top_n))

        merged = [by_id[i] for i, _ in fused if i in by_id]
        strategy = "hybrid" if sparse_ids else "dense"

        if self._reranker is not None and merged:
            order = await self._reranker.rerank(
                ctx, query, [c.content for c in merged], top_n
            )
            merged = [
                merged[i].model_copy(update={"score": s}) for i, s in order
            ]
            strategy += "+rerank"
        else:
            merged = merged[:top_n]

        # 按预算裁剪：粗估 1 token ≈ 3 字符，避免检索结果把上下文顶爆
        budget_chars = cfg.max_context_tokens * 3
        used = 0
        trimmed: list[Chunk] = []
        for c in merged:
            if used + len(c.content) > budget_chars and trimmed:
                break
            trimmed.append(c)
            used += len(c.content)
        merged = trimmed

        return RetrievalResult(
            chunks=merged,
            strategy=strategy,
            latency_ms=int((time.monotonic() - started) * 1000),
            total_tokens=sum(len(c.content) for c in merged) // 3,
        )
