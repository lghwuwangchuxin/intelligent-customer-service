"""召回率基线巡检。

🔴 HNSW 索引退化**完全无声**：不报错、不变慢，只是开始漏结果。
   唯一可靠的发现方式是定期用精确检索当 ground truth 做对比。
   这属于「质量稳定」——LLM 应用的失效大多不是崩溃，而是静默劣化。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.context import RequestContext
from ..core.telemetry import get_logger, metrics

logger = get_logger(__name__)


@dataclass(slots=True)
class RecallReport:
    tenant_id: str
    kb_id: str
    samples: int
    top_k: int
    recall: float


async def probe(
    ctx: RequestContext, store: object, pool: object, *,
    kb_id: str = "default", sample_size: int = 20, top_k: int = 10,
) -> RecallReport:
    async with pool.connection() as conn:   # type: ignore[attr-defined]
        cur = await conn.execute(
            """SELECT embedding FROM doc_chunk
                WHERE tenant_id=%s AND kb_id=%s AND embedding IS NOT NULL
                ORDER BY random() LIMIT %s""",
            (ctx.tenant_id, kb_id, sample_size),
        )
        samples = [r[0] for r in await cur.fetchall()]

    if not samples:
        return RecallReport(ctx.tenant_id, kb_id, 0, top_k, 1.0)

    hits = total = 0
    for vec in samples:
        approx = await store.search(ctx, vec, top_k, {"kb_id": kb_id})            # type: ignore[attr-defined]
        exact = await store.search(ctx, vec, top_k, {"kb_id": kb_id, "exact": True})  # type: ignore[attr-defined]
        exact_ids = {c.id for c in exact}
        if exact_ids:
            hits += len(exact_ids & {c.id for c in approx})
            total += len(exact_ids)

    recall = hits / total if total else 1.0
    metrics().recall.labels(ctx.tenant_id, kb_id).observe(recall)

    async with pool.connection() as conn:   # type: ignore[attr-defined]
        await conn.execute(
            """INSERT INTO recall_probe (tenant_id, kb_id, sample_size, top_k, recall)
               VALUES (%s,%s,%s,%s,%s)""",
            (ctx.tenant_id, kb_id, len(samples), top_k, recall),
        )
    if recall < 0.9:
        logger.warning(
            "召回率偏低 tenant=%s kb=%s recall=%.3f —— 先调大 ef_search，仍不改善则 REINDEX",
            ctx.tenant_id, kb_id, recall,
        )
    return RecallReport(ctx.tenant_id, kb_id, len(samples), top_k, recall)
