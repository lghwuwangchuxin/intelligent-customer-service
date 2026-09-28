"""pgvector 向量库。实现 core.spi.VectorStore。

🔴 ``doc_chunk`` 按 tenant_id **LIST 分区**：共享一个近似索引会让租户之间互相
   影响召回（HNSW 图是全局的，A 的向量会占掉 B 的搜索预算）。
   分区后每租户独立索引，删租户也只是 DROP 分区。
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

from ...core.config import settings
from ...core.context import RequestContext
from ...core.errors import RetrievalFailed
from ...core.telemetry import get_logger, span
from ...core.types import Chunk

logger = get_logger(__name__)


class PgVectorStore:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._ensured: set[str] = set()
        self._lock = asyncio.Lock()

    async def ensure_tenant(self, ctx: RequestContext) -> None:
        if ctx.tenant_id in self._ensured:
            return
        async with self._lock:
            if ctx.tenant_id in self._ensured:
                return
            async with self._pool.connection() as conn:
                await conn.execute("SELECT ics_ensure_tenant(%s)", (ctx.tenant_id,))
            self._ensured.add(ctx.tenant_id)

    async def upsert(
        self, ctx: RequestContext, chunks: list[Chunk], vectors: list[list[float]]
    ) -> int:
        if not chunks:
            return 0
        await self.ensure_tenant(ctx)
        from pgvector import Vector

        from ..retrieve.sparse import to_tsvector_input

        rows = [
            {
                "id": c.id, "t": ctx.tenant_id, "doc": c.doc_id, "kb": c.source or "default",
                "no": c.chunk_no, "content": c.content, "sec": c.section_path,
                "parent": c.parent_id,
                # 索引侧与查询侧同一个分词器
                "lex": to_tsvector_input(c.content),
                "meta": json.dumps({}, ensure_ascii=False),
                "emb": Vector(v.to_list() if hasattr(v, "to_list") else list(v)),
            }
            # strict=True：长度不符会让向量与切块错位对齐，是静默的数据损坏
            for c, v in zip(chunks, vectors, strict=True)
        ]
        # 🔴 UNIQUE(tenant_id, doc_id, chunk_no) 保证摄取幂等：重跑不产生重复块
        sql = """
            INSERT INTO doc_chunk
                (id, tenant_id, doc_id, kb_id, chunk_no, content, section_path,
                 parent_id, lexical_tsv, metadata, embedding)
            VALUES (%(id)s, %(t)s, %(doc)s, %(kb)s, %(no)s, %(content)s, %(sec)s,
                    %(parent)s, to_tsvector('simple', %(lex)s), %(meta)s, %(emb)s)
            ON CONFLICT (tenant_id, doc_id, chunk_no) DO UPDATE SET
                content = EXCLUDED.content, section_path = EXCLUDED.section_path,
                lexical_tsv = EXCLUDED.lexical_tsv, embedding = EXCLUDED.embedding
        """
        async with self._pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(sql, rows)
        return len(rows)

    async def search(
        self, ctx: RequestContext, vector: list[float], top_k: int, filters: dict
    ) -> list[Chunk]:
        from pgvector import Vector

        # 从库里读回的向量是 pgvector.Vector（池级注册了类型），不是 list。
        # 召回率巡检就是把查到的向量再喂回来，这里要兼容两种形态。
        values = vector.to_list() if hasattr(vector, "to_list") else list(vector)

        exact = bool(filters.get("exact"))
        sql = """
            SELECT id, doc_id, chunk_no, content, section_path, parent_id, kb_id,
                   tenant_id, embedding <=> %(v)s AS distance
              FROM doc_chunk
             WHERE tenant_id = %(t)s
               AND (%(kb)s::text IS NULL OR kb_id = %(kb)s)
               AND embedding IS NOT NULL
             ORDER BY embedding <=> %(v)s
             LIMIT %(k)s
        """
        params = {
            "t": ctx.tenant_id, "kb": filters.get("kb_id"),
            "k": top_k, "v": Vector(values),
        }
        try:
            with span("rag.vector_search", ctx):
                async with self._pool.connection() as conn, conn.transaction():
                    if exact:
                        # 精确检索：仅供召回率基线比对，绝不用于线上请求
                        await conn.execute("SET LOCAL enable_indexscan = off")
                        await conn.execute("SET LOCAL enable_bitmapscan = off")
                    cur = await conn.execute(sql, params)
                    rows = await cur.fetchall()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RetrievalFailed(str(exc)[:200]) from exc

        out: list[Chunk] = []
        for r in rows:
            ctx.assert_owns(r[7])   # 存储层 WHERE 之外的第二道闸
            out.append(Chunk(
                id=str(r[0]), tenant_id=r[7], doc_id=r[1], chunk_no=r[2], content=r[3],
                section_path=r[4], parent_id=r[5], source=r[6], score=1.0 - float(r[8]),
            ))
        if not exact and len(out) < top_k:
            logger.debug(
                "向量召回 %d < 请求 %d（ef_search=%d, iterative_scan=%s）",
                len(out), top_k, settings.retrieval.hnsw_ef_search,
                settings.retrieval.iterative_scan,
            )
        return out

    async def fetch(self, ctx: RequestContext, ids: Sequence[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """SELECT id, doc_id, chunk_no, content, section_path, parent_id,
                          kb_id, tenant_id
                     FROM doc_chunk WHERE tenant_id = %s AND id = ANY(%s::uuid[])""",
                (ctx.tenant_id, list(ids)),
            )
            rows = await cur.fetchall()
        result: dict[str, Chunk] = {}
        for r in rows:
            ctx.assert_owns(r[7])
            result[str(r[0])] = Chunk(
                id=str(r[0]), tenant_id=r[7], doc_id=r[1], chunk_no=r[2],
                content=r[3], section_path=r[4], parent_id=r[5], source=r[6],
            )
        return result

    async def delete_doc(self, ctx: RequestContext, doc_id: str) -> int:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM doc_chunk WHERE tenant_id = %s AND doc_id = %s",
                (ctx.tenant_id, doc_id),
            )
            return cur.rowcount
