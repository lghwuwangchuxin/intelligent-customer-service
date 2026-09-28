"""摄取管线：解析 → 切块 → 嵌入 → 入库。"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass

from ...core.context import RequestContext
from ...core.spi import Embedder, VectorStore
from ...core.telemetry import get_logger
from ...core.types import Chunk
from .parser import FileParser

logger = get_logger(__name__)


@dataclass(slots=True)
class IngestResult:
    doc_id: str
    chunks: int
    skipped: bool = False
    format: str = "text"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def split(text: str, *, chunk_size: int = 512, min_chars: int = 48) -> list[str]:
    """切块。

    🔴 Chonkie 的 ``RecursiveChunker`` **没有 chunk_overlap**（传了 TypeError）。
       只有 TokenChunker / SentenceChunker 有。照 2025 年教程写必错。
    """
    text = text.strip()
    if not text:
        return []
    try:
        from chonkie import RecursiveChunker

        chunker = RecursiveChunker(chunk_size=chunk_size, min_characters_per_chunk=min_chars)
        return [c.text for c in chunker(text) if c.text.strip()]
    except Exception as exc:
        # 降级必须留痕：静默降级会让检索质量莫名其妙地掉
        logger.warning("Chonkie 切块失败，退化为定长切分: %s", exc)
        size = chunk_size * 2
        return [text[i : i + size] for i in range(0, len(text), size)]


class IngestPipeline:
    def __init__(self, store: VectorStore, embedder: Embedder, pool: object) -> None:
        self._store = store
        self._embedder = embedder
        self._pool = pool
        self._parser = FileParser()

    async def ingest_bytes(
        self, ctx: RequestContext, *, doc_id: str, raw: bytes,
        filename: str = "", content_type: str = "", kb_id: str = "default",
    ) -> IngestResult:
        # 🔴 parse 是同步 CPU 密集，必须显式挪出事件循环
        parsed = await asyncio.to_thread(self._parser.parse, raw, filename, content_type)
        result = await self.ingest_text(
            ctx, doc_id=doc_id, text=parsed.text, kb_id=kb_id, title=filename
        )
        result.format = parsed.format
        return result

    async def ingest_text(
        self, ctx: RequestContext, *, doc_id: str, text: str,
        kb_id: str = "default", title: str = "",
    ) -> IngestResult:
        digest = content_hash(text)
        async with self._pool.connection() as conn:  # type: ignore[attr-defined]
            cur = await conn.execute(
                "SELECT content_hash FROM document WHERE tenant_id=%s AND doc_id=%s",
                (ctx.tenant_id, doc_id),
            )
            row = await cur.fetchone()
        if row and row[0] == digest:
            # 内容指纹未变：跳过重切分与重嵌入（嵌入是最贵的一步）
            return IngestResult(doc_id=doc_id, chunks=0, skipped=True)

        pieces = split(text)
        if not pieces:
            return IngestResult(doc_id=doc_id, chunks=0)

        chunks = [
            Chunk(
                id=str(uuid.uuid4()), tenant_id=ctx.tenant_id, doc_id=doc_id,
                chunk_no=i, content=p, source=kb_id,
            )
            for i, p in enumerate(pieces)
        ]
        # 🔴 批量嵌入，不要循环里逐条调（慢 10 倍）
        vectors = await self._embedder.embed(ctx, [c.content for c in chunks], kind="document")

        await self._store.delete_doc(ctx, doc_id)   # 文档变短时避免留下孤儿块
        written = await self._store.upsert(ctx, chunks, vectors)

        async with self._pool.connection() as conn:  # type: ignore[attr-defined]
            await conn.execute(
                """INSERT INTO document (tenant_id, doc_id, kb_id, title, content_hash, chunk_count)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (tenant_id, doc_id) DO UPDATE SET
                     kb_id=EXCLUDED.kb_id, title=EXCLUDED.title,
                     content_hash=EXCLUDED.content_hash, chunk_count=EXCLUDED.chunk_count,
                     updated_at=now()""",
                (ctx.tenant_id, doc_id, kb_id, title, digest, written),
            )
        logger.info("摄取完成 doc=%s chunks=%d", doc_id, written)
        return IngestResult(doc_id=doc_id, chunks=written)
