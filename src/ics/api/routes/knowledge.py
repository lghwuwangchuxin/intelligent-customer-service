"""知识库入口。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel, Field

from ..deps import Container, Ctx

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class RetrieveRequest(BaseModel):
    query: str
    kb_id: str = "default"
    top_n: int | None = Field(default=None, ge=1, le=50)


class IngestRequest(BaseModel):
    doc_id: str
    text: str
    title: str = ""
    kb_id: str = "default"


@router.post("/retrieve")
async def retrieve(req: RetrieveRequest, ctx: Ctx, c: Container) -> dict[str, Any]:
    result = await c.retrieval.retrieve(          # type: ignore[attr-defined]
        ctx, req.query, kb_id=req.kb_id, top_n=req.top_n
    )
    return {
        "query": req.query, "strategy": result.strategy,
        "latency_ms": result.latency_ms,
        "documents": [
            {"id": c_.id, "doc_id": c_.doc_id, "chunk_no": c_.chunk_no,
             "content": c_.content, "score": c_.score, "section": c_.section_path}
            for c_ in result.chunks
        ],
    }


@router.post("/documents")
async def ingest(req: IngestRequest, ctx: Ctx, c: Container) -> dict[str, Any]:
    r = await c.ingest.ingest_text(               # type: ignore[attr-defined]
        ctx, doc_id=req.doc_id, text=req.text, kb_id=req.kb_id, title=req.title
    )
    return {"doc_id": r.doc_id, "chunks": r.chunks, "skipped_unchanged": r.skipped}


@router.post("/documents/upload")
async def upload(file: UploadFile, ctx: Ctx, c: Container, kb_id: str = "default") -> dict[str, Any]:
    raw = await file.read()
    # 🔴 按真实格式解析。盲目 utf-8 解码会把 PDF/Word 二进制当文本入库、
    #    被嵌入、被检索、被当成答案依据，全程不报错。
    r = await c.ingest.ingest_bytes(              # type: ignore[attr-defined]
        ctx, doc_id=file.filename or "upload", raw=raw,
        filename=file.filename or "", content_type=file.content_type or "", kb_id=kb_id,
    )
    return {"doc_id": r.doc_id, "chunks": r.chunks, "format": r.format}


@router.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str, ctx: Ctx, c: Container) -> dict[str, Any]:
    return {"doc_id": doc_id, "deleted_chunks": await c.store.delete_doc(ctx, doc_id)}  # type: ignore[attr-defined]


@router.get("/documents")
async def list_docs(ctx: Ctx, c: Container, kb_id: str | None = None) -> dict[str, Any]:
    async with c.pool.connection() as conn:       # type: ignore[attr-defined]
        cur = await conn.execute(
            """SELECT doc_id, kb_id, title, chunk_count, updated_at
                 FROM document WHERE tenant_id=%s AND (%s::text IS NULL OR kb_id=%s)
                ORDER BY updated_at DESC LIMIT 200""",
            (ctx.tenant_id, kb_id, kb_id),
        )
        rows = await cur.fetchall()
    return {
        "documents": [
            {"doc_id": r[0], "kb_id": r[1], "title": r[2],
             "chunk_count": r[3], "updated_at": r[4].isoformat()}
            for r in rows
        ]
    }


@router.get("/stats")
async def stats(ctx: Ctx, c: Container) -> dict[str, Any]:
    async with c.pool.connection() as conn:       # type: ignore[attr-defined]
        cur = await conn.execute(
            "SELECT count(DISTINCT doc_id), count(*) FROM doc_chunk WHERE tenant_id=%s",
            (ctx.tenant_id,),
        )
        docs, chunks = await cur.fetchone()
        cur = await conn.execute(
            """SELECT recall, probed_at FROM recall_probe
                WHERE tenant_id=%s ORDER BY probed_at DESC LIMIT 1""",
            (ctx.tenant_id,),
        )
        probe = await cur.fetchone()
    return {
        "documents": docs, "chunks": chunks,
        "last_recall": {"recall": probe[0], "at": probe[1].isoformat()} if probe else None,
    }
