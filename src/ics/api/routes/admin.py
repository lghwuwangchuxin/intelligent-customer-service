"""管理入口：待确认操作单、长期记忆、系统信息、指标。"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Response
from pydantic import BaseModel

from ...core.config import settings
from ...model.registry import get_registry
from ..deps import Container, Ctx

router = APIRouter(tags=["admin"])


class ResolveRequest(BaseModel):
    status: Literal["approved", "rejected"]


@router.get("/pending")
async def list_pending(ctx: Ctx, c: Container) -> dict[str, Any]:
    """待人工确认的写操作。写操作只产出操作单，没有绕过路径。"""
    items = await c.pending.list(ctx)      # type: ignore[attr-defined]
    return {
        "pending": [
            {"id": i.id, "tool": i.tool, "args": i.args, "summary": i.summary,
             "status": i.status, "created_at": i.created_at.isoformat()}
            for i in items
        ]
    }


@router.post("/pending/{pid}")
async def resolve_pending(pid: str, req: ResolveRequest, ctx: Ctx, c: Container) -> dict[str, Any]:
    ok = await c.pending.resolve(ctx, pid, req.status)   # type: ignore[attr-defined]
    return {"id": pid, "status": req.status, "updated": ok}


@router.get("/memories")
async def list_memories(ctx: Ctx, c: Container, q: str | None = None) -> dict[str, Any]:
    items = await c.memories.search(ctx, q)              # type: ignore[attr-defined]
    return {"memories": [{"key": i.key, "value": i.value} for i in items]}


@router.get("/system/info")
async def system_info(c: Container) -> dict[str, Any]:
    reg = get_registry()
    return {
        "env": settings.env,
        "models": {a: reg.resolve(a).provider_model for a in reg.aliases()},
        "agents": [a.id for a in c.agents.list()],       # type: ignore[attr-defined]
        "prompts": c.prompts.refs(),                     # type: ignore[attr-defined]
        "tools": c.tools.names(),                        # type: ignore[attr-defined]
        "durable_memory": c.durable_memory,              # type: ignore[attr-defined]
        "retrieval": settings.retrieval.model_dump(),
    }


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
