"""进程入口。四类部署单元各自独立扩缩。"""
from __future__ import annotations

import os

import uvicorn


def api() -> None:
    """HTTP API。按 QPS 扩缩。"""
    uvicorn.run(
        "ics.api.main:app",
        host=os.getenv("ICS_HOST", "0.0.0.0"),
        port=int(os.getenv("ICS_PORT", "8000")),
        reload=os.getenv("ICS_RELOAD", "").lower() in {"1", "true"},
    )


def mcp_server() -> None:
    """MCP Server。无状态，可水平扩展。"""
    import asyncio

    from ics.api.wiring import build_container
    from ics.mcp_servers.knowledge_server import build_server

    async def run() -> None:
        container, stack = await build_container()
        try:
            server = build_server(container.retrieval)
            config = uvicorn.Config(
                server.streamable_http_app(stateless_http=True),
                host=os.getenv("ICS_HOST", "0.0.0.0"),
                port=int(os.getenv("ICS_MCP_PORT", "8003")),
            )
            await uvicorn.Server(config).serve()
        finally:
            await stack.aclose()

    asyncio.run(run())


def recall_probe() -> None:
    """召回率巡检。挂定时任务。"""
    import argparse
    import asyncio

    from ics.api.wiring import build_container
    from ics.core.context import RequestContext
    from ics.worker.recall_probe import probe

    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--kb-id", default="default")
    args = ap.parse_args()

    async def run() -> int:
        container, stack = await build_container()
        try:
            r = await probe(
                RequestContext(tenant_id=args.tenant), container.store,
                container.pool, kb_id=args.kb_id,
            )
            print(f"tenant={r.tenant_id} kb={r.kb_id} samples={r.samples} recall={r.recall:.4f}")
            return 0 if r.recall >= 0.9 else 1
        finally:
            await stack.aclose()

    raise SystemExit(asyncio.run(run()))
