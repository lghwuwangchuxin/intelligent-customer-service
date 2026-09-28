"""HTTP 应用。"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.errors import AppError
from ..core.telemetry import get_logger, setup_logging, setup_tracing
from .routes import admin, chat, knowledge

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from ..infra.pg import close_pool
    from .wiring import build_container

    setup_logging()
    setup_tracing()
    container, stack = await build_container()
    app.state.container = container
    app.state.stack = stack
    try:
        yield
    finally:
        await stack.aclose()
        await close_pool()


def create_app() -> FastAPI:
    app = FastAPI(title="intelligent-customer-service", version="3.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        # 🔴 只出错误码 + 本地化文案 + 调用方 detail，绝不回显原始异常 message
        locale = (request.headers.get("Accept-Language") or "zh-CN").split(",")[0]
        return JSONResponse(status_code=exc.http_status, content=exc.to_payload(locale))

    @app.get("/health")
    async def health() -> dict[str, str]:
        # 🔴 liveness 不查依赖：查依赖会把下游故障放大成雪崩式重启
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, Any]:
        """readiness 才查依赖。"""
        c = getattr(request.app.state, "container", None)
        if c is None:
            return {"status": "starting"}
        try:
            async with c.pool.connection() as conn:
                await conn.execute("SELECT 1")
            db = "ok"
        except Exception as exc:
            db = f"down: {str(exc)[:80]}"
        return {"status": "ok" if db == "ok" else "degraded", "database": db,
                "durable_memory": c.durable_memory}

    app.include_router(chat.router)
    app.include_router(knowledge.router)
    app.include_router(admin.router)
    return app


app = create_app()
