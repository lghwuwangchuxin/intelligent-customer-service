"""PostgreSQL 连接池。"""
from __future__ import annotations

from typing import Any

from psycopg_pool import AsyncConnectionPool

from ..core.config import settings
from ..core.telemetry import get_logger

logger = get_logger(__name__)

_pool: AsyncConnectionPool | None = None


async def _configure(conn: Any) -> None:
    """🔴 每条连接都要注册向量类型。

    只在启动时注册一条 → 池里其他连接取回的向量是字符串 →
    表现为"偶发类型错误"（取到哪条连接是随机的，所以偶发）。
    """
    from pgvector.psycopg import register_vector_async

    await register_vector_async(conn)
    async with conn.cursor() as cur:
        # 🔴 带过滤的 ANN 检索必开迭代扫描，否则过滤后召回不足且不报错。
        #    这里用会话级 SET（非 SET LOCAL），对该连接后续所有事务都生效。
        await cur.execute(f"SET hnsw.iterative_scan = {settings.retrieval.iterative_scan}")
        await cur.execute(f"SET hnsw.ef_search = {int(settings.retrieval.hnsw_ef_search)}")
    # 🔴 必须提交：psycopg_pool 要求 configure 返回时连接**不在事务中**，
    #    否则每条新连接都会被判为 "left in status INTRANS" 并丢弃，池永远建不起来。
    await conn.commit()


def get_pool() -> AsyncConnectionPool:
    """取池对象。open=False —— 生命周期挂到应用 lifespan，不在 import 时 open。"""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=settings.database_url.get_secret_value(),
            min_size=settings.pg_pool_min,
            max_size=settings.pg_pool_max,
            configure=_configure,
            timeout=5.0,
            max_idle=300,
            open=False,
        )
    return _pool


async def open_pool() -> AsyncConnectionPool:
    pool = get_pool()
    await pool.open(wait=True)
    logger.info("PostgreSQL 连接池已就绪 min=%d max=%d", settings.pg_pool_min, settings.pg_pool_max)
    return pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
