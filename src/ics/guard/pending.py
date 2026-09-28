"""待确认操作单存储。"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..core.context import RequestContext


@dataclass(slots=True)
class PendingAction:
    id: str
    tool: str
    args: dict[str, Any]
    summary: str
    status: str
    created_at: datetime


class PgPendingStore:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def create(
        self, ctx: RequestContext, tool: str, args: dict[str, Any], summary: str
    ) -> str:
        pid = str(uuid.uuid4())
        async with self._pool.connection() as conn:
            await conn.execute(
                """INSERT INTO pending_action
                       (id, tenant_id, user_id, tool, args, summary, status)
                   VALUES (%s,%s,%s,%s,%s,%s,'pending')""",
                (pid, ctx.tenant_id, ctx.user_id, tool,
                 json.dumps(args, ensure_ascii=False), summary),
            )
        return pid

    async def list(self, ctx: RequestContext, status: str = "pending") -> list[PendingAction]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """SELECT id, tool, args, summary, status, created_at
                     FROM pending_action WHERE tenant_id=%s AND status=%s
                    ORDER BY created_at DESC LIMIT 100""",
                (ctx.tenant_id, status),
            )
            return [
                PendingAction(str(r[0]), r[1], r[2], r[3], r[4], r[5])
                for r in await cur.fetchall()
            ]

    async def resolve(self, ctx: RequestContext, pid: str, status: str) -> bool:
        if status not in {"approved", "rejected"}:
            raise ValueError("status 只能是 approved / rejected")
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """UPDATE pending_action SET status=%s, resolved_at=now()
                    WHERE tenant_id=%s AND id=%s AND status='pending'""",
                (status, ctx.tenant_id, pid),
            )
        return cur.rowcount > 0
