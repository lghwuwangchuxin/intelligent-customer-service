"""会话索引。

checkpointer 存的是图状态，不负责"给人看的会话列表"——标题、时间、条数是产品元数据。
单独维护索引表，而不是反查 checkpointer 的内部表（那会把自己钉死在框架内部 schema 上）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..core.context import RequestContext

TITLE_MAX = 40


@dataclass(slots=True)
class ConversationSummary:
    conversation_id: str
    title: str
    agent_id: str
    message_count: int
    updated_at: datetime


def derive_title(message: str) -> str:
    text = " ".join(message.split())
    return text[:TITLE_MAX] if text else "新会话"


class ConversationIndex:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def touch(
        self, ctx: RequestContext, *, conversation_id: str, first_message: str,
        agent_id: str, delta: int = 2,
    ) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """INSERT INTO conversation
                       (tenant_id, conversation_id, user_id, title, agent_id, message_count)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (tenant_id, conversation_id) DO UPDATE SET
                     agent_id=EXCLUDED.agent_id,
                     message_count=conversation.message_count+EXCLUDED.message_count,
                     updated_at=now()""",
                (ctx.tenant_id, conversation_id, ctx.user_id,
                 derive_title(first_message), agent_id, delta),
            )

    async def list(self, ctx: RequestContext, limit: int = 50) -> list[ConversationSummary]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """SELECT conversation_id, title, agent_id, message_count, updated_at
                     FROM conversation WHERE tenant_id=%s
                    ORDER BY updated_at DESC LIMIT %s""",
                (ctx.tenant_id, limit),
            )
            return [
                ConversationSummary(r[0], r[1], r[2], r[3], r[4])
                for r in await cur.fetchall()
            ]

    async def delete(self, ctx: RequestContext, conversation_id: str) -> bool:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM conversation WHERE tenant_id=%s AND conversation_id=%s",
                (ctx.tenant_id, conversation_id),
            )
        return cur.rowcount > 0
