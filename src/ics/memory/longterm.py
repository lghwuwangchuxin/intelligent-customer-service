"""跨会话长期记忆。

🔴 命名空间**第一段必须是租户**：store 的 search 按前缀匹配，
   租户不在前缀里就等于全局可见。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.context import RequestContext
from ..core.errors import TenantRequired


@dataclass(slots=True)
class MemoryItem:
    key: str
    value: dict[str, Any]


def namespace_for(ctx: RequestContext) -> tuple[str, str, str]:
    if not ctx.user_id:
        # fail-closed：没有用户身份就不该读写"用户的"长期记忆
        raise TenantRequired("长期记忆需要用户标识")
    return (ctx.tenant_id, "memories", ctx.user_id)


class LongTermMemory:
    def __init__(self, store: Any) -> None:
        self._store = store

    async def put(self, ctx: RequestContext, key: str, value: dict[str, Any]) -> None:
        await self._store.aput(namespace_for(ctx), key, value)

    async def get(self, ctx: RequestContext, key: str) -> MemoryItem | None:
        item = await self._store.aget(namespace_for(ctx), key)
        return None if item is None else MemoryItem(key=key, value=dict(item.value))

    async def search(
        self, ctx: RequestContext, query: str | None = None, limit: int = 50
    ) -> list[MemoryItem]:
        items = await self._store.asearch(namespace_for(ctx), query=query, limit=limit)
        return [MemoryItem(key=i.key, value=dict(i.value)) for i in items]

    async def delete(self, ctx: RequestContext, key: str) -> None:
        await self._store.adelete(namespace_for(ctx), key)
