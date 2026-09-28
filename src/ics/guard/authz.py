"""工具授权与写操作闸门。"""
from __future__ import annotations

from typing import Any, Protocol

from ..core.context import RequestContext
from ..core.errors import GuardrailBlocked, PendingApproval
from ..core.telemetry import get_logger

logger = get_logger(__name__)


class ToolPolicy(Protocol):
    def allowed(self, tenant_id: str, name: str) -> bool: ...
    def is_write(self, name: str) -> bool: ...


class PendingStore(Protocol):
    async def create(
        self, ctx: RequestContext, tool: str, args: dict[str, Any], summary: str
    ) -> str: ...


async def before_tool_call(
    ctx: RequestContext,
    name: str,
    args: dict[str, Any],
    policy: ToolPolicy,
    pending: PendingStore | None = None,
) -> None:
    """工具调用前的闸门。

    🔴 **写操作只产出待确认操作单，没有绕过路径**：
       不执行，落一条 pending 记录并抛 PendingApproval，由编排层挂起等人工确认。
    """
    if not policy.allowed(ctx.tenant_id, name):
        raise GuardrailBlocked(f"工具 {name} 未对该租户开放", tool=name)

    if policy.is_write(name):
        if pending is None:
            raise GuardrailBlocked(f"写操作 {name} 缺少待确认单存储，拒绝执行", tool=name)
        pid = await pending.create(ctx, name, args, summary=f"{name}({args})")
        logger.info("写操作已转为待确认单 tool=%s pending=%s", name, pid)
        raise PendingApproval(f"pending_action={pid}", tool=name, pending_id=pid)
