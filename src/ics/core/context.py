"""请求上下文。🔴 显式传递，禁用 ContextVar —— 异步/线程池边界会静默丢失。

租户与预算都在这里：两者都必须跟着调用链走到最深处，
任何一个"从环境里取"的方案在后台任务里都会失效。
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, replace

_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class RequestContext:
    tenant_id: str                               # ← 缺失即抛，绝不给默认值
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str | None = None
    locale: str = "zh-CN"
    budget_tokens: int = 50_000
    spent_tokens: int = 0

    def __post_init__(self) -> None:
        # fail-closed：没有租户就不该有请求。给默认值等于放行跨租户读取。
        if not self.tenant_id:
            raise ValueError("tenant_id is required (fail-closed)")
        if not _TENANT_RE.match(self.tenant_id):
            raise ValueError(f"illegal tenant_id: {self.tenant_id!r}")

    @property
    def remaining(self) -> int:
        return max(self.budget_tokens - self.spent_tokens, 0)

    def spend(self, n: int) -> RequestContext:
        return replace(self, spent_tokens=self.spent_tokens + n)

    def child(self, share: int = 1, **kw: object) -> RequestContext:
        """派生给子 Agent：trace 继承，预算按份额切分（防多智能体跑飞）。"""
        return replace(
            self, budget_tokens=self.remaining // max(share, 1), spent_tokens=0, **kw
        )

    def assert_owns(self, row_tenant_id: str | None) -> None:
        """读回数据后的二次校验：存储层 WHERE 之外的纵深防御。"""
        if row_tenant_id != self.tenant_id:
            raise PermissionError(
                f"tenant mismatch: expected {self.tenant_id}, got {row_tenant_id}"
            )
