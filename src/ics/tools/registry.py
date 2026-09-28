"""工具注册表：名字 → 可调用 + 元数据（是否写操作、租户白名单）。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..core.context import RequestContext
from ..core.errors import ToolFailed

Handler = Callable[[RequestContext, dict[str, Any]], Awaitable[str]]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    # 🔴 写操作必须标出来：guard 层据此挂人工闸门
    is_write: bool = False
    allowed_tenants: set[str] | None = None   # None = 所有租户


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolFailed(f"未注册的工具: {name}", available=sorted(self._tools)) from None

    def allowed(self, tenant_id: str, name: str) -> bool:
        spec = self._tools.get(name)
        if spec is None:
            return False
        return spec.allowed_tenants is None or tenant_id in spec.allowed_tenants

    def is_write(self, name: str) -> bool:
        spec = self._tools.get(name)
        return bool(spec and spec.is_write)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def to_openai_schema(self, names: list[str]) -> list[dict[str, Any]]:
        out = []
        for n in names:
            s = self.get(n)
            out.append({
                "type": "function",
                "function": {"name": s.name, "description": s.description,
                             "parameters": s.parameters},
            })
        return out


_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
