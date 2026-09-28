"""MCP 客户端。**全仓库唯一允许 import mcp SDK 的模块。**

🔴 MCP 规范已到 2026-07-28，与 2025 年教程差异极大：
   - ``initialize`` 握手整个移除，协议无状态；能力由每请求 ``_meta`` 自带
   - 新增 ``server/discover``（服务端 MUST 实现）
   - ``Mcp-Session-Id`` 与 SSE Last-Event-ID 续传移除
   - Roots / Sampling / Logging 整体弃用

🔴 SDK（mcp 2.2.0）实测：
   - ``mcp.server.fastmcp.FastMCP`` 已改名 → ``mcp.server.mcpserver.MCPServer``
   - 客户端函数是 ``streamable_http_client``，且**只 yield 2 个值**（旧版 3 个）
   - ``Tool.inputSchema`` 已改名 → ``input_schema``（两侧都改了）
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from ..core.context import RequestContext
from ..core.errors import ToolFailed, UpstreamUnavailable
from ..core.telemetry import get_logger, span

logger = get_logger(__name__)


@dataclass(slots=True)
class RemoteTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    server: str


class MCPClient:
    def __init__(self, url: str, name: str = "") -> None:
        self._url = url
        self._name = name or url
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    async def connect(self) -> None:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        self._stack = AsyncExitStack()
        try:
            # 🔴 只解两个值
            read, write = await self._stack.enter_async_context(streamable_http_client(self._url))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.discover()   # 🔴 取代已移除的 initialize
            logger.info("已连接 MCP Server %s", self._url)
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception as exc:
            await self.aclose()
            raise UpstreamUnavailable(f"连接 MCP Server 失败: {exc}", url=self._url) from exc

    async def list_tools(self) -> list[RemoteTool]:
        if self._session is None:
            raise UpstreamUnavailable("MCP 客户端未连接", url=self._url)
        result = await self._session.list_tools()
        return [
            RemoteTool(
                name=t.name, description=t.description or "",
                input_schema=t.input_schema, server=self._name,
            )
            for t in result.tools
        ]

    async def call(self, ctx: RequestContext, name: str, args: dict[str, Any]) -> str:
        if self._session is None:
            raise UpstreamUnavailable("MCP 客户端未连接", url=self._url)
        try:
            with span("tool.mcp_call", ctx, **{"tool.name": name, "tool.server": self._name}):
                result = await self._session.call_tool(name, args)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ToolFailed(f"MCP 工具 {name} 调用失败: {exc}", tool=name) from exc
        return "\n".join(
            p for p in (getattr(c, "text", "") for c in getattr(result, "content", [])) if p
        )

    async def aclose(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None
