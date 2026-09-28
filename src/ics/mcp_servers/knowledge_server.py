"""知识库 MCP Server。

🔴 ``MCPServer`` 是 mcp 2.x 的名字（1.x 叫 ``FastMCP``，已改名）。
🔴 生产用 ``stateless_http=True``：协议自 2026-07-28 起本身无状态，
   有状态模式在多副本部署下会出问题。
🔴 MCP 跨进程边界：租户必须是**显式入参**，不能指望调用方上下文传过来。
"""
from __future__ import annotations

from typing import Any

from ..core.context import RequestContext
from ..core.telemetry import get_logger

logger = get_logger(__name__)


def build_server(retrieval: Any, name: str = "ics-knowledge") -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name=name, title="智能客服知识库",
        description="充电运营平台对外暴露的知识检索能力", version="3.0.0",
    )

    @server.tool(
        name="search_knowledge", title="知识库检索",
        description="在充电运营平台的知识库中检索资料，返回带来源标注的片段。",
    )
    async def search_knowledge(query: str, tenant_id: str, kb_id: str = "default") -> str:
        from ..rag.generate.assemble import assemble

        ctx = RequestContext(tenant_id=tenant_id)
        result = await retrieval.retrieve(ctx, query, kb_id=kb_id)
        return assemble(result.chunks)

    logger.info("MCP Server 已构建 name=%s", name)
    return server
