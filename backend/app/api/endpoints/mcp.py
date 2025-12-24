"""
MCP Tool Endpoints - Handles MCP tool listing and execution.

Bounded Context: Tool/Integration Domain
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.api.schemas import ToolExecuteRequest, ToolInfo
from app.infrastructure.factory import get_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["MCP Tools"])


@router.get("/tools", response_model=List[ToolInfo])
async def list_mcp_tools():
    """
    List all available MCP tools.
    """
    registry = get_registry()

    tools = registry.get("tool_registry").get_all()
    return [
        ToolInfo(
            name=tool.name,
            description=tool.description,
            parameters=[p.model_dump() for p in tool.parameters],
        )
        for tool in tools
    ]


@router.post("/tools/{tool_name}/execute")
async def execute_mcp_tool(tool_name: str, request: ToolExecuteRequest):
    """
    Execute a specific MCP tool.
    """
    registry = get_registry()

    tool = registry.get("tool_registry").get(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    try:
        result = await registry.get("tool_registry").execute(tool_name, **request.params)
        return result
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/{tool_name}")
async def get_tool_info(tool_name: str):
    """
    Get detailed information about a specific tool.
    """
    registry = get_registry()

    tool = registry.get("tool_registry").get(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": [p.model_dump() for p in tool.parameters],
        "schema": tool.to_schema(),
    }