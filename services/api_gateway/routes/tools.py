"""Tools/MCP routes for API Gateway."""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/tools", tags=["tools"])


class ExecuteToolRequest(BaseModel):
    """Execute tool request."""
    tool_name: str
    arguments: Dict[str, Any]
    timeout_seconds: Optional[int] = None


@router.get("")
async def list_tools(
    request: Request,
    tags: Optional[str] = None,
    name_pattern: Optional[str] = None,
):
    """
    List available tools.
    """
    client = request.app.state.mcp_client

    if not client:
        raise HTTPException(status_code=503, detail="MCP service unavailable")

    try:
        tag_list = tags.split(",") if tags else None
        tools = await client.list_tools(
            tags=tag_list,
            name_pattern=name_pattern,
        )
        return {"tools": tools, "total": len(tools)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{tool_name}")
async def get_tool_schema(request: Request, tool_name: str):
    """
    Get tool schema.
    """
    client = request.app.state.mcp_client

    if not client:
        raise HTTPException(status_code=503, detail="MCP service unavailable")

    try:
        schema = await client.get_tool_schema(tool_name)
        if not schema:
            raise HTTPException(status_code=404, detail="Tool not found")
        return schema

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
async def execute_tool(request: Request, body: ExecuteToolRequest):
    """
    Execute a tool.
    """
    client = request.app.state.mcp_client

    if not client:
        raise HTTPException(status_code=503, detail="MCP service unavailable")

    try:
        result = await client.execute_tool(
            tool_name=body.tool_name,
            arguments=body.arguments,
            timeout_seconds=body.timeout_seconds,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
