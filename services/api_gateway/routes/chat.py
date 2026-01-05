"""Chat routes for API Gateway."""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request."""
    message: str
    conversation_id: Optional[str] = None
    enable_tools: bool = True
    enable_rag: bool = True
    knowledge_base_id: Optional[str] = None
    use_multi_agent: bool = False


class ChatResponse(BaseModel):
    """Chat response."""
    message: str
    conversation_id: str
    tool_calls: List[dict] = []
    sources: List[str] = []


@router.post("", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """
    Send a chat message.

    Uses single agent or multi-agent based on configuration.
    """
    if body.use_multi_agent:
        client = request.app.state.multi_agent_client
    else:
        client = request.app.state.single_agent_client

    if not client:
        raise HTTPException(status_code=503, detail="Agent service unavailable")

    try:
        response = await client.chat(
            message=body.message,
            conversation_id=body.conversation_id,
            enable_tools=body.enable_tools,
            enable_rag=body.enable_rag,
            knowledge_base_id=body.knowledge_base_id,
        )

        return ChatResponse(
            message=response["message"],
            conversation_id=response["conversation_id"],
            tool_calls=response.get("tool_calls", []),
            sources=response.get("sources", []),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_chat(request: Request, body: ChatRequest):
    """
    Stream chat response.

    Returns Server-Sent Events.
    """
    client = request.app.state.single_agent_client

    if not client:
        raise HTTPException(status_code=503, detail="Agent service unavailable")

    async def generate():
        try:
            async for response in client.stream_chat(
                message=body.message,
                conversation_id=body.conversation_id,
                enable_tools=body.enable_tools,
                enable_rag=body.enable_rag,
            ):
                data = json.dumps({
                    "message": response["message"],
                    "conversation_id": response["conversation_id"],
                    "is_final": response["is_final"],
                })
                yield f"data: {data}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/history/{conversation_id}")
async def get_history(request: Request, conversation_id: str, limit: int = 50):
    """Get conversation history."""
    client = request.app.state.single_agent_client

    if not client:
        raise HTTPException(status_code=503, detail="Agent service unavailable")

    try:
        history = await client.get_history(
            conversation_id=conversation_id,
            limit=limit,
        )
        return {"messages": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
