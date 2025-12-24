"""
Chat Endpoints - Handles chat and RAG conversation requests.

Bounded Context: Chat/Conversation Domain
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse
from app.infrastructure.factory import get_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a chat message and get a response.

    - **message**: The user's message
    - **use_rag**: If true, uses knowledge base for context
    - **stream**: If true, returns streaming response
    """
    logger.info(
        f"[API] /chat/message - message: {request.message[:50]}..., "
        f"use_rag: {request.use_rag}, stream: {request.stream}"
    )
    registry = get_registry()

    try:
        if request.stream:
            # Return streaming response
            if request.use_rag:
                generator = registry.get("rag").stream_query(request.message)
            else:
                history = [
                    {"role": m.role, "content": m.content}
                    for m in (request.history or [])
                ]
                generator = registry.get("chat").stream_chat(request.message, history)

            return StreamingResponse(
                generator,
                media_type="text/event-stream",
            )
        else:
            # Return regular response
            if request.use_rag:
                logger.info("[API] Using RAG mode")
                response = await registry.get("rag").aquery(request.message)
                sources = registry.get("rag").get_relevant_documents(request.message)
                logger.info(
                    f"[API] RAG response - length: {len(response)}, "
                    f"sources: {len(sources) if sources else 0}"
                )
            else:
                logger.info("[API] Using direct chat mode")
                history = [
                    {"role": m.role, "content": m.content}
                    for m in (request.history or [])
                ]
                response = registry.get("chat").chat(request.message, history)
                sources = None
                logger.info(f"[API] Chat response - length: {len(response)}")

            return ChatResponse(
                response=response,
                conversation_id=request.conversation_id,
                sources=sources,
            )

    except Exception as e:
        logger.error(f"[API] Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_message(request: ChatRequest):
    """
    Stream a chat response.
    """
    registry = get_registry()

    async def generate():
        try:
            if request.use_rag:
                async for chunk in registry.get("rag").astream_query(request.message):
                    yield f"data: {chunk}\n\n"
            else:
                history = [
                    {"role": m.role, "content": m.content}
                    for m in (request.history or [])
                ]
                for chunk in registry.get("chat").stream_chat(request.message, history):
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )