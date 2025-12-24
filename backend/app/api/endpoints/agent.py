"""
Agent Endpoints - Handles ReAct and LangGraph agent requests.

Bounded Context: Agent Domain
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agent import LANGGRAPH_AVAILABLE
from app.api.schemas import (
    AgentCapabilities,
    AgentChatRequest,
    AgentChatResponse,
    LangGraphChatRequest,
    LangGraphChatResponse,
)
from app.infrastructure.factory import get_registry
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["Agent"])


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """
    Chat with the ReAct agent.

    The agent can use tools to search knowledge base, browse web, execute code, etc.
    """
    logger.info(
        f"[API] /agent/chat - message: {request.message[:50]}..., "
        f"conversation_id: {request.conversation_id}, stream: {request.stream}"
    )

    if not settings.AGENT_ENABLED:
        logger.warning("[API] Agent mode is disabled")
        raise HTTPException(status_code=403, detail="Agent mode is disabled")

    registry = get_registry()

    try:
        if request.stream:
            async def generate():
                try:
                    history = [
                        {"role": m.role, "content": m.content}
                        for m in (request.history or [])
                    ]
                    async for event in registry.get("agent").stream(
                        question=request.message,
                        conversation_id=request.conversation_id,
                        history=history,
                    ):
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )
        else:
            logger.info("[API] Agent processing request")
            history = [
                {"role": m.role, "content": m.content}
                for m in (request.history or [])
            ]
            result = await registry.get("agent").run(
                question=request.message,
                conversation_id=request.conversation_id,
                history=history,
            )

            logger.info(
                f"[API] Agent completed - iterations: {result.get('iterations', 0)}, "
                f"tool_calls: {len(result.get('tool_calls', []))}, "
                f"response_length: {len(result.get('response', ''))}"
            )
            return AgentChatResponse(
                response=result["response"],
                conversation_id=result.get("conversation_id"),
                thoughts=result.get("thoughts"),
                tool_calls=result.get("tool_calls"),
                iterations=result.get("iterations", 0),
                error=result.get("error"),
            )

    except Exception as e:
        logger.error(f"[API] Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def agent_chat_stream(request: AgentChatRequest):
    """
    Stream agent response with intermediate steps (thoughts, actions, observations).
    """
    if not settings.AGENT_ENABLED:
        raise HTTPException(status_code=403, detail="Agent mode is disabled")

    registry = get_registry()

    async def generate():
        try:
            history = [
                {"role": m.role, "content": m.content}
                for m in (request.history or [])
            ]
            async for event in registry.get("agent").stream(
                question=request.message,
                conversation_id=request.conversation_id,
                history=history,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@router.get("/memory/{conversation_id}")
async def get_agent_memory(conversation_id: str):
    """
    Get agent memory for a conversation.
    """
    registry = get_registry()

    memory = registry.get("memory_manager").get_memory(conversation_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conversation_id,
        "messages": memory.get_recent_messages(20),
        "summary": memory.summary,
        "message_count": len(memory.messages),
    }


@router.delete("/memory/{conversation_id}")
async def clear_agent_memory(conversation_id: str):
    """
    Clear agent memory for a conversation.
    """
    registry = get_registry()

    success = registry.get("memory_manager").delete_memory(conversation_id)
    return {"success": success, "conversation_id": conversation_id}


# ============== LangGraph Agent Endpoints ==============

@router.get("/capabilities", response_model=AgentCapabilities)
async def get_agent_capabilities():
    """
    Get agent capabilities and status.
    """
    registry = get_registry()

    return AgentCapabilities(
        react_agent=registry.get("agent") is not None,
        langgraph_agent=registry.get("langgraph_agent") is not None,
        planning_enabled=LANGGRAPH_AVAILABLE and registry.get("langgraph_agent") is not None,
        parallel_tools_enabled=LANGGRAPH_AVAILABLE and registry.get("langgraph_agent") is not None,
        error_recovery_enabled=LANGGRAPH_AVAILABLE and registry.get("langgraph_agent") is not None,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
        available_tools=len(registry.get("tool_registry").get_all()),
    )


@router.post("/langgraph/chat", response_model=LangGraphChatResponse)
async def langgraph_agent_chat(request: LangGraphChatRequest):
    """
    Chat with the LangGraph agent.

    The LangGraph agent provides advanced features:
    - Task planning for complex queries
    - Parallel tool execution
    - Intelligent error recovery
    - State-based workflow management
    """
    if not settings.AGENT_ENABLED:
        raise HTTPException(status_code=403, detail="Agent mode is disabled")

    registry = get_registry()
    langgraph_agent = registry.get("langgraph_agent")

    if not langgraph_agent:
        # Fallback to ReAct agent
        logger.warning("LangGraph agent not available, falling back to ReAct agent")
        history = [
            {"role": m.role, "content": m.content}
            for m in (request.history or [])
        ]
        result = await registry.get("agent").run(
            question=request.message,
            conversation_id=request.conversation_id,
            history=history,
        )
        return LangGraphChatResponse(
            response=result["response"],
            conversation_id=result.get("conversation_id"),
            thoughts=result.get("thoughts"),
            tool_calls=result.get("tool_calls"),
            iterations=result.get("iterations", 0),
            error=result.get("error"),
        )

    try:
        if request.stream:
            async def generate():
                try:
                    history = [
                        {"role": m.role, "content": m.content}
                        for m in (request.history or [])
                    ]
                    async for event in langgraph_agent.stream(
                        question=request.message,
                        conversation_id=request.conversation_id,
                        history=history,
                    ):
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )
        else:
            history = [
                {"role": m.role, "content": m.content}
                for m in (request.history or [])
            ]
            result = await langgraph_agent.run(
                question=request.message,
                conversation_id=request.conversation_id,
                history=history,
            )

            return LangGraphChatResponse(
                response=result.get("response", ""),
                conversation_id=result.get("conversation_id"),
                plan=result.get("plan"),
                thoughts=result.get("thoughts"),
                tool_calls=result.get("tool_calls"),
                iterations=result.get("iterations", 0),
                parallel_executions=result.get("parallel_executions", 0),
                error_recoveries=result.get("error_recoveries", 0),
                error=result.get("error"),
            )

    except Exception as e:
        logger.error(f"LangGraph agent chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/langgraph/chat/stream")
async def langgraph_agent_chat_stream(request: LangGraphChatRequest):
    """
    Stream LangGraph agent response with intermediate steps.
    """
    if not settings.AGENT_ENABLED:
        raise HTTPException(status_code=403, detail="Agent mode is disabled")

    registry = get_registry()
    langgraph_agent = registry.get("langgraph_agent")

    if not langgraph_agent:
        raise HTTPException(
            status_code=503,
            detail="LangGraph agent is not available. Use /agent/chat/stream instead."
        )

    async def generate():
        try:
            history = [
                {"role": m.role, "content": m.content}
                for m in (request.history or [])
            ]
            async for event in langgraph_agent.stream(
                question=request.message,
                conversation_id=request.conversation_id,
                history=history,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"LangGraph stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )