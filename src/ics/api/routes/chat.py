"""对话入口。只做协议转换。"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...agent.patterns.router import route
from ...core.telemetry import get_logger
from ..deps import Container, Ctx

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    agent_id: str | None = None      # None = 自动分派
    conversation_id: str | None = None
    kb_id: str = "default"


def _thread_id(tenant_id: str, conversation_id: str) -> str:
    # 🔴 必须带租户前缀：不同租户生成相同 conversation_id 时会读到彼此的历史
    return f"{tenant_id}:{conversation_id}"


async def _index_conversation(c: Any, ctx: Any, conversation_id: str, req: ChatRequest, agent_id: str) -> None:
    """更新会话索引。索引失败不能影响对话本身。"""
    try:
        await c.conversations.touch(
            ctx, conversation_id=conversation_id, first_message=req.message, agent_id=agent_id
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("会话索引更新失败 conversation=%s: %s", conversation_id, exc)


async def _pick_agent(c: Any, ctx: Any, req: ChatRequest) -> Any:
    if req.agent_id:
        return c.agents.get(req.agent_id)
    candidates = c.agents.list(ctx.tenant_id)
    chosen = await route(ctx, req.message, candidates, c.llm, c.prompts)
    return c.agents.get(chosen)


@router.get("/agents")
async def list_agents(ctx: Ctx, c: Container) -> dict[str, Any]:
    return {
        "agents": [
            {
                "id": a.id, "display_name": a.display_name, "description": a.description,
                "pattern": a.pattern, "tools": a.tools,
                "requires_approval": a.guard.hitl_tools,
            }
            for a in c.agents.list(ctx.tenant_id)   # type: ignore[attr-defined]
        ]
    }


@router.post("")
async def chat(req: ChatRequest, ctx: Ctx, c: Container) -> dict[str, Any]:
    cfg = await _pick_agent(c, ctx, req)
    conversation_id = req.conversation_id or str(uuid.uuid4())
    graph = c.engine.build(cfg)                     # type: ignore[attr-defined]
    config = {"configurable": {"thread_id": _thread_id(ctx.tenant_id, conversation_id)}}

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config, context=ctx,
    )
    content = ""
    tool_calls: list[dict[str, Any]] = []
    for msg in result.get("messages", []):
        if calls := getattr(msg, "tool_calls", None):
            tool_calls.extend({"name": x.get("name"), "args": x.get("args")} for x in calls)
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

    await _index_conversation(c, ctx, conversation_id, req, cfg.id)

    return {
        "content": content, "agent_id": cfg.id, "conversation_id": conversation_id,
        "tool_calls": tool_calls, "interrupted": bool(result.get("__interrupt__")),
    }


@router.post("/stream")
async def chat_stream(req: ChatRequest, ctx: Ctx, c: Container) -> StreamingResponse:
    cfg = await _pick_agent(c, ctx, req)
    conversation_id = req.conversation_id or str(uuid.uuid4())
    graph = c.engine.build(cfg)                     # type: ignore[attr-defined]
    config = {"configurable": {"thread_id": _thread_id(ctx.tenant_id, conversation_id)}}

    # 🔴 在开流**之前**写索引，而不是流结束后：
    #    用户中途点「停止」时流是被取消的，放在结束后写会整个跳过，
    #    结果是从界面发起的会话永远不出现在侧栏（实测：checkpointer 有 6 条
    #    thread，索引表只有 1 条——那唯一一条还是非流式接口写的）。
    #    会话在用户按下发送那一刻就已存在，提前写符合语义。
    await _index_conversation(c, ctx, conversation_id, req, cfg.id)

    async def events():
        def emit(payload: dict[str, Any]) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield emit({"type": "agent", "agent_id": cfg.id, "conversation_id": conversation_id})
        try:
            async for mode, chunk in graph.astream(
                {"messages": [{"role": "user", "content": req.message}]},
                config=config, context=ctx, stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    msg, _ = chunk
                    if getattr(msg, "content", None):
                        yield emit({"type": "token", "content": msg.content})
                elif mode == "updates":
                    for node in chunk:
                        yield emit({"type": "node", "node": node})
        except asyncio.CancelledError:
            # 🔴 客户端断连：重抛让任务真正结束，否则后台继续烧 token
            logger.info("流式会话被取消 conversation=%s", conversation_id)
            raise
        yield emit({"type": "done"})

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/conversations")
async def list_conversations(ctx: Ctx, c: Container) -> dict[str, Any]:
    items = await c.conversations.list(ctx)         # type: ignore[attr-defined]
    return {
        "conversations": [
            {"conversation_id": i.conversation_id, "title": i.title, "agent_id": i.agent_id,
             "message_count": i.message_count, "updated_at": i.updated_at.isoformat()}
            for i in items
        ]
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, ctx: Ctx, c: Container) -> dict[str, Any]:
    config = {"configurable": {"thread_id": _thread_id(ctx.tenant_id, conversation_id)}}
    snapshot = await c.checkpointer.aget_tuple(config)   # type: ignore[attr-defined]
    if snapshot is None:
        return {"conversation_id": conversation_id, "messages": []}
    messages = snapshot.checkpoint.get("channel_values", {}).get("messages", [])
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": getattr(m, "type", "unknown"), "content": getattr(m, "content", "")}
            for m in messages
        ],
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, ctx: Ctx, c: Container) -> dict[str, Any]:
    removed = await c.conversations.delete(ctx, conversation_id)   # type: ignore[attr-defined]
    delete_thread = getattr(c.checkpointer, "adelete_thread", None)  # type: ignore[attr-defined]
    if delete_thread is not None:
        try:
            await delete_thread(_thread_id(ctx.tenant_id, conversation_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("删除会话图状态失败: %s", exc)
    return {"deleted": removed}
