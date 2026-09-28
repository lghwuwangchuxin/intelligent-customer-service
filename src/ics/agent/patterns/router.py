"""按能力描述分派到某个 Agent。

单 Agent 与多 Agent 是**同一套机制**：单 Agent = 指定 id 直接跑；
多 Agent = 先分派再跑。不需要两套代码、两个服务、两个端口。
"""
from __future__ import annotations

import asyncio

from ...core.context import RequestContext
from ...core.telemetry import get_logger
from ...core.types import Content
from ..prompts import PromptLibrary
from ..schema import AgentConfig

logger = get_logger(__name__)

FALLBACK = "general"


async def route(
    ctx: RequestContext,
    question: str,
    candidates: list[AgentConfig],
    llm: object,
    prompts: PromptLibrary,
    *,
    prompt_ref: str = "router.v1",
    model: str = "chat-cheap",
) -> str:
    """返回选中的 agent id。路由失败回落 general —— 分派失败不该让整个请求失败。"""
    if not candidates:
        return FALLBACK
    if len(candidates) == 1:
        return candidates[0].id

    catalog = "\n".join(f"- {c.id}: {c.description}" for c in candidates)
    text = prompts.render(prompt_ref, catalog=catalog, question=question)
    valid = {c.id for c in candidates}
    try:
        resp = await llm.chat(  # type: ignore[attr-defined]
            ctx, model, [Content.text("user", text)], max_tokens=32, temperature=0.0
        )
        picked = resp.content.strip().split()[0].strip(".,:：") if resp.content.strip() else ""
        if picked in valid:
            return picked
        logger.info("路由返回未知 agent %r，回落 %s", picked, FALLBACK)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("路由失败，回落 %s: %s", FALLBACK, exc)
    return FALLBACK if FALLBACK in valid else candidates[0].id
