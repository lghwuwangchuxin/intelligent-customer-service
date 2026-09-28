"""模型网关：所有 chat 调用的唯一出口。

换模型厂商 / 换 SDK 时，只有这个文件需要改。
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from ..core.config import settings
from ..core.context import RequestContext
from ..core.errors import (
    BudgetExceeded,
    EmptyGeneration,
    ModelRateLimited,
    ModelUnavailable,
)
from ..core.telemetry import get_logger, metrics, span
from ..core.types import Content, LLMResponse, Usage
from .registry import ModelRegistry, get_registry

logger = get_logger(__name__)


def _extract_usage(resp: Any) -> Usage:
    raw = getattr(resp, "usage", None)
    if raw is None:
        return Usage()

    def g(obj: Any, name: str) -> int:
        if obj is None:
            return 0
        if isinstance(obj, dict):
            return int(obj.get(name) or 0)
        return int(getattr(obj, name, 0) or 0)

    details = getattr(raw, "completion_tokens_details", None)
    if details is None and isinstance(raw, dict):
        details = raw.get("completion_tokens_details")

    usage = Usage(
        prompt_tokens=g(raw, "prompt_tokens"),
        completion_tokens=g(raw, "completion_tokens"),
        reasoning_tokens=g(details, "reasoning_tokens"),
        # 不同厂商字段名不一，两个都试
        cached_tokens=g(raw, "prompt_cache_hit_tokens") or g(raw, "cached_tokens"),
    )
    try:
        from litellm import completion_cost

        usage.cost_usd = float(completion_cost(completion_response=resp) or 0.0)
    except Exception:  # 成本核算失败不该让业务调用失败
        pass
    return usage


class ModelGateway:
    """实现 core.spi.LLM。"""

    def __init__(self, registry: ModelRegistry | None = None, router: Any = None) -> None:
        self._registry = registry or get_registry()
        self._router = router
        self._sem = asyncio.Semaphore(settings.max_concurrency_per_model)

    def _build_router(self) -> Any:
        from litellm import Router

        entries = []
        for alias in self._registry.aliases("chat"):
            spec = self._registry.resolve(alias)
            params: dict[str, Any] = {"model": spec.provider_model}
            if settings.llm_api_base:
                params["api_base"] = settings.llm_api_base
            if key := settings.llm_api_key.get_secret_value():
                params["api_key"] = key
            entries.append({"model_name": alias, "litellm_params": params})
        return Router(
            model_list=entries,
            num_retries=2,
            timeout=settings.request_timeout_s,
        )

    @property
    def router(self) -> Any:
        if self._router is None:
            self._router = self._build_router()
        return self._router

    async def chat(
        self,
        ctx: RequestContext,
        model: str,
        messages: Sequence[Content],
        **kw: Any,
    ) -> LLMResponse:
        spec = self._registry.resolve(model)
        # 🔴 预算在发请求**之前**检查：已经超了就不该再花钱
        if ctx.remaining <= 0:
            raise BudgetExceeded(f"remaining={ctx.remaining}", tenant=ctx.tenant_id)

        max_tokens = int(kw.pop("max_tokens", 0) or spec.max_output)
        if spec.reasoning and max_tokens < 1024:
            # 推理模型预算太小会把 token 全花在思维链上，content 返回空串
            logger.warning("推理模型 %s 的 max_tokens=%d 偏小，可能返回空内容", model, max_tokens)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_openai() for m in messages],
            "max_tokens": max_tokens,
            **kw,
        }

        try:
            async with self._sem:
                with (
                    span("gen_ai.chat", ctx, **{"gen_ai.request.model": model}),
                    metrics().timed("model.chat"),
                ):
                    resp = await self.router.acompletion(**payload)
        except asyncio.CancelledError:
            raise  # 🔴 吞掉会导致任务泄漏 + token 白烧
        except Exception as exc:
            metrics().model_calls.labels(ctx.tenant_id, model, "error").inc()
            text = str(exc).lower()
            if "rate" in text and "limit" in text:
                raise ModelRateLimited(str(exc)[:200], model=model) from exc
            raise ModelUnavailable(str(exc)[:200], model=model) from exc

        choice = resp.choices[0]
        served = getattr(resp, "model", spec.provider_model) or spec.provider_model
        out = LLMResponse(
            content=choice.message.content or "",
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            usage=_extract_usage(resp),
            model=served,
            tool_calls=[
                {"id": c.get("id"), "name": c.get("function", {}).get("name"),
                 "args": c.get("function", {}).get("arguments")}
                for c in (getattr(choice.message, "tool_calls", None) or [])
                if isinstance(c, dict)
            ],
        )

        # 🔴 静默重定向检测：请求 A 实际跑 B 时，连计价都是错的
        want = spec.provider_model.split("/")[-1]
        got = served.split("/")[-1]
        if want and got and want != got:
            logger.warning("模型被重定向: 请求 %s -> 实际 %s（按实际模型计价）", want, got)
            metrics().degraded.labels("model", "alias_redirect").inc()

        u = out.usage
        m = metrics()
        m.model_calls.labels(ctx.tenant_id, model, "ok").inc()
        for kind, n in (
            ("prompt", u.prompt_tokens), ("completion", u.completion_tokens),
            ("reasoning", u.reasoning_tokens), ("cached", u.cached_tokens),
        ):
            if n:
                m.tokens.labels(ctx.tenant_id, model, kind).inc(n)
        if u.cost_usd:
            m.cost.labels(ctx.tenant_id, model).inc(u.cost_usd)

        if out.truncated_empty:
            raise EmptyGeneration(
                f"模型返回空内容且 finish_reason=length（reasoning_tokens={u.reasoning_tokens}），"
                "请调大 max_tokens",
                model=model,
            )
        return out


_gateway: ModelGateway | None = None


def get_gateway() -> ModelGateway:
    global _gateway
    if _gateway is None:
        _gateway = ModelGateway()
    return _gateway
