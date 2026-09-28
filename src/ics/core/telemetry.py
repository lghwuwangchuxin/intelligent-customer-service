"""统一埋点。属性名走 OTel GenAI 语义约定，换后端不用重埋点。

🔴 span 属性里不放完整 prompt / 响应正文 —— 体积大且含敏感信息。
   正文走可脱敏、可设保留期的独立通道。
"""
from __future__ import annotations

import logging
import sys
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from opentelemetry import trace

from .config import settings
from .context import RequestContext

tracer = trace.get_tracer("ics")

_CONFIGURED = False
_TRACING_READY = False


def setup_tracing() -> bool:
    """按配置装 OTLP 导出器。

    🔴 绑 OTel 而不是某个厂商 SDK：Langfuse / Phoenix / Jaeger / Tempo 都接 OTLP，
       换后端只改 endpoint，埋点代码一行不动——这就是「可观测的爆炸半径≈0」。
    """
    global _TRACING_READY
    if _TRACING_READY or not settings.otel_enabled:
        return _TRACING_READY
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        headers = dict(
            kv.split("=", 1)
            for kv in settings.otel_headers.get_secret_value().split(",")
            if "=" in kv
        )
        provider = TracerProvider(resource=Resource.create({"service.name": "ics"}))
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_endpoint, headers=headers or None)
            )
        )
        trace.set_tracer_provider(provider)
        _TRACING_READY = True
        get_logger(__name__).info("OTel 追踪已启用 endpoint=%s", settings.otel_endpoint)
    except Exception as exc:
        # 可观测不可用绝不能拖垮业务
        get_logger(__name__).warning("OTel 初始化失败，追踪保持关闭: %s", exc)
    return _TRACING_READY


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("tenant_id", "trace_id"):
            if (v := getattr(record, key, None)) is not None:
                payload[key] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _JsonFormatter()
        if settings.log_json
        else logging.Formatter("%(asctime)s %(levelname)-5s %(name)s | %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level.upper())
    for noisy in ("httpx", "httpcore", "LiteLLM", "urllib3", "opentelemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


@contextmanager
def span(name: str, ctx: RequestContext | None = None, **attrs: Any) -> Iterator[Any]:
    with tracer.start_as_current_span(name) as sp:
        if ctx is not None:
            sp.set_attribute("tenant.id", ctx.tenant_id)
            sp.set_attribute("trace.id", ctx.trace_id)
        for k, v in attrs.items():
            if v is not None:
                sp.set_attribute(k, v)
        try:
            yield sp
        except Exception as exc:
            sp.set_attribute("error.type", type(exc).__name__)
            raise


def traced(name: str, kind: str = "span") -> Callable[..., Any]:
    """装饰 ``async def fn(ctx, ...)``，自动开 span 并带上租户与 trace。"""

    def deco(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(fn)
        async def wrapper(ctx: RequestContext, *a: Any, **kw: Any) -> Any:
            with span(name, ctx, **{"gen_ai.operation.name": kind}):
                return await fn(ctx, *a, **kw)

        return wrapper

    return deco


def model_attributes(
    *,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    """OTel GenAI 语义约定属性。推理与缓存 token 是约定外但对成本关键的维度。"""
    attrs: dict[str, Any] = {
        "gen_ai.system": "ics",
        "gen_ai.request.model": model,
        "gen_ai.usage.input_tokens": prompt_tokens,
        "gen_ai.usage.output_tokens": completion_tokens,
    }
    if reasoning_tokens:
        attrs["ics.gen_ai.usage.reasoning_tokens"] = reasoning_tokens
    if cached_tokens:
        attrs["ics.gen_ai.usage.cached_tokens"] = cached_tokens
    return attrs


# ── 指标 ───────────────────────────────────────────────────
# 🔴 本领域三个最易漏的告警，都不报错：降级触发次数 / 缓存命中率突降 / 召回率基线下降
class _Metrics:
    def __init__(self) -> None:
        from prometheus_client import Counter, Histogram

        self.model_calls = Counter(
            "ics_model_calls_total", "模型调用", ["tenant", "model", "outcome"]
        )
        self.tokens = Counter(
            "ics_model_tokens_total", "token 用量", ["tenant", "model", "kind"]
        )
        self.cost = Counter("ics_model_cost_usd_total", "成本(USD)", ["tenant", "model"])
        self.latency = Histogram("ics_operation_seconds", "耗时", ["operation", "outcome"])
        self.cache = Counter("ics_cache_total", "缓存", ["namespace", "result"])
        # 静默劣化三件套
        self.degraded = Counter("ics_degraded_total", "降级触发", ["component", "reason"])
        self.recall = Histogram("ics_recall_ratio", "召回率基线", ["tenant", "kb"])

    @contextmanager
    def timed(self, operation: str) -> Iterator[None]:
        start = time.monotonic()
        outcome = "ok"
        try:
            yield
        except Exception:
            outcome = "error"
            raise
        finally:
            self.latency.labels(operation, outcome).observe(time.monotonic() - start)


_metrics: _Metrics | None = None


def metrics() -> _Metrics:
    global _metrics
    if _metrics is None:
        _metrics = _Metrics()
    return _metrics
