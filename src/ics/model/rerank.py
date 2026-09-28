"""重排。实现 core.spi.Reranker。

🔴 选型必须先看 HF 仓库 ``config.json`` 的 ``architectures``：
   XLMRobertaForSequenceClassification → 标准 CrossEncoder，可直接用
   Qwen3ForCausalLM                    → 生成式，当 CrossEncoder 加载不报错，
                                         但得到的是无意义分数（纯静默失败）
"""
from __future__ import annotations

import asyncio

from ..core.config import settings
from ..core.context import RequestContext
from ..core.telemetry import get_logger, span
from .registry import ModelRegistry, get_registry

logger = get_logger(__name__)

CROSS_ENCODER_ARCHS = frozenset({
    "XLMRobertaForSequenceClassification",
    "BertForSequenceClassification",
    "RobertaForSequenceClassification",
})


class CrossEncoderReranker:
    def __init__(self, alias: str = "rerank-main", registry: ModelRegistry | None = None) -> None:
        self._model_name = (registry or get_registry()).resolve(alias).provider_model
        self._model: object | None = None
        # ← 本地模型不是线程安全的，限并发
        self._sem = asyncio.Semaphore(settings.rerank_concurrency)

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder
        from transformers import AutoConfig

        archs = set(getattr(AutoConfig.from_pretrained(self._model_name), "architectures", None) or [])
        if archs and not (archs & CROSS_ENCODER_ARCHS):
            raise ValueError(
                f"{self._model_name} 的 architectures={sorted(archs)} 不是序列分类模型；"
                "当 CrossEncoder 加载会得到无意义分数"
            )
        self._model = CrossEncoder(self._model_name)
        return self._model

    async def rerank(
        self, ctx: RequestContext, query: str, docs: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        if not docs:
            return []

        def _score() -> list[float]:
            model = self._load()
            return [float(x) for x in model.predict([(query, d) for d in docs])]  # type: ignore[attr-defined]

        async with self._sem:
            with span("gen_ai.rerank", ctx, **{"gen_ai.request.model": self._model_name}):
                # 🔴 推理是 CPU/GPU 密集，必须挪出事件循环
                scores = await asyncio.to_thread(_score)
        return sorted(enumerate(scores), key=lambda kv: -kv[1])[:top_n]
