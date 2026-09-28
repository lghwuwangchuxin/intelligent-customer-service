"""模型别名表。

🔴 上层代码只认别名。写死厂商模型名会让"换厂商"变成全仓库改动，
   而且旧别名被厂商静默重定向时（deepseek-chat → deepseek-flash），
   连成本统计都会错。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from ..core.errors import ModelUnavailable


@dataclass(frozen=True, slots=True)
class ModelSpec:
    alias: str
    provider_model: str        # LiteLLM 格式：<provider>/<model>
    kind: Literal["chat", "embed", "rerank"]
    context_window: int
    max_output: int
    # 🔴 推理模型会把思维链算进输出预算，设小会导致 content 为空且 finish_reason=length
    reasoning: bool = False
    dimensions: int = 0        # 仅 embed
    max_input_tokens: int = 0  # 仅 embed；装配期校验块是否超限


_DEFAULT: dict[str, ModelSpec] = {
    "chat-main": ModelSpec(
        alias="chat-main",
        provider_model=os.getenv("ICS_MODEL_CHAT_MAIN", "ollama/qwen3"),
        kind="chat", context_window=32_768, max_output=4096,
    ),
    "chat-cheap": ModelSpec(
        alias="chat-cheap",
        provider_model=os.getenv("ICS_MODEL_CHAT_CHEAP", "ollama/qwen3"),
        kind="chat", context_window=32_768, max_output=2048,
    ),
    "embed-main": ModelSpec(
        alias="embed-main",
        provider_model=os.getenv("ICS_MODEL_EMBED_MAIN", "bge-m3"),
        kind="embed", context_window=8192, max_output=0,
        dimensions=1024,
        # 🔴 bge-m3 是 8192；bge-small-zh-v1.5 只有 512，换模型必须同步改这里
        max_input_tokens=8192,
    ),
    "rerank-main": ModelSpec(
        alias="rerank-main",
        provider_model=os.getenv("ICS_MODEL_RERANK_MAIN", "BAAI/bge-reranker-v2-m3"),
        kind="rerank", context_window=8192, max_output=0,
    ),
}


class ModelRegistry:
    def __init__(self, specs: dict[str, ModelSpec] | None = None) -> None:
        self._specs = dict(specs or _DEFAULT)

    def resolve(self, alias: str) -> ModelSpec:
        try:
            return self._specs[alias]
        except KeyError:
            raise ModelUnavailable(
                f"未登记的模型别名: {alias}", available=sorted(self._specs)
            ) from None

    def register(self, spec: ModelSpec) -> None:
        self._specs[spec.alias] = spec

    def aliases(self, kind: str | None = None) -> list[str]:
        return sorted(a for a, s in self._specs.items() if kind is None or s.kind == kind)


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
