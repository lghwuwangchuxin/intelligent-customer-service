"""Agent 配置模型。

🔑 **这份模型就是「Agent 的全部」**。代码里不存在任何一个具体 Agent。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievalCfg(BaseModel):
    knowledge_ids: list[str] = Field(default_factory=list)
    rewrite: Literal["none", "simple", "hyde", "multi"] = "simple"
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_k: int = 60
    rerank_top_n: int = 5
    max_context_tokens: int = 4000


class GuardCfg(BaseModel):
    # 内置 email；cn_mobile / cn_id_card 是 guard.pii 提供的中文场景规则
    pii_types: list[str] = Field(
        default_factory=lambda: ["email", "cn_mobile", "cn_id_card"]
    )
    hitl_tools: list[str] = Field(default_factory=list)   # 需人工确认的工具
    max_model_calls: int = 6      # 🔴 ReAct 无收敛保证，必须限步
    max_tool_calls: int = 8


class AgentConfig(BaseModel):
    id: str
    version: int = 1
    tenant_id: str = "*"                 # "*" = 所有租户可用
    display_name: str = ""
    description: str = ""
    pattern: Literal["simple", "react", "plan_execute"] = "react"

    model: str = "chat-main"             # ← 别名，永不写厂商模型名
    cheap_model: str = "chat-cheap"
    prompt_ref: str = ""                 # ← 指向 prompts/，不内联文本
    temperature: float = 0.0
    max_tokens: int = 2048               # 🔴 推理模型至少 1024，否则 content 为空

    tools: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    retrieval: RetrievalCfg | None = None
    memory: Literal["none", "window", "summary", "vector"] = "window"
    guard: GuardCfg = Field(default_factory=GuardCfg)
    budget_tokens: int = 50_000

    @property
    def key(self) -> tuple[str, int]:
        return (self.id, self.version)
