"""全局内容模型。🔴 必须在写任何业务代码前定死。

用带 discriminator 的联合类型而不是裸 ``str``：
接多模态时只需加一个 Part 子类，不必改遍所有调用点。
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ── 多模态内容 ─────────────────────────────────────────────


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    url: str | None = None
    b64: str | None = None
    mime: str = "image/png"


class ToolResultPart(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    is_error: bool = False


Part = Annotated[TextPart | ImagePart | ToolResultPart, Field(discriminator="type")]

Role = Literal["system", "user", "assistant", "tool"]


class Content(BaseModel):
    """消息体。🔴 不要用 str —— 接多模态时会推倒重来。"""

    role: Role
    parts: list[Part]

    @classmethod
    def text(cls, role: Role, s: str) -> Content:
        return cls(role=role, parts=[TextPart(text=s)])

    def as_text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    def to_openai(self) -> dict[str, object]:
        """转成厂商侧消息。纯文本回字符串（兼容性最好），否则回 parts 数组。"""
        if all(isinstance(p, TextPart) for p in self.parts):
            return {"role": self.role, "content": self.as_text()}
        blocks: list[dict[str, object]] = []
        for p in self.parts:
            if isinstance(p, TextPart):
                blocks.append({"type": "text", "text": p.text})
            elif isinstance(p, ImagePart):
                url = p.url or f"data:{p.mime};base64,{p.b64}"
                blocks.append({"type": "image_url", "image_url": {"url": url}})
            else:
                # 未知模态退化成文本说明，而不是悄悄丢掉
                blocks.append({"type": "text", "text": p.content})
        return {"role": self.role, "content": blocks}


# ── 检索 ──────────────────────────────────────────────────


class Chunk(BaseModel):
    id: str
    tenant_id: str
    doc_id: str
    chunk_no: int
    content: str
    section_path: str | None = None     # 标题路径：引用定位 + 上下文增强
    parent_id: str | None = None        # 父子块
    source: str | None = None
    score: float | None = None


class RetrievalResult(BaseModel):
    chunks: list[Chunk]
    strategy: str                        # dense / sparse / hybrid
    latency_ms: int = 0
    total_tokens: int = 0                # 供预算裁剪使用


# ── 模型响应 ───────────────────────────────────────────────


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0            # ← 看不见的成本，必须单独统计
    cached_tokens: int = 0               # ← 缓存命中率的数据源
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def billable_input(self) -> int:
        """未命中缓存的输入才按全价计。"""
        return max(0, self.prompt_tokens - self.cached_tokens)


class LLMResponse(BaseModel):
    content: str
    finish_reason: str = "stop"
    usage: Usage = Field(default_factory=Usage)
    model: str = ""                      # ← 实际返回的模型名，用于检测静默重定向
    tool_calls: list[dict] = Field(default_factory=list)

    @property
    def truncated_empty(self) -> bool:
        """🔴 推理模型把预算耗在思维链上时：content 为空且 finish_reason=length"""
        return self.finish_reason == "length" and not self.content.strip()


class ParsedDocument(BaseModel):
    text: str
    format: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()
