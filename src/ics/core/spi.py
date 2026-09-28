"""层间接口。

用 ``Protocol`` 而非 ABC：第三方实现无需继承，框架类型不渗透进核心模型。
🔴 只在已有或明确将有第二实现时才定义接口 —— 一个 SPI 只有一个实现 = 净损失。
   「收敛调用点」与「定义抽象接口」是两件事，前者永远该做，后者等第二实现。

下面每个 Protocol 都已有或即将有第二实现：
  LLM          —— LiteLLM 网关 / 未来的自托管推理服务
  Embedder     —— Ollama / OpenAI 兼容端点
  Reranker     —— CrossEncoder / 生成式重排（打分方式不同，必须抽象）
  VectorStore  —— pgvector / 未来的 Milvus（换库时爆炸半径限在一个文件）
  LexicalIndex —— PG tsvector / 进程内 BM25
  DocumentParser —— 每种格式一个实现
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from .context import RequestContext
from .types import Chunk, Content, LLMResponse, ParsedDocument


class LLM(Protocol):
    async def chat(
        self,
        ctx: RequestContext,
        model: str,
        messages: Sequence[Content],
        **kw: object,
    ) -> LLMResponse: ...


class Embedder(Protocol):
    dimensions: int
    max_input_tokens: int                 # ← 暴露它，装配期就能校验块是否超限

    async def embed(
        self,
        ctx: RequestContext,
        texts: list[str],
        kind: Literal["query", "document"] = "document",
    ) -> list[list[float]]: ...
    #      ↑ 非对称模型查询侧要加指令前缀，漏加静默降低召回


class Reranker(Protocol):
    async def rerank(
        self, ctx: RequestContext, query: str, docs: list[str], top_n: int
    ) -> list[tuple[int, float]]: ...


class VectorStore(Protocol):
    async def upsert(
        self, ctx: RequestContext, chunks: list[Chunk], vectors: list[list[float]]
    ) -> int: ...
    async def search(
        self, ctx: RequestContext, vector: list[float], top_k: int, filters: dict
    ) -> list[Chunk]: ...
    async def fetch(self, ctx: RequestContext, ids: Sequence[str]) -> dict[str, Chunk]: ...
    async def delete_doc(self, ctx: RequestContext, doc_id: str) -> int: ...


class LexicalIndex(Protocol):
    """词法检索。只返回有序 id —— 分数不参与 RRF，故无需统一量纲。"""

    async def search(
        self, ctx: RequestContext, query: str, top_k: int, filters: dict
    ) -> list[str]: ...


class DocumentParser(Protocol):
    def parse(self, raw: bytes, filename: str = "", content_type: str = "") -> ParsedDocument: ...
    # ↑ 故意是同步：诚实反映它 CPU 密集，逼调用方显式 to_thread
