"""词法（稀疏）检索。实现 core.spi.LexicalIndex。

🔴 本模块存在的唯一理由是一个静默失败：
   中文不分词直接喂给 BM25 / to_tsvector('simple')，整段中文会被当成一个 token，
   结果是**所有文档得分全 0**，混合检索静默退化为纯向量——不报错、不告警。
   实测：5 条中文文档未分词时 BM25 全 0.0；接 jieba 后正确命中 0.528。

索引侧与查询侧必须用**同一个**分词器，否则词形对不上、召回为 0。
"""
from __future__ import annotations

import re
from typing import Any

from ...core.context import RequestContext
from ...core.telemetry import get_logger

logger = get_logger(__name__)

_WORD = re.compile(r"[a-zA-Z0-9_]+")
_CJK = re.compile(r"[一-鿿]")
# jieba 会把标点单独切成 token（中文引号、逗号等）。它们匹配不到有用内容，
# 只会稀释 ts_rank 排序，必须过滤掉。
_MEANINGFUL = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英混排分词。中文走 jieba 搜索引擎模式（召回高于精确模式，适合检索）。"""
    text = text.strip()
    if not text:
        return []
    if not _CJK.search(text):
        return [t.lower() for t in _WORD.findall(text)]
    import jieba

    return [
        t.strip().lower()
        for t in jieba.cut_for_search(text)
        if t.strip() and _MEANINGFUL.search(t)
    ]


def to_tsvector_input(text: str) -> str:
    """索引侧：空格分隔的词串，喂给 to_tsvector('simple', ...)。"""
    return " ".join(tokenize(text))


def to_tsquery_input(text: str) -> str:
    """查询侧：OR 连接的 tsquery 串。

    🔴 不能用 ``plainto_tsquery``——它是 **AND** 语义，要求每个词都命中。
       检索场景里查询词几乎不可能全部出现在同一篇文档中
       （"退款多久能到账" 切出 退款/多久/能/到/账，文档里没有"多久"就整条不命中），
       结果是词法路**恒返回空**、混合检索静默退化为纯向量，且不报任何错。
       实测就踩到了这一条：strategy 一直是 dense 而不是 hybrid。

    每个词单引号包裹并转义，既避免 tsquery 语法报错，也让它只被当作词而非操作符。
    """
    tokens = tokenize(text)
    if not tokens:
        return ""
    return " | ".join("'" + t.replace("'", "''") + "'" for t in tokens)


class PgLexicalIndex:
    """PostgreSQL tsvector。与向量同库，少一个中间件。

    用 ``ts_rank_cd`` 排序。它不是严格 BM25，但 RRF 只取名次不取分数，
    换一个单调的相关性函数不影响融合结果——这正是 RRF 的好处之一。
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def search(
        self, ctx: RequestContext, query: str, top_k: int, filters: dict
    ) -> list[str]:
        tsquery = to_tsquery_input(query)
        if not tsquery:
            return []
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """SELECT id FROM doc_chunk
                    WHERE tenant_id = %(t)s
                      AND (%(kb)s::text IS NULL OR kb_id = %(kb)s)
                      AND lexical_tsv @@ to_tsquery('simple', %(q)s)
                    ORDER BY ts_rank_cd(lexical_tsv, to_tsquery('simple', %(q)s)) DESC
                    LIMIT %(k)s""",
                {"t": ctx.tenant_id, "kb": filters.get("kb_id"), "q": tsquery, "k": top_k},
            )
            return [str(r[0]) for r in await cur.fetchall()]


class MemoryLexicalIndex:
    """进程内 BM25（bm25s）。测试与小型知识库；多副本之间不共享。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._tenants: list[str] = []
        self._kbs: list[str] = []
        self._r: Any = None

    def build(self, records: list[tuple[str, str, str, str]]) -> None:
        """records: (id, tenant_id, kb_id, content)"""
        import bm25s

        self._ids = [r[0] for r in records]
        self._tenants = [r[1] for r in records]
        self._kbs = [r[2] for r in records]
        if not records:
            self._r = None
            return
        r = bm25s.BM25()
        r.index([tokenize(rec[3]) for rec in records])
        self._r = r

    async def search(
        self, ctx: RequestContext, query: str, top_k: int, filters: dict
    ) -> list[str]:
        if self._r is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        idx, scores = self._r.retrieve([tokens], k=min(len(self._ids), max(top_k * 4, top_k)))
        kb = filters.get("kb_id")
        out: list[str] = []
        for pos, score in zip(idx[0], scores[0], strict=True):
            if score <= 0:
                continue
            # 🔴 词法路也必须做租户隔离，不能只靠向量路过滤
            if self._tenants[pos] != ctx.tenant_id:
                continue
            if kb is not None and self._kbs[pos] != kb:
                continue
            out.append(self._ids[pos])
            if len(out) >= top_k:
                break
        return out
