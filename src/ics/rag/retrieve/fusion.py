"""Reciprocal Rank Fusion。

🔴 必须是 RRF 而非加权求和：向量余弦有界 [-1,1]，BM25/ts_rank 无界且随语料漂移。
   加权相加会让量纲大的一路单边支配排序，而且不报错——融合"看起来在工作"，
   实际退化成单路检索。

🔴 也不要给每一路乘权重：RRF 的全部意义是**只用名次、丢弃分数**。
   乘系数等于把量纲问题引回来一半，且让 k 的含义不可解释。
   要调整某路影响力，应调该路召回条数或 k。
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

DEFAULT_RRF_K = 60


def rrf[T: Hashable](
    rankings: Sequence[Iterable[T]], *, k: int = DEFAULT_RRF_K, top_n: int | None = None
) -> list[tuple[T, float]]:
    """把多路**有序**结果融合成一路。只读名次，不读分数。"""
    scores: dict[T, float] = {}
    for ranking in rankings:
        seen: set[T] = set()
        rank = 0
        for item in ranking:
            if item in seen:      # 单路内去重，否则重复项被记两次名次
                continue
            seen.add(item)
            rank += 1
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], repr(kv[0])))
    return ordered[:top_n] if top_n is not None else ordered
