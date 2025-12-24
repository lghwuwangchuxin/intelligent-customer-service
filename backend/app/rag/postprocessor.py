"""
RAG Post-processor Module / RAG 后处理模块
============================================

提供语义去重和多样性过滤 (MMR) 功能，优化检索结果质量。

性能优化 (v2.0)
--------------
- 使用批量嵌入 (batch embedding) 替代逐个计算
- 复用节点已有的嵌入向量，避免重复计算
- 内置嵌入缓存，减少 API 调用

核心功能
--------
1. **语义去重 (Semantic Deduplication)**
   - 基于嵌入向量的余弦相似度去除重复内容
   - 避免返回相似度过高的冗余文档
   - 贪婪算法保证保留最相关的文档

2. **MMR 多样性过滤 (Maximal Marginal Relevance)**
   - 平衡相关性与多样性
   - 避免返回内容高度相似的文档集合
   - 提升结果覆盖面，增加信息密度

工作原理
--------

### 语义去重算法

```
输入: 节点列表 [N1, N2, N3, ...]，相似度阈值 θ
输出: 去重后的节点列表

1. 计算所有节点的嵌入向量 [E1, E2, E3, ...]
2. 初始化已选集合 S = []
3. 对于每个节点 Ni:
   a. 计算 Ni 与 S 中所有节点的最大相似度
   b. 如果 max_similarity < θ，将 Ni 加入 S
   c. 否则，丢弃 Ni（视为重复）
4. 返回 S
```

相似度计算公式 (余弦相似度):
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

### MMR 算法

MMR (Maximal Marginal Relevance) 目标是选择既相关又多样的文档集合。

核心公式:
```
MMR(Di) = λ × Rel(Di, Q) - (1-λ) × max(Sim(Di, Dj))
                                    Dj∈S
其中:
- Rel(Di, Q): 文档 Di 与查询 Q 的相关性
- Sim(Di, Dj): 文档 Di 与已选文档 Dj 的相似度
- λ: 相关性与多样性的平衡系数 (0~1)
- S: 已选文档集合
```

λ 参数说明:
- λ = 1.0: 纯相关性排序（忽略多样性）
- λ = 0.5: 相关性与多样性平衡（推荐）
- λ = 0.0: 纯多样性排序（忽略相关性）

配置参数
--------
通过 config/settings.py 或 .env 配置:

```python
# 语义去重
RAG_ENABLE_DEDUP = True          # 是否启用去重
RAG_DEDUP_THRESHOLD = 0.95       # 去重阈值 (0.9-0.99)

# 最终结果
RAG_FINAL_TOP_K = 5              # 最终返回文档数
```

使用示例
--------

### 基本用法

```python
from app.rag.postprocessor import RAGPostProcessor

# 创建后处理器
processor = RAGPostProcessor(
    enable_dedup=True,
    dedup_threshold=0.95,
    enable_mmr=True,
    mmr_lambda=0.5,
    final_top_k=5
)

# 处理节点
processed_nodes = processor.process(nodes, query="用户问题")
```

### 单独使用去重器

```python
from app.rag.postprocessor import SemanticDeduplicator

dedup = SemanticDeduplicator(similarity_threshold=0.95)
unique_nodes = dedup.postprocess_nodes(nodes)
```

### 单独使用 MMR

```python
from app.rag.postprocessor import MMRPostprocessor
from llama_index.core.schema import QueryBundle

mmr = MMRPostprocessor(lambda_mult=0.5, top_n=5)
query_bundle = QueryBundle(query_str="用户问题")
diverse_nodes = mmr.postprocess_nodes(nodes, query_bundle)
```

### 获取全局实例

```python
from app.rag.postprocessor import get_post_processor

processor = get_post_processor()
result = processor.process(nodes, query)
```

处理流程
--------
```
检索结果 (10+ 节点)
       ↓
[语义去重] - 移除相似度 > 0.95 的重复文档
       ↓
[MMR 过滤] - 平衡相关性与多样性
       ↓
最终结果 (5 节点)
```

性能考虑
--------
1. **嵌入计算**: 去重和 MMR 都需要计算嵌入向量，可能较慢
2. **缓存优化**: 如果节点已有嵌入，可避免重复计算
3. **批量处理**: 尽量一次性获取所有嵌入，减少 API 调用

阈值调优建议
-----------
| 场景 | 去重阈值 | MMR λ | 说明 |
|------|---------|-------|------|
| 精确问答 | 0.95 | 0.7 | 保留更多相关文档 |
| 知识探索 | 0.90 | 0.5 | 增加结果多样性 |
| 摘要生成 | 0.85 | 0.3 | 避免信息重复 |

参考文献
--------
- Carbonell & Goldstein (1998): "The Use of MMR, Diversity-Based Reranking"
- 语义相似度: https://en.wikipedia.org/wiki/Cosine_similarity

Author: Intelligent Customer Service Team
Version: 2.0.0
"""
import logging
import hashlib
from typing import List, Optional, Dict, Tuple, Any
import numpy as np

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import Settings as LlamaSettings

from config.settings import settings
from app.core.embeddings import get_embedding_manager

logger = logging.getLogger(__name__)


# ============================================================================
# Embedding Cache and Helper Functions (Redis-backed with memory fallback)
# ============================================================================

class EmbeddingCache:
    """
    Redis-backed embedding cache with memory fallback.

    Uses the unified CacheManager for caching embeddings.
    Automatically falls back to memory cache if Redis is unavailable.

    Attributes
    ----------
    max_size : int
        Maximum cache entries (for memory fallback)

    Example
    -------
    ```python
    cache = EmbeddingCache()
    await cache.initialize()

    # Get/set embeddings
    await cache.set("text content", embedding_array)
    embedding = await cache.get("text content")
    ```
    """

    def __init__(self, max_size: int = 1000):
        self._max_size = max_size
        self._cache_manager = None
        self._initialized = False
        # Fallback memory cache for sync operations
        self._memory_cache: Dict[str, np.ndarray] = {}

    def _hash_content(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    async def initialize(self) -> None:
        """Initialize the cache manager."""
        if self._initialized:
            return
        try:
            from app.core.cache import get_cache_manager
            self._cache_manager = await get_cache_manager()
            self._initialized = True
            logger.info(f"[EmbeddingCache] Initialized with {self._cache_manager.backend.value} backend")
        except Exception as e:
            logger.warning(f"[EmbeddingCache] Failed to initialize cache manager: {e}")
            self._initialized = False

    def get(self, content: str) -> Optional[np.ndarray]:
        """Get cached embedding (sync, uses memory cache)."""
        key = self._hash_content(content)
        return self._memory_cache.get(key)

    async def aget(self, content: str) -> Optional[np.ndarray]:
        """Get cached embedding (async, uses Redis if available)."""
        if not self._initialized:
            await self.initialize()

        if self._cache_manager:
            try:
                return await self._cache_manager.get_embedding(content)
            except Exception as e:
                logger.warning(f"[EmbeddingCache] Redis get error: {e}")

        return self.get(content)

    def set(self, content: str, embedding: np.ndarray) -> None:
        """Cache embedding (sync, uses memory cache)."""
        if len(self._memory_cache) >= self._max_size:
            keys_to_remove = list(self._memory_cache.keys())[:self._max_size // 10]
            for key in keys_to_remove:
                del self._memory_cache[key]

        key = self._hash_content(content)
        self._memory_cache[key] = embedding

    async def aset(self, content: str, embedding: np.ndarray) -> None:
        """Cache embedding (async, uses Redis if available)."""
        if not self._initialized:
            await self.initialize()

        # Always set in memory for sync access
        self.set(content, embedding)

        # Also set in Redis if available
        if self._cache_manager:
            try:
                await self._cache_manager.set_embedding(content, embedding)
            except Exception as e:
                logger.warning(f"[EmbeddingCache] Redis set error: {e}")

    def get_batch(self, contents: List[str]) -> Tuple[List[np.ndarray], List[int], List[str], List[int]]:
        """
        Get cached embeddings for a batch of contents (sync).

        Returns:
            Tuple of (cached_embeddings, cached_indices, uncached_contents, uncached_indices)
        """
        cached = []
        cached_indices = []
        uncached = []
        uncached_indices = []

        for i, content in enumerate(contents):
            embedding = self.get(content)
            if embedding is not None:
                cached.append(embedding)
                cached_indices.append(i)
            else:
                uncached.append(content)
                uncached_indices.append(i)

        return cached, cached_indices, uncached, uncached_indices

    async def aget_batch(
        self,
        contents: List[str]
    ) -> Tuple[List[np.ndarray], List[int], List[str], List[int]]:
        """
        Get cached embeddings for a batch of contents (async).

        Returns:
            Tuple of (cached_embeddings, cached_indices, uncached_contents, uncached_indices)
        """
        if not self._initialized:
            await self.initialize()

        if self._cache_manager:
            try:
                return await self._cache_manager.get_embeddings_batch(contents)
            except Exception as e:
                logger.warning(f"[EmbeddingCache] Redis batch get error: {e}")

        return self.get_batch(contents)

    async def aset_batch(self, contents: List[str], embeddings: List[np.ndarray]) -> None:
        """Set cached embeddings for a batch of contents (async)."""
        if not self._initialized:
            await self.initialize()

        # Set in memory
        for content, emb in zip(contents, embeddings):
            self.set(content, emb)

        # Set in Redis if available
        if self._cache_manager:
            try:
                await self._cache_manager.set_embeddings_batch(contents, embeddings)
            except Exception as e:
                logger.warning(f"[EmbeddingCache] Redis batch set error: {e}")

    def clear(self) -> None:
        """Clear the memory cache."""
        self._memory_cache.clear()

    async def aclear(self) -> None:
        """Clear all caches (async)."""
        self._memory_cache.clear()
        if self._cache_manager:
            try:
                await self._cache_manager.clear_embeddings()
            except Exception as e:
                logger.warning(f"[EmbeddingCache] Redis clear error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {"memory_size": len(self._memory_cache)}
        if self._cache_manager:
            stats.update(self._cache_manager.get_stats())
        return stats


# Global embedding cache
_embedding_cache = EmbeddingCache()


def get_embeddings_batch(
    texts: List[str],
    use_cache: bool = True,
) -> np.ndarray:
    """
    Get embeddings for a batch of texts with caching support.

    This function:
    1. Checks cache for existing embeddings
    2. Computes embeddings for uncached texts using batch API
    3. Caches new embeddings
    4. Returns all embeddings in original order

    Args:
        texts: List of texts to embed
        use_cache: Whether to use embedding cache

    Returns:
        numpy array of embeddings (shape: [len(texts), embedding_dim])
    """
    if not texts:
        return np.array([])

    embed_manager = get_embedding_manager()

    if not use_cache:
        # Direct batch embedding without cache
        embeddings = embed_manager.embed_documents(texts)
        return np.array(embeddings)

    # Check cache
    cached_embeddings, cached_indices, uncached_texts, uncached_indices = \
        _embedding_cache.get_batch(texts)

    # If all cached, return directly
    if not uncached_texts:
        logger.debug(f"[PostProcessor] All {len(texts)} embeddings from cache")
        # Reconstruct in original order
        result = np.zeros((len(texts), cached_embeddings[0].shape[0]))
        for i, emb in zip(cached_indices, cached_embeddings):
            result[i] = emb
        return result

    # Compute uncached embeddings in batch
    logger.debug(
        f"[PostProcessor] Cache hit: {len(cached_indices)}, "
        f"computing: {len(uncached_texts)}"
    )
    new_embeddings = embed_manager.embed_documents(uncached_texts)

    # Cache new embeddings
    for text, emb in zip(uncached_texts, new_embeddings):
        _embedding_cache.set(text, np.array(emb))

    # Reconstruct full embeddings in original order
    embedding_dim = len(new_embeddings[0]) if new_embeddings else cached_embeddings[0].shape[0]
    result = np.zeros((len(texts), embedding_dim))

    for i, emb in zip(cached_indices, cached_embeddings):
        result[i] = emb
    for i, emb in zip(uncached_indices, new_embeddings):
        result[i] = np.array(emb)

    return result


class SemanticDeduplicator(BaseNodePostprocessor):
    """
    语义去重器 (Semantic Deduplicator)
    =================================

    基于嵌入向量的余弦相似度进行语义级别的文档去重。

    工作原理
    --------
    使用贪婪算法，按顺序检查每个文档：
    1. 计算当前文档与已选文档的最大相似度
    2. 如果相似度低于阈值，保留该文档
    3. 如果相似度高于阈值，视为重复并丢弃

    这种方法的优点是保留原始排序中靠前（更相关）的文档。

    Attributes
    ----------
    similarity_threshold : float
        去重阈值，默认 0.95。
        - 值越高，去重越严格（只去除几乎完全相同的文档）
        - 值越低，去重越宽松（去除更多相似文档）

    enable_dedup : bool
        是否启用去重功能，默认 True。

    Example
    -------
    ```python
    dedup = SemanticDeduplicator(
        similarity_threshold=0.95,  # 相似度 > 95% 视为重复
        enable_dedup=True
    )

    # 假设有 10 个节点，其中 3 对高度相似
    unique_nodes = dedup.postprocess_nodes(nodes)
    # 结果可能只有 7 个节点
    ```

    Note
    ----
    - 需要嵌入模型支持，依赖 LlamaSettings.embed_model
    - 时间复杂度 O(n²)，n 为节点数量
    - 建议在重排序之后、最终截断之前使用
    """

    # Pydantic fields
    similarity_threshold: float = settings.RAG_DEDUP_THRESHOLD
    enable_dedup: bool = settings.RAG_ENABLE_DEDUP

    def __init__(
        self,
        similarity_threshold: float = None,
        enable_dedup: bool = None,
        **kwargs,
    ):
        """
        初始化语义去重器。

        Parameters
        ----------
        similarity_threshold : float, optional
            去重阈值 (0.0 ~ 1.0)，默认使用配置值 RAG_DEDUP_THRESHOLD。
            推荐范围: 0.90 ~ 0.98

        enable_dedup : bool, optional
            是否启用去重，默认使用配置值 RAG_ENABLE_DEDUP。
        """
        super().__init__(
            similarity_threshold=similarity_threshold or settings.RAG_DEDUP_THRESHOLD,
            enable_dedup=enable_dedup if enable_dedup is not None else settings.RAG_ENABLE_DEDUP,
            **kwargs,
        )

        logger.info(
            f"[SemanticDedup] Initialized - threshold: {self.similarity_threshold}, "
            f"enabled: {self.enable_dedup}"
        )

    @classmethod
    def class_name(cls) -> str:
        return "SemanticDeduplicator"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        执行语义去重，移除相似度过高的重复节点。

        算法流程
        --------
        1. 批量获取所有节点的嵌入向量
        2. L2 归一化以便计算余弦相似度
        3. 贪婪迭代：
           - 对于每个候选节点，检查与已选节点的相似度
           - 如果最大相似度 < 阈值，保留该节点
           - 否则，视为重复并跳过

        Parameters
        ----------
        nodes : List[NodeWithScore]
            待去重的节点列表，通常来自重排序后的结果

        query_bundle : Optional[QueryBundle]
            查询包（本方法不使用，但接口需要）

        Returns
        -------
        List[NodeWithScore]
            去重后的节点列表，保持原始相对顺序

        Example
        -------
        输入 5 个节点，相似度矩阵:
        ```
              N1    N2    N3    N4    N5
        N1   1.0   0.97  0.60  0.55  0.45
        N2   0.97  1.0   0.58  0.50  0.40
        N3   0.60  0.58  1.0   0.96  0.35
        N4   0.55  0.50  0.96  1.0   0.30
        N5   0.45  0.40  0.35  0.30  1.0
        ```

        阈值 = 0.95 时:
        - N1 保留（第一个）
        - N2 跳过（与 N1 相似度 0.97 > 0.95）
        - N3 保留（与 N1 相似度 0.60 < 0.95）
        - N4 跳过（与 N3 相似度 0.96 > 0.95）
        - N5 保留（与所有已选节点相似度都 < 0.95）

        结果: [N1, N3, N5]
        """
        if not self.enable_dedup or len(nodes) <= 1:
            return nodes

        logger.info(f"[SemanticDedup] Deduplicating {len(nodes)} nodes")

        # 步骤 1: 使用批量嵌入获取所有节点的嵌入向量 (优化: 替代逐个计算)
        texts = [node.node.get_content() for node in nodes]

        try:
            # 尝试复用节点已有的嵌入 (如果存在)
            embeddings_list = []
            texts_to_embed = []
            indices_to_embed = []

            for i, node in enumerate(nodes):
                if hasattr(node.node, 'embedding') and node.node.embedding is not None:
                    embeddings_list.append((i, np.array(node.node.embedding)))
                else:
                    texts_to_embed.append(texts[i])
                    indices_to_embed.append(i)

            # 批量计算缺失的嵌入 (使用缓存)
            if texts_to_embed:
                logger.debug(f"[SemanticDedup] Reusing {len(embeddings_list)} embeddings, computing {len(texts_to_embed)}")
                new_embeddings = get_embeddings_batch(texts_to_embed, use_cache=True)
                for idx, emb in zip(indices_to_embed, new_embeddings):
                    embeddings_list.append((idx, emb))
            else:
                logger.debug(f"[SemanticDedup] All {len(nodes)} embeddings reused from nodes")

            # 按原始顺序重组嵌入
            embeddings_list.sort(key=lambda x: x[0])
            embeddings = np.array([emb for _, emb in embeddings_list])

        except Exception as e:
            logger.warning(f"[SemanticDedup] Failed to get embeddings: {e}")
            return nodes

        # 步骤 2: L2 归一化（用于余弦相似度计算）
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

        # 步骤 3: 贪婪去重
        selected_indices: List[int] = []
        for i, node in enumerate(nodes):
            is_duplicate = False

            for j in selected_indices:
                # 归一化后，点积 = 余弦相似度
                similarity = np.dot(embeddings[i], embeddings[j])
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    logger.debug(
                        f"[SemanticDedup] Node {i} is duplicate of {j} "
                        f"(similarity: {similarity:.3f})"
                    )
                    break

            if not is_duplicate:
                selected_indices.append(i)

        result = [nodes[i] for i in selected_indices]
        logger.info(
            f"[SemanticDedup] Deduplication complete: {len(nodes)} -> {len(result)} nodes"
        )

        return result


class MMRPostprocessor(BaseNodePostprocessor):
    """
    MMR 多样性后处理器 (Maximal Marginal Relevance)
    ==============================================

    基于 MMR 算法进行结果多样性过滤，平衡相关性与多样性。

    背景
    ----
    传统的相关性排序可能导致返回的文档内容高度相似，
    信息冗余度高。MMR 算法通过惩罚与已选文档相似的候选，
    在保证相关性的同时增加结果的多样性。

    算法原理
    --------
    MMR 分数公式:
    ```
    MMR(Di) = λ × Sim(Di, Q) - (1-λ) × max(Sim(Di, Dj))
                                        Dj∈S
    ```

    迭代选择过程:
    1. 计算所有候选文档与查询的相关性
    2. 选择相关性最高的文档作为第一个结果
    3. 对于剩余文档，计算 MMR 分数:
       - 相关性部分: 与查询的相似度
       - 多样性惩罚: 与已选文档的最大相似度
    4. 选择 MMR 分数最高的文档
    5. 重复步骤 3-4 直到选够 top_n 个

    Attributes
    ----------
    lambda_mult : float
        平衡系数 λ (0.0 ~ 1.0)，默认 0.5。
        - λ = 1.0: 纯相关性，等同于普通排序
        - λ = 0.5: 相关性与多样性平衡（推荐）
        - λ = 0.0: 纯多样性，可能牺牲相关性

    top_n : int
        最终返回的文档数量。

    Example
    -------
    ```python
    mmr = MMRPostprocessor(
        lambda_mult=0.5,  # 相关性与多样性各占一半权重
        top_n=5           # 返回 5 个结果
    )

    query_bundle = QueryBundle(query_str="如何申请退款？")
    diverse_results = mmr.postprocess_nodes(nodes, query_bundle)
    ```

    应用场景
    --------
    - **知识库问答**: 返回覆盖不同方面的答案
    - **文档摘要**: 避免内容重复，提高信息密度
    - **推荐系统**: 增加推荐结果的多样性

    Note
    ----
    - 时间复杂度 O(k × n)，k 为 top_n，n 为候选数
    - 需要嵌入模型支持
    - 通常在去重之后使用
    """

    # Pydantic fields
    lambda_mult: float = 0.5
    top_n: int = settings.RAG_FINAL_TOP_K

    def __init__(
        self,
        lambda_mult: float = 0.5,
        top_n: int = None,
        **kwargs,
    ):
        """
        初始化 MMR 后处理器。

        Parameters
        ----------
        lambda_mult : float, optional
            相关性与多样性的平衡系数 (0.0 ~ 1.0)，默认 0.5。
            - 推荐值: 0.4 ~ 0.7
            - 偏向相关性: 0.7 ~ 0.9
            - 偏向多样性: 0.3 ~ 0.5

        top_n : int, optional
            最终返回文档数，默认使用 RAG_FINAL_TOP_K。
        """
        super().__init__(
            lambda_mult=lambda_mult,
            top_n=top_n or settings.RAG_FINAL_TOP_K,
            **kwargs,
        )

        logger.info(
            f"[MMRPostprocessor] Initialized - lambda: {self.lambda_mult}, "
            f"top_n: {self.top_n}"
        )

    @classmethod
    def class_name(cls) -> str:
        return "MMRPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Apply MMR diversity filtering.

        Args:
            nodes: List of NodeWithScore objects
            query_bundle: Query bundle containing the query string

        Returns:
            Diversity-filtered list of NodeWithScore objects
        """
        if len(nodes) <= self.top_n:
            return nodes

        if query_bundle is None:
            return nodes[:self.top_n]

        logger.info(f"[MMRPostprocessor] Applying MMR to {len(nodes)} nodes")

        # 优化: 使用批量嵌入替代逐个计算
        try:
            # 收集所有需要嵌入的文本 (查询 + 文档)
            texts = [node.node.get_content() for node in nodes]
            all_texts = [query_bundle.query_str] + texts

            # 尝试复用节点已有的嵌入
            doc_embeddings_list = []
            texts_to_embed = [query_bundle.query_str]  # 查询始终需要嵌入
            indices_to_embed = [-1]  # -1 表示查询

            for i, node in enumerate(nodes):
                if hasattr(node.node, 'embedding') and node.node.embedding is not None:
                    doc_embeddings_list.append((i, np.array(node.node.embedding)))
                else:
                    texts_to_embed.append(texts[i])
                    indices_to_embed.append(i)

            # 批量计算嵌入 (使用缓存)
            new_embeddings = get_embeddings_batch(texts_to_embed, use_cache=True)

            # 分离查询嵌入和文档嵌入
            query_embedding = new_embeddings[0]

            # 重组文档嵌入
            embed_idx = 1
            for idx in indices_to_embed[1:]:  # 跳过查询
                doc_embeddings_list.append((idx, new_embeddings[embed_idx]))
                embed_idx += 1

            doc_embeddings_list.sort(key=lambda x: x[0])
            doc_embeddings = np.array([emb for _, emb in doc_embeddings_list])

            logger.debug(f"[MMRPostprocessor] Reused {len(nodes) - len(indices_to_embed) + 1} embeddings")

        except Exception as e:
            logger.warning(f"[MMRPostprocessor] Batch embedding failed: {e}, falling back to individual")
            # 回退到逐个计算
            embed_model = LlamaSettings.embed_model
            query_embedding = np.array(embed_model.get_text_embedding(query_bundle.query_str))
            doc_embeddings = np.array([
                embed_model.get_text_embedding(node.node.get_content())
                for node in nodes
            ])

        # Normalize
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        doc_embeddings = doc_embeddings / (norms + 1e-10)

        # Calculate relevance scores (similarity to query)
        relevance_scores = np.dot(doc_embeddings, query_embedding)

        # MMR selection
        selected_indices: List[int] = []
        remaining_indices = list(range(len(nodes)))

        while len(selected_indices) < self.top_n and remaining_indices:
            mmr_scores = []

            for i in remaining_indices:
                relevance = relevance_scores[i]

                # Calculate max similarity to already selected
                if selected_indices:
                    similarities = [
                        np.dot(doc_embeddings[i], doc_embeddings[j])
                        for j in selected_indices
                    ]
                    max_similarity = max(similarities)
                else:
                    max_similarity = 0.0

                # MMR score
                mmr = self.lambda_mult * relevance - (1 - self.lambda_mult) * max_similarity
                mmr_scores.append((i, mmr))

            # Select node with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        result = [nodes[i] for i in selected_indices]
        logger.info(f"[MMRPostprocessor] MMR complete: {len(nodes)} -> {len(result)} nodes")

        return result


class RAGPostProcessor:
    """
    Combined RAG post-processor that applies all post-processing steps.
    """

    def __init__(
        self,
        enable_dedup: bool = None,
        dedup_threshold: float = None,
        enable_mmr: bool = True,
        mmr_lambda: float = 0.5,
        final_top_k: int = None,
    ):
        self.enable_dedup = enable_dedup if enable_dedup is not None else settings.RAG_ENABLE_DEDUP
        self.enable_mmr = enable_mmr
        self.final_top_k = final_top_k or settings.RAG_FINAL_TOP_K

        self.deduplicator = SemanticDeduplicator(
            similarity_threshold=dedup_threshold,
            enable_dedup=self.enable_dedup,
        )
        self.mmr_processor = MMRPostprocessor(
            lambda_mult=mmr_lambda,
            top_n=self.final_top_k,
        )

        logger.info(
            f"[RAGPostProcessor] Initialized - dedup: {self.enable_dedup}, "
            f"mmr: {self.enable_mmr}, final_top_k: {self.final_top_k}"
        )

    def process(
        self,
        nodes: List[NodeWithScore],
        query: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """
        Apply all post-processing steps.

        Args:
            nodes: List of NodeWithScore objects
            query: Optional query string for MMR

        Returns:
            Post-processed list of NodeWithScore objects
        """
        logger.info(f"[RAGPostProcessor] Processing {len(nodes)} nodes")

        # Step 1: Semantic deduplication
        if self.enable_dedup:
            nodes = self.deduplicator.postprocess_nodes(nodes)

        # Step 2: MMR diversity filtering
        if self.enable_mmr and query:
            query_bundle = QueryBundle(query_str=query)
            nodes = self.mmr_processor.postprocess_nodes(nodes, query_bundle)
        else:
            nodes = nodes[:self.final_top_k]

        logger.info(f"[RAGPostProcessor] Post-processing complete: {len(nodes)} nodes")
        return nodes

    async def aprocess(
        self,
        nodes: List[NodeWithScore],
        query: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """Async version of process."""
        return self.process(nodes, query)


# Global singleton
_post_processor: Optional[RAGPostProcessor] = None


def get_post_processor() -> RAGPostProcessor:
    """Get the global post-processor instance."""
    global _post_processor
    if _post_processor is None:
        _post_processor = RAGPostProcessor()
    return _post_processor
