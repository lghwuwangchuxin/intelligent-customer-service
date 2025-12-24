"""
混合检索器模块 (Hybrid Retriever Module)
==========================================

本模块实现了向量检索与 BM25 关键词检索的混合检索策略，通过 RRF (Reciprocal Rank Fusion)
算法融合两种检索结果，以提高检索的准确性和召回率。

核心原理
--------
1. **向量检索 (Vector Search)**:
   - 基于语义相似度的稠密检索
   - 能够理解语义相近但用词不同的查询
   - 适合处理同义词、近义词、语义相关的场景

2. **BM25 关键词检索**:
   - 基于词频统计的稀疏检索
   - 对精确匹配和专业术语有更好的召回
   - 使用 jieba 分词处理中文文本
   - 支持内存 BM25 (默认) 或 ES BM25 (当启用混合存储时)

3. **RRF 融合算法**:
   - 公式: RRF_score = Σ (weight_i / (k + rank_i))
   - k 通常取 60，用于平滑排名差异
   - 通过权重控制向量和 BM25 结果的重要性

4. **ES 完整文本获取** (新增):
   - 检索后使用 chunk_id 从 ES 获取完整文本与元数据
   - 仅在启用混合存储时激活

配置参数 (config/settings.py)
-----------------------------
- RAG_ENABLE_HYBRID: bool - 是否启用混合检索 (默认 True)
- RAG_VECTOR_WEIGHT: float - 向量检索权重 (默认 0.7)
- RAG_BM25_WEIGHT: float - BM25检索权重 (默认 0.3)
- RAG_RETRIEVAL_TOP_K: int - 初始检索数量 (默认 10)

使用示例
--------
```python
from app.rag.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from llama_index.core.schema import QueryBundle, TextNode

# 方式1: 使用全局单例
retriever = get_hybrid_retriever()

# 方式2: 自定义配置
retriever = HybridRetriever(
    vector_weight=0.6,
    bm25_weight=0.4,
    top_k=15,
    enable_hybrid=True
)

# 构建 BM25 索引 (需要提供文档节点)
nodes = [TextNode(text="文档内容1"), TextNode(text="文档内容2")]
retriever.build_bm25_index(nodes)

# 单查询检索
query = QueryBundle(query_str="用户问题")
results = retriever.retrieve(query)

# 多查询检索 (适合 Query Expansion 场景)
queries = ["问题变体1", "问题变体2", "问题变体3"]
results = retriever.retrieve_multi_query(queries)

# 获取检索结果
for node_with_score in results:
    print(f"内容: {node_with_score.node.get_content()}")
    print(f"分数: {node_with_score.score}")
```

性能优化建议
------------
1. BM25 索引构建是一次性操作，建议在服务启动时完成
2. 对于大规模语料库，可考虑定期重建 BM25 索引
3. 根据实际效果调整 vector_weight 和 bm25_weight 权重

参考文献
--------
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond
- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal Rank Fusion
"""
import logging
from typing import List, Optional, Dict, Any
from collections import defaultdict

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from rank_bm25 import BM25Okapi
import jieba

from config.settings import settings
from app.rag.index_manager import get_index_manager

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    混合检索器：结合向量检索与 BM25 关键词检索。

    该检索器通过融合两种互补的检索策略来提高检索质量：
    - 向量检索擅长语义理解，能找到意思相近但用词不同的文档
    - BM25 检索擅长精确匹配，对专业术语和关键词有更好的召回

    Attributes:
        vector_weight (float): 向量检索结果的权重，范围 [0, 1]
        bm25_weight (float): BM25 检索结果的权重，范围 [0, 1]
        top_k (int): 返回的最大结果数量
        enable_hybrid (bool): 是否启用混合检索，False 时仅使用向量检索
        use_es_for_full_text (bool): 是否从 ES 获取完整文本

    Note:
        - 使用前需调用 build_bm25_index() 构建 BM25 索引
        - 若未构建 BM25 索引，将自动降级为纯向量检索
        - 权重之和不必为 1，RRF 算法会自动归一化
    """

    def __init__(
        self,
        vector_weight: float = None,
        bm25_weight: float = None,
        top_k: int = None,
        enable_hybrid: bool = None,
        use_es_for_full_text: bool = None,
    ):
        """
        初始化混合检索器。

        Args:
            vector_weight: 向量检索权重，默认从配置读取 (0.7)
            bm25_weight: BM25 检索权重，默认从配置读取 (0.3)
            top_k: 返回结果数量，默认从配置读取 (10)
            enable_hybrid: 是否启用混合检索，默认从配置读取 (True)
            use_es_for_full_text: 是否从 ES 获取完整文本，默认从配置读取
        """
        super().__init__()

        self.vector_weight = vector_weight or settings.RAG_VECTOR_WEIGHT
        self.bm25_weight = bm25_weight or settings.RAG_BM25_WEIGHT
        self.top_k = top_k or settings.RAG_RETRIEVAL_TOP_K
        self.enable_hybrid = enable_hybrid if enable_hybrid is not None else settings.RAG_ENABLE_HYBRID

        # Whether to fetch full text from ES after retrieval
        self.use_es_for_full_text = use_es_for_full_text if use_es_for_full_text is not None else (
            hasattr(settings, 'HYBRID_STORAGE_ENABLED') and settings.HYBRID_STORAGE_ENABLED
        )

        self._index_manager = get_index_manager()
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[str] = []
        self._corpus_nodes: List[TextNode] = []
        self._tokenized_corpus: List[List[str]] = []

        # ES manager for fetching full text (lazy init)
        self._es_manager = None

        logger.info(
            f"[HybridRetriever] Initialized - vector_weight: {self.vector_weight}, "
            f"bm25_weight: {self.bm25_weight}, top_k: {self.top_k}, hybrid: {self.enable_hybrid}, "
            f"use_es_for_full_text: {self.use_es_for_full_text}"
        )

    def _get_es_manager(self):
        """Lazy initialization of ES manager."""
        if self._es_manager is None and self.use_es_for_full_text:
            try:
                from app.core.hybrid_store_manager import get_hybrid_store_manager
                hybrid_store = get_hybrid_store_manager()
                if hybrid_store:
                    self._es_manager = hybrid_store.es_manager
            except Exception as e:
                logger.warning(f"[HybridRetriever] Failed to get ES manager: {e}")
        return self._es_manager

    def build_bm25_index(self, nodes: List[TextNode]) -> None:
        """
        构建 BM25 索引。

        从文档节点列表构建 BM25 索引，用于后续的关键词检索。
        使用 jieba 进行中文分词处理。

        Args:
            nodes: LlamaIndex TextNode 对象列表，每个节点包含文档内容

        Example:
            >>> nodes = [TextNode(text="产品使用说明"), TextNode(text="常见问题解答")]
            >>> retriever.build_bm25_index(nodes)

        Note:
            - 此方法会覆盖之前构建的索引
            - 对于大规模语料库，构建过程可能需要较长时间
            - 建议在服务启动时调用一次
        """
        if not nodes:
            logger.warning("[HybridRetriever] No nodes provided, BM25 index not built")
            return

        logger.info(f"[HybridRetriever] Building BM25 index from {len(nodes)} nodes")

        self._corpus_nodes = nodes
        self._corpus = [node.get_content() for node in nodes]

        # 使用 jieba 进行中文分词
        self._tokenized_corpus = [list(jieba.cut(doc)) for doc in self._corpus]

        # 构建 BM25 索引
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(f"[HybridRetriever] ✅ BM25 index built successfully with {len(nodes)} documents")

    def append_to_bm25_index(self, nodes: List[TextNode]) -> None:
        """
        向现有 BM25 索引追加新节点。

        Args:
            nodes: 要追加的新节点列表
        """
        if not nodes:
            return

        if self._bm25 is None:
            # 如果还没有索引，直接构建
            self.build_bm25_index(nodes)
            return

        logger.info(f"[HybridRetriever] Appending {len(nodes)} nodes to BM25 index")

        # 追加到现有语料库
        self._corpus_nodes.extend(nodes)
        new_corpus = [node.get_content() for node in nodes]
        self._corpus.extend(new_corpus)

        # 追加分词结果
        new_tokenized = [list(jieba.cut(doc)) for doc in new_corpus]
        self._tokenized_corpus.extend(new_tokenized)

        # 重建 BM25 索引（rank_bm25 不支持增量更新）
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(f"[HybridRetriever] BM25 index updated, total documents: {len(self._corpus_nodes)}")

    async def rebuild_bm25_from_vector_store(self) -> bool:
        """
        从向量存储中恢复 BM25 索引。

        服务重启后调用此方法可以从 Milvus/ES 恢复 BM25 索引，
        确保混合检索功能正常工作。

        Returns:
            bool: 是否成功恢复索引
        """
        logger.info("[HybridRetriever] ========== 开始恢复 BM25 索引 ==========")

        try:
            # 方案1: 尝试从 ES 获取所有已索引的文档（如果启用了混合存储）
            if self.use_es_for_full_text:
                es_manager = self._get_es_manager()
                if es_manager:
                    try:
                        # 从 ES 获取所有分块
                        logger.info("[HybridRetriever] 尝试从 Elasticsearch 恢复...")
                        chunks = await es_manager.get_all_chunks(limit=10000)
                        if chunks:
                            # 转换为 TextNode
                            nodes = [
                                TextNode(
                                    text=chunk.content,
                                    id_=chunk.chunk_id,
                                    metadata=chunk.metadata or {}
                                )
                                for chunk in chunks
                            ]
                            self.build_bm25_index(nodes)
                            logger.info(f"[HybridRetriever] ✅ 从 ES 恢复 BM25 索引成功，共 {len(nodes)} 个文档")
                            return True
                    except Exception as e:
                        logger.warning(f"[HybridRetriever] 从 ES 恢复失败: {e}")

            # 方案2: 尝试从 IndexManager 获取节点
            try:
                logger.info("[HybridRetriever] 尝试从 IndexManager 恢复...")
                nodes = await self._index_manager.get_all_nodes_async()
                if nodes:
                    self.build_bm25_index(nodes)
                    logger.info(f"[HybridRetriever] ✅ 从 IndexManager 恢复 BM25 索引成功，共 {len(nodes)} 个文档")
                    return True
                else:
                    logger.info("[HybridRetriever] IndexManager 中没有已索引的节点")
            except Exception as e:
                logger.warning(f"[HybridRetriever] 从 IndexManager 恢复失败: {e}")

            # 方案3: 检查 Milvus 集合状态
            try:
                from app.core.vector_store import get_vector_store_manager
                vector_store = get_vector_store_manager()
                stats = vector_store.get_collection_stats()
                num_entities = stats.get('num_entities', 0)

                if num_entities > 0:
                    logger.warning(
                        f"[HybridRetriever] ⚠️ Milvus 中有 {num_entities} 个实体，"
                        "但无法直接获取用于 BM25 索引。混合检索将降级为纯向量检索。"
                    )
                else:
                    logger.info("[HybridRetriever] Milvus 集合为空，无需恢复 BM25 索引")
            except Exception as e:
                logger.debug(f"[HybridRetriever] 无法获取 Milvus 状态: {e}")

            logger.info("[HybridRetriever] ========== BM25 索引恢复完成 ==========")
            return False

        except Exception as e:
            logger.error(f"[HybridRetriever] 恢复 BM25 索引时发生错误: {e}", exc_info=True)
            return False

    def is_bm25_ready(self) -> bool:
        """检查 BM25 索引是否已构建。"""
        return self._bm25 is not None and len(self._corpus_nodes) > 0

    def get_bm25_stats(self) -> Dict[str, Any]:
        """获取 BM25 索引统计信息。"""
        return {
            "is_ready": self.is_bm25_ready(),
            "num_documents": len(self._corpus_nodes) if self._corpus_nodes else 0,
            "avg_doc_length": sum(len(doc) for doc in self._corpus) / max(len(self._corpus), 1) if self._corpus else 0,
            "enable_hybrid": self.enable_hybrid,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
        }

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        执行混合检索（内部方法）。

        根据配置决定使用混合检索还是纯向量检索：
        1. 如果 enable_hybrid=False 或 BM25 索引未构建，使用纯向量检索
        2. 否则，同时执行向量检索和 BM25 检索，然后用 RRF 融合结果
        3. 如果启用了 ES 完整文本获取，从 ES 拉取完整文本和元数据

        Args:
            query_bundle: LlamaIndex QueryBundle，包含查询字符串

        Returns:
            List[NodeWithScore]: 检索结果列表，按相关性分数降序排列
        """
        query = query_bundle.query_str
        logger.info(f"[HybridRetriever] Retrieving for query: {query[:50]}...")

        if not self.enable_hybrid or self._bm25 is None:
            # 降级为纯向量检索
            logger.info("[HybridRetriever] Using vector-only retrieval")
            results = self._vector_retrieve(query_bundle)
        else:
            # 同时执行两种检索
            vector_results = self._vector_retrieve(query_bundle)
            bm25_results = self._bm25_retrieve(query)

            logger.info(
                f"[HybridRetriever] Vector results: {len(vector_results)}, "
                f"BM25 results: {len(bm25_results)}"
            )

            # 使用 RRF 融合结果
            results = self._rrf_fusion(vector_results, bm25_results)

        # 限制结果数量
        results = results[:self.top_k]
        logger.info(f"[HybridRetriever] Merged results: {len(results)}")

        # 从 ES 获取完整文本和元数据（如果启用）
        if self.use_es_for_full_text and results:
            results = self._enrich_from_es(results)

        return results

    def _enrich_from_es(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        从 ES 获取完整文本和元数据来丰富检索结果。

        流程：
        1. 收集所有 node_id (chunk_id)
        2. 批量从 ES 获取完整文本
        3. 更新节点内容和元数据

        Args:
            nodes: 检索到的节点列表

        Returns:
            List[NodeWithScore]: 更新后的节点列表
        """
        import asyncio

        es_manager = self._get_es_manager()
        if not es_manager:
            logger.debug("[HybridRetriever] ES manager not available, skipping enrichment")
            return nodes

        try:
            # Collect chunk IDs
            chunk_ids = [n.node.node_id for n in nodes if n.node.node_id]

            if not chunk_ids:
                return nodes

            # Fetch from ES (run async in sync context)
            try:
                loop = asyncio.get_event_loop()
                full_chunks = loop.run_until_complete(
                    es_manager.get_chunks_by_ids(chunk_ids)
                )
            except RuntimeError:
                # No event loop, create one
                full_chunks = asyncio.run(
                    es_manager.get_chunks_by_ids(chunk_ids)
                )

            if not full_chunks:
                logger.debug("[HybridRetriever] No chunks returned from ES")
                return nodes

            # Build map for quick lookup
            chunk_map = {c.chunk_id: c for c in full_chunks}

            # Enrich nodes with full content and metadata
            enriched_count = 0
            for node_with_score in nodes:
                node_id = node_with_score.node.node_id
                if node_id in chunk_map:
                    full_chunk = chunk_map[node_id]
                    # Update content
                    node_with_score.node.set_content(full_chunk.content)
                    # Update metadata
                    if full_chunk.metadata:
                        node_with_score.node.metadata.update(full_chunk.metadata)
                    enriched_count += 1

            logger.info(f"[HybridRetriever] Enriched {enriched_count}/{len(nodes)} nodes from ES")
            return nodes

        except Exception as e:
            logger.warning(f"[HybridRetriever] Failed to enrich from ES: {e}")
            return nodes

    def _vector_retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        执行向量相似度检索。

        通过 IndexManager 获取向量检索器，基于嵌入向量的余弦相似度
        检索最相关的文档节点。

        Args:
            query_bundle: 查询包

        Returns:
            List[NodeWithScore]: 向量检索结果
        """
        retriever = self._index_manager.get_retriever(similarity_top_k=self.top_k)
        return retriever.retrieve(query_bundle)

    def _bm25_retrieve(self, query: str) -> List[NodeWithScore]:
        """
        执行 BM25 关键词检索。

        使用 BM25 算法基于词频统计计算查询与文档的相关性分数。
        查询文本会先经过 jieba 分词处理。

        Args:
            query: 查询字符串

        Returns:
            List[NodeWithScore]: BM25 检索结果，按分数降序排列
        """
        if self._bm25 is None or not self._corpus_nodes:
            return []

        # 使用 jieba 对查询进行分词
        tokenized_query = list(jieba.cut(query))

        # 计算 BM25 分数
        scores = self._bm25.get_scores(tokenized_query)

        # 记录原始分数用于调试
        logger.debug(f"[HybridRetriever] BM25 raw scores: {scores}")

        # 创建结果列表
        # 注意：BM25Okapi 可能产生负分（当词在所有文档中出现时 IDF 为负）
        # 因此我们不过滤负分，而是返回所有有效结果并按分数排序
        results = []
        for idx, score in enumerate(scores):
            # 只排除完全没有匹配的结果（分数为0或极小值）
            # BM25 分数可以是负数，所以不能用 > 0 过滤
            if abs(score) > 1e-10:  # 有任何匹配分数的结果
                node = self._corpus_nodes[idx]
                results.append(NodeWithScore(node=node, score=float(score)))

        # 按分数降序排列（分数越高匹配越好，即使是负数也有相对排序意义）
        results.sort(key=lambda x: x.score, reverse=True)

        logger.debug(f"[HybridRetriever] BM25 returning {len(results)} results")
        return results[:self.top_k]

    def _rrf_fusion(
        self,
        vector_results: List[NodeWithScore],
        bm25_results: List[NodeWithScore],
        k: int = 60,
    ) -> List[NodeWithScore]:
        """
        使用 RRF (Reciprocal Rank Fusion) 算法融合检索结果。

        RRF 是一种经典的排名融合算法，通过倒数排名来平滑不同排名系统的差异。

        算法公式:
            RRF_score(d) = Σ (weight_i / (k + rank_i(d)))

        其中:
            - d 是文档
            - weight_i 是第 i 个排名系统的权重
            - rank_i(d) 是文档 d 在第 i 个系统中的排名（从 1 开始）
            - k 是平滑常数，通常取 60

        Args:
            vector_results: 向量检索的结果列表
            bm25_results: BM25 检索的结果列表
            k: RRF 平滑常数，默认 60。较大的 k 值使排名差异的影响更平缓

        Returns:
            List[NodeWithScore]: 融合后的结果列表，按 RRF 分数降序排列

        Example:
            假设文档 A 在向量检索中排第 1，在 BM25 中排第 3
            RRF_score(A) = 0.7/(60+1) + 0.3/(60+3) = 0.0115 + 0.0048 = 0.0163
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        node_map: Dict[str, NodeWithScore] = {}

        # 处理向量检索结果
        for rank, node_with_score in enumerate(vector_results, 1):
            node_id = node_with_score.node.node_id
            rrf_scores[node_id] += self.vector_weight * (1.0 / (k + rank))
            node_map[node_id] = node_with_score

        # 处理 BM25 检索结果
        for rank, node_with_score in enumerate(bm25_results, 1):
            node_id = node_with_score.node.node_id
            rrf_scores[node_id] += self.bm25_weight * (1.0 / (k + rank))
            if node_id not in node_map:
                node_map[node_id] = node_with_score

        # 按 RRF 分数降序排列
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 创建结果列表，使用 RRF 分数作为新分数
        results = []
        for node_id in sorted_ids:
            original = node_map[node_id]
            results.append(
                NodeWithScore(node=original.node, score=rrf_scores[node_id])
            )

        return results

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        异步检索接口。

        使用 asyncio.to_thread 将 CPU 密集型操作移至线程池执行，
        避免阻塞事件循环。

        Args:
            query_bundle: 查询包

        Returns:
            List[NodeWithScore]: 检索结果
        """
        import asyncio
        # 将同步检索操作移至线程池执行，避免阻塞事件循环
        return await asyncio.to_thread(self._retrieve, query_bundle)

    def retrieve_multi_query(
        self,
        queries: List[str],
    ) -> List[NodeWithScore]:
        """
        多查询检索并融合结果。

        适用于 Query Expansion 场景，将用户的原始查询扩展为多个变体后，
        分别检索并融合结果，以提高召回率。

        Args:
            queries: 查询字符串列表（通常包含原始查询及其扩展变体）

        Returns:
            List[NodeWithScore]: 融合后的检索结果

        Example:
            >>> queries = [
            ...     "如何退款",           # 原始查询
            ...     "退款流程是什么",      # 扩展变体 1
            ...     "怎么申请退货退款"     # 扩展变体 2
            ... ]
            >>> results = retriever.retrieve_multi_query(queries)
        """
        logger.info(f"[HybridRetriever] Multi-query retrieval with {len(queries)} queries")

        all_results: List[List[NodeWithScore]] = []

        for query in queries:
            bundle = QueryBundle(query_str=query)
            results = self._retrieve(bundle)
            all_results.append(results)

        # 使用 RRF 融合所有查询的结果
        return self._merge_multi_query_results(all_results)

    async def aretrieve_multi_query(
        self,
        queries: List[str],
    ) -> List[NodeWithScore]:
        """
        异步多查询检索并融合结果。

        并行执行多个查询检索，显著提升多查询场景的性能。

        Args:
            queries: 查询字符串列表

        Returns:
            List[NodeWithScore]: 融合后的检索结果
        """
        import asyncio
        logger.info(f"[HybridRetriever] Async multi-query retrieval with {len(queries)} queries")

        # 并行执行所有查询
        async def retrieve_single(query: str) -> List[NodeWithScore]:
            bundle = QueryBundle(query_str=query)
            return await self.aretrieve(bundle)

        tasks = [retrieve_single(q) for q in queries]
        all_results = await asyncio.gather(*tasks)

        # 使用 RRF 融合所有查询的结果
        return self._merge_multi_query_results(list(all_results))

    def _merge_multi_query_results(
        self,
        results_list: List[List[NodeWithScore]],
        k: int = 60,
    ) -> List[NodeWithScore]:
        """
        融合多个查询的检索结果。

        对每个查询的结果赋予相等权重，使用 RRF 算法融合。
        在多个查询中都排名靠前的文档会获得更高的最终分数。

        Args:
            results_list: 多个查询的结果列表
            k: RRF 平滑常数

        Returns:
            List[NodeWithScore]: 融合后的结果，限制为 top_k 个
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        node_map: Dict[str, NodeWithScore] = {}

        # 每个查询的权重相等
        weight = 1.0 / len(results_list)

        for results in results_list:
            for rank, node_with_score in enumerate(results, 1):
                node_id = node_with_score.node.node_id
                rrf_scores[node_id] += weight * (1.0 / (k + rank))
                if node_id not in node_map:
                    node_map[node_id] = node_with_score

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for node_id in sorted_ids:
            original = node_map[node_id]
            results.append(
                NodeWithScore(node=original.node, score=rrf_scores[node_id])
            )

        return results[:self.top_k]


# ============================================================================
# 全局单例模式
# ============================================================================

_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """
    获取全局混合检索器单例。

    使用单例模式确保整个应用共享同一个检索器实例，
    避免重复构建 BM25 索引。

    Returns:
        HybridRetriever: 全局检索器实例

    Example:
        >>> retriever = get_hybrid_retriever()
        >>> results = retriever.retrieve(QueryBundle(query_str="用户问题"))
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever
