"""
Jina Reranker Module.
Provides neural reranking using Jina's multilingual reranker model.
Falls back to simple score-based reranking if sentence-transformers is unavailable.
"""
import logging
from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from config.settings import settings

logger = logging.getLogger(__name__)

# Try to import CrossEncoder, fall back gracefully if not available
CROSS_ENCODER_AVAILABLE = False
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    logger.warning(
        "[JinaReranker] sentence-transformers not available, "
        "using simple score-based reranking instead"
    )


class JinaReranker(BaseNodePostprocessor):
    """
    Jina-based neural reranker for improved relevance scoring.
    Uses sentence-transformers CrossEncoder for reranking.
    """

    # Pydantic fields
    model_name: str = settings.RAG_RERANK_MODEL
    top_n: int = settings.RAG_RERANK_TOP_N
    score_threshold: float = settings.RAG_RERANK_SCORE_THRESHOLD
    enable_rerank: bool = settings.RAG_ENABLE_RERANK

    # Non-field attributes
    _cross_encoder: Optional[object] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        model: str = None,
        top_n: int = None,
        score_threshold: float = None,
        enable_rerank: bool = None,
        **kwargs,
    ):
        # Set values via Pydantic constructor
        super().__init__(
            model_name=model or settings.RAG_RERANK_MODEL,
            top_n=top_n or settings.RAG_RERANK_TOP_N,
            score_threshold=score_threshold or settings.RAG_RERANK_SCORE_THRESHOLD,
            enable_rerank=enable_rerank if enable_rerank is not None else settings.RAG_ENABLE_RERANK,
            **kwargs,
        )

        logger.info(
            f"[JinaReranker] Initialized - model: {self.model_name}, "
            f"top_n: {self.top_n}, threshold: {self.score_threshold}, enabled: {self.enable_rerank}"
        )

    def _load_model(self):
        """Lazy load the CrossEncoder model."""
        if self._cross_encoder is None:
            if not CROSS_ENCODER_AVAILABLE:
                logger.warning("[JinaReranker] CrossEncoder not available, skipping model load")
                return False

            logger.info(f"[JinaReranker] Loading CrossEncoder model: {self.model_name}")
            try:
                self._cross_encoder = CrossEncoder(
                    self.model_name,
                    max_length=512,
                    trust_remote_code=True,
                )
                logger.info("[JinaReranker] CrossEncoder model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"[JinaReranker] Failed to load model: {e}")
                return False
        return True

    @classmethod
    def class_name(cls) -> str:
        return "JinaReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Rerank nodes based on relevance to the query.

        Args:
            nodes: List of NodeWithScore objects to rerank
            query_bundle: Query bundle containing the query string

        Returns:
            Reranked list of NodeWithScore objects
        """
        if not self.enable_rerank:
            logger.info("[JinaReranker] Reranking disabled, returning original nodes")
            return nodes

        if not nodes or query_bundle is None:
            return nodes

        query = query_bundle.query_str
        logger.info(f"[JinaReranker] Reranking {len(nodes)} nodes for query: {query[:50]}...")

        # Load model if not already loaded
        if not self._load_model():
            logger.warning("[JinaReranker] Model not available, using simple score-based sorting")
            # Fall back to simple sorting by existing scores (no threshold filtering)
            # Note: We don't apply score_threshold here because it was designed for
            # CrossEncoder reranker scores (0-1), not RRF fusion scores (~0.01-0.02)
            sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
            result = sorted_nodes[:self.top_n]
            logger.info(f"[JinaReranker] Fallback returned {len(result)} nodes")
            return result

        # Prepare query-document pairs
        pairs = [
            (query, node.node.get_content())
            for node in nodes
        ]

        # Get reranker scores
        try:
            scores = self._cross_encoder.predict(pairs)
            min_score = min(scores) if len(scores) > 0 else 0
            max_score = max(scores) if len(scores) > 0 else 0
            avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0
            logger.info(
                f"[JinaReranker] Scores computed: min={min_score:.3f}, "
                f"max={max_score:.3f}, avg={avg_score:.3f}"
            )
        except Exception as e:
            logger.error(f"[JinaReranker] Failed to compute scores: {e}")
            return nodes

        # 动态阈值调整逻辑：
        # 如果所有分数都低于阈值，使用自适应阈值避免返回空结果
        effective_threshold = self.score_threshold
        all_below_threshold = all(score < self.score_threshold for score in scores)

        if all_below_threshold and len(scores) > 0:
            # 使用最大分数的一半作为自适应阈值，确保至少保留一些结果
            adaptive_threshold = max_score * 0.5
            effective_threshold = min(adaptive_threshold, self.score_threshold)
            logger.warning(
                f"[JinaReranker] ⚠️ 所有分数低于阈值 {self.score_threshold:.3f}，"
                f"使用自适应阈值 {effective_threshold:.3f}"
            )

        # Create reranked list with updated scores
        reranked = []
        filtered_count = 0
        for node, score in zip(nodes, scores):
            if score >= effective_threshold:
                reranked.append(
                    NodeWithScore(node=node.node, score=float(score))
                )
            else:
                filtered_count += 1
                logger.debug(f"[JinaReranker] Filtered node with score {score:.3f} < {effective_threshold:.3f}")

        # 安全保障：如果过滤后没有结果，至少保留分数最高的几个
        if not reranked and nodes:
            logger.warning(
                f"[JinaReranker] ⚠️ 所有节点被过滤，保留分数最高的 {min(self.top_n, len(nodes))} 个节点"
            )
            # 按原始 reranker 分数排序并保留
            scored_nodes = list(zip(nodes, scores))
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            reranked = [
                NodeWithScore(node=node.node, score=float(score))
                for node, score in scored_nodes[:self.top_n]
            ]

        # Sort by score descending
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Return top-n
        result = reranked[:self.top_n]

        logger.info(
            f"[JinaReranker] Reranking complete: {len(nodes)} -> {len(result)} nodes "
            f"(filtered {filtered_count} below threshold {effective_threshold:.3f})"
        )

        return result

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """
        Convenience method for reranking with a query string.

        Args:
            query: Query string
            nodes: List of NodeWithScore objects

        Returns:
            Reranked list of NodeWithScore objects
        """
        query_bundle = QueryBundle(query_str=query)
        return self._postprocess_nodes(nodes, query_bundle)

    async def arerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """
        Async version of rerank (runs sync under the hood).
        """
        return self.rerank(query, nodes)


class SimpleReranker(BaseNodePostprocessor):
    """
    Simple score-based reranker as fallback when Jina model is unavailable.
    Uses the original retrieval scores for ranking.
    """

    # Pydantic fields
    top_n: int = settings.RAG_RERANK_TOP_N
    score_threshold: float = 0.0

    def __init__(
        self,
        top_n: int = None,
        score_threshold: float = None,
        **kwargs,
    ):
        super().__init__(
            top_n=top_n or settings.RAG_RERANK_TOP_N,
            score_threshold=score_threshold or 0.0,
            **kwargs,
        )

        logger.info(f"[SimpleReranker] Initialized - top_n: {self.top_n}")

    @classmethod
    def class_name(cls) -> str:
        return "SimpleReranker"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Simply filter and truncate to top-n."""
        filtered = [n for n in nodes if n.score >= self.score_threshold]
        filtered.sort(key=lambda x: x.score, reverse=True)
        return filtered[:self.top_n]


# Global singleton
_jina_reranker: Optional[JinaReranker] = None


def get_jina_reranker() -> JinaReranker:
    """Get the global Jina reranker instance."""
    global _jina_reranker
    if _jina_reranker is None:
        _jina_reranker = JinaReranker()
    return _jina_reranker


def get_reranker(use_jina: bool = True) -> BaseNodePostprocessor:
    """
    Get a reranker instance.

    Args:
        use_jina: Whether to use Jina reranker (default True)

    Returns:
        Reranker instance
    """
    if use_jina and settings.RAG_ENABLE_RERANK:
        try:
            return get_jina_reranker()
        except Exception as e:
            logger.warning(f"[Reranker] Failed to create Jina reranker: {e}, using simple reranker")
            return SimpleReranker()
    return SimpleReranker()
