"""
Hybrid ES + Milvus Retriever for RAG.
=====================================

This module provides a hybrid retriever that combines:
- Milvus vector search (semantic similarity)
- Elasticsearch BM25 search (keyword matching)

Using Reciprocal Rank Fusion (RRF) to merge results.

Retrieval Flow
--------------
```
User Query
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
┌──────────────┐                  ┌──────────────┐
│  Vectorize   │                  │   ES BM25    │
│   (embed)    │                  │   Search     │
└──────────────┘                  └──────────────┘
    │                                     │
    ▼                                     ▼
┌──────────────┐                  ┌──────────────┐
│ Milvus ANN   │                  │  Keyword     │
│   Search     │                  │  Matching    │
└──────────────┘                  └──────────────┘
    │                                     │
    └────────────────┬────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │  RRF Fusion  │
              │   Scoring    │
              └──────────────┘
                     │
                     ▼
              ┌──────────────┐
              │   Reranker   │
              │  (optional)  │
              └──────────────┘
                     │
                     ▼
              ┌──────────────┐
              │ Final Top-K  │
              │   Results    │
              └──────────────┘
```

Author: Intelligent Customer Service Team
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever

from langchain_core.documents import Document as LangchainDocument
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.domain.base.entities import DocumentChunk, ChunkSearchResult
from app.domain.base.interfaces import IEmbeddingService
from config.settings import settings

logger = logging.getLogger(__name__)


class HybridESMilvusRetriever(BaseRetriever):
    """
    Hybrid retriever combining ES BM25 and Milvus vector search.

    This retriever implements LangChain's BaseRetriever interface for
    compatibility with LangChain LCEL chains and RAG pipelines.

    Attributes
    ----------
    es_manager : Any
        Elasticsearch manager for BM25 search
    vector_store : Any
        Milvus vector store for semantic search
    embedding_service : IEmbeddingService
        Service for generating query embeddings
    top_k : int
        Number of final results to return
    vector_weight : float
        Weight for vector search results in RRF
    bm25_weight : float
        Weight for BM25 search results in RRF

    Example
    -------
    ```python
    retriever = HybridESMilvusRetriever(
        es_manager=es_manager,
        vector_store=vector_store,
        embedding_service=embedding_service,
        top_k=10,
    )

    # LangChain compatible
    docs = retriever.invoke("How to reset password?")

    # Async
    docs = await retriever.ainvoke("How to reset password?")
    ```
    """

    # Pydantic config
    es_manager: Any = None
    vector_store: Any = None
    embedding_service: Any = None  # IEmbeddingService
    top_k: int = 10
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    initial_retrieval_k: int = 20
    use_reranker: bool = False
    reranker: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        es_manager: Any = None,
        vector_store: Any = None,
        embedding_service: Any = None,
        top_k: int = None,
        vector_weight: float = None,
        bm25_weight: float = None,
        use_reranker: bool = None,
        reranker: Any = None,
        **kwargs,
    ):
        """
        Initialize hybrid retriever.

        Parameters
        ----------
        es_manager : ElasticsearchManager
            ES manager for BM25 search
        vector_store : VectorStoreManager
            Milvus manager for vector search
        embedding_service : IEmbeddingService
            Embedding service for query vectorization
        top_k : int, optional
            Final number of results (default from settings)
        vector_weight : float, optional
            Vector search weight (default from settings)
        bm25_weight : float, optional
            BM25 search weight (default from settings)
        use_reranker : bool, optional
            Whether to use reranker (default from settings)
        reranker : Any, optional
            Reranker instance
        """
        super().__init__(**kwargs)

        self.es_manager = es_manager
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k = top_k or settings.RAG_FINAL_TOP_K
        self.vector_weight = vector_weight or settings.HYBRID_MILVUS_WEIGHT
        self.bm25_weight = bm25_weight or settings.HYBRID_ES_WEIGHT
        self.initial_retrieval_k = settings.HYBRID_SEARCH_TOP_K
        self.use_reranker = use_reranker if use_reranker is not None else settings.RAG_ENABLE_RERANK
        self.reranker = reranker

        logger.info(
            f"[HybridRetriever] Initialized: top_k={self.top_k}, "
            f"weights=(v={self.vector_weight}, bm25={self.bm25_weight})"
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[LangchainDocument]:
        """
        Synchronous retrieval (wraps async).

        Parameters
        ----------
        query : str
            Search query
        run_manager : CallbackManagerForRetrieverRun, optional
            Callback manager

        Returns
        -------
        List[LangchainDocument]
            Retrieved documents
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new loop if current is running
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(
                self._aget_relevant_documents(query, run_manager=run_manager)
            )
        except RuntimeError:
            # No event loop
            return asyncio.run(
                self._aget_relevant_documents(query, run_manager=run_manager)
            )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[LangchainDocument]:
        """
        Asynchronous hybrid retrieval.

        Parameters
        ----------
        query : str
            Search query
        run_manager : CallbackManagerForRetrieverRun, optional
            Callback manager

        Returns
        -------
        List[LangchainDocument]
            Retrieved documents
        """
        start_time = time.time()
        logger.info(f"[HybridRetriever] Query: {query[:100]}...")

        # Step 1: Generate query embedding
        query_vector = await self._get_query_embedding(query)

        if query_vector is None:
            logger.warning("[HybridRetriever] Failed to generate query embedding")
            # Fall back to BM25 only
            return await self._bm25_only_search(query)

        # Step 2: Parallel search in Milvus and ES
        vector_results, bm25_results = await asyncio.gather(
            self._search_milvus(query, query_vector),
            self._search_es(query),
            return_exceptions=True,
        )

        # Handle errors
        if isinstance(vector_results, Exception):
            logger.error(f"[HybridRetriever] Milvus error: {vector_results}")
            vector_results = []
        if isinstance(bm25_results, Exception):
            logger.error(f"[HybridRetriever] ES error: {bm25_results}")
            bm25_results = []

        logger.info(
            f"[HybridRetriever] Retrieved: Milvus={len(vector_results)}, ES={len(bm25_results)}"
        )

        # Step 3: RRF Fusion
        fused_results = self._rrf_fusion(vector_results, bm25_results)
        logger.info(f"[HybridRetriever] After RRF fusion: {len(fused_results)} results")

        # Step 4: Optional reranking
        if self.use_reranker and self.reranker and fused_results:
            fused_results = await self._rerank(query, fused_results)
            logger.info(f"[HybridRetriever] After reranking: {len(fused_results)} results")

        # Step 5: Get final top_k
        final_results = fused_results[:self.top_k]

        # Step 6: Fetch full content from ES if needed
        if self.es_manager:
            final_results = await self._enrich_from_es(final_results)

        # Convert to LangChain documents
        documents = self._to_langchain_documents(final_results)

        elapsed = time.time() - start_time
        logger.info(
            f"[HybridRetriever] Complete: {len(documents)} docs in {elapsed:.3f}s"
        )

        return documents

    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for query."""
        if not self.embedding_service:
            return None

        try:
            embeddings = await self.embedding_service.embed_async([query])
            return embeddings[0] if embeddings else None
        except Exception as e:
            logger.error(f"[HybridRetriever] Embedding error: {e}")
            return None

    async def _search_milvus(
        self,
        query: str,
        query_vector: List[float],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search Milvus for similar vectors.

        Returns list of (chunk_id, score, metadata) tuples.
        """
        if not self.vector_store:
            return []

        try:
            # Use vector store's retrieve method
            results = self.vector_store.retrieve(query, top_k=self.initial_retrieval_k)

            return [
                (
                    node.node.node_id,  # chunk_id
                    node.score,
                    dict(node.node.metadata),
                )
                for node in results
            ]
        except Exception as e:
            logger.error(f"[HybridRetriever] Milvus search error: {e}")
            return []

    async def _search_es(
        self,
        query: str,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search ES using BM25.

        Returns list of (chunk_id, score, metadata) tuples.
        """
        if not self.es_manager:
            return []

        try:
            results = await self.es_manager.search_by_keyword(
                query=query,
                top_k=self.initial_retrieval_k,
            )

            return [
                (
                    result.chunk_id,
                    result.score,
                    {
                        "content": result.content,
                        "doc_id": result.doc_id,
                        **result.metadata,
                    },
                )
                for result in results
            ]
        except Exception as e:
            logger.error(f"[HybridRetriever] ES search error: {e}")
            return []

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[str, float, Dict[str, Any]]],
        bm25_results: List[Tuple[str, float, Dict[str, Any]]],
        k: int = 60,
    ) -> List[Tuple[str, float, Dict[str, Any], str]]:
        """
        Reciprocal Rank Fusion.

        Parameters
        ----------
        vector_results : list
            Results from vector search
        bm25_results : list
            Results from BM25 search
        k : int
            RRF constant (default 60)

        Returns
        -------
        list
            Fused results as (chunk_id, score, metadata, source) tuples
        """
        scores: Dict[str, float] = {}
        metadata_map: Dict[str, Dict[str, Any]] = {}
        sources: Dict[str, List[str]] = {}

        # Score vector results
        for rank, (chunk_id, _, metadata) in enumerate(vector_results, 1):
            rrf_score = self.vector_weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            metadata_map[chunk_id] = metadata
            sources.setdefault(chunk_id, []).append("milvus")

        # Score BM25 results
        for rank, (chunk_id, _, metadata) in enumerate(bm25_results, 1):
            rrf_score = self.bm25_weight / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in metadata_map:
                metadata_map[chunk_id] = metadata
            else:
                # Merge metadata, prefer ES content
                if "content" in metadata:
                    metadata_map[chunk_id]["content"] = metadata["content"]
            sources.setdefault(chunk_id, []).append("es")

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [
            (
                chunk_id,
                scores[chunk_id],
                metadata_map[chunk_id],
                "+".join(sources[chunk_id]),
            )
            for chunk_id in sorted_ids
        ]

    async def _rerank(
        self,
        query: str,
        results: List[Tuple[str, float, Dict[str, Any], str]],
    ) -> List[Tuple[str, float, Dict[str, Any], str]]:
        """
        Rerank results using reranker model.

        Parameters
        ----------
        query : str
            Original query
        results : list
            Fused results

        Returns
        -------
        list
            Reranked results
        """
        if not self.reranker or not results:
            return results

        try:
            # Extract texts for reranking
            texts = [r[2].get("content", "") for r in results]

            # Rerank
            rerank_scores = await self.reranker.rerank(query, texts)

            # Combine with original results
            reranked = [
                (results[i][0], score, results[i][2], results[i][3])
                for i, score in enumerate(rerank_scores)
            ]

            # Sort by rerank score
            reranked.sort(key=lambda x: x[1], reverse=True)

            return reranked

        except Exception as e:
            logger.error(f"[HybridRetriever] Rerank error: {e}")
            return results

    async def _enrich_from_es(
        self,
        results: List[Tuple[str, float, Dict[str, Any], str]],
    ) -> List[Tuple[str, float, Dict[str, Any], str]]:
        """
        Fetch full content from ES for results that only have partial metadata.
        """
        if not self.es_manager:
            return results

        # Find results that need enrichment (missing content)
        needs_enrichment = [
            r for r in results if not r[2].get("content")
        ]

        if not needs_enrichment:
            return results

        try:
            chunk_ids = [r[0] for r in needs_enrichment]
            full_chunks = await self.es_manager.get_chunks_by_ids(chunk_ids)

            # Build lookup map
            chunk_map = {c.chunk_id: c for c in full_chunks}

            # Enrich results
            enriched = []
            for chunk_id, score, metadata, source in results:
                if chunk_id in chunk_map:
                    chunk = chunk_map[chunk_id]
                    metadata = {
                        **metadata,
                        "content": chunk.content,
                        "doc_id": chunk.doc_id,
                        **chunk.metadata,
                    }
                enriched.append((chunk_id, score, metadata, source))

            return enriched

        except Exception as e:
            logger.error(f"[HybridRetriever] Enrich error: {e}")
            return results

    async def _bm25_only_search(self, query: str) -> List[LangchainDocument]:
        """Fallback to BM25-only search when embedding fails."""
        if not self.es_manager:
            return []

        try:
            results = await self.es_manager.search_by_keyword(
                query=query,
                top_k=self.top_k,
            )

            return [
                LangchainDocument(
                    page_content=r.content,
                    metadata={
                        "chunk_id": r.chunk_id,
                        "doc_id": r.doc_id,
                        "score": r.score,
                        "source": "es_bm25",
                        **r.metadata,
                    },
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"[HybridRetriever] BM25 fallback error: {e}")
            return []

    def _to_langchain_documents(
        self,
        results: List[Tuple[str, float, Dict[str, Any], str]],
    ) -> List[LangchainDocument]:
        """Convert results to LangChain documents."""
        documents = []
        for chunk_id, score, metadata, source in results:
            content = metadata.pop("content", "")
            doc = LangchainDocument(
                page_content=content,
                metadata={
                    "chunk_id": chunk_id,
                    "score": score,
                    "retrieval_source": source,
                    **metadata,
                },
            )
            documents.append(doc)
        return documents


class HybridLlamaIndexRetriever(LlamaBaseRetriever):
    """
    LlamaIndex compatible hybrid retriever.

    For use with LlamaIndex RAG pipelines and query engines.

    Example
    -------
    ```python
    retriever = HybridLlamaIndexRetriever(
        es_manager=es_manager,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )

    # Use with LlamaIndex
    nodes = retriever.retrieve("How to reset password?")
    ```
    """

    def __init__(
        self,
        es_manager: Any = None,
        vector_store: Any = None,
        embedding_service: Any = None,
        top_k: int = None,
        vector_weight: float = None,
        bm25_weight: float = None,
    ):
        """Initialize LlamaIndex hybrid retriever."""
        super().__init__()

        self._hybrid_retriever = HybridESMilvusRetriever(
            es_manager=es_manager,
            vector_store=vector_store,
            embedding_service=embedding_service,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes for query.

        Parameters
        ----------
        query_bundle : QueryBundle
            LlamaIndex query bundle

        Returns
        -------
        List[NodeWithScore]
            Retrieved nodes with scores
        """
        query = query_bundle.query_str

        # Use hybrid retriever
        docs = self._hybrid_retriever._get_relevant_documents(query)

        # Convert to LlamaIndex nodes
        nodes = []
        for doc in docs:
            node = TextNode(
                text=doc.page_content,
                id_=doc.metadata.get("chunk_id", ""),
                metadata={
                    k: v for k, v in doc.metadata.items()
                    if k not in ["chunk_id", "score"]
                },
            )
            score = doc.metadata.get("score", 0.5)
            nodes.append(NodeWithScore(node=node, score=score))

        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Async retrieve nodes for query.

        Parameters
        ----------
        query_bundle : QueryBundle
            LlamaIndex query bundle

        Returns
        -------
        List[NodeWithScore]
            Retrieved nodes with scores
        """
        query = query_bundle.query_str

        # Use hybrid retriever async
        docs = await self._hybrid_retriever._aget_relevant_documents(query)

        # Convert to LlamaIndex nodes
        nodes = []
        for doc in docs:
            node = TextNode(
                text=doc.page_content,
                id_=doc.metadata.get("chunk_id", ""),
                metadata={
                    k: v for k, v in doc.metadata.items()
                    if k not in ["chunk_id", "score"]
                },
            )
            score = doc.metadata.get("score", 0.5)
            nodes.append(NodeWithScore(node=node, score=score))

        return nodes


# ==================== Factory Functions ====================

def create_hybrid_retriever(
    es_manager: Any = None,
    vector_store: Any = None,
    embedding_service: Any = None,
    top_k: int = None,
    vector_weight: float = None,
    bm25_weight: float = None,
    framework: str = "langchain",
) -> Any:
    """
    Create hybrid retriever for specified framework.

    Parameters
    ----------
    es_manager : ElasticsearchManager
        ES manager instance
    vector_store : VectorStoreManager
        Vector store instance
    embedding_service : IEmbeddingService
        Embedding service instance
    top_k : int, optional
        Number of results
    vector_weight : float, optional
        Vector search weight
    bm25_weight : float, optional
        BM25 search weight
    framework : str
        Target framework: "langchain" or "llamaindex"

    Returns
    -------
    BaseRetriever or LlamaBaseRetriever
        Framework-appropriate retriever
    """
    if framework == "llamaindex":
        return HybridLlamaIndexRetriever(
            es_manager=es_manager,
            vector_store=vector_store,
            embedding_service=embedding_service,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
    else:
        return HybridESMilvusRetriever(
            es_manager=es_manager,
            vector_store=vector_store,
            embedding_service=embedding_service,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
