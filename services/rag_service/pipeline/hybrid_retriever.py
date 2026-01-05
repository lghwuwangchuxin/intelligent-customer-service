"""Hybrid retrieval module combining vector and keyword search."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from services.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedDocument:
    """Retrieved document with metadata."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # 'vector', 'bm25', or 'hybrid'


@dataclass
class RetrievalResult:
    """Result of retrieval operation."""
    documents: List[RetrievedDocument]
    query: str
    total_candidates: int = 0
    latency_ms: int = 0


class HybridRetriever:
    """
    Hybrid retriever combining vector and BM25 search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from:
    - Dense vector search (semantic similarity)
    - BM25/keyword search (lexical matching)
    """

    def __init__(
        self,
        vector_store=None,
        elasticsearch_client=None,
        embedding_model=None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Vector store client (e.g., Milvus, Qdrant)
            elasticsearch_client: Elasticsearch client for BM25
            embedding_model: Embedding model for query vectorization
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            rrf_k: RRF parameter k (default 60)
        """
        self.vector_store = vector_store
        self.es_client = elasticsearch_client
        self.embedding_model = embedding_model
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        knowledge_base_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        expanded_queries: Optional[List[str]] = None,
        hyde_passage: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Perform hybrid retrieval.

        Args:
            query: Search query
            top_k: Number of results to return
            knowledge_base_id: Filter by knowledge base
            filters: Additional filters
            expanded_queries: Expanded queries for multi-query retrieval
            hyde_passage: HyDE generated passage for embedding

        Returns:
            RetrievalResult with fused documents
        """
        import time
        start_time = time.time()

        # Prepare queries for retrieval
        queries = [query]
        if expanded_queries:
            queries.extend(expanded_queries)

        # Run vector and BM25 search in parallel
        tasks = []

        # Vector search
        if self.vector_store and self.embedding_model:
            # Use HyDE passage if available
            embed_text = hyde_passage if hyde_passage else query
            tasks.append(self._vector_search(embed_text, top_k * 2, knowledge_base_id, filters))

        # BM25 search for all queries
        if self.es_client:
            for q in queries:
                tasks.append(self._bm25_search(q, top_k * 2, knowledge_base_id, filters))

        # Execute searches
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        vector_docs = []
        bm25_docs = []

        idx = 0
        if self.vector_store and self.embedding_model:
            if not isinstance(results[idx], Exception):
                vector_docs = results[idx]
            else:
                logger.warning(f"Vector search failed: {results[idx]}")
            idx += 1

        if self.es_client:
            for i in range(len(queries)):
                if idx + i < len(results) and not isinstance(results[idx + i], Exception):
                    bm25_docs.extend(results[idx + i])
                elif idx + i < len(results):
                    logger.warning(f"BM25 search failed: {results[idx + i]}")

        # Fuse results using RRF
        fused_docs = self._rrf_fusion(vector_docs, bm25_docs, top_k)

        latency_ms = int((time.time() - start_time) * 1000)

        return RetrievalResult(
            documents=fused_docs,
            query=query,
            total_candidates=len(vector_docs) + len(bm25_docs),
            latency_ms=latency_ms,
        )

    async def _vector_search(
        self,
        query: str,
        top_k: int,
        knowledge_base_id: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievedDocument]:
        """Perform vector similarity search."""
        try:
            # Get query embedding
            query_embedding = await self.embedding_model.embed(query)

            # Build filter expression
            filter_expr = self._build_vector_filter(knowledge_base_id, filters)

            # Search vector store
            results = await self.vector_store.search(
                vector=query_embedding,
                top_k=top_k,
                filter_expr=filter_expr,
            )

            return [
                RetrievedDocument(
                    id=r.id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                    source="vector",
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        knowledge_base_id: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievedDocument]:
        """Perform BM25 keyword search using Elasticsearch."""
        try:
            # Build ES query
            es_query = self._build_es_query(query, knowledge_base_id, filters)

            # Search Elasticsearch
            index_name = f"kb_{knowledge_base_id}" if knowledge_base_id else "knowledge_base"
            response = await self.es_client.search(
                index=index_name,
                body={
                    "query": es_query,
                    "size": top_k,
                    "_source": ["content", "metadata"],
                },
            )

            return [
                RetrievedDocument(
                    id=hit["_id"],
                    content=hit["_source"].get("content", ""),
                    score=hit["_score"],
                    metadata=hit["_source"].get("metadata", {}),
                    source="bm25",
                )
                for hit in response["hits"]["hits"]
            ]
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []

    def _rrf_fusion(
        self,
        vector_docs: List[RetrievedDocument],
        bm25_docs: List[RetrievedDocument],
        top_k: int,
    ) -> List[RetrievedDocument]:
        """
        Fuse results using Reciprocal Rank Fusion.

        RRF Score = sum(1 / (k + rank_i)) for each ranking list

        Args:
            vector_docs: Documents from vector search
            bm25_docs: Documents from BM25 search
            top_k: Number of results to return

        Returns:
            Fused and re-ranked documents
        """
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, RetrievedDocument] = {}

        # Score vector results
        for rank, doc in enumerate(vector_docs, start=1):
            rrf_scores[doc.id] += self.vector_weight / (self.rrf_k + rank)
            doc_map[doc.id] = doc

        # Score BM25 results (deduplicate by id)
        seen_bm25 = set()
        bm25_rank = 0
        for doc in bm25_docs:
            if doc.id not in seen_bm25:
                bm25_rank += 1
                rrf_scores[doc.id] += self.bm25_weight / (self.rrf_k + bm25_rank)
                seen_bm25.add(doc.id)
                if doc.id not in doc_map:
                    doc_map[doc.id] = doc

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build result list
        results = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id]
            results.append(RetrievedDocument(
                id=doc.id,
                content=doc.content,
                score=rrf_scores[doc_id],
                metadata=doc.metadata,
                source="hybrid",
            ))

        return results

    def _build_vector_filter(
        self,
        knowledge_base_id: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Build filter expression for vector store."""
        conditions = []

        if knowledge_base_id:
            conditions.append(f'knowledge_base_id == "{knowledge_base_id}"')

        if filters:
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f'{key} == {value}')

        return " && ".join(conditions) if conditions else None

    def _build_es_query(
        self,
        query: str,
        knowledge_base_id: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build Elasticsearch query."""
        must = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "title^2"],
                    "type": "best_fields",
                }
            }
        ]

        filter_clauses = []

        if knowledge_base_id:
            filter_clauses.append({
                "term": {"metadata.knowledge_base_id": knowledge_base_id}
            })

        if filters:
            for key, value in filters.items():
                filter_clauses.append({
                    "term": {f"metadata.{key}": value}
                })

        return {
            "bool": {
                "must": must,
                "filter": filter_clauses,
            }
        }


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self):
        self.documents = []

    async def search(
        self,
        vector: List[float],
        top_k: int,
        filter_expr: Optional[str] = None,
    ):
        """Mock search returning empty results."""
        return []

    async def insert(self, documents: List[Dict[str, Any]]):
        """Mock insert."""
        self.documents.extend(documents)


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    async def embed(self, text: str) -> List[float]:
        """Generate mock embedding."""
        import hashlib
        # Generate deterministic mock embedding based on text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_bytes[:self.dimension]]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for batch."""
        return [await self.embed(text) for text in texts]
