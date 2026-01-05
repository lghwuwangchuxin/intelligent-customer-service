"""
Hybrid Store Manager - Coordinates ES and Milvus for unified storage.
======================================================================

This module provides a unified interface for hybrid storage operations
combining Elasticsearch (text + metadata) with Milvus (vector search).

Architecture
------------
```
                    HybridStoreManager
                           │
           ┌───────────────┼───────────────┐
           ▼               │               ▼
    ElasticsearchManager   │      VectorStoreManager
    (Text + Metadata)      │      (Vector Search)
           │               │               │
           ▼               ▼               ▼
    ┌──────────────┐   ┌───────┐   ┌──────────────┐
    │ Elasticsearch │   │ chunk │   │    Milvus    │
    │   (BM25)      │◀──│  _id  │──▶│   (ANN)      │
    └──────────────┘   └───────┘   └──────────────┘
```

Data Flow
---------
**Indexing:**
1. DocumentProcessor creates chunks with unique chunk_ids
2. HybridStoreManager generates embeddings
3. Stores text + metadata + embedding in ES (for BM25 and backup)
4. Stores embedding + chunk_id in Milvus (for ANN search)

**Retrieval:**
1. Query is vectorized
2. Milvus returns Top-K chunk_ids with scores
3. (Optional) ES BM25 search returns chunk_ids with scores
4. RRF fusion combines results
5. ES retrieves full text + metadata for final chunk_ids

Author: Intelligent Customer Service Team
Version: 1.0.0
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.domain.base.entities import (
    DocumentChunk,
    ChunkSearchResult,
    HybridIndexResult,
)
from app.domain.base.interfaces import IHybridStore, IEmbeddingService
from config.settings import settings

logger = logging.getLogger(__name__)


class HybridStoreManager(IHybridStore):
    """
    Hybrid Store Manager - Coordinates ES and Milvus operations.

    This class implements the IHybridStore interface and provides:
    - Unified indexing to both ES and Milvus
    - Hybrid search combining vector and BM25
    - Consistent deletion across both stores
    - Health monitoring for both backends

    Attributes
    ----------
    es_manager : ElasticsearchManager
        Elasticsearch manager for text and metadata storage
    vector_store : VectorStoreManager
        Milvus vector store for semantic search
    embedding_service : IEmbeddingService
        Embedding service for generating vectors

    Example
    -------
    ```python
    hybrid_store = HybridStoreManager(
        es_manager=es_manager,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )

    # Index documents
    result = await hybrid_store.index_chunks(chunks)

    # Hybrid search
    results = await hybrid_store.hybrid_search(
        query="How to reset password?",
        query_vector=embedding,
        top_k=10,
    )
    ```
    """

    def __init__(
        self,
        es_manager: Any,  # ElasticsearchManager
        vector_store: Any,  # VectorStoreManager
        embedding_service: IEmbeddingService,
        vector_weight: float = None,
        bm25_weight: float = None,
    ):
        """
        Initialize HybridStoreManager.

        Parameters
        ----------
        es_manager : ElasticsearchManager
            Elasticsearch manager instance
        vector_store : VectorStoreManager
            Milvus vector store instance
        embedding_service : IEmbeddingService
            Embedding service for generating vectors
        vector_weight : float, optional
            Weight for vector search results (default from settings)
        bm25_weight : float, optional
            Weight for BM25 search results (default from settings)
        """
        self.es_manager = es_manager
        self.vector_store = vector_store
        self.embedding_service = embedding_service

        # Search weights
        self.vector_weight = vector_weight or settings.HYBRID_MILVUS_WEIGHT
        self.bm25_weight = bm25_weight or settings.HYBRID_ES_WEIGHT

        logger.info(
            f"[HybridStore] Initialized with weights: "
            f"vector={self.vector_weight}, bm25={self.bm25_weight}"
        )

    async def index_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 50,
        generate_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Index document chunks to both ES and Milvus.

        Parameters
        ----------
        chunks : List[DocumentChunk]
            List of document chunks to index
        batch_size : int
            Number of chunks per batch
        generate_embeddings : bool
            Whether to generate embeddings (if not already present)

        Returns
        -------
        Dict[str, Any]
            Result with counts for ES and Milvus indexing
        """
        if not chunks:
            return {
                "success": True,
                "chunks_processed": 0,
                "es_indexed": 0,
                "milvus_indexed": 0,
            }

        start_time = time.time()
        logger.info(f"[HybridStore] Indexing {len(chunks)} chunks...")

        # Ensure all chunks have unique IDs
        for chunk in chunks:
            if not chunk.chunk_id:
                chunk.chunk_id = str(uuid.uuid4())

        # Generate embeddings if needed
        chunks_needing_embedding = [c for c in chunks if c.embedding is None]
        if generate_embeddings and chunks_needing_embedding:
            logger.info(f"[HybridStore] Generating embeddings for {len(chunks_needing_embedding)} chunks")
            await self._generate_embeddings(chunks_needing_embedding, batch_size)

        # Verify all chunks have embeddings
        chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
        if len(chunks_with_embeddings) < len(chunks):
            logger.warning(
                f"[HybridStore] {len(chunks) - len(chunks_with_embeddings)} chunks "
                "without embeddings, will only index to ES"
            )

        # Index to ES and Milvus in parallel
        es_result, milvus_result = await asyncio.gather(
            self._index_to_es(chunks, batch_size),
            self._index_to_milvus(chunks_with_embeddings, batch_size),
            return_exceptions=True,
        )

        # Handle results
        es_count = 0 if isinstance(es_result, Exception) else es_result
        milvus_count = 0 if isinstance(milvus_result, Exception) else milvus_result

        if isinstance(es_result, Exception):
            logger.error(f"[HybridStore] ES indexing failed: {es_result}")
        if isinstance(milvus_result, Exception):
            logger.error(f"[HybridStore] Milvus indexing failed: {milvus_result}")

        elapsed = time.time() - start_time

        result = {
            "success": es_count > 0 or milvus_count > 0,
            "chunks_processed": len(chunks),
            "es_indexed": es_count,
            "milvus_indexed": milvus_count,
            "processing_time_ms": round(elapsed * 1000, 2),
            "es_error": str(es_result) if isinstance(es_result, Exception) else None,
            "milvus_error": str(milvus_result) if isinstance(milvus_result, Exception) else None,
        }

        logger.info(
            f"[HybridStore] Indexing complete: ES={es_count}, Milvus={milvus_count}, "
            f"time={elapsed:.2f}s"
        )

        return result

    async def _generate_embeddings(
        self,
        chunks: List[DocumentChunk],
        batch_size: int,
    ) -> None:
        """Generate embeddings for chunks in batches."""
        texts = [chunk.content for chunk in chunks]

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            try:
                embeddings = await self.embedding_service.embed_async(batch_texts)

                for chunk, embedding in zip(batch_chunks, embeddings):
                    chunk.embedding = embedding

                logger.debug(f"[HybridStore] Generated embeddings for batch {i//batch_size + 1}")

            except Exception as e:
                logger.error(f"[HybridStore] Embedding generation failed for batch: {e}")
                # Continue with other batches

    async def _index_to_es(
        self,
        chunks: List[DocumentChunk],
        batch_size: int,
    ) -> int:
        """Index chunks to Elasticsearch."""
        if not self.es_manager:
            logger.warning("[HybridStore] ES manager not available, skipping ES indexing")
            return 0

        try:
            indexed_ids = await self.es_manager.bulk_index_chunks(chunks, batch_size)
            return len(indexed_ids)
        except Exception as e:
            logger.error(f"[HybridStore] ES indexing error: {e}")
            raise

    async def _index_to_milvus(
        self,
        chunks: List[DocumentChunk],
        batch_size: int,
    ) -> int:
        """Index chunk embeddings to Milvus."""
        if not self.vector_store:
            logger.warning("[HybridStore] Vector store not available, skipping Milvus indexing")
            return 0

        if not chunks:
            return 0

        try:
            # Convert DocumentChunk to LlamaIndex TextNode for Milvus
            from llama_index.core.schema import TextNode

            nodes = []
            for chunk in chunks:
                if chunk.embedding:
                    node = TextNode(
                        text=chunk.content,
                        id_=chunk.chunk_id,
                        metadata={
                            **chunk.metadata,
                            "doc_id": chunk.doc_id,
                            "chunk_index": chunk.chunk_index,
                        },
                        embedding=chunk.embedding,
                    )
                    nodes.append(node)

            if nodes:
                # Use vector store's add_nodes method
                ids = self.vector_store.add_nodes(nodes)
                return len(ids)

            return 0

        except Exception as e:
            logger.error(f"[HybridStore] Milvus indexing error: {e}")
            raise

    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = None,
        bm25_weight: float = None,
    ) -> List[ChunkSearchResult]:
        """
        Perform hybrid search combining vector and BM25 results.

        Parameters
        ----------
        query : str
            Text query for BM25 search
        query_vector : List[float]
            Query embedding for vector search
        top_k : int
            Maximum number of results
        filters : Dict[str, Any], optional
            Metadata filters for ES
        vector_weight : float, optional
            Weight for vector results (overrides default)
        bm25_weight : float, optional
            Weight for BM25 results (overrides default)

        Returns
        -------
        List[ChunkSearchResult]
            Fused search results with scores
        """
        start_time = time.time()

        vector_weight = vector_weight or self.vector_weight
        bm25_weight = bm25_weight or self.bm25_weight

        # Determine initial retrieval count (higher for fusion)
        initial_top_k = min(top_k * 3, settings.HYBRID_SEARCH_TOP_K)

        logger.info(
            f"[HybridStore] Hybrid search: query='{query[:50]}...', "
            f"top_k={top_k}, weights=(v={vector_weight}, bm25={bm25_weight})"
        )

        # Search both stores in parallel
        vector_results, bm25_results = await asyncio.gather(
            self._search_milvus(query_vector, initial_top_k),
            self._search_es_bm25(query, initial_top_k, filters),
            return_exceptions=True,
        )

        # Handle errors
        if isinstance(vector_results, Exception):
            logger.error(f"[HybridStore] Milvus search failed: {vector_results}")
            vector_results = []
        if isinstance(bm25_results, Exception):
            logger.error(f"[HybridStore] ES BM25 search failed: {bm25_results}")
            bm25_results = []

        # Fuse results using RRF
        fused_results = self._rrf_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=top_k,
        )

        # Get full content from ES for top results
        if fused_results:
            chunk_ids = [r.chunk.chunk_id for r in fused_results]
            full_chunks = await self._get_chunks_from_es(chunk_ids)

            # Update results with full content
            chunk_map = {c.chunk_id: c for c in full_chunks}
            for result in fused_results:
                if result.chunk.chunk_id in chunk_map:
                    full_chunk = chunk_map[result.chunk.chunk_id]
                    result.chunk.content = full_chunk.content
                    result.chunk.metadata = full_chunk.metadata

        elapsed = time.time() - start_time
        logger.info(
            f"[HybridStore] Hybrid search complete: {len(fused_results)} results "
            f"in {elapsed:.3f}s"
        )

        return fused_results

    async def _search_milvus(
        self,
        query_vector: List[float],
        top_k: int,
    ) -> List[ChunkSearchResult]:
        """
        Search Milvus for similar vectors using direct vector query.

        Parameters
        ----------
        query_vector : List[float]
            Query embedding vector
        top_k : int
            Number of results to return

        Returns
        -------
        List[ChunkSearchResult]
            Search results with chunk_id and similarity score
        """
        if not self.vector_store:
            return []

        try:
            from pymilvus import Collection, connections

            # Connect to Milvus
            connections.connect(
                alias="hybrid_search",
                host=self.vector_store.host if hasattr(self.vector_store, 'host') else settings.MILVUS_HOST,
                port=self.vector_store.port if hasattr(self.vector_store, 'port') else settings.MILVUS_PORT,
            )

            collection_name = (
                self.vector_store.collection_name
                if hasattr(self.vector_store, 'collection_name')
                else settings.MILVUS_COLLECTION
            )
            collection = Collection(collection_name, using="hybrid_search")
            collection.load()

            # Search by vector
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }

            results = collection.search(
                data=[query_vector],
                anns_field="vector",  # The vector field name
                param=search_params,
                limit=top_k,
                output_fields=["id", "text"],  # Fields to return
            )

            # Convert to ChunkSearchResult
            chunk_results = []
            for hits in results:
                for hit in hits:
                    chunk_id = str(hit.id) if hit.id else str(hit.entity.get("id", ""))
                    text = hit.entity.get("text", "") if hasattr(hit, 'entity') else ""

                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id="",  # Will be filled from ES
                        content=text,
                        metadata={},
                    )
                    chunk_results.append(ChunkSearchResult(
                        chunk=chunk,
                        score=float(hit.score) if hasattr(hit, 'score') else float(hit.distance),
                        source="milvus_vector",
                    ))

            logger.debug(f"[HybridStore] Milvus search returned {len(chunk_results)} results")
            return chunk_results

        except Exception as e:
            logger.error(f"[HybridStore] Milvus search error: {e}")
            return []
        finally:
            # Disconnect to avoid connection leak
            try:
                connections.disconnect("hybrid_search")
            except Exception:
                pass

    async def _search_es_bm25(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkSearchResult]:
        """Search ES using BM25."""
        if not self.es_manager:
            return []

        try:
            results = await self.es_manager.search_by_keyword(
                query=query,
                top_k=top_k,
                filters=filters,
            )

            # Convert ES results to ChunkSearchResult
            chunk_results = []
            for result in results:
                chunk = DocumentChunk(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    content=result.content,
                    metadata=result.metadata,
                )
                chunk_results.append(ChunkSearchResult(
                    chunk=chunk,
                    score=result.score,
                    source="es_bm25",
                ))

            return chunk_results

        except Exception as e:
            logger.error(f"[HybridStore] ES BM25 search error: {e}")
            return []

    async def _get_chunks_from_es(
        self,
        chunk_ids: List[str],
    ) -> List[DocumentChunk]:
        """Get full chunk content from ES by IDs."""
        if not self.es_manager or not chunk_ids:
            return []

        try:
            return await self.es_manager.get_chunks_by_ids(chunk_ids)
        except Exception as e:
            logger.error(f"[HybridStore] ES get chunks error: {e}")
            return []

    def _rrf_fusion(
        self,
        vector_results: List[ChunkSearchResult],
        bm25_results: List[ChunkSearchResult],
        vector_weight: float,
        bm25_weight: float,
        top_k: int,
        k: int = 60,  # RRF constant
    ) -> List[ChunkSearchResult]:
        """
        Reciprocal Rank Fusion (RRF) for combining search results.

        Formula: score = Σ(weight / (k + rank))

        Parameters
        ----------
        vector_results : List[ChunkSearchResult]
            Results from vector search
        bm25_results : List[ChunkSearchResult]
            Results from BM25 search
        vector_weight : float
            Weight for vector results
        bm25_weight : float
            Weight for BM25 results
        top_k : int
            Number of final results
        k : int
            RRF constant (default 60)

        Returns
        -------
        List[ChunkSearchResult]
            Fused and re-ranked results
        """
        # Build rank maps
        chunk_scores: Dict[str, float] = {}
        chunk_data: Dict[str, ChunkSearchResult] = {}

        # Score vector results
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result.chunk.chunk_id
            rrf_score = vector_weight / (k + rank)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            chunk_data[chunk_id] = result

        # Score BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.chunk.chunk_id
            rrf_score = bm25_weight / (k + rank)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

        # Sort by fused score
        sorted_ids = sorted(
            chunk_scores.keys(),
            key=lambda x: chunk_scores[x],
            reverse=True,
        )[:top_k]

        # Build final results
        fused_results = []
        for chunk_id in sorted_ids:
            result = chunk_data[chunk_id]
            fused_results.append(ChunkSearchResult(
                chunk=result.chunk,
                score=chunk_scores[chunk_id],
                source="hybrid",
            ))

        return fused_results

    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document from both ES and Milvus.

        Parameters
        ----------
        doc_id : str
            Document ID to delete

        Returns
        -------
        Dict[str, Any]
            Deletion results from both stores
        """
        logger.info(f"[HybridStore] Deleting document: {doc_id}")

        es_deleted = 0
        milvus_deleted = 0

        # Get chunk IDs from ES first
        chunk_ids = []
        if self.es_manager:
            try:
                # Get all chunks for this doc
                chunks = await self._get_chunks_by_doc_id(doc_id)
                chunk_ids = [c.chunk_id for c in chunks]

                # Delete from ES
                es_deleted = await self.es_manager.delete_by_doc_id(doc_id)
            except Exception as e:
                logger.error(f"[HybridStore] ES deletion error: {e}")

        # Delete from Milvus by chunk IDs
        if self.vector_store and chunk_ids:
            try:
                # Milvus deletion by IDs
                # Note: VectorStoreManager may need enhancement for this
                from pymilvus import Collection, connections

                connections.connect(
                    host=self.vector_store.host,
                    port=self.vector_store.port,
                )
                collection = Collection(self.vector_store.collection_name)

                # Delete by expression
                expr = f"id in {chunk_ids}"
                collection.delete(expr)
                milvus_deleted = len(chunk_ids)

            except Exception as e:
                logger.error(f"[HybridStore] Milvus deletion error: {e}")

        return {
            "success": es_deleted > 0 or milvus_deleted > 0,
            "doc_id": doc_id,
            "es_deleted": es_deleted,
            "milvus_deleted": milvus_deleted,
        }

    async def _get_chunks_by_doc_id(self, doc_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document from ES."""
        if not self.es_manager:
            return []

        try:
            # Search ES for chunks with this doc_id
            results = await self.es_manager.search_by_keyword(
                query="*",
                top_k=1000,
                filters={"doc_id": doc_id},
            )

            return [
                DocumentChunk(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    content=r.content,
                    metadata=r.metadata,
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"[HybridStore] Get chunks by doc_id error: {e}")
            return []

    async def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete document information from both stores.

        Parameters
        ----------
        doc_id : str
            Document ID

        Returns
        -------
        Dict[str, Any], optional
            Document info including chunks, metadata, and storage status
        """
        if not self.es_manager:
            return None

        try:
            # Get document metadata from ES
            metadata = await self.es_manager.get_document_metadata(doc_id)

            if metadata:
                # Get chunk count from ES
                chunks = await self._get_chunks_by_doc_id(doc_id)

                return {
                    "doc_id": doc_id,
                    "metadata": metadata,
                    "chunk_count": len(chunks),
                    "es_status": "indexed",
                    "milvus_status": "indexed" if chunks else "unknown",
                }

            return None

        except Exception as e:
            logger.error(f"[HybridStore] Get document info error: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of both storage backends.

        Returns
        -------
        Dict[str, Any]
            Combined health status for ES and Milvus
        """
        es_health = {"status": "unavailable"}
        milvus_health = {"status": "unavailable"}

        # Check ES
        if self.es_manager:
            try:
                es_health = await self.es_manager.health_check()
            except Exception as e:
                es_health = {"status": "error", "error": str(e)}

        # Check Milvus
        if self.vector_store:
            try:
                stats = self.vector_store.get_collection_stats()
                if "error" not in stats:
                    milvus_health = {
                        "status": "healthy",
                        "collection": self.vector_store.collection_name,
                        "num_entities": stats.get("num_entities", 0),
                    }
                else:
                    milvus_health = {"status": "error", "error": stats["error"]}
            except Exception as e:
                milvus_health = {"status": "error", "error": str(e)}

        # Overall status
        es_ok = es_health.get("status") == "healthy"
        milvus_ok = milvus_health.get("status") == "healthy"

        return {
            "status": "healthy" if es_ok and milvus_ok else "degraded",
            "elasticsearch": es_health,
            "milvus": milvus_health,
            "hybrid_enabled": settings.HYBRID_STORAGE_ENABLED,
        }

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from both storage backends.

        Returns
        -------
        Dict[str, Any]
            Combined statistics
        """
        es_stats = {}
        milvus_stats = {}

        if self.es_manager:
            try:
                es_stats = await self.es_manager.get_stats()
            except Exception as e:
                es_stats = {"error": str(e)}

        if self.vector_store:
            try:
                milvus_stats = self.vector_store.get_collection_stats()
            except Exception as e:
                milvus_stats = {"error": str(e)}

        return {
            "elasticsearch": es_stats,
            "milvus": milvus_stats,
            "hybrid_enabled": settings.HYBRID_STORAGE_ENABLED,
            "weights": {
                "vector": self.vector_weight,
                "bm25": self.bm25_weight,
            },
        }


# ==================== Factory and Singleton ====================

_hybrid_store_manager: Optional[HybridStoreManager] = None


def get_hybrid_store_manager() -> Optional[HybridStoreManager]:
    """
    Get global hybrid store manager instance.

    Returns None if hybrid storage is not enabled or not initialized.

    Returns
    -------
    HybridStoreManager, optional
        Hybrid store manager singleton
    """
    global _hybrid_store_manager
    return _hybrid_store_manager


def init_hybrid_store_manager(
    es_manager: Any,
    vector_store: Any,
    embedding_service: IEmbeddingService,
) -> HybridStoreManager:
    """
    Initialize the global hybrid store manager.

    Parameters
    ----------
    es_manager : ElasticsearchManager
        Elasticsearch manager instance
    vector_store : VectorStoreManager
        Vector store manager instance
    embedding_service : IEmbeddingService
        Embedding service instance

    Returns
    -------
    HybridStoreManager
        Initialized hybrid store manager
    """
    global _hybrid_store_manager

    _hybrid_store_manager = HybridStoreManager(
        es_manager=es_manager,
        vector_store=vector_store,
        embedding_service=embedding_service,
    )

    logger.info("[HybridStore] Global manager initialized")
    return _hybrid_store_manager
