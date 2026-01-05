"""
Elasticsearch Manager - Document storage and full-text search.

Provides:
- ES connection management with authentication
- Index lifecycle management (create, delete, alias)
- Document CRUD operations with bulk support
- Full-text search with BM25
- Health checks and diagnostics
"""

import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

from config.settings import settings

logger = logging.getLogger(__name__)

# Try to import elasticsearch
try:
    from elasticsearch import AsyncElasticsearch, NotFoundError, ConflictError
    from elasticsearch.helpers import async_bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    AsyncElasticsearch = None
    logger.warning("[ES] elasticsearch package not installed. Install with: pip install elasticsearch>=8.0.0")


@dataclass
class DocumentChunk:
    """
    Document chunk entity for ES storage.

    This represents a single chunk of a document with all metadata
    needed for storage in ES and association with Milvus.
    """
    chunk_id: str  # UUID, used as primary key and Milvus association
    doc_id: str  # Original document ID
    content: str  # Chunk text content
    metadata: Dict[str, Any] = field(default_factory=dict)  # Rich metadata
    chunk_index: int = 0  # Chunk sequence number
    chunk_total: int = 1  # Total chunks in document
    embedding: Optional[List[float]] = None  # Optional embedding vector
    indexed_at: Optional[datetime] = None  # Indexing timestamp

    def to_es_doc(self) -> Dict[str, Any]:
        """Convert to Elasticsearch document format."""
        doc = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "chunk_total": self.chunk_total,
            "indexed_at": self.indexed_at or datetime.utcnow().isoformat(),
        }
        # Only include embedding if present (for ES hybrid search)
        if self.embedding:
            doc["content_vector"] = self.embedding
        return doc

    @classmethod
    def from_es_doc(cls, doc: Dict[str, Any]) -> "DocumentChunk":
        """Create from Elasticsearch document."""
        source = doc.get("_source", doc)
        return cls(
            chunk_id=source.get("chunk_id", doc.get("_id", "")),
            doc_id=source.get("doc_id", ""),
            content=source.get("content", ""),
            metadata=source.get("metadata", {}),
            chunk_index=source.get("chunk_index", 0),
            chunk_total=source.get("chunk_total", 1),
            embedding=source.get("content_vector"),
            indexed_at=source.get("indexed_at"),
        )


@dataclass
class ESSearchResult:
    """Search result from Elasticsearch."""
    chunk: DocumentChunk
    score: float
    highlight: Optional[Dict[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "content": self.chunk.content,
            "metadata": self.chunk.metadata,
            "score": self.score,
            "highlight": self.highlight,
        }


class ElasticsearchManager:
    """
    Elasticsearch Manager for document storage and search.

    Responsibilities:
    1. Connection management with authentication and SSL
    2. Index lifecycle (create, delete, alias management)
    3. Document CRUD with bulk operations
    4. Full-text search with BM25 and filters
    5. Health monitoring
    """

    # Default index mapping with Chinese analyzer support
    DEFAULT_MAPPING = {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 256}
                    }
                },
                "content_vector": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "keyword"},
                        "file_type": {"type": "keyword"},
                        "file_path": {"type": "keyword"},
                        "file_size": {"type": "long"},
                        "created_at": {"type": "date"},
                        "device_id": {"type": "keyword"},
                        "category": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "author": {"type": "keyword"},
                        "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}
                    }
                },
                "chunk_index": {"type": "integer"},
                "chunk_total": {"type": "integer"},
                "indexed_at": {"type": "date"}
            }
        },
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1,
            "refresh_interval": "1s",
            "analysis": {
                "analyzer": {
                    "ik_max_word": {
                        "type": "custom",
                        "tokenizer": "ik_max_word"
                    },
                    "ik_smart": {
                        "type": "custom",
                        "tokenizer": "ik_smart"
                    }
                }
            }
        }
    }

    # Fallback mapping without IK analyzer (for ES without IK plugin)
    FALLBACK_MAPPING = {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "content": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": 256}
                    }
                },
                "content_vector": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "keyword"},
                        "file_type": {"type": "keyword"},
                        "file_path": {"type": "keyword"},
                        "file_size": {"type": "long"},
                        "created_at": {"type": "date"},
                        "device_id": {"type": "keyword"},
                        "category": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "author": {"type": "keyword"},
                        "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}
                    }
                },
                "chunk_index": {"type": "integer"},
                "chunk_total": {"type": "integer"},
                "indexed_at": {"type": "date"}
            }
        },
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1,
            "refresh_interval": "1s"
        }
    }

    def __init__(
        self,
        hosts: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_name: Optional[str] = None,
        use_ssl: bool = False,
        verify_certs: bool = True,
        ca_certs: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        embedding_dim: int = 768,
    ):
        """
        Initialize Elasticsearch Manager.

        Args:
            hosts: ES hosts (comma-separated), defaults to settings
            username: ES username for authentication
            password: ES password for authentication
            index_name: Index name for chunks, defaults to settings
            use_ssl: Whether to use SSL
            verify_certs: Whether to verify SSL certificates
            ca_certs: Path to CA certificates
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            embedding_dim: Dimension of embedding vectors
        """
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError(
                "elasticsearch package is not installed. "
                "Install with: pip install elasticsearch>=8.0.0"
            )

        self.hosts = (hosts or settings.ELASTICSEARCH_HOSTS).split(",")
        self.username = username or settings.ELASTICSEARCH_USERNAME
        self.password = password or settings.ELASTICSEARCH_PASSWORD
        self.index_name = index_name or settings.ELASTICSEARCH_CHUNK_INDEX
        self.use_ssl = use_ssl if use_ssl else settings.ELASTICSEARCH_USE_SSL
        self.verify_certs = verify_certs if verify_certs else settings.ELASTICSEARCH_VERIFY_CERTS
        self.ca_certs = ca_certs or settings.ELASTICSEARCH_CA_CERTS
        self.timeout = timeout or settings.ELASTICSEARCH_TIMEOUT
        self.max_retries = max_retries or settings.ELASTICSEARCH_MAX_RETRIES
        self.embedding_dim = embedding_dim

        # Update mapping with correct embedding dimension
        self._update_mapping_dim(embedding_dim)

        # Create ES client
        self._client: Optional[AsyncElasticsearch] = None
        self._is_initialized = False
        self._use_ik_analyzer = True  # Will be checked on init

        logger.info(
            f"[ES] Manager created - hosts: {self.hosts}, "
            f"index: {self.index_name}, ssl: {self.use_ssl}"
        )

    def _update_mapping_dim(self, dim: int) -> None:
        """Update embedding dimension in mappings."""
        self.DEFAULT_MAPPING["mappings"]["properties"]["content_vector"]["dims"] = dim
        self.FALLBACK_MAPPING["mappings"]["properties"]["content_vector"]["dims"] = dim

    @property
    def client(self) -> AsyncElasticsearch:
        """Get or create ES client."""
        if self._client is None:
            # Build connection options
            options = {
                "hosts": self.hosts,
                "request_timeout": self.timeout,
                "max_retries": self.max_retries,
                "retry_on_timeout": True,
            }

            # Add authentication if provided
            if self.username and self.password:
                options["basic_auth"] = (self.username, self.password)

            # Add SSL options
            if self.use_ssl:
                options["use_ssl"] = True
                options["verify_certs"] = self.verify_certs
                if self.ca_certs:
                    options["ca_certs"] = self.ca_certs

            self._client = AsyncElasticsearch(**options)
            logger.debug(f"[ES] Client created for hosts: {self.hosts}")

        return self._client

    async def initialize(self) -> bool:
        """
        Initialize ES connection and ensure index exists.

        Returns:
            True if initialization successful
        """
        try:
            # Test connection
            info = await self.client.info()
            version = info.get("version", {}).get("number", "unknown")
            logger.info(f"[ES] Connected to Elasticsearch {version}")

            # Check if IK analyzer is available
            await self._check_ik_analyzer()

            # Ensure index exists
            await self.ensure_index()

            self._is_initialized = True
            logger.info(f"[ES] Initialization complete - index: {self.index_name}")
            return True

        except Exception as e:
            logger.error(f"[ES] Initialization failed: {e}")
            self._is_initialized = False
            raise

    async def _check_ik_analyzer(self) -> None:
        """Check if IK analyzer plugin is installed."""
        try:
            # Try to analyze with IK
            await self.client.indices.analyze(
                body={"analyzer": "ik_smart", "text": "测试"}
            )
            self._use_ik_analyzer = True
            logger.info("[ES] IK analyzer plugin detected")
        except Exception:
            self._use_ik_analyzer = False
            logger.warning("[ES] IK analyzer not available, using standard analyzer")

    async def ensure_index(self) -> bool:
        """
        Ensure the index exists with correct mapping.

        Returns:
            True if index exists or was created
        """
        try:
            exists = await self.client.indices.exists(index=self.index_name)
            if exists:
                logger.debug(f"[ES] Index {self.index_name} already exists")
                return True

            # Create index with appropriate mapping
            mapping = self.DEFAULT_MAPPING if self._use_ik_analyzer else self.FALLBACK_MAPPING
            await self.client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"[ES] Created index {self.index_name}")
            return True

        except Exception as e:
            logger.error(f"[ES] Failed to ensure index: {e}")
            # Try with fallback mapping
            try:
                await self.client.indices.create(
                    index=self.index_name,
                    body=self.FALLBACK_MAPPING
                )
                logger.info(f"[ES] Created index {self.index_name} with fallback mapping")
                return True
            except Exception as e2:
                logger.error(f"[ES] Failed to create index with fallback: {e2}")
                raise

    async def close(self) -> None:
        """Close ES client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("[ES] Connection closed")

    # ==================== Document Operations ====================

    async def index_chunk(self, chunk: DocumentChunk) -> str:
        """
        Index a single document chunk.

        Args:
            chunk: DocumentChunk to index

        Returns:
            Chunk ID
        """
        doc = chunk.to_es_doc()
        await self.client.index(
            index=self.index_name,
            id=chunk.chunk_id,
            body=doc,
            refresh="wait_for"
        )
        logger.debug(f"[ES] Indexed chunk: {chunk.chunk_id}")
        return chunk.chunk_id

    async def bulk_index_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 500,
        refresh: bool = True,
    ) -> Dict[str, Any]:
        """
        Bulk index multiple document chunks.

        Args:
            chunks: List of DocumentChunk to index
            batch_size: Number of chunks per batch
            refresh: Whether to refresh index after indexing

        Returns:
            Indexing statistics
        """
        if not chunks:
            return {"success": 0, "failed": 0, "total": 0}

        success_count = 0
        failed_count = 0
        failed_ids = []

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Prepare bulk actions
            actions = []
            for chunk in batch:
                action = {
                    "_index": self.index_name,
                    "_id": chunk.chunk_id,
                    "_source": chunk.to_es_doc()
                }
                actions.append(action)

            try:
                # Execute bulk operation
                success, failed = await async_bulk(
                    self.client,
                    actions,
                    raise_on_error=False,
                    raise_on_exception=False,
                )
                success_count += success
                if failed:
                    failed_count += len(failed)
                    failed_ids.extend([f.get("index", {}).get("_id", "") for f in failed])

            except Exception as e:
                logger.error(f"[ES] Bulk indexing error for batch {i//batch_size}: {e}")
                failed_count += len(batch)

            logger.debug(f"[ES] Indexed batch {i//batch_size + 1}: {len(batch)} chunks")

        # Refresh index if requested
        if refresh:
            await self.client.indices.refresh(index=self.index_name)

        result = {
            "success": success_count,
            "failed": failed_count,
            "total": len(chunks),
            "failed_ids": failed_ids[:10] if failed_ids else [],  # Limit failed IDs
        }

        logger.info(
            f"[ES] Bulk indexing complete - success: {success_count}, "
            f"failed: {failed_count}, total: {len(chunks)}"
        )
        return result

    async def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a single chunk by ID.

        Args:
            chunk_id: Chunk ID to retrieve

        Returns:
            DocumentChunk or None if not found
        """
        try:
            result = await self.client.get(index=self.index_name, id=chunk_id)
            return DocumentChunk.from_es_doc(result)
        except NotFoundError:
            logger.debug(f"[ES] Chunk not found: {chunk_id}")
            return None
        except Exception as e:
            logger.error(f"[ES] Error getting chunk {chunk_id}: {e}")
            return None

    async def get_chunks_by_ids(
        self,
        chunk_ids: List[str],
        include_source: bool = True,
    ) -> List[DocumentChunk]:
        """
        Get multiple chunks by IDs.

        Args:
            chunk_ids: List of chunk IDs
            include_source: Whether to include full source

        Returns:
            List of DocumentChunk (preserves order)
        """
        if not chunk_ids:
            return []

        try:
            result = await self.client.mget(
                index=self.index_name,
                body={"ids": chunk_ids},
                _source=include_source,
            )

            # Build chunks preserving order
            chunks = []
            docs_map = {
                doc["_id"]: doc
                for doc in result.get("docs", [])
                if doc.get("found", False)
            }

            for chunk_id in chunk_ids:
                if chunk_id in docs_map:
                    chunks.append(DocumentChunk.from_es_doc(docs_map[chunk_id]))

            logger.debug(f"[ES] Retrieved {len(chunks)}/{len(chunk_ids)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"[ES] Error getting chunks by IDs: {e}")
            return []

    async def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a single chunk.

        Args:
            chunk_id: Chunk ID to delete

        Returns:
            True if deleted
        """
        try:
            await self.client.delete(
                index=self.index_name,
                id=chunk_id,
                refresh="wait_for"
            )
            logger.debug(f"[ES] Deleted chunk: {chunk_id}")
            return True
        except NotFoundError:
            logger.debug(f"[ES] Chunk not found for deletion: {chunk_id}")
            return False
        except Exception as e:
            logger.error(f"[ES] Error deleting chunk {chunk_id}: {e}")
            return False

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            doc_id: Document ID

        Returns:
            Number of deleted chunks
        """
        try:
            result = await self.client.delete_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "term": {"doc_id": doc_id}
                    }
                },
                refresh=True,
            )
            deleted = result.get("deleted", 0)
            logger.info(f"[ES] Deleted {deleted} chunks for doc_id: {doc_id}")
            return deleted
        except Exception as e:
            logger.error(f"[ES] Error deleting chunks for doc {doc_id}: {e}")
            return 0

    async def delete_all(self) -> int:
        """
        Delete all documents in the index.

        Returns:
            Number of deleted documents
        """
        try:
            result = await self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}},
                refresh=True,
            )
            deleted = result.get("deleted", 0)
            logger.warning(f"[ES] Deleted all {deleted} documents from index")
            return deleted
        except Exception as e:
            logger.error(f"[ES] Error deleting all documents: {e}")
            return 0

    # ==================== Search Operations ====================

    async def search_by_keyword(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        highlight: bool = True,
        min_score: Optional[float] = None,
    ) -> List[ESSearchResult]:
        """
        Search chunks using BM25 full-text search.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"file_type": "pdf"})
            highlight: Whether to return highlighted snippets
            min_score: Minimum relevance score

        Returns:
            List of ESSearchResult
        """
        # Build query
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["content^3", "metadata.title^2", "metadata.source"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
        ]

        # Add filters
        filter_clauses = self._build_filters(filters) if filters else []

        search_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses,
                }
            },
            "size": top_k,
            "_source": True,
        }

        # Add minimum score
        if min_score:
            search_body["min_score"] = min_score

        # Add highlighting
        if highlight:
            search_body["highlight"] = {
                "fields": {
                    "content": {
                        "fragment_size": 200,
                        "number_of_fragments": 3,
                        "pre_tags": ["<em>"],
                        "post_tags": ["</em>"]
                    }
                }
            }

        try:
            result = await self.client.search(
                index=self.index_name,
                body=search_body,
            )

            hits = result.get("hits", {}).get("hits", [])
            results = []
            for hit in hits:
                chunk = DocumentChunk.from_es_doc(hit)
                score = hit.get("_score", 0.0)
                highlight_data = hit.get("highlight") if highlight else None
                results.append(ESSearchResult(
                    chunk=chunk,
                    score=score,
                    highlight=highlight_data,
                ))

            logger.debug(f"[ES] BM25 search found {len(results)} results for: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"[ES] Search error: {e}")
            return []

    async def search_by_vector(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> List[ESSearchResult]:
        """
        Search chunks using vector similarity (kNN).

        Args:
            vector: Query embedding vector
            top_k: Number of results
            filters: Metadata filters
            min_score: Minimum similarity score

        Returns:
            List of ESSearchResult
        """
        # Build kNN query
        knn_query = {
            "field": "content_vector",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": top_k * 10,
        }

        # Add filters
        if filters:
            knn_query["filter"] = {"bool": {"filter": self._build_filters(filters)}}

        search_body = {
            "knn": knn_query,
            "size": top_k,
            "_source": True,
        }

        if min_score:
            search_body["min_score"] = min_score

        try:
            result = await self.client.search(
                index=self.index_name,
                body=search_body,
            )

            hits = result.get("hits", {}).get("hits", [])
            results = []
            for hit in hits:
                chunk = DocumentChunk.from_es_doc(hit)
                score = hit.get("_score", 0.0)
                results.append(ESSearchResult(chunk=chunk, score=score))

            logger.debug(f"[ES] Vector search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"[ES] Vector search error: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> List[ESSearchResult]:
        """
        Hybrid search combining BM25 and vector similarity.

        Uses ES 8.x RRF (Reciprocal Rank Fusion) or manual fusion.

        Args:
            query: Text query for BM25
            vector: Query embedding for vector search
            top_k: Number of results
            filters: Metadata filters
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores

        Returns:
            List of ESSearchResult sorted by fused score
        """
        # Run both searches in parallel
        import asyncio
        bm25_results, vector_results = await asyncio.gather(
            self.search_by_keyword(query, top_k=top_k * 2, filters=filters, highlight=False),
            self.search_by_vector(vector, top_k=top_k * 2, filters=filters),
        )

        # RRF fusion
        k = 60  # RRF constant
        chunk_scores: Dict[str, float] = {}
        chunk_data: Dict[str, ESSearchResult] = {}

        # Add BM25 scores
        for rank, result in enumerate(bm25_results):
            chunk_id = result.chunk.chunk_id
            rrf_score = bm25_weight * (1.0 / (k + rank + 1))
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            chunk_data[chunk_id] = result

        # Add vector scores
        for rank, result in enumerate(vector_results):
            chunk_id = result.chunk.chunk_id
            rrf_score = vector_weight * (1.0 / (k + rank + 1))
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

        # Sort by fused score
        sorted_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)
        results = []
        for chunk_id in sorted_ids[:top_k]:
            result = chunk_data[chunk_id]
            result.score = chunk_scores[chunk_id]
            results.append(result)

        logger.debug(
            f"[ES] Hybrid search: BM25={len(bm25_results)}, "
            f"Vector={len(vector_results)}, Fused={len(results)}"
        )
        return results

    def _build_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build ES filter clauses from filter dict."""
        clauses = []
        for key, value in filters.items():
            if key.startswith("metadata.") or key in ["doc_id", "chunk_id"]:
                field = key
            else:
                field = f"metadata.{key}"

            if isinstance(value, list):
                clauses.append({"terms": {field: value}})
            elif isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    clauses.append({"range": {field: value}})
            else:
                clauses.append({"term": {field: value}})
        return clauses

    # ==================== Statistics & Health ====================

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Index statistics dict
        """
        try:
            stats = await self.client.indices.stats(index=self.index_name)
            primaries = stats.get("_all", {}).get("primaries", {})

            return {
                "index_name": self.index_name,
                "doc_count": primaries.get("docs", {}).get("count", 0),
                "size_bytes": primaries.get("store", {}).get("size_in_bytes", 0),
                "size_mb": round(primaries.get("store", {}).get("size_in_bytes", 0) / 1024 / 1024, 2),
            }
        except Exception as e:
            logger.error(f"[ES] Error getting stats: {e}")
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """
        Check ES cluster and index health.

        Returns:
            Health status dict
        """
        try:
            cluster_health = await self.client.cluster.health()
            index_exists = await self.client.indices.exists(index=self.index_name)

            stats = await self.get_stats() if index_exists else {}

            return {
                "status": "healthy" if cluster_health.get("status") != "red" else "unhealthy",
                "cluster_status": cluster_health.get("status"),
                "cluster_name": cluster_health.get("cluster_name"),
                "index_exists": index_exists,
                "index_name": self.index_name,
                "doc_count": stats.get("doc_count", 0),
                "is_initialized": self._is_initialized,
            }
        except Exception as e:
            logger.error(f"[ES] Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_initialized": False,
            }


# Singleton instance
_es_manager: Optional[ElasticsearchManager] = None


def get_es_manager() -> Optional[ElasticsearchManager]:
    """Get the singleton ES manager instance."""
    global _es_manager
    if _es_manager is None and settings.ELASTICSEARCH_ENABLED:
        _es_manager = ElasticsearchManager()
    return _es_manager


async def init_es_manager() -> Optional[ElasticsearchManager]:
    """Initialize and return the ES manager."""
    manager = get_es_manager()
    if manager:
        await manager.initialize()
    return manager
