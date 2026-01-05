"""Elasticsearch client for BM25 search."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from services.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ESSearchResult:
    """Elasticsearch search result."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class ElasticsearchClient:
    """
    Elasticsearch client for keyword/BM25 search.

    Provides:
    - Index management
    - Document indexing
    - BM25 search
    """

    def __init__(
        self,
        hosts: str = "http://localhost:9200",
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_prefix: str = "knowledge_base",
    ):
        """
        Initialize Elasticsearch client.

        Args:
            hosts: Elasticsearch hosts
            username: Username for authentication
            password: Password for authentication
            index_prefix: Index name prefix
        """
        self.hosts = hosts
        self.username = username
        self.password = password
        self.index_prefix = index_prefix
        self._client = None

    @property
    def indices(self):
        """Proxy to internal client's indices API."""
        if self._client:
            return self._client.indices
        return None

    async def index(self, index: str, id: str, body: Dict[str, Any]):
        """Index a document using raw ES API."""
        if not self._client:
            await self.connect()
        if self._client:
            await self._client.index(index=index, id=id, body=body)

    async def delete(self, index: str, id: str):
        """Delete a document using raw ES API."""
        if not self._client:
            await self.connect()
        if self._client:
            try:
                await self._client.delete(index=index, id=id)
            except Exception as e:
                logger.warning(f"Delete failed: {e}")

    async def get(self, index: str, id: str) -> Dict[str, Any]:
        """Get a document by ID."""
        if not self._client:
            await self.connect()
        if self._client:
            return await self._client.get(index=index, id=id)
        return {}

    async def bulk(self, operations: List[Dict[str, Any]]):
        """Bulk index operations."""
        if not self._client:
            await self.connect()
        if self._client:
            await self._client.bulk(operations=operations)

    async def connect(self):
        """Connect to Elasticsearch."""
        try:
            from elasticsearch import AsyncElasticsearch

            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)

            self._client = AsyncElasticsearch(
                hosts=[self.hosts],
                basic_auth=auth,
                verify_certs=False,
            )

            # Test connection
            info = await self._client.info()
            logger.info(f"Connected to Elasticsearch: {info['version']['number']}")

        except ImportError:
            logger.warning("elasticsearch not installed, using mock client")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")

    def _get_index_name(self, knowledge_base_id: Optional[str] = None) -> str:
        """Get index name."""
        if knowledge_base_id:
            return f"{self.index_prefix}_{knowledge_base_id}"
        return self.index_prefix

    async def ensure_index(self, knowledge_base_id: Optional[str] = None):
        """Ensure index exists with proper mappings."""
        if not self._client:
            await self.connect()

        if not self._client:
            return

        index_name = self._get_index_name(knowledge_base_id)

        try:
            exists = await self._client.indices.exists(index=index_name)
            if not exists:
                await self._client.indices.create(
                    index=index_name,
                    body={
                        "settings": {
                            "analysis": {
                                "analyzer": {
                                    "chinese": {
                                        "type": "custom",
                                        "tokenizer": "ik_max_word",
                                        "filter": ["lowercase"],
                                    }
                                }
                            }
                        },
                        "mappings": {
                            "properties": {
                                "content": {
                                    "type": "text",
                                    "analyzer": "chinese",
                                },
                                "title": {
                                    "type": "text",
                                    "analyzer": "chinese",
                                    "boost": 2.0,
                                },
                                "chunk_id": {"type": "keyword"},
                                "knowledge_base_id": {"type": "keyword"},
                                "metadata": {"type": "object", "enabled": True},
                                "created_at": {"type": "date"},
                            }
                        },
                    },
                )
                logger.info(f"Created Elasticsearch index: {index_name}")

        except Exception as e:
            logger.error(f"Failed to create index: {e}")

    async def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        knowledge_base_id: Optional[str] = None,
    ) -> bool:
        """
        Index a document.

        Args:
            doc_id: Document ID
            content: Document content
            metadata: Document metadata
            knowledge_base_id: Knowledge base ID

        Returns:
            True if successful
        """
        if not self._client:
            await self.connect()

        if not self._client:
            return False

        await self.ensure_index(knowledge_base_id)
        index_name = self._get_index_name(knowledge_base_id)

        try:
            await self._client.index(
                index=index_name,
                id=doc_id,
                body={
                    "content": content,
                    "title": metadata.get("title", "") if metadata else "",
                    "chunk_id": doc_id,
                    "knowledge_base_id": knowledge_base_id or "",
                    "metadata": metadata or {},
                },
            )
            return True

        except Exception as e:
            logger.error(f"Elasticsearch index error: {e}")
            return False

    async def search(
        self,
        index: str,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute search query.

        Args:
            index: Index name
            body: Search body

        Returns:
            Search response
        """
        if not self._client:
            await self.connect()

        if not self._client:
            return {"hits": {"hits": [], "total": {"value": 0}}}

        try:
            return await self._client.search(index=index, body=body)
        except Exception as e:
            logger.error(f"Elasticsearch search error: {e}")
            return {"hits": {"hits": [], "total": {"value": 0}}}

    async def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        knowledge_base_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ESSearchResult]:
        """
        Perform BM25 search.

        Args:
            query: Search query
            top_k: Number of results
            knowledge_base_id: Filter by knowledge base
            filters: Additional filters

        Returns:
            List of search results
        """
        index_name = self._get_index_name(knowledge_base_id)

        # Build query
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
                "term": {"knowledge_base_id": knowledge_base_id}
            })

        if filters:
            for key, value in filters.items():
                filter_clauses.append({
                    "term": {f"metadata.{key}": value}
                })

        body = {
            "query": {
                "bool": {
                    "must": must,
                    "filter": filter_clauses,
                }
            },
            "size": top_k,
            "_source": ["content", "metadata"],
        }

        response = await self.search(index=index_name, body=body)

        return [
            ESSearchResult(
                id=hit["_id"],
                content=hit["_source"].get("content", ""),
                score=hit["_score"],
                metadata=hit["_source"].get("metadata", {}),
            )
            for hit in response["hits"]["hits"]
        ]

    async def delete_document(
        self,
        doc_id: str,
        knowledge_base_id: Optional[str] = None,
    ) -> bool:
        """Delete a document."""
        if not self._client:
            return False

        index_name = self._get_index_name(knowledge_base_id)

        try:
            await self._client.delete(index=index_name, id=doc_id)
            return True
        except Exception as e:
            logger.error(f"Elasticsearch delete error: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self._client:
            return {"status": "disconnected"}

        try:
            stats = await self._client.indices.stats(index=f"{self.index_prefix}*")
            return {
                "status": "connected",
                "indices": list(stats["indices"].keys()),
                "total_docs": stats["_all"]["primaries"]["docs"]["count"],
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Close connection."""
        if self._client:
            await self._client.close()
