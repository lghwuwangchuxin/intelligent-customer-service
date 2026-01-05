"""Milvus vector store client."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from services.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Search result from Milvus."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class MilvusClient:
    """
    Milvus vector store client.

    Provides:
    - Collection management
    - Vector insertion
    - Similarity search
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "knowledge_base",
        dimension: int = 768,
    ):
        """
        Initialize Milvus client.

        Args:
            host: Milvus host
            port: Milvus port
            collection_name: Default collection name
            dimension: Vector dimension
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self._client = None
        self._collection = None

    async def connect(self):
        """Connect to Milvus."""
        try:
            from pymilvus import connections, Collection, utility

            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )

            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self._collection = Collection(self.collection_name)
                self._collection.load()
                logger.info(f"Connected to Milvus collection: {self.collection_name}")
            else:
                await self._create_collection()

        except ImportError:
            logger.warning("pymilvus not installed, using mock client")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")

    async def _create_collection(self):
        """Create collection with schema."""
        from pymilvus import (
            Collection,
            FieldSchema,
            CollectionSchema,
            DataType,
        )

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="knowledge_base_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(fields=fields, description="Knowledge base documents")
        self._collection = Collection(name=self.collection_name, schema=schema)

        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index(field_name="embedding", index_params=index_params)
        self._collection.load()

        logger.info(f"Created Milvus collection: {self.collection_name}")

    async def insert(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Insert documents into Milvus.

        Args:
            documents: List of documents with id, content, embedding, metadata

        Returns:
            List of inserted IDs
        """
        if not self._collection:
            await self.connect()

        if not self._collection:
            logger.warning("Milvus not connected, skipping insert")
            return []

        try:
            data = [
                [doc["id"] for doc in documents],
                [doc["content"][:65535] for doc in documents],
                [doc["embedding"] for doc in documents],
                [doc.get("knowledge_base_id", "") for doc in documents],
                [doc.get("metadata", {}) for doc in documents],
            ]

            self._collection.insert(data)
            self._collection.flush()

            return [doc["id"] for doc in documents]

        except Exception as e:
            logger.error(f"Milvus insert error: {e}")
            return []

    async def search(
        self,
        vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            vector: Query vector
            top_k: Number of results
            filter_expr: Filter expression

        Returns:
            List of search results
        """
        if not self._collection:
            await self.connect()

        if not self._collection:
            return []

        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }

            results = self._collection.search(
                data=[vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["content", "metadata", "knowledge_base_id"],
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append(SearchResult(
                        id=hit.id,
                        content=hit.entity.get("content", ""),
                        score=hit.score,
                        metadata=hit.entity.get("metadata", {}),
                    ))

            return search_results

        except Exception as e:
            logger.error(f"Milvus search error: {e}")
            return []

    async def delete(self, ids: List[str]) -> bool:
        """Delete documents by ID."""
        if not self._collection:
            return False

        try:
            expr = f'id in {ids}'
            self._collection.delete(expr)
            return True
        except Exception as e:
            logger.error(f"Milvus delete error: {e}")
            return False

    async def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            from pymilvus import utility, connections

            # Ensure connected
            if not connections.has_connection("default"):
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                )

            return utility.has_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to check collection: {e}")
            return False

    async def create_collection(
        self,
        name: str,
        dimension: int = 768,
        metric_type: str = "COSINE",
    ):
        """Create a new collection."""
        self.collection_name = name
        self.dimension = dimension
        await self._create_collection()

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._collection:
            return {"status": "disconnected"}

        try:
            return {
                "status": "connected",
                "collection": self.collection_name,
                "num_entities": self._collection.num_entities,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Close connection."""
        try:
            from pymilvus import connections
            connections.disconnect("default")
        except Exception:
            pass
