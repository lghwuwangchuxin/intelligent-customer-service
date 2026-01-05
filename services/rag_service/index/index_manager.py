"""Index manager for document indexing and management."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from services.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """Document representation."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class IndexStats:
    """Index statistics."""
    total_documents: int
    total_chunks: int
    index_size_bytes: int
    last_updated: Optional[datetime] = None


class IndexManager:
    """
    Manages document indexing and storage.

    Coordinates:
    - Vector store (Milvus/Qdrant) for embeddings
    - Elasticsearch for BM25/keyword search
    - Document metadata storage
    """

    def __init__(
        self,
        vector_store=None,
        elasticsearch_client=None,
        embedding_model=None,
        collection_name: str = "knowledge_base",
    ):
        """
        Initialize index manager.

        Args:
            vector_store: Vector store client
            elasticsearch_client: Elasticsearch client
            embedding_model: Embedding model
            collection_name: Default collection name
        """
        self.vector_store = vector_store
        self.es_client = elasticsearch_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self._initialized = False

    async def initialize(self):
        """Initialize index and storage."""
        if self._initialized:
            return

        # Create vector collection if needed
        if self.vector_store:
            await self._ensure_vector_collection()

        # Create ES index if needed
        if self.es_client:
            await self._ensure_es_index()

        self._initialized = True
        logger.info("Index manager initialized")

    async def _ensure_vector_collection(self):
        """Ensure vector collection exists."""
        try:
            exists = await self.vector_store.has_collection(self.collection_name)
            if not exists:
                await self.vector_store.create_collection(
                    name=self.collection_name,
                    dimension=768,  # Default embedding dimension
                    metric_type="COSINE",
                )
                logger.info(f"Created vector collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure vector collection: {e}")
            raise

    async def _ensure_es_index(self):
        """Ensure Elasticsearch index exists."""
        try:
            exists = await self.es_client.indices.exists(index=self.collection_name)
            if not exists:
                await self.es_client.indices.create(
                    index=self.collection_name,
                    body={
                        "settings": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0,
                            "analysis": {
                                "analyzer": {
                                    "text_analyzer": {
                                        "type": "standard",
                                    }
                                }
                            }
                        },
                        "mappings": {
                            "properties": {
                                "content": {
                                    "type": "text",
                                    "analyzer": "text_analyzer",
                                },
                                "metadata": {
                                    "type": "object",
                                    "enabled": True,
                                },
                                "created_at": {
                                    "type": "date",
                                },
                            }
                        }
                    }
                )
                logger.info(f"Created ES index: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure ES index: {e}")
            raise

    async def index_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
    ) -> str:
        """
        Index a single document.

        Args:
            content: Document content
            metadata: Document metadata
            document_id: Optional document ID
            knowledge_base_id: Knowledge base ID

        Returns:
            Document ID
        """
        await self.initialize()

        doc_id = document_id or str(uuid.uuid4())
        metadata = metadata or {}
        metadata["knowledge_base_id"] = knowledge_base_id
        metadata["indexed_at"] = datetime.utcnow().isoformat()

        # Generate embedding
        embedding = None
        if self.embedding_model:
            try:
                embedding = await self.embedding_model.embed(content)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        # Index to vector store
        if self.vector_store and embedding:
            try:
                await self.vector_store.insert(
                    collection_name=self.collection_name,
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata],
                )
            except Exception as e:
                logger.error(f"Failed to index to vector store: {e}")
                raise

        # Index to Elasticsearch
        if self.es_client:
            try:
                index_name = f"kb_{knowledge_base_id}" if knowledge_base_id else self.collection_name
                await self.es_client.index(
                    index=index_name,
                    id=doc_id,
                    body={
                        "content": content,
                        "metadata": metadata,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                )
            except Exception as e:
                logger.error(f"Failed to index to Elasticsearch: {e}")
                raise

        logger.info(f"Indexed document: {doc_id}")
        return doc_id

    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        knowledge_base_id: Optional[str] = None,
        batch_size: int = 100,
    ) -> List[str]:
        """
        Index multiple documents in batches.

        Args:
            documents: List of documents with 'content' and optional 'metadata'
            knowledge_base_id: Knowledge base ID
            batch_size: Batch size for indexing

        Returns:
            List of document IDs
        """
        await self.initialize()

        all_ids = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = await self._index_batch(batch, knowledge_base_id)
            all_ids.extend(ids)
            logger.info(f"Indexed batch {i // batch_size + 1}, total: {len(all_ids)}")

        return all_ids

    async def _index_batch(
        self,
        documents: List[Dict[str, Any]],
        knowledge_base_id: Optional[str],
    ) -> List[str]:
        """Index a batch of documents."""
        ids = []
        contents = []
        metadatas = []

        for doc in documents:
            doc_id = doc.get("id") or str(uuid.uuid4())
            content = doc["content"]
            metadata = doc.get("metadata", {})
            metadata["knowledge_base_id"] = knowledge_base_id
            metadata["indexed_at"] = datetime.utcnow().isoformat()

            ids.append(doc_id)
            contents.append(content)
            metadatas.append(metadata)

        # Generate embeddings in batch
        embeddings = None
        if self.embedding_model:
            try:
                embeddings = await self.embedding_model.embed_batch(contents)
            except Exception as e:
                logger.warning(f"Failed to generate embeddings: {e}")

        # Index to vector store
        if self.vector_store and embeddings:
            try:
                await self.vector_store.insert(
                    collection_name=self.collection_name,
                    ids=ids,
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas,
                )
            except Exception as e:
                logger.error(f"Failed to index batch to vector store: {e}")
                raise

        # Index to Elasticsearch
        if self.es_client:
            try:
                index_name = f"kb_{knowledge_base_id}" if knowledge_base_id else self.collection_name
                operations = []
                for doc_id, content, metadata in zip(ids, contents, metadatas):
                    operations.append({"index": {"_index": index_name, "_id": doc_id}})
                    operations.append({
                        "content": content,
                        "metadata": metadata,
                        "created_at": datetime.utcnow().isoformat(),
                    })
                await self.es_client.bulk(operations=operations)
            except Exception as e:
                logger.error(f"Failed to index batch to Elasticsearch: {e}")
                raise

        return ids

    async def delete_document(
        self,
        document_id: str,
        knowledge_base_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a document.

        Args:
            document_id: Document ID
            knowledge_base_id: Knowledge base ID

        Returns:
            True if deleted successfully
        """
        await self.initialize()

        success = True

        # Delete from vector store
        if self.vector_store:
            try:
                await self.vector_store.delete(
                    collection_name=self.collection_name,
                    ids=[document_id],
                )
            except Exception as e:
                logger.error(f"Failed to delete from vector store: {e}")
                success = False

        # Delete from Elasticsearch
        if self.es_client:
            try:
                index_name = f"kb_{knowledge_base_id}" if knowledge_base_id else self.collection_name
                await self.es_client.delete(
                    index=index_name,
                    id=document_id,
                )
            except Exception as e:
                logger.error(f"Failed to delete from Elasticsearch: {e}")
                success = False

        return success

    async def get_document(
        self,
        document_id: str,
        knowledge_base_id: Optional[str] = None,
    ) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            document_id: Document ID
            knowledge_base_id: Knowledge base ID

        Returns:
            Document if found, None otherwise
        """
        await self.initialize()

        # Try Elasticsearch first
        if self.es_client:
            try:
                index_name = f"kb_{knowledge_base_id}" if knowledge_base_id else self.collection_name
                result = await self.es_client.get(
                    index=index_name,
                    id=document_id,
                )
                source = result["_source"]
                return Document(
                    id=document_id,
                    content=source.get("content", ""),
                    metadata=source.get("metadata", {}),
                )
            except Exception as e:
                logger.debug(f"Document not found in ES: {e}")

        return None

    async def list_documents(
        self,
        knowledge_base_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[Document], int]:
        """
        List documents with pagination.

        Args:
            knowledge_base_id: Filter by knowledge base
            page: Page number (1-based)
            page_size: Page size

        Returns:
            Tuple of (documents, total_count)
        """
        await self.initialize()

        if not self.es_client:
            return [], 0

        try:
            index_name = f"kb_{knowledge_base_id}" if knowledge_base_id else self.collection_name
            query = {"match_all": {}}

            if knowledge_base_id:
                query = {
                    "term": {"metadata.knowledge_base_id": knowledge_base_id}
                }

            result = await self.es_client.search(
                index=index_name,
                body={
                    "query": query,
                    "from": (page - 1) * page_size,
                    "size": page_size,
                    "sort": [{"created_at": "desc"}],
                },
            )

            documents = []
            for hit in result["hits"]["hits"]:
                source = hit["_source"]
                documents.append(Document(
                    id=hit["_id"],
                    content=source.get("content", ""),
                    metadata=source.get("metadata", {}),
                ))

            total = result["hits"]["total"]["value"]
            return documents, total

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return [], 0

    async def get_stats(
        self,
        knowledge_base_id: Optional[str] = None,
    ) -> IndexStats:
        """
        Get index statistics.

        Args:
            knowledge_base_id: Filter by knowledge base

        Returns:
            IndexStats
        """
        await self.initialize()

        total_documents = 0
        index_size_bytes = 0

        if self.es_client:
            try:
                index_name = f"kb_{knowledge_base_id}" if knowledge_base_id else self.collection_name
                stats = await self.es_client.indices.stats(index=index_name)
                index_stats = stats["indices"].get(index_name, {})
                total_documents = index_stats.get("primaries", {}).get("docs", {}).get("count", 0)
                index_size_bytes = index_stats.get("primaries", {}).get("store", {}).get("size_in_bytes", 0)
            except Exception as e:
                logger.error(f"Failed to get ES stats: {e}")

        return IndexStats(
            total_documents=total_documents,
            total_chunks=total_documents,  # Same as documents for now
            index_size_bytes=index_size_bytes,
            last_updated=datetime.utcnow(),
        )
