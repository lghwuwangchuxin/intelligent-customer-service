"""
Domain Interfaces (Ports) - Abstract contracts for domain services.

These interfaces define the contracts that concrete implementations must fulfill.
This enables dependency inversion and makes the domain layer independent of
infrastructure details.
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)


@runtime_checkable
class IEmbeddingService(Protocol):
    """Interface for embedding services (text to vector)."""

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...

    async def embed_async(self, texts: List[str]) -> List[List[float]]:
        """Embed texts asynchronously."""
        ...

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts synchronously."""
        ...

    async def warmup(self) -> None:
        """Warmup the embedding model."""
        ...


@runtime_checkable
class ILLMService(Protocol):
    """Interface for LLM services."""

    @property
    def provider(self) -> str:
        """Get LLM provider name."""
        ...

    @property
    def model(self) -> str:
        """Get model name."""
        ...

    @property
    def supports_tool_calling(self) -> bool:
        """Check if model supports tool calling."""
        ...

    async def ainvoke(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> str:
        """Invoke LLM asynchronously."""
        ...

    def invoke(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> str:
        """Invoke LLM synchronously."""
        ...

    def stream(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream LLM response."""
        ...

    async def astream(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response asynchronously."""
        ...

    def get_info(self) -> Dict[str, Any]:
        """Get LLM service info."""
        ...


@runtime_checkable
class IVectorStore(Protocol):
    """Interface for vector store operations."""

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Add documents to vector store."""
        ...

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search similar documents."""
        ...

    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        ...

    def delete_collection(self) -> bool:
        """Delete entire collection."""
        ...

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        ...


class IDocumentProcessor(ABC):
    """Interface for document processing."""

    @abstractmethod
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a file and return chunks with metadata."""
        ...

    @abstractmethod
    def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Process text and return chunks with metadata."""
        ...

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        ...


class IRAGService(ABC):
    """Interface for RAG (Retrieval Augmented Generation) service."""

    @abstractmethod
    async def aquery(self, query: str, **kwargs) -> str:
        """Query with RAG asynchronously."""
        ...

    @abstractmethod
    def query(self, query: str, **kwargs) -> str:
        """Query with RAG synchronously."""
        ...

    @abstractmethod
    async def astream_query(
        self,
        query: str,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream query response asynchronously."""
        ...

    @abstractmethod
    def get_relevant_documents(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get relevant documents for a query."""
        ...

    @abstractmethod
    async def async_add_knowledge(
        self,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add knowledge to the RAG index asynchronously."""
        ...

    @abstractmethod
    async def async_index_directory(
        self,
        directory_path: str,
    ) -> Dict[str, Any]:
        """Index all documents in a directory asynchronously."""
        ...


class IChatService(ABC):
    """Interface for basic chat service (without RAG)."""

    @abstractmethod
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send a chat message and get response."""
        ...

    @abstractmethod
    def stream_chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """Stream chat response."""
        ...

    @abstractmethod
    async def achat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send a chat message and get response asynchronously."""
        ...


class IAgentService(ABC):
    """Interface for agent services (ReAct, LangGraph, etc.)."""

    @abstractmethod
    async def run(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Run agent and get response."""
        ...

    @abstractmethod
    async def stream(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream agent response with intermediate steps."""
        ...


@runtime_checkable
class IToolRegistry(Protocol):
    """Interface for tool registry."""

    def register(self, tool: Any) -> None:
        """Register a tool."""
        ...

    def get(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        ...

    def get_all(self) -> List[Any]:
        """Get all registered tools."""
        ...

    async def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        ...

    def initialize_default_tools(self, **dependencies) -> None:
        """Initialize default tools with dependencies."""
        ...


class IMemoryManager(ABC):
    """Interface for conversation memory management."""

    @abstractmethod
    def get_memory(self, conversation_id: str) -> Optional[Any]:
        """Get memory for a conversation."""
        ...

    @abstractmethod
    def create_memory(self, conversation_id: str) -> Any:
        """Create new memory for a conversation."""
        ...

    @abstractmethod
    def delete_memory(self, conversation_id: str) -> bool:
        """Delete memory for a conversation."""
        ...

    @abstractmethod
    def summarize_if_needed(self, conversation_id: str) -> bool:
        """Summarize memory if threshold is reached."""
        ...


@runtime_checkable
class IDocumentStore(Protocol):
    """
    Interface for document storage (Elasticsearch implementation).

    This interface defines the contract for document storage operations,
    primarily implemented by Elasticsearch to store document chunks with
    rich metadata and full-text search capabilities.

    The document store works in conjunction with the vector store (Milvus):
    - Document store: Stores text content, metadata, and optional embeddings
    - Vector store: Stores embeddings with chunk_id for semantic search
    """

    async def store_chunks(
        self,
        chunks: List[Any],  # List[DocumentChunk]
        batch_size: int = 50,
    ) -> List[str]:
        """
        Store document chunks.

        Args:
            chunks: List of DocumentChunk entities to store
            batch_size: Number of chunks per batch for bulk indexing

        Returns:
            List of chunk_ids that were successfully stored
        """
        ...

    async def get_chunk(self, chunk_id: str) -> Optional[Any]:
        """
        Get a single chunk by ID.

        Args:
            chunk_id: The unique chunk identifier

        Returns:
            DocumentChunk if found, None otherwise
        """
        ...

    async def get_chunks_by_ids(
        self,
        chunk_ids: List[str],
    ) -> List[Any]:  # List[DocumentChunk]
        """
        Get multiple chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of DocumentChunk entities (may be fewer than requested if some not found)
        """
        ...

    async def search_by_keyword(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:  # List[ChunkSearchResult]
        """
        Search chunks using keyword/BM25 search.

        Args:
            query: Search query text
            top_k: Maximum number of results to return
            filters: Optional metadata filters (e.g., {"file_type": "pdf"})

        Returns:
            List of ChunkSearchResult with scores
        """
        ...

    async def search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:  # List[ChunkSearchResult]
        """
        Search chunks using vector similarity (if embeddings stored in ES).

        Args:
            query_vector: Query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            List of ChunkSearchResult with scores
        """
        ...

    async def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a single chunk by ID.

        Args:
            chunk_id: The chunk ID to delete

        Returns:
            True if deleted, False otherwise
        """
        ...

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            doc_id: The parent document ID

        Returns:
            Number of chunks deleted
        """
        ...

    async def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get aggregated metadata for a document.

        Args:
            doc_id: The document ID

        Returns:
            Document metadata including chunk count, sources, etc.
        """
        ...

    async def health_check(self) -> Dict[str, Any]:
        """
        Check document store health.

        Returns:
            Health status including connection state, index info, etc.
        """
        ...

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get document store statistics.

        Returns:
            Statistics including document count, index size, etc.
        """
        ...


class IHybridStore(ABC):
    """
    Interface for hybrid storage (ES + Milvus coordination).

    This interface defines operations that span both document store (ES)
    and vector store (Milvus), coordinating consistent operations across both.
    """

    @abstractmethod
    async def index_chunks(
        self,
        chunks: List[Any],  # List[DocumentChunk]
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        """
        Index chunks to both ES and Milvus.

        Args:
            chunks: List of DocumentChunk entities with embeddings
            batch_size: Number of chunks per batch

        Returns:
            Result dict with counts for ES and Milvus indexing
        """
        ...

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List[Any]:  # List[ChunkSearchResult]
        """
        Perform hybrid search combining vector and BM25 results.

        Args:
            query: Text query for BM25 search
            query_vector: Query embedding for vector search
            top_k: Maximum number of results
            filters: Optional metadata filters
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 search results

        Returns:
            List of ChunkSearchResult with fused scores
        """
        ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document from both ES and Milvus.

        Args:
            doc_id: The document ID to delete

        Returns:
            Result dict with deletion counts from both stores
        """
        ...

    @abstractmethod
    async def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete document information from both stores.

        Args:
            doc_id: The document ID

        Returns:
            Document info including chunks, metadata, and vector status
        """
        ...

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of both storage backends.

        Returns:
            Combined health status for ES and Milvus
        """
        ...
