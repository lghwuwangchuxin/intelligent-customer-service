"""
Domain Entities and Value Objects.

These are the core domain models that represent business concepts.
They are framework-agnostic and can be used across all layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """Domain entity representing a chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        role = data.get("role", "user")
        if isinstance(role, str):
            role = MessageRole(role)
        return cls(
            role=role,
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Document:
    """Domain entity representing a document or text chunk."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: Optional[float] = None

    @property
    def source(self) -> Optional[str]:
        """Get document source from metadata."""
        return self.metadata.get("source") or self.metadata.get("file_name")

    @property
    def title(self) -> Optional[str]:
        """Get document title from metadata."""
        return self.metadata.get("title")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass
class SearchResult:
    """Domain entity representing a search result."""
    documents: List[Document]
    query: str
    total_count: int = 0
    search_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [doc.to_dict() for doc in self.documents],
            "total_count": self.total_count,
            "search_time_ms": self.search_time_ms,
        }


class ToolCallStatus(str, Enum):
    """Tool call status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ToolCall:
    """Domain entity representing a tool call."""
    id: str
    name: str
    arguments: Dict[str, Any]
    status: ToolCallStatus = ToolCallStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }


class ThoughtType(str, Enum):
    """Agent thought type enumeration."""
    THINKING = "thinking"
    PLANNING = "planning"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    CONCLUSION = "conclusion"


@dataclass
class AgentThought:
    """Domain entity representing an agent thought."""
    type: ThoughtType
    content: str
    iteration: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    tool_call: Optional[ToolCall] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.type.value,
            "content": self.content,
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.tool_call:
            result["tool_call"] = self.tool_call.to_dict()
        return result


@dataclass
class AgentResult:
    """Domain entity representing an agent execution result."""
    response: str
    thoughts: List[AgentThought] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    iterations: int = 0
    conversation_id: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response": self.response,
            "thoughts": [t.to_dict() for t in self.thoughts],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "iterations": self.iterations,
            "conversation_id": self.conversation_id,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class KnowledgeAddResult:
    """Domain entity representing result of adding knowledge."""
    success: bool
    num_documents: int = 0
    num_chunks: int = 0
    source: Optional[str] = None
    error: Optional[str] = None
    duplicates_skipped: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "num_documents": self.num_documents,
            "num_chunks": self.num_chunks,
            "source": self.source,
            "error": self.error,
            "duplicates_skipped": self.duplicates_skipped,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class DocumentChunk:
    """
    Domain entity representing a document chunk for hybrid ES+Milvus storage.

    This entity is the core data structure that bridges Elasticsearch and Milvus:
    - chunk_id: Unique identifier linking ES document to Milvus vector
    - doc_id: Reference to the original parent document
    - content: The text content stored in ES for retrieval
    - embedding: The vector stored in Milvus for semantic search
    - metadata: Rich metadata stored in ES for filtering and display
    """
    chunk_id: str                    # UUID, primary key linking ES and Milvus
    doc_id: str                      # Original document ID
    content: str                     # Text content of the chunk
    metadata: Dict[str, Any] = field(default_factory=dict)  # Rich metadata
    chunk_index: int = 0             # Position within the document
    chunk_total: int = 1             # Total chunks in the document
    embedding: Optional[List[float]] = None  # Vector embedding
    indexed_at: Optional[datetime] = None    # When indexed

    def __post_init__(self):
        """Set indexed_at if not provided."""
        if self.indexed_at is None:
            self.indexed_at = datetime.now()

    @property
    def source(self) -> Optional[str]:
        """Get source from metadata."""
        return self.metadata.get("source") or self.metadata.get("file_path")

    @property
    def file_type(self) -> Optional[str]:
        """Get file type from metadata."""
        return self.metadata.get("file_type")

    @property
    def title(self) -> Optional[str]:
        """Get title from metadata."""
        return self.metadata.get("title")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "chunk_total": self.chunk_total,
            "embedding": self.embedding,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
        }

    def to_es_document(self) -> Dict[str, Any]:
        """Convert to Elasticsearch document format."""
        doc = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "chunk_total": self.chunk_total,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else datetime.now().isoformat(),
        }
        if self.embedding:
            doc["content_vector"] = self.embedding
        return doc

    def to_milvus_record(self) -> Dict[str, Any]:
        """Convert to Milvus record format (vector + chunk_id only)."""
        if not self.embedding:
            raise ValueError("Embedding is required for Milvus storage")
        return {
            "chunk_id": self.chunk_id,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create from dictionary."""
        indexed_at = data.get("indexed_at")
        if indexed_at and isinstance(indexed_at, str):
            indexed_at = datetime.fromisoformat(indexed_at)
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            chunk_index=data.get("chunk_index", 0),
            chunk_total=data.get("chunk_total", 1),
            embedding=data.get("embedding"),
            indexed_at=indexed_at,
        )

    @classmethod
    def from_es_document(cls, doc: Dict[str, Any]) -> "DocumentChunk":
        """Create from Elasticsearch document."""
        source = doc.get("_source", doc)
        indexed_at = source.get("indexed_at")
        if indexed_at and isinstance(indexed_at, str):
            indexed_at = datetime.fromisoformat(indexed_at.replace("Z", "+00:00"))
        return cls(
            chunk_id=source["chunk_id"],
            doc_id=source["doc_id"],
            content=source["content"],
            metadata=source.get("metadata", {}),
            chunk_index=source.get("chunk_index", 0),
            chunk_total=source.get("chunk_total", 1),
            embedding=source.get("content_vector"),
            indexed_at=indexed_at,
        )


@dataclass
class ChunkSearchResult:
    """Domain entity representing a chunk search result with score."""
    chunk: DocumentChunk
    score: float
    source: str = "unknown"  # "milvus", "es_bm25", "es_vector", "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "source": self.source,
        }


@dataclass
class HybridIndexResult:
    """Domain entity representing result of hybrid indexing operation."""
    success: bool
    doc_id: str
    chunks_indexed: int = 0
    es_indexed: int = 0
    milvus_indexed: int = 0
    error: Optional[str] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "doc_id": self.doc_id,
            "chunks_indexed": self.chunks_indexed,
            "es_indexed": self.es_indexed,
            "milvus_indexed": self.milvus_indexed,
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
        }
