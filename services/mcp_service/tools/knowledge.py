"""
Knowledge Base Tools - MCP tools for knowledge base operations.
"""
from typing import Any, List, Optional, Dict

from services.common.logging import get_logger
from .base import BaseTool, ToolParameter

logger = get_logger(__name__)


class KnowledgeSearchTool(BaseTool):
    """Tool for searching knowledge base."""

    name = "knowledge_search"
    description = (
        "Search the knowledge base for relevant documents and information. "
        "Use this tool when you need to find specific information from the "
        "company's documentation, FAQs, or other knowledge sources."
    )
    tags = ["knowledge", "search", "rag"]
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="Search query to find relevant information",
            required=True,
        ),
        ToolParameter(
            name="top_k",
            type="integer",
            description="Number of results to return (1-10)",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="knowledge_base_id",
            type="string",
            description="Specific knowledge base to search",
            required=False,
        ),
    ]

    def __init__(self, rag_client=None):
        """
        Initialize knowledge search tool.

        Args:
            rag_client: HTTP client for RAG service
        """
        super().__init__()
        self.rag_client = rag_client

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        knowledge_base_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute knowledge search.

        Args:
            query: Search query
            top_k: Number of results
            knowledge_base_id: Optional knowledge base ID

        Returns:
            Search results
        """
        top_k = min(max(1, top_k), 10)

        if not self.rag_client:
            return {
                "results": [],
                "message": "RAG service not configured",
            }

        try:
            # Call RAG service via HTTP
            response = await self.rag_client.retrieve(
                query=query,
                top_k=top_k,
                knowledge_base_id=knowledge_base_id,
            )

            # Response is a dict from HTTP client
            documents = response.get("documents", [])

            results = [
                {
                    "id": doc.get("id", ""),
                    "content": doc.get("content", ""),
                    "score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {}),
                    "source": doc.get("source", ""),
                }
                for doc in documents
            ]

            logger.info(f"Knowledge search returned {len(results)} results for '{query[:50]}...'")

            return {
                "results": results,
                "query": query,
                "transformed_query": response.get("transformed_query"),
                "latency_ms": response.get("latency_ms", 0),
            }
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return {
                "results": [],
                "error": str(e),
            }


class KnowledgeAddTool(BaseTool):
    """Tool for adding content to knowledge base."""

    name = "knowledge_add"
    description = (
        "Add new text content to the knowledge base. "
        "Use this to store important information that should be "
        "available for future searches."
    )
    tags = ["knowledge", "index"]
    parameters = [
        ToolParameter(
            name="content",
            type="string",
            description="Content to add to the knowledge base",
            required=True,
        ),
        ToolParameter(
            name="knowledge_base_id",
            type="string",
            description="Knowledge base to add to",
            required=False,
        ),
        ToolParameter(
            name="metadata",
            type="object",
            description="Additional metadata (title, source, etc.)",
            required=False,
            default={},
        ),
    ]

    def __init__(self, rag_client=None):
        super().__init__()
        self.rag_client = rag_client

    async def execute(
        self,
        content: str,
        knowledge_base_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Add content to knowledge base."""
        if not content.strip():
            return {
                "success": False,
                "error": "Content cannot be empty",
            }

        if not self.rag_client:
            return {
                "success": False,
                "message": "RAG service not configured",
            }

        try:
            response = await self.rag_client.index_document(
                content=content,
                knowledge_base_id=knowledge_base_id,
                metadata=metadata or {},
            )

            logger.info(f"Added document to knowledge base: {response.get('document_id', 'unknown')}")

            return {
                "success": response.get("success", False),
                "document_id": response.get("document_id", ""),
                "chunks_created": response.get("chunks_created", 0),
            }
        except Exception as e:
            logger.error(f"Failed to add to knowledge base: {e}")
            return {
                "success": False,
                "error": str(e),
            }


class KnowledgeStatsTool(BaseTool):
    """Tool for getting knowledge base statistics."""

    name = "knowledge_stats"
    description = (
        "Get statistics about the knowledge base, including "
        "document count and collection information."
    )
    tags = ["knowledge", "stats"]
    parameters = [
        ToolParameter(
            name="knowledge_base_id",
            type="string",
            description="Specific knowledge base to get stats for",
            required=False,
        ),
    ]

    def __init__(self, rag_client=None):
        super().__init__()
        self.rag_client = rag_client

    async def execute(
        self,
        knowledge_base_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if not self.rag_client:
            return {
                "success": False,
                "error": "RAG service not configured",
            }

        try:
            response = await self.rag_client.get_stats(knowledge_base_id)

            return {
                "success": True,
                "total_documents": response.get("total_documents", 0),
                "total_chunks": response.get("total_chunks", 0),
                "index_size_bytes": response.get("index_size_bytes", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {
                "success": False,
                "error": str(e),
            }
