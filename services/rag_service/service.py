"""RAG Service - Main service implementation."""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from services.common.logging import get_logger
from services.common.config import get_service_config, ServiceConfig

from .pipeline import QueryTransformer, HybridRetriever, Reranker, PostProcessor
from .pipeline.query_transform import SimpleLLMClient
from .pipeline.hybrid_retriever import RetrievedDocument
from .index import IndexManager, DocumentProcessor
from .storage import MilvusClient, ElasticsearchClient, EmbeddingService

logger = get_logger(__name__)


@dataclass
class RetrieveConfig:
    """Configuration for retrieval."""
    top_k: int = 10
    enable_query_transform: bool = True
    enable_rerank: bool = True
    enable_postprocess: bool = True
    hybrid_alpha: float = 0.5  # 0=BM25 only, 1=vector only
    rerank_top_k: int = 5


@dataclass
class RetrieveResult:
    """Result of retrieval."""
    documents: List[Dict[str, Any]]
    query: str
    transformed_query: Optional[str] = None
    expanded_queries: List[str] = field(default_factory=list)
    latency_ms: int = 0
    total_candidates: int = 0


@dataclass
class IndexResult:
    """Result of indexing."""
    document_id: str
    success: bool
    error: Optional[str] = None


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) Service.

    Provides:
    - Query transformation (HyDE, expansion, rewriting)
    - Hybrid retrieval (vector + BM25)
    - Reranking
    - Post-processing (dedup, MMR)
    - Document indexing and management
    """

    def __init__(
        self,
        milvus_client: Optional[MilvusClient] = None,
        elasticsearch_client: Optional[ElasticsearchClient] = None,
        embedding_service: Optional[EmbeddingService] = None,
        llm_client=None,
        config: Optional[ServiceConfig] = None,
    ):
        """
        Initialize RAG service.

        Args:
            milvus_client: Milvus vector store client
            elasticsearch_client: Elasticsearch client for BM25
            embedding_service: Embedding service
            llm_client: LLM client for query transformation
            config: Service configuration
        """
        self.config = config or get_service_config("rag-service")
        self.milvus_client = milvus_client
        self.es_client = elasticsearch_client
        self.embedding_service = embedding_service
        self.llm_client = llm_client

        # Initialize pipeline components
        self.query_transformer = QueryTransformer(
            llm_client=llm_client,
            enable_hyde=self.config.rag_enable_hyde,
            enable_expansion=False,  # Disable by default for performance
        )

        self.hybrid_retriever = HybridRetriever(
            vector_store=milvus_client,
            elasticsearch_client=elasticsearch_client,
            embedding_model=embedding_service,
            vector_weight=self.config.rag_vector_weight,
            bm25_weight=self.config.rag_bm25_weight,
        )

        self.reranker = Reranker(
            model_name=self.config.rag_rerank_model,
            use_api=False,
            api_key=None,
        )

        self.postprocessor = PostProcessor(
            embedding_model=embedding_service,
            enable_dedup=True,
            enable_mmr=True,
        )

        # Initialize index manager
        self.index_manager = IndexManager(
            vector_store=milvus_client,
            elasticsearch_client=elasticsearch_client,
            embedding_model=embedding_service,
        )

        # Document processor
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.rag_chunk_size,
            chunk_overlap=self.config.rag_chunk_overlap,
        )

        self._initialized = False

    @classmethod
    def from_config(cls, config: Optional[ServiceConfig] = None) -> "RAGService":
        """
        Create RAG service from configuration.

        Args:
            config: Service configuration

        Returns:
            RAGService instance
        """
        if config is None:
            config = get_service_config("rag-service")

        # Initialize LLM client
        llm_client = None
        if config.llm_base_url:
            llm_client = SimpleLLMClient(
                base_url=config.llm_base_url,
                model=config.llm_model,
            )

        # Initialize Milvus client
        milvus_client = MilvusClient(
            host=config.milvus_host,
            port=config.milvus_port,
            collection_name=config.milvus_collection,
        )

        # Initialize Elasticsearch client
        es_client = ElasticsearchClient(
            hosts=config.elasticsearch_url,
            username=config.elasticsearch_username or None,
            password=config.elasticsearch_password or None,
        )

        # Initialize embedding service
        embedding_service = EmbeddingService(
            model=config.embedding_model,
            base_url=config.embedding_base_url,
            batch_size=config.embedding_batch_size,
        )

        return cls(
            milvus_client=milvus_client,
            elasticsearch_client=es_client,
            embedding_service=embedding_service,
            llm_client=llm_client,
            config=config,
        )

    async def initialize(self):
        """Initialize service and dependencies."""
        if self._initialized:
            return

        # Connect to storage backends
        if self.milvus_client:
            await self.milvus_client.connect()

        if self.es_client:
            await self.es_client.connect()

        # Warmup embedding service
        if self.embedding_service:
            await self.embedding_service.warmup()

        # Initialize index manager
        await self.index_manager.initialize()

        self._initialized = True
        logger.info("RAG Service initialized")

    async def retrieve(
        self,
        query: str,
        knowledge_base_id: Optional[str] = None,
        config: Optional[RetrieveConfig] = None,
    ) -> RetrieveResult:
        """
        Retrieve documents for a query.

        Full RAG pipeline:
        1. Query transformation (HyDE, expansion)
        2. Hybrid retrieval (vector + BM25 with RRF)
        3. Reranking
        4. Post-processing (dedup, MMR)

        Args:
            query: Search query
            knowledge_base_id: Filter by knowledge base
            config: Retrieval configuration

        Returns:
            RetrieveResult with documents
        """
        await self.initialize()

        start_time = time.time()
        config = config or RetrieveConfig()

        transformed_query = query
        expanded_queries = []
        hyde_passage = None

        # Step 1: Query transformation
        if config.enable_query_transform:
            try:
                transform_result = await self.query_transformer.transform(query)
                transformed_query = transform_result.transformed_query
                expanded_queries = transform_result.expanded_queries
                hyde_passage = transform_result.hyde_passage
                logger.debug(f"Query transformed: {query} -> {transformed_query}")
            except Exception as e:
                logger.warning(f"Query transformation failed: {e}")

        # Step 2: Hybrid retrieval
        retrieval_result = await self.hybrid_retriever.retrieve(
            query=transformed_query,
            top_k=config.top_k * 2,  # Get more for reranking
            knowledge_base_id=knowledge_base_id,
            expanded_queries=expanded_queries,
            hyde_passage=hyde_passage,
        )

        documents = retrieval_result.documents
        total_candidates = retrieval_result.total_candidates

        # Step 3: Reranking
        if config.enable_rerank and documents:
            try:
                rerank_result = await self.reranker.rerank(
                    query=query,  # Use original query for reranking
                    documents=documents,
                    top_k=config.rerank_top_k,
                )
                documents = rerank_result.documents
                logger.debug(f"Reranked: {rerank_result.original_count} -> {rerank_result.reranked_count}")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        # Step 4: Post-processing
        if config.enable_postprocess and documents:
            try:
                postprocess_result = await self.postprocessor.process(
                    documents=documents,
                    query=query,
                    top_k=config.top_k,
                )
                documents = postprocess_result.documents
                logger.debug(f"Post-processed, removed {postprocess_result.removed_duplicates} duplicates")
            except Exception as e:
                logger.warning(f"Post-processing failed: {e}")

        latency_ms = int((time.time() - start_time) * 1000)

        # Convert to dict format
        doc_dicts = [
            {
                "id": doc.id,
                "content": doc.content,
                "score": doc.score,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        return RetrieveResult(
            documents=doc_dicts,
            query=query,
            transformed_query=transformed_query if transformed_query != query else None,
            expanded_queries=expanded_queries,
            latency_ms=latency_ms,
            total_candidates=total_candidates,
        )

    async def index_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        knowledge_base_id: Optional[str] = None,
        chunk: bool = True,
    ) -> IndexResult:
        """
        Index a document.

        Args:
            content: Document content
            metadata: Document metadata
            knowledge_base_id: Knowledge base ID
            chunk: Whether to chunk the document

        Returns:
            IndexResult
        """
        await self.initialize()

        try:
            if chunk:
                # Process and chunk document
                processed = self.document_processor.process(content, metadata)

                # Index each chunk
                doc_ids = []
                for i, chunk_doc in enumerate(processed.chunks):
                    chunk_metadata = dict(chunk_doc.metadata)
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(processed.chunks)

                    doc_id = await self.index_manager.index_document(
                        content=chunk_doc.content,
                        metadata=chunk_metadata,
                        knowledge_base_id=knowledge_base_id,
                    )
                    doc_ids.append(doc_id)

                return IndexResult(
                    document_id=doc_ids[0] if doc_ids else "",
                    success=True,
                )
            else:
                # Index as single document
                doc_id = await self.index_manager.index_document(
                    content=content,
                    metadata=metadata,
                    knowledge_base_id=knowledge_base_id,
                )
                return IndexResult(document_id=doc_id, success=True)

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return IndexResult(
                document_id="",
                success=False,
                error=str(e),
            )

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
            True if successful
        """
        await self.initialize()
        return await self.index_manager.delete_document(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
        )

    async def list_documents(
        self,
        knowledge_base_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """
        List documents.

        Args:
            knowledge_base_id: Filter by knowledge base
            page: Page number
            page_size: Page size

        Returns:
            Dictionary with documents and total count
        """
        await self.initialize()

        documents, total = await self.index_manager.list_documents(
            knowledge_base_id=knowledge_base_id,
            page=page,
            page_size=page_size,
        )

        return {
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    async def get_stats(
        self,
        knowledge_base_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get index statistics.

        Args:
            knowledge_base_id: Filter by knowledge base

        Returns:
            Statistics dictionary
        """
        await self.initialize()

        stats = await self.index_manager.get_stats(knowledge_base_id)

        return {
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks,
            "index_size_bytes": stats.index_size_bytes,
            "last_updated": stats.last_updated.isoformat() if stats.last_updated else None,
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status
        """
        status = {
            "status": "healthy",
            "components": {},
        }

        # Check Milvus
        if self.milvus_client:
            try:
                milvus_stats = await self.milvus_client.get_stats()
                status["components"]["milvus"] = milvus_stats.get("status", "unknown")
                if milvus_stats.get("status") != "connected":
                    status["status"] = "degraded"
            except Exception as e:
                status["components"]["milvus"] = f"unhealthy: {e}"
                status["status"] = "degraded"

        # Check Elasticsearch
        if self.es_client:
            try:
                es_stats = await self.es_client.get_stats()
                status["components"]["elasticsearch"] = es_stats.get("status", "unknown")
                if es_stats.get("status") != "connected":
                    status["status"] = "degraded"
            except Exception as e:
                status["components"]["elasticsearch"] = f"unhealthy: {e}"
                status["status"] = "degraded"

        # Check embedding service
        if self.embedding_service:
            status["components"]["embedding"] = self.embedding_service.get_info()

        return status
