"""
Service Factory - Creates service instances with proper configuration.

Uses Factory pattern to encapsulate service creation logic,
making it easy to swap implementations and manage dependencies.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar

from app.core.vector_store import VectorStoreManager
from config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceFactory:
    """
    Factory for creating service instances.

    Encapsulates all service creation logic in one place,
    making it easy to:
    - Swap implementations (e.g., different LLM providers)
    - Manage dependencies between services
    - Configure services from settings
    """

    @staticmethod
    def create_llm_manager(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **overrides,
    ) -> "LLMManager":
        """
        Create an LLM Manager instance.

        Args:
            provider: LLM provider (default from settings)
            model: Model name (default from settings)
            **overrides: Additional configuration overrides

        Returns:
            Configured LLMManager instance
        """
        from app.core.llm_manager import LLMManager

        config = {
            "provider": provider or settings.LLM_PROVIDER,
            "model": model or settings.LLM_MODEL,
            "base_url": overrides.get("base_url", settings.LLM_BASE_URL),
            "api_key": overrides.get("api_key", settings.LLM_API_KEY),
            "temperature": overrides.get("temperature", settings.LLM_TEMPERATURE),
            "max_tokens": overrides.get("max_tokens", settings.LLM_MAX_TOKENS),
        }

        logger.debug(f"Creating LLMManager with provider={config['provider']}, model={config['model']}")
        return LLMManager(**config)

    @staticmethod
    def create_vector_store(
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        **overrides,
    ) -> "VectorStoreManager":
        """
        Create a Vector Store Manager instance.

        Args:
            host: Milvus host (default from settings)
            port: Milvus port (default from settings)
            collection_name: Collection name (default from settings)
            **overrides: Additional configuration overrides

        Returns:
            Configured VectorStoreManager instance
        """
        from app.core.vector_store import VectorStoreManager

        config = {
            "host": host or settings.MILVUS_HOST,
            "port": port or settings.MILVUS_PORT,
            "collection_name": collection_name or settings.MILVUS_COLLECTION,
            "embedding_model": overrides.get("embedding_model", settings.EMBEDDING_MODEL),
            "embedding_base_url": overrides.get("embedding_base_url", settings.EMBEDDING_BASE_URL),
        }

        logger.debug(f"Creating VectorStoreManager with host={config['host']}:{config['port']}")
        return VectorStoreManager(**config)

    @staticmethod
    def create_document_processor(
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> "DocumentProcessor":
        """
        Create a Document Processor instance.

        Args:
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks

        Returns:
            Configured DocumentProcessor instance
        """
        from app.core.document_processor import DocumentProcessor

        config = {
            "chunk_size": chunk_size or settings.RAG_CHUNK_SIZE,
            "chunk_overlap": chunk_overlap or settings.RAG_CHUNK_OVERLAP,
        }

        logger.debug(f"Creating DocumentProcessor with chunk_size={config['chunk_size']}")
        return DocumentProcessor(**config)

    @staticmethod
    def create_rag_service(
        llm_manager: "LLMManager",
        vector_store: "VectorStoreManager",
        document_processor: "DocumentProcessor",
        top_k: Optional[int] = None,
    ) -> "RAGService":
        """
        Create a RAG Service instance.

        Args:
            llm_manager: LLM Manager dependency
            vector_store: Vector Store dependency
            document_processor: Document Processor dependency
            top_k: Number of documents to retrieve

        Returns:
            Configured RAGService instance
        """
        from app.services.rag_service import RAGService

        logger.debug("Creating RAGService")
        return RAGService(
            llm_manager=llm_manager,
            vector_store_manager=vector_store,
            document_processor=document_processor,
            top_k=top_k or settings.RAG_TOP_K,
        )

    @staticmethod
    def create_chat_service(
        llm_manager: "LLMManager",
    ) -> "ChatService":
        """
        Create a Chat Service instance.

        Args:
            llm_manager: LLM Manager dependency

        Returns:
            Configured ChatService instance
        """
        from app.services.rag_service import ChatService

        logger.debug("Creating ChatService")
        return ChatService(llm_manager)

    @staticmethod
    def create_tool_registry(
        vector_store: "VectorStoreManager",
        document_processor: "DocumentProcessor",
        rag_service: "RAGService",
    ) -> "ToolRegistry":
        """
        Create a Tool Registry instance with default tools.

        Args:
            vector_store: Vector Store dependency
            document_processor: Document Processor dependency
            rag_service: RAG Service dependency

        Returns:
            Configured ToolRegistry instance with default tools
        """
        from app.mcp.registry import ToolRegistry

        logger.debug("Creating ToolRegistry")
        registry = ToolRegistry()
        registry.initialize_default_tools(
            vector_store=vector_store,
            document_processor=document_processor,
            rag_service=rag_service,
        )
        return registry

    @staticmethod
    def create_memory_manager(
        llm_manager: "LLMManager",
        max_messages: Optional[int] = None,
        summary_threshold: Optional[int] = None,
        persist_path: Optional[str] = None,
    ) -> "MemoryManager":
        """
        Create a Memory Manager instance.

        Args:
            llm_manager: LLM Manager dependency
            max_messages: Maximum messages to keep
            summary_threshold: Threshold for summarization
            persist_path: Path for persisting conversation memories

        Returns:
            Configured MemoryManager instance
        """
        from app.agent.memory import MemoryManager

        logger.debug("Creating MemoryManager")
        return MemoryManager(
            llm_manager=llm_manager,
            max_messages=max_messages or settings.AGENT_MEMORY_MAX_MESSAGES,
            summary_threshold=summary_threshold or settings.AGENT_MEMORY_SUMMARY_THRESHOLD,
            persist_path=persist_path or getattr(settings, 'AGENT_MEMORY_PERSIST_PATH', None),
        )

    @staticmethod
    def create_store_manager(
        store_type: Optional[str] = None,
        storage_path: Optional[str] = None,
    ) -> "StoreManager":
        """
        Create a Store Manager instance for long-term memory.

        Args:
            store_type: Type of store ("memory" or "persistent")
            storage_path: Path for persistent storage

        Returns:
            Configured StoreManager instance
        """
        from app.agent.store import StoreManager, create_memory_store

        # Get settings with fallbacks
        store_type = store_type or getattr(settings, 'AGENT_STORE_TYPE', 'persistent')
        storage_path = storage_path or getattr(settings, 'AGENT_STORE_PATH', './data/memory_store')

        logger.debug(f"Creating StoreManager with type={store_type}, path={storage_path}")

        store = create_memory_store(
            store_type=store_type,
            storage_path=storage_path,
        )

        return StoreManager(store)

    @staticmethod
    def create_react_agent(
        llm_manager: "LLMManager",
        tool_registry: "ToolRegistry",
        memory_manager: "MemoryManager",
        max_iterations: Optional[int] = None,
    ) -> "ReActAgent":
        """
        Create a ReAct Agent instance.

        Args:
            llm_manager: LLM Manager dependency
            tool_registry: Tool Registry dependency
            memory_manager: Memory Manager dependency
            max_iterations: Maximum agent iterations

        Returns:
            Configured ReActAgent instance
        """
        from app.agent.react_agent import ReActAgent

        logger.debug("Creating ReActAgent")
        return ReActAgent(
            llm_manager=llm_manager,
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            max_iterations=max_iterations or settings.AGENT_MAX_ITERATIONS,
        )

    @staticmethod
    def create_langgraph_agent(
        llm_manager: "LLMManager",
        tool_registry: "ToolRegistry",
        memory_manager: "MemoryManager",
        store_manager: Optional["StoreManager"] = None,
        max_iterations: Optional[int] = None,
    ) -> Optional["LangGraphAgent"]:
        """
        Create a LangGraph Agent instance if available.

        Args:
            llm_manager: LLM Manager dependency
            tool_registry: Tool Registry dependency
            memory_manager: Memory Manager dependency (short-term)
            store_manager: Store Manager dependency (long-term memory)
            max_iterations: Maximum agent iterations

        Returns:
            Configured LangGraphAgent instance or None if not available
        """
        from app.agent import LANGGRAPH_AVAILABLE, create_langgraph_agent

        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph is not available")
            return None

        try:
            logger.debug(f"Creating LangGraphAgent with store_manager={store_manager is not None}")
            return create_langgraph_agent(
                llm_manager=llm_manager,
                tool_registry=tool_registry,
                memory_manager=memory_manager,
                store_manager=store_manager,
                max_iterations=max_iterations or settings.AGENT_MAX_ITERATIONS,
                enable_planning=settings.AGENT_ENABLE_PLANNING,
                enable_parallel_tools=settings.AGENT_ENABLE_PARALLEL_TOOLS,
                tool_timeout=settings.AGENT_TOOL_TIMEOUT,
                max_tool_concurrency=settings.AGENT_MAX_TOOL_CONCURRENCY,
            )
        except Exception as e:
            logger.warning(f"Failed to create LangGraphAgent: {e}")
            return None

    @staticmethod
    def create_upload_service(
        rag_service: "RAGService",
        upload_dir: Optional[str] = None,
    ) -> "UploadService":
        """
        Create an Upload Service instance.

        Args:
            rag_service: RAG Service dependency
            upload_dir: Upload directory path

        Returns:
            Configured UploadService instance
        """
        from app.services.upload_service import UploadService

        logger.debug("Creating UploadService")
        return UploadService(
            rag_service=rag_service,
            upload_dir=upload_dir or settings.KNOWLEDGE_BASE_PATH,
        )

    @staticmethod
    def create_elasticsearch_manager(
        hosts: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_name: Optional[str] = None,
        **overrides,
    ) -> Optional["ElasticsearchManager"]:
        """
        Create an Elasticsearch Manager instance.

        Args:
            hosts: ES hosts (default from settings)
            username: ES username (default from settings)
            password: ES password (default from settings)
            index_name: Index name (default from settings)
            **overrides: Additional configuration overrides

        Returns:
            Configured ElasticsearchManager instance or None if ES is disabled
        """
        if not settings.ELASTICSEARCH_ENABLED:
            logger.info("Elasticsearch is disabled, skipping ES manager creation")
            return None

        from app.core.elasticsearch_manager import ElasticsearchManager

        config = {
            "hosts": hosts or settings.ELASTICSEARCH_HOSTS,
            "username": username or settings.ELASTICSEARCH_USERNAME,
            "password": password or settings.ELASTICSEARCH_PASSWORD,
            "index_name": index_name or settings.ELASTICSEARCH_CHUNK_INDEX,
            "use_ssl": overrides.get("use_ssl", settings.ELASTICSEARCH_USE_SSL),
            "verify_certs": overrides.get("verify_certs", settings.ELASTICSEARCH_VERIFY_CERTS),
            "ca_certs": overrides.get("ca_certs", settings.ELASTICSEARCH_CA_CERTS),
            "timeout": overrides.get("timeout", settings.ELASTICSEARCH_TIMEOUT),
            "max_retries": overrides.get("max_retries", settings.ELASTICSEARCH_MAX_RETRIES),
        }

        logger.debug(f"Creating ElasticsearchManager with hosts={config['hosts']}")
        return ElasticsearchManager(**config)

    @staticmethod
    def create_embedding_service() -> "IEmbeddingService":
        """
        Create an Embedding Service instance.

        Returns:
            Configured EmbeddingManager instance implementing IEmbeddingService
        """
        from app.core.embeddings import get_embedding_manager

        logger.debug("Creating EmbeddingService")
        return get_embedding_manager()

    @staticmethod
    def create_hybrid_store_manager(
        es_manager: Any,
        vector_store: "VectorStoreManager",
        embedding_service: "IEmbeddingService",
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> Optional["HybridStoreManager"]:
        """
        Create a Hybrid Store Manager instance.

        Args:
            es_manager: Elasticsearch manager dependency
            vector_store: Vector store dependency
            embedding_service: Embedding service dependency
            vector_weight: Weight for vector search (default from settings)
            bm25_weight: Weight for BM25 search (default from settings)

        Returns:
            Configured HybridStoreManager instance or None if hybrid storage is disabled
        """
        if not settings.HYBRID_STORAGE_ENABLED:
            logger.info("Hybrid storage is disabled, skipping hybrid store creation")
            return None

        if es_manager is None:
            logger.warning("ES manager not available, skipping hybrid store creation")
            return None

        from app.core.hybrid_store_manager import init_hybrid_store_manager

        logger.debug("Creating HybridStoreManager via init_hybrid_store_manager")
        return init_hybrid_store_manager(
            es_manager=es_manager,
            vector_store=vector_store,
            embedding_service=embedding_service,
        )

    @staticmethod
    def create_hybrid_retriever(
        es_manager: Any,
        vector_store: "VectorStoreManager",
        embedding_service: "IEmbeddingService",
        top_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> Optional["HybridESMilvusRetriever"]:
        """
        Create a Hybrid ES+Milvus Retriever instance.

        Args:
            es_manager: Elasticsearch manager dependency
            vector_store: Vector store dependency
            embedding_service: Embedding service dependency
            top_k: Number of results to return
            vector_weight: Weight for vector search
            bm25_weight: Weight for BM25 search

        Returns:
            Configured HybridESMilvusRetriever instance or None if not available
        """
        if not settings.HYBRID_STORAGE_ENABLED or es_manager is None:
            logger.info("Hybrid retriever not available, returning None")
            return None

        from app.rag.hybrid_es_retriever import HybridESMilvusRetriever

        logger.debug("Creating HybridESMilvusRetriever")
        return HybridESMilvusRetriever(
            es_manager=es_manager,
            vector_store=vector_store,
            embedding_service=embedding_service,
            top_k=top_k or settings.RAG_FINAL_TOP_K,
            vector_weight=vector_weight or settings.HYBRID_MILVUS_WEIGHT,
            bm25_weight=bm25_weight or settings.HYBRID_ES_WEIGHT,
        )