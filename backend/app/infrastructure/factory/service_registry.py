"""
Service Registry - Manages service instances and their lifecycle.

Implements the Service Locator pattern for centralized service management.
Provides both sync and async initialization paths.

Includes support for:
- Core services (LLM, Vector Store, Document Processor)
- RAG services (RAG, Chat, Hybrid Retrieval)
- ES services (Elasticsearch Manager, Hybrid Store)
- Agent services (Tool Registry, Memory, ReAct, LangGraph)
"""

import logging
from typing import Any, Dict, Optional

from config.settings import settings
from .service_factory import ServiceFactory

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Central registry for managing service instances.

    Features:
    - Singleton pattern for global access
    - Lazy initialization support
    - Async warmup for embedding models
    - Dependency tracking and proper initialization order
    - Elasticsearch and hybrid storage integration
    """

    _instance: Optional["ServiceRegistry"] = None

    def __new__(cls) -> "ServiceRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry (only runs once due to singleton)."""
        if hasattr(self, "_services"):
            return

        self._services: Dict[str, Any] = {}
        self._initialized: bool = False
        self._embedding_warmed_up: bool = False
        self._es_initialized: bool = False

    def _ensure_core_services(self) -> None:
        """Ensure core services are created (lazy initialization)."""
        if "llm" not in self._services:
            self._services["llm"] = ServiceFactory.create_llm_manager()

        if "vector_store" not in self._services:
            self._services["vector_store"] = ServiceFactory.create_vector_store()

        if "document_processor" not in self._services:
            self._services["document_processor"] = ServiceFactory.create_document_processor()

        if "embedding_service" not in self._services:
            self._services["embedding_service"] = ServiceFactory.create_embedding_service()

    def _ensure_es_services(self) -> None:
        """Ensure Elasticsearch and hybrid storage services are created."""
        if self._es_initialized:
            return

        self._ensure_core_services()

        # Create ES manager (may return None if disabled)
        if "es_manager" not in self._services:
            try:
                self._services["es_manager"] = ServiceFactory.create_elasticsearch_manager()
                if self._services["es_manager"]:
                    logger.info("[ServiceRegistry] Elasticsearch manager created")
            except Exception as e:
                logger.warning(f"[ServiceRegistry] Failed to create ES manager: {e}")
                self._services["es_manager"] = None

        # Create hybrid store manager (may return None if disabled or ES unavailable)
        if "hybrid_store" not in self._services:
            try:
                self._services["hybrid_store"] = ServiceFactory.create_hybrid_store_manager(
                    es_manager=self._services.get("es_manager"),
                    vector_store=self._services["vector_store"],
                    embedding_service=self._services["embedding_service"],
                )
                if self._services["hybrid_store"]:
                    logger.info("[ServiceRegistry] Hybrid store manager created")
            except Exception as e:
                logger.warning(f"[ServiceRegistry] Failed to create hybrid store: {e}")
                self._services["hybrid_store"] = None

        # Create hybrid retriever (may return None if disabled or ES unavailable)
        if "hybrid_retriever" not in self._services:
            try:
                self._services["hybrid_retriever"] = ServiceFactory.create_hybrid_retriever(
                    es_manager=self._services.get("es_manager"),
                    vector_store=self._services["vector_store"],
                    embedding_service=self._services["embedding_service"],
                )
                if self._services["hybrid_retriever"]:
                    logger.info("[ServiceRegistry] Hybrid retriever created")
            except Exception as e:
                logger.warning(f"[ServiceRegistry] Failed to create hybrid retriever: {e}")
                self._services["hybrid_retriever"] = None

        self._es_initialized = True

    def _ensure_rag_services(self) -> None:
        """Ensure RAG-related services are created."""
        self._ensure_core_services()

        if "rag" not in self._services:
            self._services["rag"] = ServiceFactory.create_rag_service(
                llm_manager=self._services["llm"],
                vector_store=self._services["vector_store"],
                document_processor=self._services["document_processor"],
            )

        if "chat" not in self._services:
            self._services["chat"] = ServiceFactory.create_chat_service(
                llm_manager=self._services["llm"],
            )

    def _ensure_agent_services(self) -> None:
        """Ensure agent-related services are created."""
        self._ensure_rag_services()

        if "tool_registry" not in self._services:
            self._services["tool_registry"] = ServiceFactory.create_tool_registry(
                vector_store=self._services["vector_store"],
                document_processor=self._services["document_processor"],
                rag_service=self._services["rag"],
            )

        if "memory_manager" not in self._services:
            self._services["memory_manager"] = ServiceFactory.create_memory_manager(
                llm_manager=self._services["llm"],
            )

        if "store_manager" not in self._services:
            self._services["store_manager"] = ServiceFactory.create_store_manager()
            logger.info("[ServiceRegistry] Store manager created for long-term memory")

        if "agent" not in self._services:
            self._services["agent"] = ServiceFactory.create_react_agent(
                llm_manager=self._services["llm"],
                tool_registry=self._services["tool_registry"],
                memory_manager=self._services["memory_manager"],
            )

        if "langgraph_agent" not in self._services:
            self._services["langgraph_agent"] = ServiceFactory.create_langgraph_agent(
                llm_manager=self._services["llm"],
                tool_registry=self._services["tool_registry"],
                memory_manager=self._services["memory_manager"],
                store_manager=self._services.get("store_manager"),
            )

    def _ensure_upload_services(self) -> None:
        """Ensure upload-related services are created."""
        self._ensure_rag_services()

        if "upload_service" not in self._services:
            self._services["upload_service"] = ServiceFactory.create_upload_service(
                rag_service=self._services["rag"],
            )

    async def async_init(self) -> Dict[str, Any]:
        """
        Initialize all services asynchronously.

        This should be called during application startup.
        Warms up embedding models before creating other services.

        Returns:
            Dict of all initialized services
        """
        if self._initialized:
            return self.get_all()

        logger.info("[ServiceRegistry] Starting async initialization...")

        # 1. Warmup embedding model first
        if not self._embedding_warmed_up:
            logger.info("[ServiceRegistry] Warming up embedding model...")
            from app.core.embeddings import get_embedding_manager

            embedding_manager = get_embedding_manager()
            await embedding_manager.warmup()
            self._embedding_warmed_up = True
            logger.info("[ServiceRegistry] Embedding model warmed up")

        # 2. Create all services in proper order
        logger.info("[ServiceRegistry] Creating core services...")
        self._ensure_core_services()

        # 3. Initialize ES and hybrid storage services
        if settings.ELASTICSEARCH_ENABLED:
            logger.info("[ServiceRegistry] Creating Elasticsearch services...")
            self._ensure_es_services()

            # Initialize ES index if ES manager is available
            es_manager = self._services.get("es_manager")
            if es_manager:
                try:
                    await es_manager.ensure_index()
                    logger.info("[ServiceRegistry] Elasticsearch index ensured")
                except Exception as e:
                    logger.warning(f"[ServiceRegistry] Failed to ensure ES index: {e}")

        logger.info("[ServiceRegistry] Creating RAG services...")
        self._ensure_rag_services()

        logger.info("[ServiceRegistry] Creating agent services...")
        self._ensure_agent_services()

        logger.info("[ServiceRegistry] Creating upload services...")
        self._ensure_upload_services()

        self._initialized = True
        logger.info("[ServiceRegistry] All services initialized")

        return self.get_all()

    def get(self, name: str) -> Optional[Any]:
        """
        Get a service by name.

        Triggers lazy initialization if service doesn't exist.

        Args:
            name: Service name

        Returns:
            Service instance or None if not found
        """
        if name not in self._services:
            # Trigger lazy initialization based on service type
            if name in ("llm", "vector_store", "document_processor", "embedding_service"):
                self._ensure_core_services()
            elif name in ("es_manager", "hybrid_store", "hybrid_retriever"):
                self._ensure_es_services()
            elif name in ("rag", "chat"):
                self._ensure_rag_services()
            elif name in ("tool_registry", "memory_manager", "store_manager", "agent", "langgraph_agent"):
                self._ensure_agent_services()
            elif name == "upload_service":
                self._ensure_upload_services()

        return self._services.get(name)

    def get_all(self) -> Dict[str, Any]:
        """
        Get all service instances.

        Triggers full initialization if not done.

        Returns:
            Dict of all services
        """
        if not self._initialized:
            logger.debug("[ServiceRegistry] Lazy sync initialization...")
            self._ensure_core_services()
            if settings.ELASTICSEARCH_ENABLED:
                self._ensure_es_services()
            self._ensure_rag_services()
            self._ensure_agent_services()
            self._ensure_upload_services()

        return self._services.copy()

    def register(self, name: str, service: Any) -> None:
        """
        Register a service instance manually.

        Useful for testing or custom service implementations.

        Args:
            name: Service name
            service: Service instance
        """
        logger.debug(f"[ServiceRegistry] Registering service: {name}")
        self._services[name] = service

    def unregister(self, name: str) -> Optional[Any]:
        """
        Unregister a service.

        Args:
            name: Service name

        Returns:
            The removed service or None
        """
        return self._services.pop(name, None)

    def reset(self) -> None:
        """
        Reset the registry (useful for testing).

        Clears all services and resets initialization state.
        """
        logger.info("[ServiceRegistry] Resetting registry")
        self._services.clear()
        self._initialized = False
        self._embedding_warmed_up = False
        self._es_initialized = False

    def update_llm(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Update LLM configuration and reinitialize dependent services.

        Args:
            provider: LLM provider
            model: Model name
            api_key: API key
            base_url: Base URL
            **kwargs: Additional configuration

        Returns:
            New LLM manager instance
        """
        logger.info(f"[ServiceRegistry] Updating LLM to {provider}/{model}")

        # Create new LLM manager
        self._services["llm"] = ServiceFactory.create_llm_manager(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

        # Reset dependent services
        for name in ("rag", "chat", "agent", "langgraph_agent", "memory_manager"):
            if name in self._services:
                del self._services[name]

        # Reinitialize dependent services
        self._ensure_rag_services()
        self._ensure_agent_services()

        return self._services["llm"]

    @property
    def is_initialized(self) -> bool:
        """Check if registry is fully initialized."""
        return self._initialized

    @property
    def is_embedding_warmed_up(self) -> bool:
        """Check if embedding model is warmed up."""
        return self._embedding_warmed_up

    @property
    def is_es_initialized(self) -> bool:
        """Check if Elasticsearch services are initialized."""
        return self._es_initialized

    @property
    def is_hybrid_storage_available(self) -> bool:
        """Check if hybrid storage (ES + Milvus) is available."""
        return (
            self._es_initialized
            and self._services.get("es_manager") is not None
            and self._services.get("hybrid_store") is not None
        )


# Global registry instance
_registry: Optional[ServiceRegistry] = None


def get_registry() -> ServiceRegistry:
    """
    Get the global service registry instance.

    Returns:
        ServiceRegistry singleton instance
    """
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
    return _registry


# Convenience functions for backward compatibility
async def async_init_services() -> Dict[str, Any]:
    """Initialize all services asynchronously."""
    return await get_registry().async_init()


def get_services() -> Dict[str, Any]:
    """Get all service instances."""
    return get_registry().get_all()
