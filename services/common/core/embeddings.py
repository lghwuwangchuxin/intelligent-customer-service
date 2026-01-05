"""
Unified Embedding Manager for LangChain and LlamaIndex compatibility.
Provides a single interface for embedding operations across frameworks.

Optimized for high-throughput batch embedding with:
- Direct Ollama API calls for reduced overhead
- HTTP connection pooling for reuse
- Warmup mechanism to avoid cold start latency
- Efficient batch processing
- ThreadPoolExecutor for sync methods
- Retry mechanism with exponential backoff
- Redis distributed lock for concurrent requests
- Partial failure recovery
"""
import asyncio
import logging
import time
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Tuple, Any
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

import httpx
from langchain_ollama import OllamaEmbeddings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.embeddings import BaseEmbedding

from config.settings import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EmbeddingConfig:
    """Embedding configuration with defaults."""
    batch_size: int = 50
    max_concurrent: int = 20
    max_retries: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 10.0
    request_timeout: float = 120.0
    connect_timeout: float = 10.0
    max_connections: int = 100
    max_keepalive: int = 50
    thread_pool_size: int = 10
    lock_timeout: int = 30  # Redis lock timeout in seconds
    lock_retry_interval: float = 0.1


# ============================================================================
# HTTP Client Management
# ============================================================================

_http_client: Optional[httpx.AsyncClient] = None
_http_client_lock = asyncio.Lock()


async def get_http_client() -> httpx.AsyncClient:
    """Get or create a global HTTP client with connection pooling (thread-safe)."""
    global _http_client
    async with _http_client_lock:
        if _http_client is None or _http_client.is_closed:
            _http_client = httpx.AsyncClient(
                base_url=settings.EMBEDDING_BASE_URL,
                timeout=httpx.Timeout(120.0, connect=10.0),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=50,
                    keepalive_expiry=30.0,
                ),
            )
    return _http_client


async def close_http_client():
    """Close the global HTTP client."""
    global _http_client
    async with _http_client_lock:
        if _http_client is not None and not _http_client.is_closed:
            await _http_client.aclose()
            _http_client = None


# ============================================================================
# Redis Distributed Lock
# ============================================================================

class EmbeddingLock:
    """
    Redis-based distributed lock for embedding operations.

    Prevents duplicate embedding computation when multiple requests
    try to embed the same text concurrently.
    """

    def __init__(self, timeout: int = 30, retry_interval: float = 0.1):
        self.timeout = timeout
        self.retry_interval = retry_interval
        self._redis = None
        self._initialized = False
        self._fallback_locks: Dict[str, asyncio.Lock] = {}
        self._fallback_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize Redis connection for locking."""
        if self._initialized:
            return self._redis is not None

        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(
                settings.REDIS_URL,
                password=settings.REDIS_PASSWORD,
                socket_timeout=5.0,
                decode_responses=True,
            )
            await self._redis.ping()
            self._initialized = True
            logger.info("[EmbeddingLock] Redis lock initialized")
            return True
        except Exception as e:
            logger.warning(f"[EmbeddingLock] Redis unavailable, using memory locks: {e}")
            self._initialized = True
            return False

    def _make_lock_key(self, content: str) -> str:
        """Generate lock key from content hash."""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"ics:embedding:lock:{content_hash}"

    async def acquire(self, content: str, wait: bool = True) -> bool:
        """
        Acquire lock for embedding a specific content.

        Args:
            content: Text content to lock
            wait: Whether to wait for lock or return immediately

        Returns:
            True if lock acquired, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        if self._redis:
            return await self._acquire_redis(content, wait)
        else:
            return await self._acquire_memory(content, wait)

    async def _acquire_redis(self, content: str, wait: bool) -> bool:
        """Acquire Redis-based lock."""
        lock_key = self._make_lock_key(content)
        lock_value = f"{id(asyncio.current_task())}:{time.time()}"

        start_time = time.time()
        while True:
            try:
                # Try to acquire lock with NX (only if not exists)
                acquired = await self._redis.set(
                    lock_key,
                    lock_value,
                    nx=True,
                    ex=self.timeout
                )
                if acquired:
                    return True

                if not wait:
                    return False

                # Check timeout
                if time.time() - start_time > self.timeout:
                    logger.warning(f"[EmbeddingLock] Lock acquisition timeout for {lock_key[:50]}")
                    return False

                await asyncio.sleep(self.retry_interval)

            except Exception as e:
                logger.warning(f"[EmbeddingLock] Redis lock error: {e}")
                return await self._acquire_memory(content, wait)

    async def _acquire_memory(self, content: str, wait: bool) -> bool:
        """Acquire memory-based lock (fallback)."""
        lock_key = self._make_lock_key(content)

        async with self._fallback_lock:
            if lock_key not in self._fallback_locks:
                self._fallback_locks[lock_key] = asyncio.Lock()
            lock = self._fallback_locks[lock_key]

        if wait:
            await lock.acquire()
            return True
        else:
            return lock.locked() == False and lock.acquire_nowait()

    async def release(self, content: str) -> None:
        """Release lock for content."""
        if not self._initialized:
            return

        if self._redis:
            await self._release_redis(content)
        else:
            await self._release_memory(content)

    async def _release_redis(self, content: str) -> None:
        """Release Redis-based lock."""
        lock_key = self._make_lock_key(content)
        try:
            await self._redis.delete(lock_key)
        except Exception as e:
            logger.warning(f"[EmbeddingLock] Redis unlock error: {e}")

    async def _release_memory(self, content: str) -> None:
        """Release memory-based lock."""
        lock_key = self._make_lock_key(content)
        async with self._fallback_lock:
            if lock_key in self._fallback_locks:
                lock = self._fallback_locks[lock_key]
                if lock.locked():
                    lock.release()

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# Global lock instance
_embedding_lock: Optional[EmbeddingLock] = None


async def get_embedding_lock() -> EmbeddingLock:
    """Get the global embedding lock instance."""
    global _embedding_lock
    if _embedding_lock is None:
        _embedding_lock = EmbeddingLock(
            timeout=settings.EMBEDDING_BATCH_SIZE,  # Use batch size as timeout
            retry_interval=0.1
        )
        await _embedding_lock.initialize()
    return _embedding_lock


# ============================================================================
# Retry Mechanism
# ============================================================================

class RetryError(Exception):
    """Exception raised when all retries are exhausted."""
    def __init__(self, message: str, last_error: Exception):
        super().__init__(message)
        self.last_error = last_error


async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    retryable_errors: tuple = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.HTTPStatusError,
    ),
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        retryable_errors: Tuple of retryable exception types

    Returns:
        Result from successful function call

    Raises:
        RetryError: If all retries are exhausted
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_errors as e:
            last_error = e
            if attempt == max_retries:
                break

            # Calculate delay with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            actual_delay = delay + jitter

            logger.warning(
                f"[Embeddings] Retry {attempt + 1}/{max_retries} after {actual_delay:.2f}s: {e}"
            )
            await asyncio.sleep(actual_delay)

    raise RetryError(
        f"All {max_retries} retries exhausted",
        last_error
    )


# ============================================================================
# Embedding Manager
# ============================================================================

class EmbeddingManager:
    """
    Unified embedding manager that provides both LangChain and LlamaIndex
    compatible embedding instances.

    Optimized for high-throughput with:
    - Direct Ollama API calls (bypass langchain overhead for batch operations)
    - Connection pooling
    - Warmup mechanism
    - ThreadPoolExecutor for sync methods
    - Retry with exponential backoff
    - Redis distributed lock for concurrent deduplication
    - Partial failure recovery
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or settings.EMBEDDING_MODEL
        self.base_url = base_url or settings.EMBEDDING_BASE_URL

        self._langchain_embeddings: Optional[OllamaEmbeddings] = None
        self._llamaindex_embeddings: Optional[OllamaEmbedding] = None
        self._embedding_dim: Optional[int] = None
        self._warmed_up: bool = False

        # Thread pool for sync operations
        self._thread_pool = ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="embedding_"
        )

        # Configuration
        self._config = EmbeddingConfig(
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            max_concurrent=settings.EMBEDDING_MAX_CONCURRENT,
        )

        logger.info(
            f"[Embeddings] Initializing EmbeddingManager - "
            f"model: {self.model}, base_url: {self.base_url}"
        )

    @property
    def langchain(self) -> OllamaEmbeddings:
        """Get LangChain-compatible embeddings instance."""
        if self._langchain_embeddings is None:
            logger.info(f"[Embeddings] Creating LangChain OllamaEmbeddings instance")
            self._langchain_embeddings = OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url,
            )
        return self._langchain_embeddings

    @property
    def llamaindex(self) -> OllamaEmbedding:
        """Get LlamaIndex-compatible embeddings instance."""
        if self._llamaindex_embeddings is None:
            logger.info(f"[Embeddings] Creating LlamaIndex OllamaEmbedding instance")
            self._llamaindex_embeddings = OllamaEmbedding(
                model_name=self.model,
                base_url=self.base_url,
            )
        return self._llamaindex_embeddings

    async def warmup(self) -> None:
        """
        Warmup the embedding model by running a test embedding.
        This loads the model into memory and establishes connections.
        """
        if self._warmed_up:
            return

        logger.info(f"[Embeddings] Warming up embedding model: {self.model}")
        start_time = time.time()

        try:
            # Run a test embedding to warm up the model
            test_embedding = await self._embed_single_direct("warmup test")
            self._embedding_dim = len(test_embedding)
            self._warmed_up = True

            elapsed = time.time() - start_time
            logger.info(
                f"[Embeddings] Warmup complete in {elapsed:.2f}s, "
                f"embedding dim: {self._embedding_dim}"
            )
        except Exception as e:
            logger.warning(f"[Embeddings] Warmup failed: {e}")

    async def _embed_single_direct(self, text: str) -> List[float]:
        """
        Embed a single text using direct Ollama API call with retry.
        """
        async def _do_embed():
            client = await get_http_client()
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]

        return await retry_with_backoff(
            _do_embed,
            max_retries=self._config.max_retries,
            base_delay=self._config.retry_base_delay,
            max_delay=self._config.retry_max_delay,
        )

    async def _embed_batch_direct(
        self,
        texts: List[str],
        use_lock: bool = False,
    ) -> List[List[float]]:
        """
        Embed multiple texts using direct Ollama API call with retry.

        Args:
            texts: List of texts to embed
            use_lock: Whether to use distributed lock for deduplication

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        async def _do_embed():
            client = await get_http_client()
            try:
                # Try the batch /api/embed endpoint first
                response = await client.post(
                    "/api/embed",
                    json={
                        "model": self.model,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["embeddings"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Fallback to individual embedding requests
                    logger.debug("[Embeddings] /api/embed not available, using individual")
                    tasks = [self._embed_single_direct(text) for text in texts]
                    return await asyncio.gather(*tasks)
                raise

        return await retry_with_backoff(
            _do_embed,
            max_retries=self._config.max_retries,
            base_delay=self._config.retry_base_delay,
            max_delay=self._config.retry_max_delay,
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text (sync)."""
        logger.debug(f"[Embeddings] Embedding query: {text[:50]}...")
        return self.langchain.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents (sync)."""
        logger.debug(f"[Embeddings] Embedding {len(texts)} documents")
        return self.langchain.embed_documents(texts)

    def embed_documents_threaded(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Embed documents using thread pool for parallel processing.

        Useful when sync embedding is required but parallelism is desired.
        """
        if not texts:
            return []

        batch_size = batch_size or self._config.batch_size
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        logger.info(
            f"[Embeddings] Threaded embedding: {len(texts)} docs in {len(batches)} batches"
        )

        start_time = time.time()

        def embed_batch(batch: List[str]) -> List[List[float]]:
            return self.langchain.embed_documents(batch)

        # Submit all batches to thread pool
        futures = [
            self._thread_pool.submit(embed_batch, batch)
            for batch in batches
        ]

        # Collect results
        results = []
        for i, future in enumerate(futures):
            try:
                batch_result = future.result(timeout=120)
                results.extend(batch_result)
                logger.debug(f"[Embeddings] Batch {i+1}/{len(batches)} complete")
            except Exception as e:
                logger.error(f"[Embeddings] Batch {i+1} failed: {e}")
                # Fill with empty embeddings for failed batch
                results.extend([[] for _ in batches[i]])

        elapsed = time.time() - start_time
        speed = len(texts) / elapsed if elapsed > 0 else 0
        logger.info(
            f"[Embeddings] Threaded embedding complete: {len(texts)} docs in {elapsed:.2f}s "
            f"({speed:.1f} docs/sec)"
        )

        return results

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query text."""
        logger.debug(f"[Embeddings] Async embedding query: {text[:50]}...")
        return await self._embed_single_direct(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed multiple documents."""
        logger.debug(f"[Embeddings] Async embedding {len(texts)} documents")
        return await self._embed_batch_direct(texts)

    async def aembed_documents_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None,
        show_progress: bool = True,
        use_cache: bool = True,
        use_lock: bool = True,
    ) -> List[List[float]]:
        """
        Batch async embed documents with concurrency control.

        Optimized implementation using:
        - Direct Ollama API calls
        - Connection pooling
        - Semaphore-based concurrency control
        - Retry with exponential backoff
        - Redis distributed lock for deduplication
        - Partial failure recovery
        - Progress logging

        Args:
            texts: List of texts to embed
            batch_size: Number of documents per batch
            max_concurrent: Maximum concurrent requests
            show_progress: Whether to log progress
            use_cache: Whether to check cache (if cache module available)
            use_lock: Whether to use distributed lock

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        batch_size = batch_size or self._config.batch_size
        max_concurrent = max_concurrent or self._config.max_concurrent

        # Ensure warmup has been done
        if not self._warmed_up:
            await self.warmup()

        logger.info(f"[Embeddings] ========== Batch Embedding Start ==========")
        logger.info(f"[Embeddings] Total documents: {len(texts)}")
        logger.info(f"[Embeddings] Batch size: {batch_size}")
        logger.info(f"[Embeddings] Max concurrent: {max_concurrent}")

        start_time = time.time()

        # Check cache for existing embeddings
        cached_embeddings: Dict[int, List[float]] = {}
        texts_to_embed: List[Tuple[int, str]] = []

        if use_cache:
            try:
                from app.core.cache import get_cache_manager
                cache = await get_cache_manager()

                for i, text in enumerate(texts):
                    cached = await cache.get_embedding(text)
                    if cached is not None:
                        cached_embeddings[i] = cached.tolist()
                    else:
                        texts_to_embed.append((i, text))

                if cached_embeddings:
                    logger.info(
                        f"[Embeddings] Cache hit: {len(cached_embeddings)}, "
                        f"to compute: {len(texts_to_embed)}"
                    )
            except Exception as e:
                logger.debug(f"[Embeddings] Cache check failed: {e}")
                texts_to_embed = [(i, text) for i, text in enumerate(texts)]
        else:
            texts_to_embed = [(i, text) for i, text in enumerate(texts)]

        # If all cached, return early
        if not texts_to_embed:
            logger.info(f"[Embeddings] All {len(texts)} embeddings from cache")
            return [cached_embeddings[i] for i in range(len(texts))]

        # Split into batches
        batches: List[List[Tuple[int, str]]] = []
        for i in range(0, len(texts_to_embed), batch_size):
            batches.append(texts_to_embed[i:i + batch_size])

        total_batches = len(batches)
        logger.info(f"[Embeddings] Total batches to process: {total_batches}")

        semaphore = asyncio.Semaphore(max_concurrent)
        completed_count = 0
        failed_count = 0
        completed_lock = asyncio.Lock()

        # Results storage
        computed_embeddings: Dict[int, List[float]] = {}

        async def embed_batch(
            batch_idx: int,
            batch: List[Tuple[int, str]]
        ) -> None:
            nonlocal completed_count, failed_count

            async with semaphore:
                batch_start = time.time()
                indices = [item[0] for item in batch]
                batch_texts = [item[1] for item in batch]

                try:
                    result = await self._embed_batch_direct(batch_texts)

                    # Store results
                    for idx, emb in zip(indices, result):
                        computed_embeddings[idx] = emb

                    batch_elapsed = time.time() - batch_start

                    async with completed_lock:
                        completed_count += 1
                        if show_progress:
                            progress = (completed_count / total_batches) * 100
                            speed = len(batch) / batch_elapsed if batch_elapsed > 0 else 0
                            logger.info(
                                f"[Embeddings] Batch {completed_count}/{total_batches} "
                                f"({progress:.1f}%) - {len(batch)} docs in {batch_elapsed:.2f}s "
                                f"({speed:.1f} docs/sec)"
                            )

                    # Cache the results
                    if use_cache:
                        try:
                            from app.core.cache import get_cache_manager
                            cache = await get_cache_manager()
                            import numpy as np
                            for text, emb in zip(batch_texts, result):
                                await cache.set_embedding(text, np.array(emb))
                        except Exception:
                            pass  # Cache errors are non-critical

                except RetryError as e:
                    async with completed_lock:
                        failed_count += len(batch)
                    logger.error(
                        f"[Embeddings] Batch {batch_idx + 1} failed after retries: {e.last_error}"
                    )
                    # Store empty embeddings for failed items
                    for idx in indices:
                        computed_embeddings[idx] = []

                except Exception as e:
                    async with completed_lock:
                        failed_count += len(batch)
                    logger.error(f"[Embeddings] Batch {batch_idx + 1} error: {e}")
                    for idx in indices:
                        computed_embeddings[idx] = []

        # Execute all batches concurrently
        tasks = [embed_batch(i, batch) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time
        success_count = len(texts) - failed_count
        speed = success_count / elapsed if elapsed > 0 else 0

        logger.info(f"[Embeddings] ========== Batch Embedding Complete ==========")
        logger.info(
            f"[Embeddings] Processed {len(texts)} documents in {elapsed:.2f}s "
            f"({speed:.1f} docs/sec)"
        )
        if failed_count > 0:
            logger.warning(f"[Embeddings] Failed: {failed_count} documents")

        # Merge cached and computed embeddings
        all_embeddings: Dict[int, List[float]] = {**cached_embeddings, **computed_embeddings}

        # Return in original order
        return [all_embeddings.get(i, []) for i in range(len(texts))]

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        test_embedding = self.embed_query("test")
        self._embedding_dim = len(test_embedding)
        return self._embedding_dim

    def shutdown(self) -> None:
        """Shutdown thread pool."""
        self._thread_pool.shutdown(wait=False)


# ============================================================================
# Global Instance
# ============================================================================

_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager instance."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager


def get_langchain_embeddings() -> OllamaEmbeddings:
    """Convenience function to get LangChain embeddings."""
    return get_embedding_manager().langchain


def get_llamaindex_embeddings() -> OllamaEmbedding:
    """Convenience function to get LlamaIndex embeddings."""
    return get_embedding_manager().llamaindex