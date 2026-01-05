"""Embedding service using Ollama or other providers."""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional

import httpx

from services.common.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Embedding service for text vectorization.

    Supports:
    - Ollama embeddings
    - OpenAI embeddings
    - Custom embedding APIs
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        batch_size: int = 50,
        dimension: int = 768,
        timeout: float = 120.0,
    ):
        """
        Initialize embedding service.

        Args:
            model: Embedding model name
            base_url: API base URL
            api_key: API key for authentication
            batch_size: Batch size for embedding
            dimension: Embedding dimension
            timeout: Request timeout
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.dimension = dimension
        self.timeout = timeout

        # Cache for embeddings
        self._cache: Dict[str, List[float]] = {}
        self._cache_max_size = 1000

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate embedding
        embeddings = await self.embed_batch([text])
        if embeddings:
            # Cache result
            if len(self._cache) < self._cache_max_size:
                self._cache[cache_key] = embeddings[0]
            return embeddings[0]

        # Return zero vector on failure
        return [0.0] * self.dimension

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check if using Ollama
        if "11434" in self.base_url or "ollama" in self.base_url.lower():
            return await self._embed_ollama(texts)
        else:
            return await self._embed_openai(texts)

    async def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        embeddings = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for text in texts:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={
                            "model": self.model,
                            "prompt": text,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings.append(data.get("embedding", [0.0] * self.dimension))

                except Exception as e:
                    logger.error(f"Ollama embedding error: {e}")
                    embeddings.append([0.0] * self.dimension)

        return embeddings

    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI-compatible API."""
        embeddings = []

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]

                try:
                    response = await client.post(
                        f"{self.base_url}/embeddings",
                        headers=headers,
                        json={
                            "model": self.model,
                            "input": batch,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    for item in data.get("data", []):
                        embeddings.append(item.get("embedding", [0.0] * self.dimension))

                except Exception as e:
                    logger.error(f"OpenAI embedding error: {e}")
                    embeddings.extend([[0.0] * self.dimension] * len(batch))

        return embeddings

    async def warmup(self):
        """Warmup embedding model."""
        try:
            await self.embed("warmup test")
            logger.info(f"Embedding model warmed up: {self.model}")
        except Exception as e:
            logger.warning(f"Embedding warmup failed: {e}")

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()

    def get_info(self) -> Dict[str, Any]:
        """Get embedding service info."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "cache_size": len(self._cache),
        }
