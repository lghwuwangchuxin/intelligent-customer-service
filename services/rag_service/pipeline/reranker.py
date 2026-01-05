"""Reranker module for improving retrieval quality."""

import asyncio
from typing import List, Optional, Tuple
from dataclasses import dataclass

from services.common.logging import get_logger
from .hybrid_retriever import RetrievedDocument

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Result of reranking operation."""
    documents: List[RetrievedDocument]
    original_count: int
    reranked_count: int
    latency_ms: int = 0


class Reranker:
    """
    Document reranker using cross-encoder models.

    Reranking improves precision by:
    1. Taking top-K candidates from retrieval
    2. Using a cross-encoder to score query-document pairs
    3. Re-sorting by cross-encoder scores
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        top_k: int = 5,
        batch_size: int = 32,
        device: str = "cpu",
        use_api: bool = False,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model name
            top_k: Number of documents to return after reranking
            batch_size: Batch size for inference
            device: Device for model inference
            use_api: Whether to use API instead of local model
            api_key: API key for reranker service
            api_url: API URL for reranker service
        """
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self.device = device
        self.use_api = use_api
        self.api_key = api_key
        self.api_url = api_url or "https://api.jina.ai/v1/rerank"

        self._model = None
        self._initialized = False

    async def initialize(self):
        """Initialize the reranker model."""
        if self._initialized:
            return

        if self.use_api:
            self._initialized = True
            return

        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
            self._initialized = True
            logger.info(f"Reranker model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    def _load_model(self):
        """Load cross-encoder model (blocking)."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )
        except ImportError:
            logger.warning("sentence-transformers not installed, using mock reranker")
            self._model = None

    async def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Override default top_k

        Returns:
            RerankResult with reranked documents
        """
        import time
        start_time = time.time()

        if not documents:
            return RerankResult(
                documents=[],
                original_count=0,
                reranked_count=0,
                latency_ms=0,
            )

        top_k = top_k or self.top_k
        original_count = len(documents)

        # Initialize model if needed
        if not self._initialized:
            await self.initialize()

        # Score documents
        if self.use_api:
            scored_docs = await self._rerank_with_api(query, documents)
        elif self._model:
            scored_docs = await self._rerank_with_model(query, documents)
        else:
            # Fallback: keep original order
            scored_docs = [(doc, doc.score) for doc in documents]

        # Sort by reranker score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Build result
        reranked = []
        for doc, score in scored_docs[:top_k]:
            reranked.append(RetrievedDocument(
                id=doc.id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
                source=doc.source,
            ))

        latency_ms = int((time.time() - start_time) * 1000)

        return RerankResult(
            documents=reranked,
            original_count=original_count,
            reranked_count=len(reranked),
            latency_ms=latency_ms,
        )

    async def _rerank_with_model(
        self,
        query: str,
        documents: List[RetrievedDocument],
    ) -> List[Tuple[RetrievedDocument, float]]:
        """Rerank using local cross-encoder model."""
        # Prepare pairs
        pairs = [(query, doc.content) for doc in documents]

        # Score in batches
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, batch_size=self.batch_size)
        )

        return list(zip(documents, scores))

    async def _rerank_with_api(
        self,
        query: str,
        documents: List[RetrievedDocument],
    ) -> List[Tuple[RetrievedDocument, float]]:
        """Rerank using Jina reranker API."""
        import httpx

        if not self.api_key:
            logger.warning("No API key for reranker, returning original scores")
            return [(doc, doc.score) for doc in documents]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model_name,
                        "query": query,
                        "documents": [doc.content for doc in documents],
                        "top_n": len(documents),
                    },
                )
                response.raise_for_status()
                result = response.json()

                # Map scores back to documents
                scored = []
                for item in result.get("results", []):
                    idx = item["index"]
                    score = item["relevance_score"]
                    scored.append((documents[idx], score))

                return scored

        except Exception as e:
            logger.error(f"Reranker API error: {e}")
            return [(doc, doc.score) for doc in documents]


class LLMReranker:
    """
    LLM-based reranker for cases where cross-encoder is unavailable.

    Uses LLM to score relevance of documents to query.
    """

    def __init__(self, llm_client=None):
        """
        Initialize LLM reranker.

        Args:
            llm_client: LLM client for scoring
        """
        self.llm_client = llm_client

    async def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = 5,
    ) -> RerankResult:
        """
        Rerank documents using LLM.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of documents to return

        Returns:
            RerankResult with reranked documents
        """
        import time
        start_time = time.time()

        if not documents or not self.llm_client:
            return RerankResult(
                documents=documents[:top_k] if documents else [],
                original_count=len(documents) if documents else 0,
                reranked_count=min(len(documents), top_k) if documents else 0,
                latency_ms=0,
            )

        # Score each document
        tasks = [
            self._score_document(query, doc)
            for doc in documents
        ]
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        # Pair documents with scores
        scored_docs = []
        for doc, score in zip(documents, scores):
            if isinstance(score, Exception):
                scored_docs.append((doc, doc.score))
            else:
                scored_docs.append((doc, score))

        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Build result
        reranked = []
        for doc, score in scored_docs[:top_k]:
            reranked.append(RetrievedDocument(
                id=doc.id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
                source=doc.source,
            ))

        latency_ms = int((time.time() - start_time) * 1000)

        return RerankResult(
            documents=reranked,
            original_count=len(documents),
            reranked_count=len(reranked),
            latency_ms=latency_ms,
        )

    async def _score_document(self, query: str, doc: RetrievedDocument) -> float:
        """Score a single document's relevance to query."""
        prompt = f"""Rate the relevance of the following document to the query on a scale of 0 to 10.
Only respond with a number.

Query: {query}

Document: {doc.content[:500]}

Relevance score (0-10):"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=5)
            score = float(response.strip())
            return min(max(score, 0), 10) / 10.0  # Normalize to 0-1
        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")
            return doc.score
