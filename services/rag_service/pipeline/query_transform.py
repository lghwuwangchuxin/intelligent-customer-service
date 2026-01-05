"""Query transformation module for RAG pipeline."""

import asyncio
from typing import List, Optional
from dataclasses import dataclass

from services.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TransformedQuery:
    """Transformed query result."""
    original_query: str
    transformed_query: str
    expanded_queries: List[str]
    hyde_passage: Optional[str] = None


class QueryTransformer:
    """
    Query transformer with HyDE and query expansion.

    Techniques:
    - HyDE (Hypothetical Document Embeddings): Generate hypothetical answer
    - Query Expansion: Generate related queries
    - Query Rewriting: Improve query clarity
    """

    def __init__(
        self,
        llm_client=None,
        enable_hyde: bool = True,
        enable_expansion: bool = True,
        expansion_count: int = 3,
    ):
        """
        Initialize query transformer.

        Args:
            llm_client: LLM client for query transformation
            enable_hyde: Enable HyDE transformation
            enable_expansion: Enable query expansion
            expansion_count: Number of expanded queries to generate
        """
        self.llm_client = llm_client
        self.enable_hyde = enable_hyde
        self.enable_expansion = enable_expansion
        self.expansion_count = expansion_count

    async def transform(self, query: str) -> TransformedQuery:
        """
        Transform query for better retrieval.

        Args:
            query: Original user query

        Returns:
            TransformedQuery with transformations applied
        """
        result = TransformedQuery(
            original_query=query,
            transformed_query=query,
            expanded_queries=[],
        )

        if not self.llm_client:
            logger.debug("No LLM client, skipping query transformation")
            return result

        tasks = []

        # HyDE transformation
        if self.enable_hyde:
            tasks.append(self._generate_hyde_passage(query))

        # Query expansion
        if self.enable_expansion:
            tasks.append(self._expand_query(query))

        # Query rewriting
        tasks.append(self._rewrite_query(query))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            idx = 0
            if self.enable_hyde:
                if not isinstance(results[idx], Exception):
                    result.hyde_passage = results[idx]
                idx += 1

            if self.enable_expansion:
                if not isinstance(results[idx], Exception):
                    result.expanded_queries = results[idx]
                idx += 1

            if not isinstance(results[idx], Exception):
                result.transformed_query = results[idx]

        except Exception as e:
            logger.error(f"Query transformation failed: {e}")

        return result

    async def _generate_hyde_passage(self, query: str) -> Optional[str]:
        """
        Generate hypothetical document passage using HyDE.

        HyDE improves retrieval by:
        1. Generate a hypothetical answer to the query
        2. Use this answer for embedding-based retrieval
        3. Real documents similar to the hypothetical answer are retrieved
        """
        prompt = f"""Based on the following question, write a short paragraph (2-3 sentences)
that would be a good answer. Write as if you are answering from a knowledge base document.

Question: {query}

Answer paragraph:"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=200)
            return response.strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return None

    async def _expand_query(self, query: str) -> List[str]:
        """
        Expand query into related queries.

        Query expansion improves recall by:
        1. Generate semantically related queries
        2. Use multiple queries for retrieval
        3. Combine results from all queries
        """
        prompt = f"""Given the following search query, generate {self.expansion_count} related
but different search queries that would help find relevant information.
Return only the queries, one per line.

Original query: {query}

Related queries:"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=200)
            queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
            return queries[:self.expansion_count]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []

    async def _rewrite_query(self, query: str) -> str:
        """
        Rewrite query for better clarity and retrieval.

        Query rewriting improves precision by:
        1. Remove ambiguity
        2. Add context
        3. Improve structure
        """
        prompt = f"""Rewrite the following search query to be clearer and more specific
for searching a knowledge base. Keep the same meaning but improve clarity.
Return only the rewritten query.

Original query: {query}

Rewritten query:"""

        try:
            response = await self.llm_client.generate(prompt, max_tokens=100)
            rewritten = response.strip()
            # Fallback to original if response is empty or too different
            if not rewritten or len(rewritten) > len(query) * 3:
                return query
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return query


class SimpleLLMClient:
    """Simple LLM client for query transformation."""

    def __init__(self, base_url: str, model: str = "qwen2.5:7b"):
        """
        Initialize LLM client.

        Args:
            base_url: LLM API base URL
            model: Model name
        """
        self.base_url = base_url.rstrip('/')
        self.model = model

    async def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text from prompt."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.3,
                    }
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")
