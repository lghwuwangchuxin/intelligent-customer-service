"""
Query Transformation Module.
Implements HyDE (Hypothetical Document Embeddings) and Query Expansion.
"""
import logging
from typing import List, Optional

from llama_index.core import Settings as LlamaSettings
from llama_index.core.schema import QueryBundle

from config.settings import settings

logger = logging.getLogger(__name__)

# HyDE prompt template for Chinese customer service domain
HYDE_PROMPT_TEMPLATE = """你是一个专业的客服知识库文档作者。根据用户的问题，请撰写一段可能出现在知识库中的文档内容，用于回答该问题。

用户问题: {query}

请直接写出文档内容，不要包含任何解释或前缀。文档内容:"""

# Query expansion prompt template
QUERY_EXPANSION_TEMPLATE = """你是一个查询优化专家。请将以下用户问题改写成{num_queries}个不同角度的查询，以便更全面地检索相关文档。

原始问题: {query}

请用以下格式输出，每行一个查询:
1. [查询1]
2. [查询2]
...

改写后的查询:"""


class QueryTransformer:
    """
    Query transformer that implements HyDE and Query Expansion.
    """

    def __init__(
        self,
        enable_hyde: bool = None,
        enable_expansion: bool = None,
        expansion_num: int = None,
    ):
        self.enable_hyde = enable_hyde if enable_hyde is not None else settings.RAG_ENABLE_HYDE
        self.enable_expansion = enable_expansion if enable_expansion is not None else settings.RAG_ENABLE_QUERY_EXPANSION
        self.expansion_num = expansion_num or settings.RAG_QUERY_EXPANSION_NUM

        self._llm = LlamaSettings.llm

        logger.info(
            f"[QueryTransform] Initialized - HyDE: {self.enable_hyde}, "
            f"Expansion: {self.enable_expansion}, Expansion num: {self.expansion_num}"
        )

    async def transform(self, query: str) -> List[str]:
        """
        Transform a query using enabled strategies.

        Args:
            query: Original user query

        Returns:
            List of transformed queries
        """
        logger.info(f"[QueryTransform] Transforming query: {query[:50]}...")

        queries = [query]  # Always include original

        # Apply HyDE
        if self.enable_hyde:
            try:
                hyde_doc = await self._generate_hyde_document(query)
                if hyde_doc:
                    queries.append(hyde_doc)
                    logger.info(f"[QueryTransform] Generated HyDE document: {hyde_doc[:50]}...")
            except Exception as e:
                logger.warning(f"[QueryTransform] HyDE generation failed: {e}")

        # Apply Query Expansion
        if self.enable_expansion:
            try:
                expanded = await self._expand_query(query)
                if expanded:
                    queries.extend(expanded)
                    logger.info(f"[QueryTransform] Generated {len(expanded)} expanded queries")
            except Exception as e:
                logger.warning(f"[QueryTransform] Query expansion failed: {e}")

        logger.info(f"[QueryTransform] Total queries after transformation: {len(queries)}")
        return queries

    def transform_sync(self, query: str) -> List[str]:
        """
        Synchronous version of transform.
        """
        logger.info(f"[QueryTransform] Sync transforming query: {query[:50]}...")

        queries = [query]

        if self.enable_hyde:
            try:
                hyde_doc = self._generate_hyde_document_sync(query)
                if hyde_doc:
                    queries.append(hyde_doc)
            except Exception as e:
                logger.warning(f"[QueryTransform] HyDE generation failed: {e}")

        if self.enable_expansion:
            try:
                expanded = self._expand_query_sync(query)
                if expanded:
                    queries.extend(expanded)
            except Exception as e:
                logger.warning(f"[QueryTransform] Query expansion failed: {e}")

        return queries

    async def _generate_hyde_document(self, query: str) -> Optional[str]:
        """Generate a hypothetical document using HyDE."""
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
        response = await self._llm.acomplete(prompt)
        return response.text.strip() if response else None

    def _generate_hyde_document_sync(self, query: str) -> Optional[str]:
        """Synchronous HyDE document generation."""
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
        response = self._llm.complete(prompt)
        return response.text.strip() if response else None

    async def _expand_query(self, query: str) -> List[str]:
        """Expand query into multiple variations."""
        prompt = QUERY_EXPANSION_TEMPLATE.format(
            query=query,
            num_queries=self.expansion_num
        )
        response = await self._llm.acomplete(prompt)

        if not response:
            return []

        # Parse the numbered list
        return self._parse_expanded_queries(response.text)

    def _expand_query_sync(self, query: str) -> List[str]:
        """Synchronous query expansion."""
        prompt = QUERY_EXPANSION_TEMPLATE.format(
            query=query,
            num_queries=self.expansion_num
        )
        response = self._llm.complete(prompt)

        if not response:
            return []

        return self._parse_expanded_queries(response.text)

    def _parse_expanded_queries(self, text: str) -> List[str]:
        """Parse expanded queries from LLM response."""
        queries = []
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering patterns like "1.", "1)", "[1]", etc.
            import re
            cleaned = re.sub(r'^[\d]+[\.\)\]]\s*', '', line)
            cleaned = re.sub(r'^\[[\d]+\]\s*', '', cleaned)

            if cleaned and len(cleaned) > 3:  # Minimum length check
                queries.append(cleaned)

        return queries[:self.expansion_num]

    def create_query_bundles(self, queries: List[str]) -> List[QueryBundle]:
        """
        Create QueryBundle objects from a list of query strings.

        Args:
            queries: List of query strings

        Returns:
            List of QueryBundle objects
        """
        return [QueryBundle(query_str=q) for q in queries]


# Global singleton
_query_transformer: Optional[QueryTransformer] = None


def get_query_transformer() -> QueryTransformer:
    """Get the global query transformer instance."""
    global _query_transformer
    if _query_transformer is None:
        _query_transformer = QueryTransformer()
    return _query_transformer
