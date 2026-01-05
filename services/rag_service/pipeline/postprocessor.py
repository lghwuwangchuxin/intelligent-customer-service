"""Post-processing module for retrieval results."""

from typing import List, Optional, Set
from dataclasses import dataclass

from services.common.logging import get_logger
from .hybrid_retriever import RetrievedDocument

logger = get_logger(__name__)


@dataclass
class PostProcessResult:
    """Result of post-processing operation."""
    documents: List[RetrievedDocument]
    removed_duplicates: int
    latency_ms: int = 0


class PostProcessor:
    """
    Post-processor for retrieval results.

    Applies:
    - Semantic deduplication
    - Maximum Marginal Relevance (MMR) for diversity
    - Content filtering
    - Metadata enrichment
    """

    def __init__(
        self,
        enable_dedup: bool = True,
        dedup_threshold: float = 0.95,
        enable_mmr: bool = True,
        mmr_lambda: float = 0.5,
        min_content_length: int = 50,
        max_content_length: int = 5000,
        embedding_model=None,
    ):
        """
        Initialize post-processor.

        Args:
            enable_dedup: Enable semantic deduplication
            dedup_threshold: Similarity threshold for deduplication
            enable_mmr: Enable MMR for diversity
            mmr_lambda: MMR lambda parameter (0=diversity, 1=relevance)
            min_content_length: Minimum content length to keep
            max_content_length: Maximum content length to keep
            embedding_model: Embedding model for similarity calculation
        """
        self.enable_dedup = enable_dedup
        self.dedup_threshold = dedup_threshold
        self.enable_mmr = enable_mmr
        self.mmr_lambda = mmr_lambda
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.embedding_model = embedding_model

    async def process(
        self,
        documents: List[RetrievedDocument],
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> PostProcessResult:
        """
        Post-process retrieval results.

        Args:
            documents: Documents to process
            query: Original query (for MMR)
            top_k: Maximum documents to return

        Returns:
            PostProcessResult with processed documents
        """
        import time
        start_time = time.time()

        if not documents:
            return PostProcessResult(
                documents=[],
                removed_duplicates=0,
                latency_ms=0,
            )

        original_count = len(documents)

        # Filter by content length
        documents = self._filter_by_length(documents)

        # Remove duplicates
        removed_duplicates = 0
        if self.enable_dedup:
            documents, removed_duplicates = await self._deduplicate(documents)

        # Apply MMR for diversity
        if self.enable_mmr and query and self.embedding_model:
            documents = await self._apply_mmr(documents, query, top_k)
        elif top_k:
            documents = documents[:top_k]

        latency_ms = int((time.time() - start_time) * 1000)

        return PostProcessResult(
            documents=documents,
            removed_duplicates=removed_duplicates,
            latency_ms=latency_ms,
        )

    def _filter_by_length(
        self,
        documents: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """Filter documents by content length."""
        filtered = []
        for doc in documents:
            content_len = len(doc.content)
            if self.min_content_length <= content_len <= self.max_content_length:
                filtered.append(doc)
            else:
                logger.debug(
                    f"Filtered document {doc.id}: length {content_len} "
                    f"outside range [{self.min_content_length}, {self.max_content_length}]"
                )
        return filtered

    async def _deduplicate(
        self,
        documents: List[RetrievedDocument],
    ) -> tuple[List[RetrievedDocument], int]:
        """
        Remove semantically duplicate documents.

        Uses either embedding similarity or text overlap.
        """
        if len(documents) <= 1:
            return documents, 0

        # Use embedding-based deduplication if model available
        if self.embedding_model:
            return await self._embedding_dedup(documents)
        else:
            return self._text_overlap_dedup(documents)

    async def _embedding_dedup(
        self,
        documents: List[RetrievedDocument],
    ) -> tuple[List[RetrievedDocument], int]:
        """Deduplicate using embedding similarity."""
        try:
            # Get embeddings for all documents
            contents = [doc.content for doc in documents]
            embeddings = await self.embedding_model.embed_batch(contents)

            # Find duplicates
            unique_docs = []
            unique_embeddings = []
            removed = 0

            for doc, emb in zip(documents, embeddings):
                is_duplicate = False
                for unique_emb in unique_embeddings:
                    similarity = self._cosine_similarity(emb, unique_emb)
                    if similarity > self.dedup_threshold:
                        is_duplicate = True
                        removed += 1
                        break

                if not is_duplicate:
                    unique_docs.append(doc)
                    unique_embeddings.append(emb)

            return unique_docs, removed

        except Exception as e:
            logger.warning(f"Embedding dedup failed, falling back to text overlap: {e}")
            return self._text_overlap_dedup(documents)

    def _text_overlap_dedup(
        self,
        documents: List[RetrievedDocument],
    ) -> tuple[List[RetrievedDocument], int]:
        """Deduplicate using text overlap (Jaccard similarity)."""
        unique_docs = []
        seen_shingles: List[Set[str]] = []
        removed = 0

        for doc in documents:
            # Create shingles (n-grams)
            shingles = self._create_shingles(doc.content, n=3)

            # Check for duplicates
            is_duplicate = False
            for seen in seen_shingles:
                similarity = self._jaccard_similarity(shingles, seen)
                if similarity > self.dedup_threshold:
                    is_duplicate = True
                    removed += 1
                    break

            if not is_duplicate:
                unique_docs.append(doc)
                seen_shingles.append(shingles)

        return unique_docs, removed

    async def _apply_mmr(
        self,
        documents: List[RetrievedDocument],
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """
        Apply Maximum Marginal Relevance for diversity.

        MMR = λ * sim(d, q) - (1 - λ) * max(sim(d, d'))

        Balances relevance to query and diversity from selected documents.
        """
        if len(documents) <= 1:
            return documents

        top_k = top_k or len(documents)

        try:
            # Get embeddings
            contents = [doc.content for doc in documents]
            doc_embeddings = await self.embedding_model.embed_batch(contents)
            query_embedding = await self.embedding_model.embed(query)

            # Calculate query similarities
            query_sims = [
                self._cosine_similarity(query_embedding, doc_emb)
                for doc_emb in doc_embeddings
            ]

            # Select documents using MMR
            selected_indices = []
            remaining_indices = list(range(len(documents)))

            while len(selected_indices) < top_k and remaining_indices:
                best_idx = None
                best_score = float('-inf')

                for idx in remaining_indices:
                    # Relevance to query
                    relevance = query_sims[idx]

                    # Maximum similarity to already selected
                    max_sim = 0
                    for sel_idx in selected_indices:
                        sim = self._cosine_similarity(
                            doc_embeddings[idx],
                            doc_embeddings[sel_idx]
                        )
                        max_sim = max(max_sim, sim)

                    # MMR score
                    mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)

            return [documents[i] for i in selected_indices]

        except Exception as e:
            logger.warning(f"MMR failed, returning top documents: {e}")
            return documents[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def _create_shingles(text: str, n: int = 3) -> Set[str]:
        """Create n-gram shingles from text."""
        words = text.lower().split()
        if len(words) < n:
            return {text.lower()}
        return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}

    @staticmethod
    def _jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


class ContentEnricher:
    """
    Enrich document content and metadata.

    Can add:
    - Title extraction
    - Summary generation
    - Entity extraction
    - Metadata normalization
    """

    def __init__(self, llm_client=None):
        """
        Initialize content enricher.

        Args:
            llm_client: LLM client for content enrichment
        """
        self.llm_client = llm_client

    async def enrich(
        self,
        documents: List[RetrievedDocument],
        add_summary: bool = False,
        add_title: bool = False,
    ) -> List[RetrievedDocument]:
        """
        Enrich document content and metadata.

        Args:
            documents: Documents to enrich
            add_summary: Generate summaries
            add_title: Extract titles

        Returns:
            Enriched documents
        """
        if not documents:
            return documents

        enriched = []
        for doc in documents:
            new_metadata = dict(doc.metadata)

            if add_title and 'title' not in new_metadata:
                title = self._extract_title(doc.content)
                new_metadata['title'] = title

            if add_summary and 'summary' not in new_metadata and self.llm_client:
                summary = await self._generate_summary(doc.content)
                new_metadata['summary'] = summary

            enriched.append(RetrievedDocument(
                id=doc.id,
                content=doc.content,
                score=doc.score,
                metadata=new_metadata,
                source=doc.source,
            ))

        return enriched

    def _extract_title(self, content: str) -> str:
        """Extract title from content."""
        # Simple heuristic: first line or first sentence
        lines = content.strip().split('\n')
        first_line = lines[0].strip() if lines else ""

        # If first line looks like a title (short, no punctuation at end)
        if first_line and len(first_line) < 100 and not first_line.endswith(('.', '?', '!')):
            return first_line

        # Otherwise, extract first sentence
        sentences = content.split('.')
        return sentences[0].strip()[:100] if sentences else content[:50]

    async def _generate_summary(self, content: str) -> str:
        """Generate summary using LLM."""
        if not self.llm_client:
            return content[:200]

        prompt = f"""Summarize the following text in 1-2 sentences:

{content[:1000]}

Summary:"""

        try:
            summary = await self.llm_client.generate(prompt, max_tokens=100)
            return summary.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return content[:200]
