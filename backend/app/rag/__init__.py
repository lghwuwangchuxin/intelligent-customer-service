"""
Enhanced RAG Module with LlamaIndex.
Provides advanced retrieval capabilities including:
- Query transformation (HyDE, Query Expansion)
- Hybrid retrieval (Vector + BM25)
- Jina Reranker
- Post-processing (Dedup, MMR)
"""

# Lazy imports to avoid circular dependencies and allow graceful degradation
def __getattr__(name):
    """Lazy import for RAG components."""
    if name == "EnhancedRAGQueryEngine":
        from app.rag.query_engine import EnhancedRAGQueryEngine
        return EnhancedRAGQueryEngine
    elif name == "QueryTransformer":
        from app.rag.query_transform import QueryTransformer
        return QueryTransformer
    elif name == "HybridRetriever":
        from app.rag.hybrid_retriever import HybridRetriever
        return HybridRetriever
    elif name == "JinaReranker":
        from app.rag.reranker import JinaReranker
        return JinaReranker
    elif name == "RAGPostProcessor":
        from app.rag.postprocessor import RAGPostProcessor
        return RAGPostProcessor
    elif name == "IndexManager":
        from app.rag.index_manager import IndexManager
        return IndexManager
    elif name == "get_query_engine":
        from app.rag.query_engine import get_query_engine
        return get_query_engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnhancedRAGQueryEngine",
    "QueryTransformer",
    "HybridRetriever",
    "JinaReranker",
    "RAGPostProcessor",
    "IndexManager",
    "get_query_engine",
]
