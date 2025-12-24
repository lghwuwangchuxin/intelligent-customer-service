"""
Knowledge Base Tools - MCP tools for knowledge base operations.

Query Flow:
1. Vector search (semantic search in Milvus)
2. Fallback to knowledge base file search (text search)
3. If no results, Agent will use other tools (web search, etc.)

性能优化
--------
- Redis 缓存（带内存回退）
- 缓存命中时直接返回，避免重复搜索
- 缓存统计和监控

Version: 1.2.0 (Redis 缓存集成)
"""
import logging
import asyncio
import re
import time
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from collections import OrderedDict

import jieba

from app.mcp.tools.base import BaseMCPTool, ToolParameter
from config.settings import settings

logger = logging.getLogger(__name__)

# Relevance score threshold for fallback
RELEVANCE_THRESHOLD = 0.3

# 缓存配置
CACHE_MAX_SIZE = 100  # 最大缓存条目数
CACHE_TTL = 300  # 缓存过期时间（秒）


class SearchResultCache:
    """
    Redis-backed search result cache with memory fallback.

    使用统一的 CacheManager 进行缓存管理。
    Redis 不可用时自动回退到内存缓存。

    Attributes
    ----------
    max_size : int
        最大缓存条目数（内存回退时）

    ttl : float
        缓存过期时间（秒）

    Example
    -------
    ```python
    cache = SearchResultCache()
    await cache.initialize()

    # 设置缓存
    await cache.aset("如何退款", 5, [{"content": "..."}])

    # 获取缓存
    results = await cache.aget("如何退款", 5)
    if results is not None:
        print("缓存命中!")
    ```
    """

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: float = CACHE_TTL):
        self.max_size = max_size
        self.ttl = ttl
        self._cache_manager = None
        self._initialized = False
        # Fallback memory cache
        self._memory_cache: OrderedDict[str, Tuple[List[dict], float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    async def initialize(self) -> None:
        """Initialize the cache manager."""
        if self._initialized:
            return
        try:
            from app.core.cache import get_cache_manager
            self._cache_manager = await get_cache_manager()
            self._initialized = True
            logger.info(f"[SearchCache] Initialized with {self._cache_manager.backend.value} backend")
        except Exception as e:
            logger.warning(f"[SearchCache] Failed to initialize cache manager: {e}")
            self._initialized = False

    def _hash_key(self, query: str, top_k: int) -> str:
        """生成缓存键。"""
        content = f"{query.strip().lower()}:{top_k}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, top_k: int = 5) -> Optional[List[dict]]:
        """获取缓存结果（同步，使用内存缓存）。"""
        key = self._hash_key(query, top_k)

        if key not in self._memory_cache:
            self._misses += 1
            return None

        results, timestamp = self._memory_cache[key]

        # 检查是否过期
        if time.time() - timestamp > self.ttl:
            del self._memory_cache[key]
            self._misses += 1
            return None

        # LRU: 移动到末尾
        self._memory_cache.move_to_end(key)
        self._hits += 1
        return results

    async def aget(self, query: str, top_k: int = 5) -> Optional[List[dict]]:
        """获取缓存结果（异步，优先使用 Redis）。"""
        if not self._initialized:
            await self.initialize()

        if self._cache_manager:
            try:
                results = await self._cache_manager.get_search_results(query, top_k)
                if results is not None:
                    self._hits += 1
                    return results
                self._misses += 1
                return None
            except Exception as e:
                logger.warning(f"[SearchCache] Redis get error: {e}")

        return self.get(query, top_k)

    def set(self, query: str, top_k: int, results: List[dict]) -> None:
        """设置缓存结果（同步，使用内存缓存）。"""
        key = self._hash_key(query, top_k)

        if key in self._memory_cache:
            del self._memory_cache[key]

        while len(self._memory_cache) >= self.max_size:
            self._memory_cache.popitem(last=False)

        self._memory_cache[key] = (results, time.time())

    async def aset(self, query: str, top_k: int, results: List[dict]) -> None:
        """设置缓存结果（异步，同时写入 Redis 和内存）。"""
        if not self._initialized:
            await self.initialize()

        # Always set in memory for sync access
        self.set(query, top_k, results)

        # Also set in Redis if available
        if self._cache_manager:
            try:
                await self._cache_manager.set_search_results(query, results, top_k)
            except Exception as e:
                logger.warning(f"[SearchCache] Redis set error: {e}")

    def clear(self) -> None:
        """清空内存缓存。"""
        self._memory_cache.clear()
        self._hits = 0
        self._misses = 0

    async def aclear(self) -> None:
        """清空所有缓存（异步）。"""
        self.clear()
        if self._cache_manager:
            try:
                await self._cache_manager.clear_search()
            except Exception as e:
                logger.warning(f"[SearchCache] Redis clear error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息。"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        stats = {
            "memory_size": len(self._memory_cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
        }
        if self._cache_manager:
            stats["backend"] = self._cache_manager.backend.value
        return stats


# 全局缓存实例
_search_cache: Optional[SearchResultCache] = None


def get_search_cache() -> SearchResultCache:
    """获取全局搜索缓存实例。"""
    global _search_cache
    if _search_cache is None:
        _search_cache = SearchResultCache()
    return _search_cache


async def get_search_cache_async() -> SearchResultCache:
    """获取已初始化的全局搜索缓存实例（异步）。"""
    cache = get_search_cache()
    if not cache._initialized:
        await cache.initialize()
    return cache


class KnowledgeSearchTool(BaseMCPTool):
    """
    Search the knowledge base for relevant information.

    Query Flow:
    1. Vector search (semantic search via RAG service or vector store)
    2. If no high-confidence results, fallback to text search in knowledge base files
    3. Returns combined results sorted by relevance
    """

    name = "knowledge_search"
    description = (
        "Search the knowledge base for relevant documents and information. "
        "Use this tool when you need to find specific information from the "
        "company's documentation, FAQs, or other knowledge sources. "
        "This tool uses vector similarity search with text search fallback."
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query to find relevant information",
            required=True,
        ),
        ToolParameter(
            name="top_k",
            type="integer",
            description="Number of results to return (1-10)",
            required=False,
            default=5,
        ),
    ]

    def __init__(self, rag_service=None, vector_store=None):
        super().__init__()
        self.rag_service = rag_service
        self.vector_store = vector_store
        self._kb_path = Path(settings.KNOWLEDGE_BASE_PATH)
        # Make it absolute if relative
        if not self._kb_path.is_absolute():
            self._kb_path = Path(__file__).parent.parent.parent.parent / self._kb_path

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True,
    ) -> List[dict]:
        """
        Search knowledge base for relevant documents with fallback logic and caching.

        Query Flow:
        1. Check cache for existing results
        2. Vector search (semantic search)
        3. If no results or low confidence, fallback to text search
        4. Cache and return combined results

        Args:
            query: Search query string.
            top_k: Number of results to return.
            use_cache: Whether to use result caching.

        Returns:
            List of relevant document chunks with metadata.
        """
        top_k = min(max(1, top_k), 10)  # Clamp to 1-10
        start_time = time.time()

        # Step 0: Check cache (using async Redis cache)
        cache = await get_search_cache_async()
        if use_cache:
            cached_results = await cache.aget(query, top_k)
            if cached_results is not None:
                elapsed_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"[KnowledgeSearch] Cache hit for '{query[:30]}...' "
                    f"({len(cached_results)} results, {elapsed_ms}ms, backend={cache.get_stats().get('backend', 'memory')})"
                )
                # 添加缓存标记
                for r in cached_results:
                    r["cached"] = True
                return cached_results

        results = []

        # Step 1: Vector search (semantic search)
        logger.info(f"[KnowledgeSearch] Step 1: Vector search for '{query[:50]}...'")
        vector_results = await self._vector_search(query, top_k)

        if vector_results:
            results.extend(vector_results)
            # Check if results have good confidence scores
            high_confidence = any(
                r.get('score', 0) and r.get('score', 0) > RELEVANCE_THRESHOLD
                for r in results
            )

            if high_confidence:
                elapsed_ms = int((time.time() - start_time) * 1000)
                logger.info(
                    f"[KnowledgeSearch] Vector search returned {len(results)} "
                    f"high-confidence results ({elapsed_ms}ms)"
                )
                final_results = results[:top_k]
                if use_cache:
                    await cache.aset(query, top_k, final_results)
                return final_results

        # Step 2: Fallback to text search in knowledge base files
        logger.info(f"[KnowledgeSearch] Step 2: Text search fallback for '{query[:50]}...'")
        text_results = await self._text_search(query, top_k)

        if text_results:
            # Deduplicate and merge results
            existing_contents = {r.get('content', '')[:100] for r in results}
            for result in text_results:
                content_key = result.get('content', '')[:100]
                if content_key not in existing_contents:
                    results.append(result)
                    existing_contents.add(content_key)

        elapsed_ms = int((time.time() - start_time) * 1000)

        if results:
            logger.info(
                f"[KnowledgeSearch] Combined search returned {len(results)} results ({elapsed_ms}ms)"
            )
            final_results = results[:top_k]
            if use_cache:
                await cache.aset(query, top_k, final_results)
            return final_results

        # No results found
        logger.info(f"[KnowledgeSearch] No results found for '{query[:50]}...' ({elapsed_ms}ms)")
        return [{
            "content": "未找到相关知识库内容。",
            "source": "system",
            "score": 0,
            "note": "建议使用网络搜索工具获取更多信息"
        }]

    async def _vector_search(self, query: str, top_k: int) -> List[dict]:
        """Execute vector similarity search."""
        try:
            if self.rag_service:
                # Use RAG service for search (sync method, run in thread pool)
                docs = await asyncio.to_thread(
                    self.rag_service.get_relevant_documents, query
                )
                docs = docs[:top_k] if docs else []
            elif self.vector_store:
                # Direct vector store search
                docs = await asyncio.to_thread(
                    self.vector_store.similarity_search, query, k=top_k
                )
            else:
                return []

            results = []
            for doc in docs:
                if isinstance(doc, dict):
                    results.append({
                        "content": doc.get("content", doc.get("page_content", "")),
                        "source": doc.get("source", "unknown"),
                        "title": doc.get("title", ""),
                        "score": doc.get("score", 0),
                        "search_type": "vector"
                    })
                else:
                    # LangChain Document object
                    results.append({
                        "content": getattr(doc, "page_content", str(doc)),
                        "source": doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") else "unknown",
                        "title": doc.metadata.get("title", "") if hasattr(doc, "metadata") else "",
                        "score": doc.metadata.get("score", 0) if hasattr(doc, "metadata") else 0,
                        "search_type": "vector"
                    })

            return results

        except Exception as e:
            logger.error(f"[KnowledgeSearch] Vector search error: {e}")
            return []

    async def _text_search(self, query: str, top_k: int) -> List[dict]:
        """Execute text search in knowledge base files as fallback."""
        try:
            if not self._kb_path.exists():
                logger.warning(f"[KnowledgeSearch] Knowledge base path not found: {self._kb_path}")
                return []

            results = []
            # Prepare search terms using jieba for Chinese tokenization
            # Filter out stopwords and short tokens
            stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '什么', '如何', '怎么', '怎样', '吗', '呢', '啊', '吧'}
            keywords = [kw.lower() for kw in jieba.cut(query) if len(kw) >= 2 and kw not in stopwords]
            logger.debug(f"[KnowledgeSearch] Text search keywords (jieba): {keywords}")

            # Search in text files
            text_extensions = {'.txt', '.md', '.json', '.csv'}
            for file_path in self._kb_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                    try:
                        content = await asyncio.to_thread(
                            file_path.read_text, encoding='utf-8'
                        )
                        # Calculate relevance score based on keyword matches
                        content_lower = content.lower()
                        matches = sum(1 for kw in keywords if kw in content_lower)

                        if matches > 0:
                            relevance = matches / len(keywords) if keywords else 0

                            # Extract relevant snippets
                            snippets = self._extract_snippets(content, keywords, max_snippets=3)

                            for snippet in snippets:
                                results.append({
                                    "content": snippet,
                                    "source": file_path.name,
                                    "title": f"From {file_path.name}",
                                    "score": relevance,
                                    "search_type": "text"
                                })

                    except Exception as e:
                        logger.debug(f"[KnowledgeSearch] Error reading {file_path}: {e}")

            # Sort by relevance score
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"[KnowledgeSearch] Text search error: {e}")
            return []

    def _extract_snippets(self, content: str, keywords: List[str], max_snippets: int = 3) -> List[str]:
        """Extract relevant text snippets containing keywords."""
        snippets = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                # Get context (line before and after)
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                snippet = '\n'.join(lines[start:end]).strip()

                # Limit snippet length
                if len(snippet) > 500:
                    snippet = snippet[:500] + '...'

                if snippet and snippet not in snippets:
                    snippets.append(snippet)

                if len(snippets) >= max_snippets:
                    break

        return snippets


class KnowledgeAddTextTool(BaseMCPTool):
    """
    Add text content to the knowledge base.
    Useful for adding information during conversations.
    """

    name = "knowledge_add_text"
    description = (
        "Add new text content to the knowledge base. "
        "Use this to store important information that should be "
        "available for future searches."
    )
    parameters = [
        ToolParameter(
            name="text",
            type="string",
            description="The text content to add to the knowledge base",
            required=True,
        ),
        ToolParameter(
            name="title",
            type="string",
            description="A title or label for this content",
            required=False,
            default="",
        ),
        ToolParameter(
            name="source",
            type="string",
            description="Source identifier for this content",
            required=False,
            default="user_input",
        ),
    ]

    def __init__(self, rag_service=None):
        super().__init__()
        self.rag_service = rag_service

    async def execute(
        self,
        text: str,
        title: str = "",
        source: str = "user_input",
    ) -> dict:
        """
        Add text to the knowledge base.

        Args:
            text: Content to add.
            title: Optional title.
            source: Source identifier.

        Returns:
            Dict with success status and details.
        """
        if not text.strip():
            return {"success": False, "error": "Text cannot be empty"}

        try:
            if not self.rag_service:
                return {"success": False, "error": "RAG service not configured"}

            metadata = {"title": title, "source": source}
            # Use sync method in thread pool
            result = await asyncio.to_thread(
                self.rag_service.add_knowledge,
                text=text,
                metadata=metadata,
            )

            logger.info(f"Added text to knowledge base: {title or 'untitled'}")
            return {
                "success": True,
                "message": "Text added to knowledge base",
                "details": result,
            }

        except Exception as e:
            logger.error(f"Failed to add text to knowledge base: {e}")
            return {"success": False, "error": str(e)}


class KnowledgeStatsTool(BaseMCPTool):
    """
    Get statistics about the knowledge base.
    """

    name = "knowledge_stats"
    description = (
        "Get statistics about the knowledge base, including "
        "document count and collection information."
    )
    parameters = []

    def __init__(self, vector_store=None):
        super().__init__()
        self.vector_store = vector_store

    async def execute(self) -> dict:
        """
        Get knowledge base statistics.

        Returns:
            Dict with collection statistics.
        """
        try:
            if not self.vector_store:
                return {"error": "Vector store not configured"}

            stats = self.vector_store.get_collection_stats()
            return {
                "success": True,
                "stats": stats,
            }

        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {"success": False, "error": str(e)}
