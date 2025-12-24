"""
Web Search Tool - MCP tool for searching the web.
Uses DuckDuckGo for privacy-friendly web searches.
"""
import logging
from typing import List

from app.mcp.tools.base import BaseMCPTool, ToolParameter

logger = logging.getLogger(__name__)


class WebSearchTool(BaseMCPTool):
    """
    Search the web for current information.
    Uses DuckDuckGo search API.
    """

    name = "web_search"
    description = (
        "Search the web for current information, news, or topics not in the knowledge base. "
        "Use this when you need up-to-date information or when the knowledge base "
        "doesn't have the answer."
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query",
            required=True,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results (1-10)",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="region",
            type="string",
            description="Region for search results (e.g., 'cn-zh' for China, 'us-en' for US)",
            required=False,
            default="cn-zh",
        ),
    ]

    def __init__(self):
        super().__init__()
        self._ddgs = None

    def _get_ddgs(self):
        """Lazy load DuckDuckGo search client."""
        if self._ddgs is None:
            try:
                from duckduckgo_search import DDGS
                self._ddgs = DDGS()
            except ImportError:
                raise ImportError(
                    "duckduckgo-search is required. "
                    "Install with: pip install duckduckgo-search"
                )
        return self._ddgs

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        region: str = "cn-zh",
    ) -> List[dict]:
        """
        Search the web using DuckDuckGo.

        Args:
            query: Search query.
            max_results: Maximum number of results.
            region: Search region.

        Returns:
            List of search results with title, link, and snippet.
        """
        max_results = min(max(1, max_results), 10)

        try:
            ddgs = self._get_ddgs()

            # Run search in thread pool since it's synchronous
            import asyncio
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(ddgs.text(
                    query,
                    region=region,
                    max_results=max_results,
                ))
            )

            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "link": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })

            logger.info(f"Web search for '{query}' returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return [{"error": str(e)}]


class WebFetchTool(BaseMCPTool):
    """
    Fetch and extract content from a web page.
    """

    name = "web_fetch"
    description = (
        "Fetch and extract the main content from a web page URL. "
        "Use this when you need to read the full content of a specific web page."
    )
    parameters = [
        ToolParameter(
            name="url",
            type="string",
            description="The URL to fetch",
            required=True,
        ),
        ToolParameter(
            name="max_length",
            type="integer",
            description="Maximum content length to return",
            required=False,
            default=5000,
        ),
    ]

    async def execute(self, url: str, max_length: int = 5000) -> dict:
        """
        Fetch content from a URL.

        Args:
            url: URL to fetch.
            max_length: Maximum content length.

        Returns:
            Dict with title and content.
        """
        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Get title
            title = soup.title.string if soup.title else ""

            # Get main content
            content = soup.get_text(separator="\n", strip=True)

            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + "..."

            logger.info(f"Fetched content from: {url}")
            return {
                "success": True,
                "url": url,
                "title": title,
                "content": content,
            }

        except Exception as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
            }
