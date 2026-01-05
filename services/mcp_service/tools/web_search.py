"""
Web Search Tools - MCP tools for searching the web.
Uses Baidu search for Chinese web searches.
"""
import re
import urllib.parse
from typing import List, Optional, Dict, Any

import httpx
from bs4 import BeautifulSoup

from services.common.logging import get_logger
from .base import BaseTool, ToolParameter

logger = get_logger(__name__)


class BaiduSearchClient:
    """
    Baidu search client.
    Parses Baidu search result pages to get search results.
    """

    BASE_URL = "https://www.baidu.com/s"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 5) -> List[dict]:
        """
        Execute Baidu search.

        Args:
            query: Search query.
            max_results: Maximum number of results.

        Returns:
            List of search results with title, link, snippet.
        """
        params = {
            "wd": query,
            "rn": min(max_results * 2, 20),
            "ie": "utf-8",
        }

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=self.HEADERS)
                response.raise_for_status()

            return self._parse_results(response.text, max_results)

        except httpx.TimeoutException:
            logger.error(f"Baidu search timeout for query: {query}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"Baidu search HTTP error: {e}")
            raise

    def _parse_results(self, html: str, max_results: int) -> List[dict]:
        """Parse Baidu search result page."""
        soup = BeautifulSoup(html, "html.parser")
        results = []

        content_left = soup.find("div", {"id": "content_left"})
        if not content_left:
            logger.warning("Could not find content_left in Baidu response")
            return results

        result_items = content_left.find_all("div", class_=re.compile(r"result|c-container"))

        for item in result_items:
            if len(results) >= max_results:
                break

            result = self._parse_result_item(item)
            if result:
                results.append(result)

        return results

    def _parse_result_item(self, item) -> Optional[dict]:
        """Parse single search result item."""
        try:
            title_tag = item.find("h3")
            if not title_tag:
                return None

            link_tag = title_tag.find("a")
            if not link_tag:
                return None

            title = link_tag.get_text(strip=True)
            link = link_tag.get("href", "")

            if not title or not link:
                return None

            snippet = ""
            snippet_selectors = [
                "span.content-right_8Zs40",
                "div.c-abstract",
                "div.c-span-last",
                "span[class*='content']",
            ]

            for selector in snippet_selectors:
                snippet_tag = item.select_one(selector)
                if snippet_tag:
                    snippet = snippet_tag.get_text(strip=True)
                    break

            if not snippet:
                for tag in item.find_all("h3"):
                    tag.decompose()
                snippet = item.get_text(separator=" ", strip=True)[:300]

            snippet = re.sub(r'\s+', ' ', snippet).strip()

            return {
                "title": title,
                "link": link,
                "snippet": snippet[:500] if snippet else "",
            }

        except Exception as e:
            logger.debug(f"Error parsing result item: {e}")
            return None


class WebSearchTool(BaseTool):
    """
    Search the web for current information.
    Uses Baidu search for Chinese web searches.
    """

    name = "web_search"
    description = (
        "Search the web for current information, news, or topics not in the knowledge base. "
        "Use this when you need up-to-date information or when the knowledge base "
        "doesn't have the answer. Uses Baidu search engine."
    )
    tags = ["web", "search"]
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
    ]

    def __init__(self, timeout: float = 10.0):
        super().__init__()
        self._client = BaiduSearchClient(timeout=timeout)

    async def execute(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[dict]:
        """
        Search the web using Baidu.

        Args:
            query: Search query.
            max_results: Maximum number of results.

        Returns:
            List of search results with title, link, and snippet.
        """
        max_results = min(max(1, max_results), 10)

        try:
            results = await self._client.search(query, max_results)

            logger.info(f"Baidu search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Baidu search failed: {e}")
            return [{"error": f"Search failed: {str(e)}"}]


class WebFetchTool(BaseTool):
    """
    Fetch and extract content from a web page.
    """

    name = "web_fetch"
    description = (
        "Fetch and extract the main content from a web page URL. "
        "Use this when you need to read the full content of a specific web page."
    )
    tags = ["web", "fetch"]
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
    timeout_seconds = 30

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
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                )
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
