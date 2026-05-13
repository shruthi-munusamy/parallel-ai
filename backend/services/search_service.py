"""
search_service.py - Tavily web search

Tavily is purpose-built for AI agents — returns clean, LLM-optimized snippets
without the scraping/rate-limit issues of DuckDuckGo.

Portfolio note: production would use Google Custom Search or Bing.
The abstraction here means swapping providers is a one-file change.
"""

import logging
import os
from tavily import TavilyClient

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set")
        self.client = TavilyClient(api_key=api_key)
        logger.info("✓ Search service ready (Tavily)")

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                include_raw_content=False,
            )
            return [
                {
                    "url":     r.get("url", ""),
                    "title":   r.get("title", ""),
                    "snippet": r.get("content", ""),
                    "date":    r.get("published_date", None),
                }
                for r in response.get("results", [])
                if r.get("url")
            ]
        except Exception as e:
            logger.error(f"Search error for '{query}': {e}")
            return []
