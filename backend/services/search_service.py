"""
search_service.py - Web search service

Production: swap PROVIDER to "google" or "bing" with respective API keys.
Development/portfolio: uses DuckDuckGo (no key required) or mock fallback.

The abstraction here is intentional — all five pipeline components call
search through this single interface, so changing providers requires
editing only this file.
"""

import asyncio
import logging
import time
import os

logger = logging.getLogger(__name__)

# Set SEARCH_PROVIDER=duckduckgo in .env to enable live search
PROVIDER = os.getenv("SEARCH_PROVIDER", "mock")


class SearchService:
    def __init__(self):
        logger.info(f"✓ Search service ready (provider: {PROVIDER})")

    def _sync_ddg(self, query: str, max_results: int) -> list[dict]:
        from duckduckgo_search import DDGS
        time.sleep(1.5)
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "url":     r.get("href", ""),
                "title":   r.get("title", ""),
                "snippet": r.get("body", ""),
                "date":    r.get("published", None),
            }
            for r in results if r.get("href")
        ]

    def _mock(self, query: str, max_results: int) -> list[dict]:
        """
        Realistic mock for local development and CI.
        Structured to exercise the full pipeline (retriever, stance, synthesis).
        """
        templates = [
            {
                "url":     "https://pubmed.ncbi.nlm.nih.gov/mock1",
                "title":   f"Clinical evidence: {query}",
                "snippet": f"A randomized controlled trial found evidence directly relevant to: {query}. Results were statistically significant (p<0.05).",
                "date":    "2024-01",
            },
            {
                "url":     "https://www.nejm.org/mock2",
                "title":   f"Systematic review: {query}",
                "snippet": f"Meta-analysis of 12 studies examining {query} found mixed results across populations.",
                "date":    "2023-11",
            },
            {
                "url":     "https://www.reuters.com/mock3",
                "title":   f"Experts dispute claims about {query}",
                "snippet": f"Several researchers have questioned the evidence base for {query}, citing methodological concerns.",
                "date":    "2024-02",
            },
        ]
        return templates[:max_results]

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        try:
            if PROVIDER == "duckduckgo":
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self._sync_ddg, query, max_results
                )
            return self._mock(query, max_results)
        except Exception as e:
            logger.error(f"Search error for '{query}': {e}")
            return self._mock(query, max_results)
