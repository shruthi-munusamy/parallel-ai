"""
search_service.py - Web search with graceful degradation.

Provider: Tavily, which returns clean extracted snippets rather than raw SERP
HTML, so there is no scraping or rate-limit layer to maintain.

Degradation contract: a missing API key is a configuration state, not an error.
Without TAVILY_API_KEY this serves a small built-in corpus so the pipeline,
the demo and the test suite all run end to end offline. `is_mock` is surfaced
through /health so a mock-backed result is never mistaken for a live one.

Swapping providers is a one-class change: anything returning
{url, title, snippet, date} satisfies the rest of the pipeline.
"""

import asyncio
import logging
from typing import Any

from backend.config import config
from backend.services.mock_corpus import search_mock_corpus

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, api_key: str | None = None):
        key = config.TAVILY_API_KEY if api_key is None else api_key
        self.is_mock = not key
        self.client: Any = None

        if self.is_mock:
            logger.warning(
                "TAVILY_API_KEY not set - serving mock corpus. "
                "Results are illustrative, not live web data."
            )
            return

        try:
            from tavily import TavilyClient

            self.client = TavilyClient(api_key=key)
            logger.info("Search service ready (Tavily)")
        except Exception as e:
            # A bad key or missing package should degrade, not crash startup.
            logger.error("Tavily init failed (%s) - falling back to mock corpus", e)
            self.is_mock = True

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Returns [{url, title, snippet, date}]. Never raises: a failed search
        yields [] so one bad sub-query cannot abort the whole verification.
        """
        if self.is_mock:
            return search_mock_corpus(query, max_results)

        try:
            # TavilyClient.search is blocking. Calling it directly inside an
            # async def would stall the event loop and serialize every
            # concurrent sub-query, so it runs on a worker thread.
            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                max_results=max_results,
                include_raw_content=False,
            )
        except Exception as e:
            logger.error("Search failed for %r: %s", query, e)
            return []

        return [
            {
                "url": r.get("url", ""),
                "title": r.get("title", "") or "Untitled",
                "snippet": r.get("content", "") or "",
                "date": r.get("published_date"),
            }
            for r in (response or {}).get("results", [])
            if r.get("url")
        ]

    async def search_many(self, queries: list[str], max_results: int = 5) -> list[dict]:
        """
        Run sub-queries concurrently and merge, preserving first-seen order and
        deduplicating by URL.

        Concurrency matters here: decomposition produces up to 6 sub-queries, so
        sequential search would make the search stage ~6x the latency of one
        call and blow the end-to-end budget on its own.
        """
        if not queries:
            return []

        batches = await asyncio.gather(
            *(self.search(q, max_results) for q in queries),
            return_exceptions=True,
        )

        merged: list[dict] = []
        seen: set[str] = set()
        for query, batch in zip(queries, batches):
            if isinstance(batch, BaseException):
                logger.warning("Sub-query %r failed: %s", query, batch)
                continue
            for result in batch:
                url = result["url"]
                if url not in seen:
                    seen.add(url)
                    merged.append(result)
        return merged
