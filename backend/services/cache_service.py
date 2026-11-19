"""
In-process TTL + LRU cache for verification results.

Entries expire after CACHE_TTL_S; the least recently used is evicted beyond
CACHE_MAX_ENTRIES. Per-process only, so multiple workers don't share it.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

from backend.config import config

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self, ttl_s: float | None = None, max_entries: int | None = None):
        self.ttl_s = config.CACHE_TTL_S if ttl_s is None else ttl_s
        self.max_entries = config.CACHE_MAX_ENTRIES if max_entries is None else max_entries
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def _key(claim: str) -> str:
        # Normalized so trivial whitespace and case differences share an entry.
        normalized = " ".join(claim.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def get(self, claim: str) -> Any | None:
        key = self._key(claim)
        entry = self._store.get(key)

        if entry is None:
            self._misses += 1
            return None

        stored_at, value = entry
        if time.monotonic() - stored_at > self.ttl_s:
            del self._store[key]
            self._misses += 1
            logger.debug("Cache entry expired for claim hash %s", key[:8])
            return None

        self._store.move_to_end(key)
        self._hits += 1
        return value

    async def set(self, claim: str, result: Any) -> None:
        key = self._key(claim)
        self._store[key] = (time.monotonic(), result)
        self._store.move_to_end(key)

        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self._evictions += 1

    async def clear(self) -> None:
        self._store.clear()

    async def get_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._store),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_s,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }
