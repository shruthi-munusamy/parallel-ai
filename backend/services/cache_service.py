"""Result caching"""
import hashlib
import logging

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.cache = {}
    
    def _hash(self, claim):
        return hashlib.md5(claim.lower().encode()).hexdigest()
    
    async def get(self, claim):
        key = self._hash(claim)
        return self.cache.get(key)
    
    async def set(self, claim, result):
        key = self._hash(claim)
        self.cache[key] = result
    
    async def get_stats(self):
        return {'total': len(self.cache)}
