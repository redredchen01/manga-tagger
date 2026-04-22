"""Redis cache manager for application-wide caching."""

import hashlib
import json
from typing import Any, Optional

import redis.asyncio as redis

from app.core.config import settings


class CacheManager:
    """Async Redis cache manager with fallback for when Redis is unavailable."""

    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.enabled = settings.REDIS_ENABLED
        self.default_ttl = settings.CACHE_TTL

    async def connect(self) -> None:
        """Connect to Redis server."""
        if not self.enabled:
            return
        try:
            self.redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
            )
            # Test connection with a simple get operation
            await self.redis.get("__test_connection__")
        except Exception:
            # Redis not available - continue without caching
            self.redis = None
            self.enabled = False

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self.redis = None

    def _make_key(self, prefix: str, data: str) -> str:
        """Generate cache key from prefix and data hash."""
        hash_obj = hashlib.md5(data.encode()).hexdigest()
        return f"{prefix}:{hash_obj}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.enabled or not self.redis:
            return None
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception:
            pass
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        if not self.enabled or not self.redis:
            return
        try:
            ttl = ttl or self.default_ttl
            await self.redis.setex(key, ttl, json.dumps(value))
        except Exception:
            pass

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        if not self.enabled or not self.redis:
            return
        try:
            await self.redis.delete(key)
        except Exception:
            pass

    async def clear_prefix(self, prefix: str) -> None:
        """Clear all keys with given prefix."""
        if not self.enabled or not self.redis:
            return
        try:
            pattern = f"{prefix}:*"
            async for key in self.redis.scan_iter(match=pattern):
                await self.redis.delete(key)
        except Exception:
            pass


cache_manager = CacheManager()
