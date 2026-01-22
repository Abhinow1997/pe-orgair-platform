# src/pe_orgair/infrastructure/cache.py
"""Redis caching with TTL and pub/sub."""
from typing import Any, Optional, TypeVar, Callable
from functools import wraps
import json
import redis
import structlog

from pe_orgair.config.settings import settings

logger = structlog.get_logger()
T = TypeVar("T")


class CacheManager:
    """Redis cache manager with TTL support."""
    
    def __init__(self):
        self._client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self._pubsub = self._client.pubsub()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        value = self._client.get(key)
        if value:
            logger.debug("cache_hit", key=key)
            return json.loads(value)
        logger.debug("cache_miss", key=key)
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set cached value with TTL."""
        self._client.setex(key, ttl, json.dumps(value))
        logger.debug("cache_set", key=key, ttl=ttl)
    
    def delete(self, key: str) -> None:
        """Delete cached value."""
        self._client.delete(key)
        logger.debug("cache_delete", key=key)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        keys = self._client.keys(pattern)
        if keys:
            count = self._client.delete(*keys)
            logger.info("cache_invalidated", pattern=pattern, count=count)
            return count
        return 0
    
    def publish(self, channel: str, message: dict) -> None:
        """Publish message to channel."""
        self._client.publish(channel, json.dumps(message))
        logger.debug("pubsub_published", channel=channel)


cache = CacheManager()


def cached(key_template: str, ttl: int = 3600):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = key_template.format(*args, **kwargs)
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator