"""缓存与幂等。

🔴 两者对故障的态度**相反**：
   缓存 fail-open（后端挂了当未命中，回源）——它只是可选加速，不该拖垮业务；
   幂等 fail-closed（后端挂了抛错）——吞掉异常＝当作"没处理过"＝放行重复执行。

🔴 缓存键在构造时就把租户编进去，不靠调用方自觉拼接。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..core.config import settings
from ..core.context import RequestContext
from ..core.errors import IdempotencyUnavailable
from ..core.telemetry import get_logger, metrics

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CacheKey:
    namespace: str
    tenant_id: str
    digest: str

    @classmethod
    def build(cls, namespace: str, ctx: RequestContext, *parts: Any) -> CacheKey:
        blob = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
        return cls(
            namespace=namespace,
            tenant_id=ctx.tenant_id,
            digest=hashlib.sha256(blob.encode()).hexdigest()[:32],
        )

    def render(self) -> str:
        # 租户段放最前：便于按租户批量失效，也便于一眼看出归属
        return f"ics:{self.tenant_id}:{self.namespace}:{self.digest}"


class CacheBackend(Protocol):
    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: bytes, ttl: int) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def set_if_absent(self, key: str, value: bytes, ttl: int) -> bool: ...


class MemoryBackend:
    """进程内。单副本与测试用；多副本之间互不可见。"""

    def __init__(self) -> None:
        self._d: dict[str, tuple[bytes, float]] = {}
        self._lock = asyncio.Lock()

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    async def get(self, key: str) -> bytes | None:
        async with self._lock:
            hit = self._d.get(key)
            if hit is None or hit[1] <= self._now():
                self._d.pop(key, None)
                return None
            return hit[0]

    async def set(self, key: str, value: bytes, ttl: int) -> None:
        async with self._lock:
            self._d[key] = (value, self._now() + ttl)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._d.pop(key, None)

    async def set_if_absent(self, key: str, value: bytes, ttl: int) -> bool:
        async with self._lock:
            hit = self._d.get(key)
            if hit is not None and hit[1] > self._now():
                return False
            self._d[key] = (value, self._now() + ttl)
            return True


class RedisBackend:
    def __init__(self, url: str) -> None:
        self._url = url
        self._c: Any = None

    async def _conn(self) -> Any:
        if self._c is None:
            import redis.asyncio as redis

            self._c = redis.from_url(self._url, decode_responses=False)
        return self._c

    async def get(self, key: str) -> bytes | None:
        return await (await self._conn()).get(key)

    async def set(self, key: str, value: bytes, ttl: int) -> None:
        await (await self._conn()).set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        await (await self._conn()).delete(key)

    async def set_if_absent(self, key: str, value: bytes, ttl: int) -> bool:
        # 🔴 必须用原子 SET NX：先 GET 再 SET 之间有竞态，并发下两个请求都会认为自己是第一个
        return bool(await (await self._conn()).set(key, value, ex=ttl, nx=True))

    async def aclose(self) -> None:
        if self._c is not None:
            await self._c.aclose()
            self._c = None


class Cache:
    def __init__(self, backend: CacheBackend) -> None:
        self._b = backend

    async def get_json(self, key: CacheKey) -> Any | None:
        try:
            raw = await self._b.get(key.render())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("缓存读取失败，降级为未命中: %s", exc)
            metrics().degraded.labels("cache", "read_error").inc()
            return None
        metrics().cache.labels(key.namespace, "hit" if raw else "miss").inc()
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    async def set_json(self, key: CacheKey, value: Any, ttl: int | None = None) -> None:
        try:
            await self._b.set(
                key.render(),
                json.dumps(value, ensure_ascii=False, default=str).encode(),
                ttl or settings.cache_ttl_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("缓存写入失败（不影响业务）: %s", exc)


class Idempotency:
    def __init__(self, backend: CacheBackend, ttl: int = 3600) -> None:
        self._b = backend
        self._ttl = ttl

    async def claim(self, key: CacheKey) -> bool:
        try:
            return await self._b.set_if_absent(key.render(), b"1", self._ttl)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 🔴 不能吞：吞掉等于当作"没处理过"，直接放行重复执行
            raise IdempotencyUnavailable("幂等后端不可用，拒绝执行", key=key.render()) from exc


_cache: Cache | None = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        backend: CacheBackend = (
            RedisBackend(settings.redis_url.get_secret_value())
            if settings.env != "local"
            else MemoryBackend()
        )
        _cache = Cache(backend)
    return _cache
