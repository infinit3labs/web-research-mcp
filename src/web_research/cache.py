"""Small in-memory TTL + size-bounded cache.

Single-process, no external dependency. Callers are responsible for building
cache keys that never embed credential material — this module only stores
whatever key/value pairs it's given.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable, Generic, TypeVar

V = TypeVar("V")


class TTLCache(Generic[V]):
    """LRU cache with a per-entry TTL. Not thread-safe — intended for a
    single asyncio event loop, where cache methods never yield mid-operation.
    """

    def __init__(
        self,
        max_entries: int,
        ttl_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_entries = max(0, max_entries)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._clock = clock
        self._store: "OrderedDict[str, tuple[float, V]]" = OrderedDict()

    def get(self, key: str) -> V | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= self._clock():
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: V) -> None:
        # A zero TTL or zero capacity means "caching disabled" — every read
        # misses, so don't bother storing anything.
        if self.ttl_seconds <= 0 or self.max_entries <= 0:
            return
        self._store[key] = (self._clock() + self.ttl_seconds, value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
