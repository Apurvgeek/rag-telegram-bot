"""
Query Cache
===========
Simple LRU cache: identical query strings reuse previously retrieved chunks,
avoiding redundant embedding + DB reads.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List

from src.retriever import RetrievedChunk, get_retriever


def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


@lru_cache(maxsize=256)
def cached_retrieve(cache_key: str, query: str) -> tuple[RetrievedChunk, ...]:
    """Retrieve and cache results. cache_key is md5(query) to satisfy hashability."""
    return tuple(get_retriever().retrieve(query))


def retrieve_with_cache(query: str) -> List[RetrievedChunk]:
    key = _cache_key(query)
    return list(cached_retrieve(key, query))
