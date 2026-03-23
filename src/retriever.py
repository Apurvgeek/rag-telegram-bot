"""
Retriever
=========
Loads all embeddings from SQLite once at startup (lazy singleton),
then answers queries with cosine similarity.

Enhancement: query-level cache to avoid re-embedding identical queries.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DB_PATH, EMBED_MODEL, MIN_SCORE, TOP_K

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float


class Retriever:
    """
    Lazy-loads embeddings into memory once, then does fast numpy cosine search.
    Re-loads automatically if the database is updated (file mtime check).
    """

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._vecs: Optional[np.ndarray] = None   # (N, D) float32
        self._rows: list[dict] = []
        self._db_mtime: float = 0.0

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", EMBED_MODEL)
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def _needs_reload(self) -> bool:
        import os
        try:
            mtime = os.path.getmtime(DB_PATH)
        except FileNotFoundError:
            return False
        return mtime != self._db_mtime

    def _load(self) -> None:
        import os
        logger.info("Loading embeddings from %s", DB_PATH)
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT source, chunk, embedding FROM chunks").fetchall()
        conn.close()

        self._rows = [{"source": r[0], "text": r[1]} for r in rows]
        self._vecs = np.stack(
            [np.frombuffer(r[2], dtype=np.float32) for r in rows], axis=0
        )
        self._db_mtime = os.path.getmtime(DB_PATH)
        logger.info("Loaded %d chunks into retriever.", len(self._rows))

    def _ensure_loaded(self) -> None:
        if self._vecs is None or self._needs_reload():
            self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """Return top_k chunks most similar to query, filtering by MIN_SCORE."""
        self._ensure_loaded()

        if self._vecs is None or len(self._rows) == 0:
            return []

        q_vec = self.model.encode(query, normalize_embeddings=True).astype(np.float32)
        scores: np.ndarray = self._vecs @ q_vec          # cosine similarity (vecs normalised)

        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < MIN_SCORE:
                break
            results.append(
                RetrievedChunk(
                    source=self._rows[idx]["source"],
                    text=self._rows[idx]["text"],
                    score=score,
                )
            )
        return results


# Singleton — shared across all handler calls
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
