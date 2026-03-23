"""
Ingestion pipeline
==================
Reads every .txt / .md file in KNOWLEDGE_DIR, splits into overlapping chunks,
embeds them with sentence-transformers, and persists to SQLite.

Run once (or whenever documents change):
    python -m src.ingest
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    EMBED_MODEL,
    KNOWLEDGE_DIR,
)

logger = logging.getLogger(__name__)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window character-level splitter with sentence boundary awareness."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # carry-over overlap
            carry = current[-overlap:] if overlap else ""
            current = (carry + "\n\n" + para).strip()

    if current:
        chunks.append(current)

    return chunks


# ── Database helpers ──────────────────────────────────────────────────────────

def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            source   TEXT    NOT NULL,
            chunk    TEXT    NOT NULL,
            hash     TEXT    NOT NULL UNIQUE,
            embedding BLOB   NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)")
    conn.commit()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def ingest(force: bool = False) -> int:
    """
    Embed all documents in KNOWLEDGE_DIR and store in SQLite.
    Skips chunks whose hash already exists unless force=True.
    Returns the number of new chunks added.
    """
    docs = list(KNOWLEDGE_DIR.glob("**/*.txt")) + list(KNOWLEDGE_DIR.glob("**/*.md"))
    if not docs:
        logger.warning("No .txt / .md files found in %s", KNOWLEDGE_DIR)
        return 0

    logger.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    conn = sqlite3.connect(DB_PATH)
    _init_db(conn)

    added = 0
    for doc in docs:
        text = doc.read_text(encoding="utf-8", errors="ignore")
        chunks = _split(text)
        logger.info("  %s → %d chunks", doc.name, len(chunks))

        for chunk in chunks:
            h = _hash(chunk)
            if not force:
                exists = conn.execute(
                    "SELECT 1 FROM chunks WHERE hash = ?", (h,)
                ).fetchone()
                if exists:
                    continue

            vec = model.encode(chunk, normalize_embeddings=True)
            conn.execute(
                "INSERT OR REPLACE INTO chunks (source, chunk, hash, embedding) VALUES (?,?,?,?)",
                (doc.name, chunk, h, vec.astype(np.float32).tobytes()),
            )
            added += 1

    conn.commit()
    conn.close()
    logger.info("Ingestion complete — %d new chunks stored.", added)
    return added


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
