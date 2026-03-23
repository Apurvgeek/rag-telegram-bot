"""
Configuration — loads from .env via python-dotenv.
All tuneable constants live here so handlers stay clean.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Telegram ─────────────────────────────────────────────────────────────────
BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]

# ── RAG knobs ─────────────────────────────────────────────────────────────────
KNOWLEDGE_DIR: Path = Path(os.getenv("KNOWLEDGE_DIR", "knowledge_base"))
DB_PATH: str = os.getenv("DB_PATH", "rag_store.db")

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))          # characters per chunk
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K: int = int(os.getenv("TOP_K", "3"))                      # chunks to retrieve
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.20"))       # cosine similarity floor

# ── Embedding model (local, no API key needed) ────────────────────────────────
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── LLM backend ───────────────────────────────────────────────────────────────
# Options: "ollama" | "openai" | "huggingface"
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Optional enhancements ────────────────────────────────────────────────────
HISTORY_LENGTH: int = int(os.getenv("HISTORY_LENGTH", "3"))    # turns to keep per user
SHOW_SOURCES: bool = os.getenv("SHOW_SOURCES", "true").lower() == "true"
