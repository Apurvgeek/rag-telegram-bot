"""
LLM Client
==========
Unified interface for three backends:
  • ollama  — local model via HTTP (default, no API key)
  • openai  — OpenAI API
  • huggingface — local HF pipeline (fallback)

Switch via LLM_BACKEND in .env.
"""
from __future__ import annotations

import logging

from src.config import (
    LLM_BACKEND,
    LLM_MODEL,
    OLLAMA_URL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a concise, helpful assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so honestly — do not make things up.
Keep your answer under 200 words unless the question requires more detail."""


def _build_prompt(context: str, history: list[dict], query: str) -> str:
    history_text = ""
    if history:
        turns = "\n".join(
            f"User: {t['user']}\nAssistant: {t['assistant']}" for t in history
        )
        history_text = f"\n\n--- Conversation history ---\n{turns}"

    return (
        f"Context:\n{context}"
        f"{history_text}"
        f"\n\nUser question: {query}"
        f"\n\nAnswer:"
    )


# ── Backend: Ollama ────────────────────────────────────────────────────────────

def _ollama(prompt: str) -> str:
    import json
    import urllib.request

    payload = json.dumps(
        {"model": LLM_MODEL, "prompt": prompt, "stream": False, "system": SYSTEM_PROMPT}
    ).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


# ── Backend: OpenAI ────────────────────────────────────────────────────────────

def _openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ── Backend: HuggingFace (local pipeline fallback) ─────────────────────────────

_hf_pipeline = None


def _huggingface(prompt: str) -> str:
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline

        logger.info("Loading HuggingFace pipeline: %s", LLM_MODEL)
        _hf_pipeline = pipeline(
            "text-generation",
            model=LLM_MODEL,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
        )
    result = _hf_pipeline(SYSTEM_PROMPT + "\n\n" + prompt)
    return result[0]["generated_text"].split("Answer:")[-1].strip()


# ── Public API ────────────────────────────────────────────────────────────────

def generate(context: str, history: list[dict], query: str) -> str:
    """
    Generate an answer given retrieved context, conversation history, and query.
    Raises RuntimeError on unknown backend.
    """
    prompt = _build_prompt(context, history, query)
    logger.debug("LLM prompt length: %d chars", len(prompt))

    backend = LLM_BACKEND.lower()
    if backend == "ollama":
        return _ollama(prompt)
    elif backend == "openai":
        return _openai(prompt)
    elif backend == "huggingface":
        return _huggingface(prompt)
    else:
        raise RuntimeError(f"Unknown LLM_BACKEND: {LLM_BACKEND!r}")
