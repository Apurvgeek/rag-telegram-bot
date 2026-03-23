"""
Telegram Handlers
=================
/start  — welcome message
/help   — usage instructions
/ask    — RAG query (also triggered by plain text messages)
"""
from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from src.cache import retrieve_with_cache
from src.config import SHOW_SOURCES
from src.history import get_history_manager
from src.llm_client import generate

logger = logging.getLogger(__name__)

WELCOME = (
    "👋 *Welcome to the Mini-RAG Bot!*\n\n"
    "I can answer questions from my knowledge base using retrieval-augmented generation.\n\n"
    "Type /help to see available commands."
)

HELP_TEXT = (
    "📖 *Commands*\n\n"
    "/ask \\<your question\\> — ask anything from the knowledge base\n"
    "/help — show this message\n\n"
    "You can also just type your question without any command.\n\n"
    "_Example:_ `/ask What is the refund policy?`"
)

NO_CONTEXT = (
    "⚠️ I couldn't find relevant information in the knowledge base for that question. "
    "Try rephrasing, or ask something else."
)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME, parse_mode="Markdown")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Extract query from /ask <query> or plain text
    if update.message.text.startswith("/ask"):
        parts = update.message.text.split(maxsplit=1)
        query = parts[1].strip() if len(parts) > 1 else ""
    else:
        query = update.message.text.strip()

    if not query:
        await update.message.reply_text(
            "Please provide a question. Example: `/ask What is the return policy?`",
            parse_mode="Markdown",
        )
        return

    user_id = update.effective_user.id
    await update.message.chat.send_action(ChatAction.TYPING)

    # ── Retrieve ────────────────────────────────────────────────────────────
    chunks = retrieve_with_cache(query)

    if not chunks:
        await update.message.reply_text(NO_CONTEXT)
        return

    # ── Build context string ──────────────────────────────────────────────
    context_text = "\n\n---\n\n".join(c.text for c in chunks)

    # ── Get history and generate answer ──────────────────────────────────
    history_mgr = get_history_manager()
    history = history_mgr.get(user_id)

    try:
        answer = generate(context_text, history, query)
    except Exception as exc:
        logger.exception("LLM generation failed: %s", exc)
        await update.message.reply_text(
            "⚠️ The language model returned an error. Please check your setup."
        )
        return

    # ── Store turn in history ─────────────────────────────────────────────
    history_mgr.add(user_id, query, answer)

    # ── Format reply ──────────────────────────────────────────────────────
    reply = answer

    if SHOW_SOURCES and chunks:
        sources = sorted({c.source for c in chunks})
        src_line = "  •  ".join(f"`{s}`" for s in sources)
        reply += f"\n\n📎 _Sources: {src_line}_"

    await update.message.reply_text(reply)
    logger.info("user=%d  query=%r  chunks=%d", user_id, query[:60], len(chunks))
