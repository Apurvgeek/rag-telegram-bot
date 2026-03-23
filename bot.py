"""
Mini-RAG Telegram Bot
Entry point — initialises the bot and registers handlers.
"""
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from src.config import BOT_TOKEN
from src.handlers import ask_handler, help_handler, start_handler

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("ask", ask_handler))

    # Fallback: plain text messages treated as /ask queries
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask_handler))

    logger.info("Bot started — polling for updates.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
