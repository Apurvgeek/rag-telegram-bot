"""
History Manager
===============
Maintains a bounded, per-user conversation history in memory.
Thread-safe for python-telegram-bot's async context.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

from src.config import HISTORY_LENGTH


class HistoryManager:
    """
    Stores the last HISTORY_LENGTH (user, assistant) turn pairs per Telegram user_id.
    """

    def __init__(self, maxlen: int = HISTORY_LENGTH) -> None:
        self._maxlen = maxlen
        self._store: Dict[int, Deque[dict]] = {}

    def get(self, user_id: int) -> List[dict]:
        """Return conversation history as a list of {user, assistant} dicts."""
        return list(self._store.get(user_id, deque()))

    def add(self, user_id: int, user_msg: str, assistant_msg: str) -> None:
        if user_id not in self._store:
            self._store[user_id] = deque(maxlen=self._maxlen)
        self._store[user_id].append({"user": user_msg, "assistant": assistant_msg})

    def clear(self, user_id: int) -> None:
        self._store.pop(user_id, None)


# Singleton
_history = HistoryManager()


def get_history_manager() -> HistoryManager:
    return _history
