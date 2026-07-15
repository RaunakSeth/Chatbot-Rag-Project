"""
Stage 5 — Humanization.

Implementation note (per spec):
  NO dedicated model.  Humanization is achieved entirely through:

  1. Session-level conversation memory  → injected as history into Stage 4.
  2. Tone instructions in the system prompt  → handled in stage4_generation.py.
  3. Follow-up-question pattern for ambiguous queries  → driven by Stage 4 model.

This module provides the session memory store (server-side, per session-ID).
It is intentionally simple — a dict of lists kept in memory.
For production, swap _store for Redis or a SQLite backend.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)

# ── In-memory session store ───────────────────────────────────────────────────

HistoryTurn = dict  # {"role": "user" | "assistant", "content": str}

_store: dict[str, list[HistoryTurn]] = defaultdict(list)
_lock = Lock()


def get_history(session_id: str, max_turns: int = 6) -> list[HistoryTurn]:
    """Return the last `max_turns` turns for the session (both user+assistant)."""
    with _lock:
        history = _store[session_id]
        # Each turn is 2 messages (user + assistant), so slice accordingly
        cutoff = max_turns * 2
        return list(history[-cutoff:])


def append_turn(session_id: str, user_message: str, assistant_response: str) -> None:
    """Append a user+assistant turn to the session history."""
    with _lock:
        _store[session_id].append({"role": "user", "content": user_message})
        _store[session_id].append({"role": "assistant", "content": assistant_response})
    logger.debug(
        "Session '%s' now has %d messages.", session_id, len(_store[session_id])
    )


def clear_session(session_id: str) -> None:
    """Clear the history for a given session."""
    with _lock:
        _store.pop(session_id, None)
    logger.info("Cleared session '%s'.", session_id)


def session_exists(session_id: str) -> bool:
    with _lock:
        return session_id in _store and len(_store[session_id]) > 0
