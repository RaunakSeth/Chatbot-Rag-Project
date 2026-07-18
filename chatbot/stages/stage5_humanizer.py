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
import os
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)

# ── In-memory session store (fallback) ────────────────────────────────────────

HistoryTurn = dict  # {"role": "user" | "assistant", "content": str}

_store: dict[str, list[HistoryTurn]] = defaultdict(list)
_lock = Lock()


def get_history(session_id: str, max_turns: int = 6) -> list[HistoryTurn]:
    """Return the last `max_turns` turns for the session (both user+assistant)."""
    use_supabase = bool(os.getenv("SUPABASE_URL", "").strip())
    history = []
    
    if use_supabase:
        try:
            from chatbot.db import get_client
            sb = get_client()
            res = sb.table("sessions").select("messages").eq("session_id", session_id).maybe_single().execute()
            if res and hasattr(res, "data") and res.data:
                history = res.data.get("messages", [])
        except Exception as exc:
            logger.error("Failed to fetch session from Supabase: %s", exc)
            with _lock:
                history = _store[session_id]
    else:
        with _lock:
            history = _store[session_id]

    cutoff = max_turns * 2
    return list(history[-cutoff:])


def append_turn(session_id: str, client_id: str, user_message: str, assistant_response: str) -> None:
    """Append a user+assistant turn to the session history."""
    use_supabase = bool(os.getenv("SUPABASE_URL", "").strip())
    
    with _lock:
        _store[session_id].append({"role": "user", "content": user_message})
        _store[session_id].append({"role": "assistant", "content": assistant_response})
        local_history = _store[session_id].copy()

    if use_supabase:
        try:
            from chatbot.db import get_client
            sb = get_client()
            
            # Fetch existing to append (since Supabase jsonb doesn't easily append via simple rest)
            # Actually we can just overwrite with the local_history since we cache it,
            # BUT if we have multi-dyno, we should fetch first.
            res = sb.table("sessions").select("messages").eq("session_id", session_id).maybe_single().execute()
            messages = []
            if res and hasattr(res, "data") and res.data:
                messages = res.data.get("messages", [])
            
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_response})
            
            sb.table("sessions").upsert({
                "session_id": session_id,
                "client_id": client_id,
                "messages": messages
            }, on_conflict="session_id").execute()
            
        except Exception as exc:
            logger.error("Failed to save session to Supabase: %s", exc)

    logger.debug("Session '%s' updated.", session_id)


def clear_session(session_id: str) -> None:
    """Clear the history for a given session."""
    with _lock:
        _store.pop(session_id, None)
        
    if os.getenv("SUPABASE_URL", "").strip():
        try:
            from chatbot.db import get_client
            get_client().table("sessions").delete().eq("session_id", session_id).execute()
        except Exception as exc:
            logger.error("Failed to delete session from Supabase: %s", exc)
            
    logger.info("Cleared session '%s'.", session_id)


def session_exists(session_id: str) -> bool:
    if os.getenv("SUPABASE_URL", "").strip():
        try:
            from chatbot.db import get_client
            res = get_client().table("sessions").select("session_id").eq("session_id", session_id).maybe_single().execute()
            return bool(res and hasattr(res, "data") and res.data)
        except Exception:
            return False
    with _lock:
        return session_id in _store and len(_store[session_id]) > 0
