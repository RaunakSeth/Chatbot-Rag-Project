"""
Groq API client — unified LLM wrapper for all pipeline stages.

Models used:
  Stage 1 & 2  : llama-3.1-8b-instant  (14,400 RPD free, fastest)
  Stage 4      : llama-3.3-70b-versatile (1,000 RPD free, best quality)

Free tier limits (as of 2026):
  llama-3.1-8b-instant  : 30 RPM / 6,000 TPM / 14,400 RPD
  llama-3.3-70b-versatile : 30 RPM / 12,000 TPM / 1,000 RPD
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Default models ────────────────────────────────────────────────────────────

FAST_MODEL   = "llama-3.1-8b-instant"      # stages 1 & 2 (14,400 RPD)
QUALITY_MODEL = "llama-3.3-70b-versatile"  # stage 4     (1,000 RPD)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ── Core chat function ────────────────────────────────────────────────────────

def chat_completion(
    messages: list[dict[str, str]],
    model: str = FAST_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
    groq_api_key: str | None = None,
    timeout: float = 30.0,
) -> str:
    """
    Call the Groq chat completions endpoint and return the assistant's reply.

    Parameters
    ----------
    messages      : OpenAI-style message list (role + content dicts).
    model         : Groq model tag.
    temperature   : Sampling temperature (0 = deterministic).
    max_tokens    : Max tokens to generate.
    groq_api_key  : API key; falls back to GROQ_API_KEY env var.
    timeout       : HTTP request timeout in seconds.

    Returns
    -------
    The assistant message content as a plain string.

    Raises
    ------
    RuntimeError  : If the API call fails after exhausting retries.
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        resp = httpx.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as exc:
        logger.error("Groq API error %s: %s", exc.response.status_code, exc.response.text)
        raise RuntimeError(f"Groq API returned {exc.response.status_code}: {exc.response.text}") from exc
    except Exception as exc:
        logger.error("Groq API call failed: %s", exc)
        raise RuntimeError(f"Groq API call failed: {exc}") from exc


async def chat_completion_async(
    messages: list[dict[str, str]],
    model: str = QUALITY_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
    groq_api_key: str | None = None,
    timeout: float = 30.0,
) -> str:
    """Async version of chat_completion — for use in FastAPI async routes."""
    import asyncio
    return await asyncio.to_thread(
        chat_completion,
        messages,
        model,
        temperature,
        max_tokens,
        groq_api_key,
        timeout,
    )
