"""
Stage 1 — Business-Scope Classifier.

Model: Groq llama-3.1-8b-instant (14,400 req/day free)
Load time: ~0 ms (API call, no local weights)

Output contract (strict JSON):
  {
    "in_scope": bool,
    "category": "pricing|services|booking|general_faq|out_of_scope",
    "confidence": float  # 0.0 – 1.0
  }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from chatbot.llm import FAST_MODEL, chat_completion

logger = logging.getLogger(__name__)

CATEGORIES = Literal["pricing", "services", "booking", "general_faq", "out_of_scope"]

_SYSTEM_PROMPT = """\
You are a lenient business-scope classifier. Decide whether a user's message could REASONABLY be related to the business.
Context matters: If the business is a law firm or clinic, queries about injuries, accidents, crimes, or medical help are highly relevant and IN SCOPE.
When in doubt, always assume the message is IN SCOPE.

CRITICAL INSTRUCTION: Standard conversational replies, greetings, and continuations (e.g., "sure", "yes", "no", "please do so", "hi", "thanks", "ok") are ALWAYS IN SCOPE because they are part of a normal business conversation. Categorize them as 'general_faq'.

Categories:
- pricing        : cost, fees, plans, discounts
- services       : what the business offers or can help with (e.g. legal cases, treatments)
- booking        : appointments, availability
- general_faq    : hours, location, contact, and conversational fillers/continuations
- out_of_scope   : completely unrelated (e.g., politics, random trivia, sports)

Respond with ONLY a JSON object, no prose, no markdown:
{"in_scope": true/false, "category": "<one of the 5 above>", "confidence": 0.0-1.0}
"""

_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Business: TechFix Computer Repair\n"
            "Message: How much does it cost to fix a cracked laptop screen?"
        ),
    },
    {
        "role": "assistant",
        "content": '{"in_scope": true, "category": "pricing", "confidence": 0.97}',
    },
    {
        "role": "user",
        "content": (
            "Business: Duncan Firm, P.A.\n"
            "Message: I got hit by a car, which of your associates can help me?"
        ),
    },
    {
        "role": "assistant",
        "content": '{"in_scope": true, "category": "services", "confidence": 0.98}',
    },
    {
        "role": "user",
        "content": (
            "Business: TechFix Computer Repair\n"
            "Message: Who won the 2024 US election?"
        ),
    },
    {
        "role": "assistant",
        "content": '{"in_scope": false, "category": "out_of_scope", "confidence": 0.99}',
    },
]


@dataclass
class ClassifierResult:
    in_scope: bool
    category: str
    confidence: float
    raw: str = ""


def classify(
    message: str,
    business_name: str,
    model_id: str = FAST_MODEL,
    groq_api_key: str | None = None,
    max_new_tokens: int = 80,
) -> ClassifierResult:
    """
    Classify whether `message` is in-scope for `business_name`.

    Never raises — falls back to safe default (in_scope=True) on error so a
    classifier glitch doesn't block the whole pipeline.
    """
    user_turn = f"Business: {business_name}\nMessage: {message}"
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *_FEW_SHOT,
        {"role": "user", "content": user_turn},
    ]

    try:
        raw = chat_completion(
            messages=messages,
            model=model_id,
            temperature=0.0,
            max_tokens=max_new_tokens,
            groq_api_key=groq_api_key,
        )
        return _parse_result(raw)
    except Exception as exc:
        logger.error("Stage 1 classifier error: %s", exc, exc_info=True)
        # Safe default: pass through (let safety guardrail catch real issues)
        return ClassifierResult(
            in_scope=True,
            category="general_faq",
            confidence=0.0,
            raw=str(exc),
        )


def _parse_result(raw: str) -> ClassifierResult:
    """Extract JSON from the model output, tolerating minor formatting noise."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
        return ClassifierResult(
            in_scope=bool(data.get("in_scope", True)),
            category=str(data.get("category", "general_faq")),
            confidence=float(data.get("confidence", 0.5)),
            raw=raw,
        )
    except json.JSONDecodeError:
        logger.warning("Stage 1: could not parse JSON from model output: %r", raw)
        return ClassifierResult(in_scope=True, category="general_faq", confidence=0.0, raw=raw)
