"""
Pipeline Orchestrator — runs all 5 stages in the correct order.

Stages 1 (scope) and 2 (safety) run CONCURRENTLY (asyncio.gather).
Stages 3–4 run sequentially after the gate.

Hard gate rule:
  if NOT in_scope OR NOT safe  →  return refusal immediately, skip stages 3–4.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

from chatbot.config import ClientConfig, get_app_settings
from chatbot.stages import stage1_classifier as s1
from chatbot.stages import stage2_safety as s2
from chatbot.stages import stage3_retrieval as s3
from chatbot.stages import stage4_generation as s4
from chatbot.stages import stage5_humanizer as s5

logger = logging.getLogger(__name__)


# ── Pipeline response ─────────────────────────────────────────────────────────

@dataclass
class PipelineResponse:
    answer: str
    blocked: bool = False
    block_reason: str = ""
    category: str = ""
    safety_flags: list[str] = field(default_factory=list)
    retrieved_count: int = 0
    model_used: str = ""


# ── Async runners ─────────────────────────────────────────────────────────────

async def _run_classifier(message: str, config: ClientConfig, groq_api_key: str, history: list[dict]) -> s1.ClassifierResult:
    return await asyncio.to_thread(
        s1.classify,
        message,
        config.business_name,
        config.classifier_model,
        groq_api_key,
        history,
    )


async def _run_safety(message: str, config: ClientConfig, groq_api_key: str, history: list[dict]) -> s2.SafetyResult:
    return await asyncio.to_thread(
        s2.check_safety,
        message,
        config.safety_model,
        groq_api_key,
        history,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(
    message: str,
    session_id: str,
    config: ClientConfig,
    clients_root: str = "./clients",
) -> PipelineResponse:
    """Execute the 5-stage pipeline for `message` under `config`."""
    settings = get_app_settings()
    groq_api_key = os.getenv("GROQ_API_KEY", "")

    # ── Prep: load session history ───────────────────────────────────────────
    history = s5.get_history(session_id, config.session.max_history_turns)

    # ── Stages 1 + 2 (parallel) ───────────────────────────────────────────────
    logger.info("[%s] Running stages 1+2 concurrently …", session_id)
    classifier_result, safety_result = await asyncio.gather(
        _run_classifier(message, config, groq_api_key, history),
        _run_safety(message, config, groq_api_key, history),
    )

    logger.info(
        "[%s] Stage1: in_scope=%s category=%s confidence=%.2f",
        session_id, classifier_result.in_scope,
        classifier_result.category, classifier_result.confidence,
    )
    logger.info(
        "[%s] Stage2: safe=%s flags=%s",
        session_id, safety_result.safe, safety_result.flags,
    )

    # ── Hard gate ─────────────────────────────────────────────────────────────
    if not classifier_result.in_scope or not safety_result.safe:
        block_reason = []
        if not classifier_result.in_scope:
            block_reason.append(f"out-of-scope ({classifier_result.category})")
        if not safety_result.safe:
            block_reason.append(f"safety flags: {safety_result.flags}")

        logger.info("[%s] BLOCKED — %s", session_id, " | ".join(block_reason))
        return PipelineResponse(
            answer=config.resolved_refusal(),
            blocked=True,
            block_reason=" | ".join(block_reason),
            category=classifier_result.category,
            safety_flags=safety_result.flags,
        )

    # ── Stage 3: Retrieval ────────────────────────────────────────────────────
    logger.info("[%s] Stage 3: retrieving context …", session_id)
    lancedb_path = str(config.lancedb_path(clients_root))
    chunks = await asyncio.to_thread(
        s3.retrieve,
        message,
        lancedb_path,
        config.retrieval.top_k,
        config.retrieval.score_threshold,
        config.embedding_model,
        config.client_id,   # passed for Supabase mode
    )
    logger.info("[%s] Stage 3: retrieved %d chunks.", session_id, len(chunks))

    # Determine booking URL to pass to the AI prompt
    booking_cfg = getattr(config, "booking_config", {})
    booking_url = booking_cfg.get("booking_url", "")
    if not booking_url:
        # Default to our dynamic booking form
        base = os.getenv("BASE_URL", "http://localhost:10000").rstrip("/")
        booking_url = f"{base}/book/{config.client_id}"

    # ── Stage 4: Generation ───────────────────────────────────────────────────
    logger.info("[%s] Stage 4: generating answer …", session_id)
    gen = await s4.generate_async(
        question=message,
        chunks=chunks,
        business_name=config.business_name,
        model_tag=config.generation_model,
        tone=config.tone,
        workflow_instructions=getattr(config, "workflow_instructions", ""),
        booking_url=booking_url,
        history=history,
        groq_api_key=groq_api_key,
    )

    # ── Stage 5 (post): store turn ────────────────────────────────────────────
    s5.append_turn(session_id, config.client_id, message, gen.answer)
    logger.info("[%s] Stage 5: session updated.", session_id)

    return PipelineResponse(
        answer=gen.answer,
        blocked=False,
        category=classifier_result.category,
        safety_flags=[],
        retrieved_count=len(chunks),
        model_used=gen.model_used,
    )
