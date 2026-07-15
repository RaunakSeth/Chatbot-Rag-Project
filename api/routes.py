"""
API route definitions.

Endpoints:
  POST /chat              - Main chat endpoint
  DELETE /sessions/{id}   - Clear a session's conversation history
  GET  /health            - Health check (Ollama ping)
  GET  /clients/{id}      - Fetch a client's config (sans secrets)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, Field

from chatbot.config import load_config, get_app_settings
from chatbot.pipeline import run_pipeline
from chatbot.stages.stage5_humanizer import clear_session, session_exists

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    client_id: str = Field(..., description="Client identifier (must match a config in clients/)")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session identifier for conversation memory. Auto-generated if omitted.",
    )
    message: str = Field(..., min_length=1, max_length=4096, description="User message")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    blocked: bool
    block_reason: str = ""
    category: str = ""
    retrieved_count: int = 0
    model_used: str = ""
    latency_ms: float = 0.0


class SessionInfo(BaseModel):
    session_id: str
    exists: bool


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    details: dict[str, Any] = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint — runs all 5 pipeline stages and returns the answer.
    """
    settings = get_app_settings()
    t0 = time.perf_counter()

    # Load client config
    try:
        config = load_config(request.client_id, settings.clients_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Run pipeline
    try:
        result = await run_pipeline(
            message=request.message,
            session_id=request.session_id,
            config=config,
            clients_root=settings.clients_dir,
        )
    except Exception as exc:
        logger.error("Pipeline error for client '%s': %s", request.client_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal pipeline error.") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "chat client=%s session=%s blocked=%s latency=%.0fms",
        request.client_id,
        request.session_id,
        result.blocked,
        latency_ms,
    )

    return ChatResponse(
        session_id=request.session_id,
        answer=result.answer,
        blocked=result.blocked,
        block_reason=result.block_reason,
        category=result.category,
        retrieved_count=result.retrieved_count,
        model_used=result.model_used,
        latency_ms=round(latency_ms, 1),
    )


@router.delete("/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def delete_session(
    session_id: str = Path(..., description="Session ID to clear"),
) -> SessionInfo:
    """Clear a session's conversation history."""
    clear_session(session_id)
    return SessionInfo(session_id=session_id, exists=False)


@router.get("/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def get_session(
    session_id: str = Path(..., description="Session ID to check"),
) -> SessionInfo:
    """Check whether a session exists."""
    return SessionInfo(session_id=session_id, exists=session_exists(session_id))


@router.get("/health", response_model=HealthResponse, tags=["Admin"])
async def health_check() -> HealthResponse:
    """Health check — pings Ollama to verify it's reachable."""
    settings = get_app_settings()
    try:
        import ollama
        client = ollama.Client(host=settings.ollama_base_url)
        models = client.list()
        ollama_ok = True
        details = {"ollama_url": settings.ollama_base_url, "models_available": len(models.models)}
    except Exception as exc:
        ollama_ok = False
        details = {"error": str(exc), "ollama_url": settings.ollama_base_url}

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_reachable=ollama_ok,
        details=details,
    )


@router.get("/clients/{client_id}", tags=["Admin"])
async def get_client_config(
    client_id: str = Path(..., description="Client ID"),
) -> dict:
    """Return a sanitized view of a client's config."""
    settings = get_app_settings()
    try:
        config = load_config(client_id, settings.clients_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "client_id": config.client_id,
        "business_name": config.business_name,
        "hardware_tier": config.hardware_tier,
        "tone": config.tone,
        "generation_model": config.generation_model_ollama,
        "embedding_model": config.embedding_model,
        "retrieval": config.retrieval.model_dump(),
        "session": config.session.model_dump(),
    }
