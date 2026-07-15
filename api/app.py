"""
FastAPI application entry point.

Run locally:
  uvicorn api.app:app --host 0.0.0.0 --port 8000

Cloud (Render.com):
  Render reads render.yaml and runs this automatically.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.admin_routes import admin_router

# ── Logging setup ─────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Business AI Chatbot API starting …")

    # Verify Groq API key is present
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        logger.info("✅ GROQ_API_KEY found — LLM inference ready.")
    else:
        logger.warning("⚠️  GROQ_API_KEY not set — chat will fail. Set it in .env")

    # Verify Supabase (optional — falls back to LanceDB locally)
    if os.getenv("SUPABASE_URL", ""):
        logger.info("✅ SUPABASE_URL found — using cloud vector store.")
    else:
        logger.info("ℹ️  SUPABASE_URL not set — using local LanceDB.")

    logger.info("🟢 Server ready.")
    yield
    logger.info("🛑 Business AI Chatbot API shutting down.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Business AI Chatbot API",
    description=(
        "Multi-tenant RAG chatbot. "
        "Each client has an isolated knowledge base. "
        "Powered by Groq + Supabase pgvector."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(router)
app.include_router(admin_router)

# ── Static files & root page ──────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the chat UI on the same origin as the API (no CORS issues)."""
    return FileResponse(os.path.join(_BASE_DIR, "index.html"))

# Mount static AFTER explicit routes so it doesn't shadow them
app.mount("/static", StaticFiles(directory=_BASE_DIR), name="static")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn
    from chatbot.config import get_app_settings
    settings = get_app_settings()
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
