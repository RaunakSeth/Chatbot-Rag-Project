"""
Admin API routes — onboarding, client management, re-indexing.

Endpoints:
  GET  /admin                → serves admin_ui.html
  GET  /clients              → list all clients
  POST /onboard              → create/update a client (URL + PDF upload)
  DELETE /clients/{id}       → delete a client and all its data
  POST /clients/{id}/reindex → re-crawl/re-process a client's data
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
admin_router = APIRouter(prefix="", tags=["Admin"])

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")  # optional password protection


# ── Simple auth dependency (optional) ────────────────────────────────────────

def _check_admin(x_admin_secret: str | None = None) -> None:
    """If ADMIN_SECRET is set, verify it matches the header."""
    if _ADMIN_SECRET and x_admin_secret != _ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid admin secret.")


# ── Serve admin UI ────────────────────────────────────────────────────────────

@admin_router.get("/admin", include_in_schema=False)
async def serve_admin() -> FileResponse:
    """Serve the admin UI HTML page."""
    return FileResponse(os.path.join(_BASE_DIR, "api", "admin_ui.html"))


# ── List clients ──────────────────────────────────────────────────────────────

@admin_router.get("/clients")
async def list_clients() -> list[dict[str, Any]]:
    """Return a list of all onboarded clients (used by the frontend dropdown)."""
    if os.getenv("SUPABASE_URL", "").strip():
        from chatbot.db import list_client_configs, count_client_chunks
        configs = list_client_configs()
        return [
            {
                **c,
                "chunk_count": count_client_chunks(c["client_id"]),
            }
            for c in configs
        ]
    else:
        # Local mode — scan clients/ directory
        clients_dir = Path(os.getenv("CLIENTS_DIR", "./clients"))
        results = []
        if clients_dir.exists():
            for client_dir in sorted(clients_dir.iterdir()):
                cfg_file = client_dir / "config.yaml"
                if cfg_file.exists():
                    import yaml
                    with cfg_file.open() as f:
                        data = yaml.safe_load(f)
                    results.append({
                        "client_id": data.get("client_id", client_dir.name),
                        "business_name": data.get("business_name", client_dir.name),
                        "tone": data.get("tone", "friendly"),
                        "hardware_tier": data.get("hardware_tier", "A"),
                    })
        return results


# ── Onboard a new client ──────────────────────────────────────────────────────

class OnboardStatus(BaseModel):
    client_id: str
    status: str
    message: str
    chunk_count: int = 0


@admin_router.post("/onboard", response_model=OnboardStatus)
async def onboard_client(
    background_tasks: BackgroundTasks,
    client_id: str = Form(...),
    business_name: str = Form(...),
    tone: str = Form("friendly"),
    hardware_tier: str = Form("A"),
    refusal_message: str = Form("I can only answer questions about {business_name}."),
    website_url: str = Form(""),
    max_pages: int = Form(30),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(64),
    top_k: int = Form(5),
    pdfs: list[UploadFile] = File(default=[]),
) -> OnboardStatus:
    """
    Onboard a new client by:
    1. Saving config to Supabase (or YAML locally)
    2. Crawling the website URL (if provided)
    3. Parsing uploaded PDFs (if any)
    4. Embedding all text and storing in the vector DB
    """
    from chatbot.config import ClientConfig, RetrievalConfig, SessionConfig, save_config
    from chatbot.retrieval.chunker import chunk_text

    # Validate inputs
    if not website_url and not pdfs:
        raise HTTPException(status_code=400, detail="Provide at least a website URL or PDF files.")

    # Save config first
    config = ClientConfig(
        client_id=client_id,
        business_name=business_name,
        hardware_tier=hardware_tier,  # type: ignore
        tone=tone,  # type: ignore
        refusal_message=refusal_message,
        retrieval=RetrievalConfig(top_k=top_k, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    )

    use_supabase = bool(os.getenv("SUPABASE_URL", "").strip())
    if use_supabase:
        from chatbot.db import save_client_config
        save_client_config({
            "client_id": client_id,
            "business_name": business_name,
            "hardware_tier": hardware_tier,
            "tone": tone,
            "refusal_message": refusal_message,
            "retrieval_top_k": top_k,
            "score_threshold": 0.35,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_history_turns": 6,
        })
    else:
        clients_root = os.getenv("CLIENTS_DIR", "./clients")
        save_config(config, clients_root)

    # Save uploaded PDFs to temp dir
    pdf_paths: list[Path] = []
    tmp_dir = tempfile.mkdtemp()
    for pdf in pdfs:
        if pdf.filename and pdf.filename.endswith(".pdf"):
            dest = Path(tmp_dir) / pdf.filename
            dest.write_bytes(await pdf.read())
            pdf_paths.append(dest)

    # Run indexing as a background task so the response returns immediately
    background_tasks.add_task(
        _run_indexing,
        client_id=client_id,
        website_url=website_url,
        max_pages=max_pages,
        pdf_paths=pdf_paths,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_supabase=use_supabase,
        clients_root=os.getenv("CLIENTS_DIR", "./clients"),
    )

    return OnboardStatus(
        client_id=client_id,
        status="indexing",
        message=f"✅ Config saved. Indexing started in background for '{business_name}'.",
    )


async def _run_indexing(
    client_id: str,
    website_url: str,
    max_pages: int,
    pdf_paths: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    use_supabase: bool,
    clients_root: str,
) -> None:
    """Background indexing task — crawl, parse, embed, store."""
    from chatbot.retrieval.chunker import chunk_text
    from chatbot.stages.stage3_retrieval import index_chunks

    all_chunks: list[dict] = []

    # Website crawl
    if website_url:
        try:
            logger.info("[onboard:%s] Crawling %s …", client_id, website_url)
            from onboarding.crawler import crawl_website
            pages = await asyncio.to_thread(crawl_website, website_url, max_pages)
            for page in pages:
                chunks = chunk_text(page["text"], source=f"website:{page['url']}",
                                    chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                all_chunks.extend(chunks)
            logger.info("[onboard:%s] Crawled %d pages → %d chunks.", client_id, len(pages), len(all_chunks))
        except Exception as exc:
            logger.error("[onboard:%s] Crawl failed: %s", client_id, exc)

    # PDF parsing
    for pdf_path in pdf_paths:
        try:
            logger.info("[onboard:%s] Parsing PDF: %s …", client_id, pdf_path.name)
            from onboarding.pdf_parser import parse_pdf_directory
            pages = await asyncio.to_thread(parse_pdf_directory, pdf_path.parent)
            for page in pages:
                chunks = chunk_text(page["text"], source=page["source"],
                                    chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                all_chunks.extend(chunks)
        except Exception as exc:
            logger.error("[onboard:%s] PDF parse failed: %s", client_id, exc)

    if not all_chunks:
        logger.error("[onboard:%s] No chunks generated — nothing to index.", client_id)
        return

    # Deduplicate
    seen: set[str] = set()
    unique = [c for c in all_chunks if not (c["chunk_id"] in seen or seen.add(c["chunk_id"]))]

    # Index
    lancedb_path = str(Path(clients_root) / client_id / "lancedb")
    try:
        count = await asyncio.to_thread(
            index_chunks, unique, lancedb_path, "BAAI/bge-m3", 32, client_id
        )
        logger.info("[onboard:%s] ✅ Indexing complete — %d chunks stored.", client_id, count)
    except Exception as exc:
        logger.error("[onboard:%s] Indexing failed: %s", client_id, exc)


# ── Delete client ─────────────────────────────────────────────────────────────

@admin_router.delete("/clients/{client_id}")
async def delete_client(client_id: str) -> dict:
    """Delete a client's config and all its indexed data."""
    if os.getenv("SUPABASE_URL", "").strip():
        from chatbot.db import delete_client_chunks, get_client
        delete_client_chunks(client_id)
        get_client().table("clients").delete().eq("client_id", client_id).execute()
    else:
        import shutil
        clients_root = Path(os.getenv("CLIENTS_DIR", "./clients"))
        client_dir = clients_root / client_id
        if client_dir.exists():
            shutil.rmtree(client_dir)
    return {"status": "deleted", "client_id": client_id}


# ── Re-index client ───────────────────────────────────────────────────────────

@admin_router.post("/clients/{client_id}/reindex")
async def reindex_client(
    client_id: str,
    background_tasks: BackgroundTasks,
    website_url: str = Form(""),
    max_pages: int = Form(30),
) -> dict:
    """Re-crawl and re-index a client's website."""
    if not website_url:
        raise HTTPException(status_code=400, detail="website_url is required for re-indexing.")

    use_supabase = bool(os.getenv("SUPABASE_URL", "").strip())
    if use_supabase:
        from chatbot.db import delete_client_chunks
        delete_client_chunks(client_id)

    background_tasks.add_task(
        _run_indexing,
        client_id=client_id,
        website_url=website_url,
        max_pages=max_pages,
        pdf_paths=[],
        chunk_size=512,
        chunk_overlap=64,
        use_supabase=use_supabase,
        clients_root=os.getenv("CLIENTS_DIR", "./clients"),
    )
    return {"status": "reindexing", "client_id": client_id}


# ── Bulk Import Excel ─────────────────────────────────────────────────────────

@admin_router.post("/import-clients")
async def import_clients_excel(
    background_tasks: BackgroundTasks,
    excel_file: UploadFile = File(...),
) -> dict:
    """
    Bulk onboard clients via an uploaded .xlsx file.
    Expected columns: client_id, business_name, website_url, tone, hardware_tier
    """
    if not excel_file.filename or not excel_file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported.")
    
    import openpyxl
    
    # Save the file to temp
    tmp_path = Path(tempfile.gettempdir()) / excel_file.filename
    tmp_path.write_bytes(await excel_file.read())
    
    try:
        wb = openpyxl.load_workbook(tmp_path)
        sheet = wb.active
        headers = [str(cell.value).strip() if cell.value else "" for cell in sheet[1]]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read Excel file: {str(e)}")
    
    from chatbot.config import ClientConfig, save_config
    use_supabase = bool(os.getenv("SUPABASE_URL", "").strip())
    
    clients_imported = 0
    
    for row in sheet.iter_rows(min_row=2, values_only=True):
        if not any(row):  # skip empty rows
            continue
            
        row_dict = dict(zip(headers, row))
        client_id = row_dict.get("client_id")
        business_name = row_dict.get("business_name")
        website_url = row_dict.get("website_url", "")
        
        if not client_id or not business_name:
            continue
            
        client_id = str(client_id).strip()
        business_name = str(business_name).strip()
        website_url = str(website_url).strip() if website_url else ""
        
        tone = str(row_dict.get("tone", "friendly")).strip().lower()
        if tone not in ["friendly", "formal", "concise"]:
            tone = "friendly"
            
        hardware_tier = str(row_dict.get("hardware_tier", "A")).strip().upper()
        if hardware_tier not in ["A", "B"]:
            hardware_tier = "A"
        
        # Save config
        config = ClientConfig(
            client_id=client_id,
            business_name=business_name,
            hardware_tier=hardware_tier, # type: ignore
            tone=tone, # type: ignore
        )
        
        if use_supabase:
            from chatbot.db import save_client_config
            save_client_config({
                "client_id": client_id,
                "business_name": business_name,
                "hardware_tier": hardware_tier,
                "tone": tone,
                "refusal_message": config.refusal_message,
                "retrieval_top_k": config.retrieval.top_k,
                "score_threshold": config.retrieval.score_threshold,
                "chunk_size": config.retrieval.chunk_size,
                "chunk_overlap": config.retrieval.chunk_overlap,
                "max_history_turns": config.session.max_history_turns,
            })
        else:
            clients_root = os.getenv("CLIENTS_DIR", "./clients")
            save_config(config, clients_root)
            
        # Trigger background task for each
        if website_url:
            background_tasks.add_task(
                _run_indexing,
                client_id=client_id,
                website_url=website_url,
                max_pages=30,
                pdf_paths=[],
                chunk_size=512,
                chunk_overlap=64,
                use_supabase=use_supabase,
                clients_root=os.getenv("CLIENTS_DIR", "./clients"),
            )
        clients_imported += 1

    return {"status": "success", "message": f"Started import for {clients_imported} clients in background."}

