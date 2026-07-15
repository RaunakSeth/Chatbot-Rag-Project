"""
Client onboarding CLI.

Steps (per spec §7):
  1. Crawl website URL (optional)
  2. Parse PDFs from a directory (optional)
  3. Chunk all text
  4. Embed chunks → LanceDB
  5. Write clients/<client_id>/config.yaml

Usage:
  python -m onboarding.onboard \\
    --client-id acme_corp \\
    --business-name "Acme Corp" \\
    --url https://acmecorp.com \\
    --pdf-dir ./docs/acme \\
    --tone friendly \\
    --tier A
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from chatbot.config import ClientConfig, RetrievalConfig, SessionConfig, save_config
from chatbot.retrieval.chunker import chunk_text
from chatbot.stages.stage3_retrieval import index_chunks
from onboarding.crawler import crawl_website
from onboarding.pdf_parser import parse_pdf_directory

app = typer.Typer(help="Onboard a new client into the Business AI Chatbot system.")
console = Console()


@app.command()
def main(
    client_id: str = typer.Option(..., help="Unique identifier for this client (e.g. acme_corp)"),
    business_name: str = typer.Option(..., help="Human-readable business name"),
    url: str | None = typer.Option(None, help="Website URL to crawl"),
    pdf_dir: Path | None = typer.Option(None, help="Directory containing PDF files to ingest"),
    tone: str = typer.Option("friendly", help="Response tone: friendly | formal | concise"),
    tier: str = typer.Option("A", help="Hardware tier: A (CPU) | B (GPU)"),
    clients_root: str = typer.Option("./clients", help="Root directory for client data"),
    max_pages: int = typer.Option(50, help="Max website pages to crawl"),
    chunk_size: int = typer.Option(512, help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(64, help="Chunk overlap in characters"),
    top_k: int = typer.Option(5, help="Retrieval top-k"),
    refusal_message: str = typer.Option(
        "I can only answer questions about {business_name}.",
        help="Refusal message template (use {business_name} as placeholder)",
    ),
    max_history_turns: int = typer.Option(6, help="Max conversation history turns"),
) -> None:
    """End-to-end onboarding: crawl → parse → embed → index → save config."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    console.print(
        Panel.fit(
            f"[bold cyan]Onboarding:[/bold cyan] [green]{business_name}[/green] "
            f"([dim]{client_id}[/dim])\n"
            f"Tier: [yellow]{tier}[/yellow]  |  Tone: [yellow]{tone}[/yellow]",
            title="🤖 Business AI Chatbot",
        )
    )

    if not url and not pdf_dir:
        console.print("[red]Error:[/red] Provide at least --url or --pdf-dir.")
        raise typer.Exit(code=1)

    all_chunks: list[dict] = []

    # ── Step 1: Website crawl ─────────────────────────────────────────────────
    if url:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            console=console, transient=True
        ) as progress:
            progress.add_task(f"Crawling {url} (max {max_pages} pages)…", total=None)
            pages = crawl_website(url, max_pages=max_pages)

        console.print(f"  ✅ Crawled [bold]{len(pages)}[/bold] pages from {url}")
        for page in pages:
            chunks = chunk_text(
                page["text"],
                source=f"website:{page['url']}",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(chunks)
        console.print(f"     → {len(all_chunks)} chunks from website.")

    # ── Step 2: PDF parsing ───────────────────────────────────────────────────
    if pdf_dir:
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            console.print(f"[red]PDF directory not found:[/red] {pdf_dir}")
            raise typer.Exit(code=1)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            console=console, transient=True
        ) as progress:
            progress.add_task(f"Parsing PDFs in {pdf_dir}…", total=None)
            pdf_pages = parse_pdf_directory(pdf_dir)

        console.print(f"  ✅ Parsed [bold]{len(pdf_pages)}[/bold] pages from PDFs in {pdf_dir}")
        pre_count = len(all_chunks)
        for pdf_page in pdf_pages:
            chunks = chunk_text(
                pdf_page["text"],
                source=pdf_page["source"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(chunks)
        console.print(f"     → {len(all_chunks) - pre_count} chunks from PDFs.")

    if not all_chunks:
        console.print("[red]No chunks generated — nothing to index. Aborting.[/red]")
        raise typer.Exit(code=1)

    # ── Step 3: Deduplicate chunks ────────────────────────────────────────────
    seen_ids: set[str] = set()
    unique_chunks = []
    for c in all_chunks:
        if c["chunk_id"] not in seen_ids:
            seen_ids.add(c["chunk_id"])
            unique_chunks.append(c)
    console.print(
        f"  📦 Total unique chunks: [bold]{len(unique_chunks)}[/bold] "
        f"({len(all_chunks) - len(unique_chunks)} duplicates removed)"
    )

    # ── Step 4: Embed + index ────────────────────────────────────────────────
    lancedb_path = str(Path(clients_root) / client_id / "lancedb")
    console.print(f"  🔍 Embedding and indexing into LanceDB at [dim]{lancedb_path}[/dim]…")
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        console=console, transient=True
    ) as progress:
        progress.add_task("Embedding chunks with BGE-M3…", total=None)
        total_rows = index_chunks(unique_chunks, lancedb_path)
    console.print(f"  ✅ LanceDB index built: [bold]{total_rows}[/bold] rows total.")

    # ── Step 5: Write config ──────────────────────────────────────────────────
    config = ClientConfig(
        client_id=client_id,
        business_name=business_name,
        hardware_tier=tier,  # type: ignore[arg-type]
        tone=tone,  # type: ignore[arg-type]
        refusal_message=refusal_message,
        retrieval=RetrievalConfig(
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        session=SessionConfig(max_history_turns=max_history_turns),
    )
    config_path = save_config(config, clients_root)
    console.print(f"  ✅ Config written to [dim]{config_path}[/dim]")

    console.print(
        Panel.fit(
            f"[bold green]Onboarding complete![/bold green]\n\n"
            f"Client ID : [cyan]{client_id}[/cyan]\n"
            f"Config    : [dim]{config_path}[/dim]\n"
            f"LanceDB   : [dim]{lancedb_path}[/dim]\n\n"
            f"Start the API server:\n"
            f"  [bold]uvicorn api.app:app --host 0.0.0.0 --port 8000[/bold]\n\n"
            f"Then chat:\n"
            f'  curl -X POST http://localhost:8000/chat \\\n'
            f'    -H "Content-Type: application/json" \\\n'
            f'    -d \'{{"client_id":"{client_id}","session_id":"s1","message":"Hello"}}\'',
            title="✅ Done",
        )
    )


if __name__ == "__main__":
    app()
