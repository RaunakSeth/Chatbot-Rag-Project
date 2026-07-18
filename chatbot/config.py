"""
Client configuration schema and loader.

Each client has a YAML config file at:
  clients/<client_id>/config.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


# ── Sub-models ────────────────────────────────────────────────────────────────

class RetrievalConfig(BaseModel):
    top_k: int = Field(5, ge=1, le=20)
    score_threshold: float = Field(0.35, ge=0.0, le=1.0)
    chunk_size: int = Field(512, ge=64, le=8192)
    chunk_overlap: int = Field(64, ge=0, le=512)


class SessionConfig(BaseModel):
    max_history_turns: int = Field(6, ge=1, le=20)


# ── Root config ───────────────────────────────────────────────────────────────

class ClientConfig(BaseModel):
    client_id: str
    business_name: str
    hardware_tier: Literal["A", "B"] = "A"
    tone: Literal["friendly", "formal", "concise"] = "friendly"
    website_url: str = ""
    is_active_demo: bool = False
    refusal_message: str = "I can only answer questions about {business_name}."
    prohibited_topics: list[str] = Field(default_factory=list)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)

    @field_validator("refusal_message", mode="before")
    @classmethod
    def _inject_business_name(cls, v: str, info):
        # Delay resolution — we do it at load time after all fields are set.
        return v

    def resolved_refusal(self) -> str:
        """Return refusal message with {business_name} filled in."""
        return self.refusal_message.format(business_name=self.business_name)

    # ── Model selection helpers ────────────────────────────────────────────────

    @property
    def classifier_model(self) -> str:
        """Fast model for scope classification (Stage 1)."""
        return "llama-3.1-8b-instant"

    @property
    def safety_model(self) -> str:
        """Fast model for safety guardrail (Stage 2)."""
        return "llama-3.1-8b-instant"

    @property
    def embedding_model(self) -> str:
        """Embedding model — same for both tiers."""
        return "BAAI/bge-small-en-v1.5"

    @property
    def generation_model(self) -> str:
        """Quality model for generation (Stage 4)."""
        if self.hardware_tier == "B":
            return "llama-3.3-70b-versatile"
        return "llama-3.3-70b-versatile"  # same — Groq free tier handles both

    # Keep backward-compat alias
    @property
    def generation_model_ollama(self) -> str:
        return self.generation_model

    # ── LanceDB path ──────────────────────────────────────────────────────────

    def lancedb_path(self, clients_root: str | Path = "./clients") -> Path:
        return Path(clients_root) / self.client_id / "lancedb"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_config(client_id: str, clients_root: str | Path = "./clients") -> ClientConfig:
    """
    Load and validate a client's config.

    Cloud mode  (SUPABASE_URL set): reads from Supabase `clients` table.
    Local mode  (no SUPABASE_URL) : reads from clients/<id>/config.yaml.
    """
    import os
    if os.getenv("SUPABASE_URL", "").strip():
        # ── Supabase path ──────────────────────────────────────────────────────
        from chatbot.db import load_client_config as _sb_load
        data = _sb_load(client_id)
        if data is None:
            raise FileNotFoundError(
                f"No config found for client '{client_id}' in Supabase. "
                "Run onboarding first."
            )
        # Map DB columns → ClientConfig fields
        return ClientConfig(
            client_id=data["client_id"],
            business_name=data["business_name"],
            hardware_tier=data.get("hardware_tier", "A"),
            tone=data.get("tone", "friendly"),
            website_url=data.get("website_url", ""),
            is_active_demo=data.get("is_active_demo", False),
            refusal_message=data.get("refusal_message", "I can only answer questions about {business_name}."),
            retrieval=RetrievalConfig(
                top_k=data.get("retrieval_top_k", 5),
                score_threshold=data.get("score_threshold", 0.35),
                chunk_size=data.get("chunk_size", 512),
                chunk_overlap=data.get("chunk_overlap", 64),
            ),
            session=SessionConfig(max_history_turns=data.get("max_history_turns", 6)),
        )
    else:
        # ── Local YAML path ────────────────────────────────────────────────────
        config_path = Path(clients_root) / client_id / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No config found for client '{client_id}' at {config_path}. "
                "Run the onboarding command first."
            )
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return ClientConfig.model_validate(raw)



def save_config(config: ClientConfig, clients_root: str | Path = "./clients") -> Path:
    """Serialize and write a client config to disk."""
    out_path = Path(clients_root) / config.client_id / "config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(), fh, allow_unicode=True, sort_keys=False)
    return out_path


# ── Environment-level settings ────────────────────────────────────────────────

class AppSettings(BaseModel):
    hardware_tier: Literal["A", "B"] = "A"
    ollama_base_url: str = "http://localhost:11434"
    clients_dir: str = "./clients"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_prefix": ""}  # reads from env directly


def get_app_settings() -> AppSettings:
    from dotenv import load_dotenv
    load_dotenv()
    return AppSettings(
        hardware_tier=os.getenv("HARDWARE_TIER", "A"),  # type: ignore[arg-type]
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        clients_dir=os.getenv("CLIENTS_DIR", "./clients"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
    )
