"""
One-time migration script.
Reads existing local LanceDB chunks and uploads them to Supabase pgvector.

Usage:
  1. Ensure SUPABASE_URL and SUPABASE_KEY are in your .env file
  2. Run: python -m chatbot.db.migrate_lancedb
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not os.getenv("SUPABASE_URL"):
        logger.error("SUPABASE_URL not set in .env. Run this script when setting up cloud DB.")
        sys.exit(1)

    clients_dir = Path(os.getenv("CLIENTS_DIR", "./clients"))
    if not clients_dir.exists():
        logger.info("No clients directory found. Nothing to migrate.")
        return

    from chatbot.db import upsert_chunks, save_client_config
    import yaml
    
    total_chunks = 0
    total_clients = 0

    for client_dir in sorted(clients_dir.iterdir()):
        if not client_dir.is_dir():
            continue

        client_id = client_dir.name
        
        # Migrate Config
        cfg_path = client_dir / "config.yaml"
        if cfg_path.exists():
            with cfg_path.open() as f:
                data = yaml.safe_load(f)
            
            save_client_config({
                "client_id": client_id,
                "business_name": data.get("business_name", client_id),
                "hardware_tier": data.get("hardware_tier", "A"),
                "tone": data.get("tone", "friendly"),
                "refusal_message": data.get("refusal_message", "I can only answer questions about {business_name}."),
                "retrieval_top_k": data.get("retrieval", {}).get("top_k", 5),
                "score_threshold": data.get("retrieval", {}).get("score_threshold", 0.35),
                "chunk_size": data.get("retrieval", {}).get("chunk_size", 512),
                "chunk_overlap": data.get("retrieval", {}).get("chunk_overlap", 64),
                "max_history_turns": data.get("session", {}).get("max_history_turns", 6),
            })
            logger.info("✅ Migrated config for: %s", client_id)
            total_clients += 1
        
        # Migrate LanceDB chunks
        lancedb_path = client_dir / "lancedb"
        if lancedb_path.exists():
            import lancedb
            try:
                db = lancedb.connect(str(lancedb_path))
                if "chunks" in db.table_names():
                    table = db.open_table("chunks")
                    df = table.to_pandas()
                    
                    if not df.empty:
                        # Convert pandas DataFrame rows to dicts
                        rows_to_upsert = []
                        for _, row in df.iterrows():
                            # pyarrow vector is a numpy array in pandas, convert to list
                            embedding = row["vector"].tolist() if hasattr(row["vector"], "tolist") else list(row["vector"])
                            rows_to_upsert.append({
                                "chunk_id": row["chunk_id"],
                                "text": row["text"],
                                "source": row.get("source", ""),
                                "embedding": embedding,
                            })
                        
                        count = upsert_chunks(client_id, rows_to_upsert)
                        logger.info("✅ Migrated %d chunks for: %s", count, client_id)
                        total_chunks += count
            except Exception as e:
                logger.error("❌ Failed to migrate LanceDB for %s: %s", client_id, e)

    logger.info("---")
    logger.info("Migration complete! Migrated %d clients and %d chunks total.", total_clients, total_chunks)

if __name__ == "__main__":
    main()
