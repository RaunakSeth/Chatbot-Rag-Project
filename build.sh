#!/usr/bin/env bash
# build.sh — runs during Render build phase (before startCommand)
# Downloads the embedding model so runtime never needs internet for it.
set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Pre-downloading embedding model (BAAI/bge-small-en-v1.5)..."
python - <<'PYEOF'
import sys
try:
    from fastembed import TextEmbedding
    print("Downloading fastembed model (BAAI/bge-small-en-v1.5) ...")
    _ = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("Model cached successfully.")
    sys.exit(0)
except Exception as e:
    print(f"[ERROR] Failed to pre-download model: {e}", flush=True)
PYEOF

echo "==> Build complete."
