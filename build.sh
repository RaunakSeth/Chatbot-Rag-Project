#!/usr/bin/env bash
# build.sh — runs during Render build phase (before startCommand)
# Downloads the embedding model so runtime never needs internet for it.
set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Pre-downloading embedding model (BAAI/bge-small-en-v1.5)..."
python - <<'PYEOF'
import sys

# Try primary HuggingFace, fall back to mirror
for endpoint in ["https://huggingface.co", "https://hf-mirror.com"]:
    try:
        import os
        os.environ["HF_ENDPOINT"] = endpoint
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            "BAAI/bge-small-en-v1.5",
            ignore_patterns=["*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"Model cached at: {path} (via {endpoint})")
        sys.exit(0)
    except Exception as e:
        print(f"[WARN] Failed via {endpoint}: {e}", flush=True)

print("[ERROR] Could not download model from any source. Build will continue but embeddings will fail at runtime.")
PYEOF

echo "==> Build complete."
