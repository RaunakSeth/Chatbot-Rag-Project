FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download bge-small embedding model at build time so runtime has zero delay.
# bge-small is ~133MB and uses ~280MB RAM — fits comfortably in 16GB HF Spaces.
RUN python -c "\
from FlagEmbedding import BGEM3FlagModel; \
print('Downloading BAAI/bge-small-en-v1.5 ...'); \
BGEM3FlagModel('BAAI/bge-small-en-v1.5', use_fp16=False); \
print('Model cached.')" || echo "Model download failed — will retry at runtime."

# Copy project files
COPY . .

# HuggingFace Spaces runs containers as user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER 1000

# HuggingFace Spaces routes external traffic to port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server on port 7860
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
