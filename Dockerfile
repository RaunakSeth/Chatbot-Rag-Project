FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download fastembed model at build time so runtime has zero delay.
# fastembed uses ONNX Runtime which takes very little RAM compared to PyTorch.
RUN python -c "\
from fastembed import TextEmbedding; \
print('Downloading fastembed model (BAAI/bge-small-en-v1.5) ...'); \
_ = TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
print('Model cached.')" || echo "Model download failed — will retry at runtime."

# Copy project files
COPY . .

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server on port 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
