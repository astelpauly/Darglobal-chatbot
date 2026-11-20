FROM python:3.11-slim

# system deps for onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*

# Hugging Face model cache: MUST be outside /tmp, because Cloud Run mounts
# /tmp as empty tmpfs at runtime and wipes any build-time downloads.
# PYTHONUNBUFFERED=1 forces print() to flush immediately so logs show up live.
ENV HF_HOME=/app/hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    ORT_THREADS=2 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY embed.py retrieve.py rerank.py app.py index.html ./
COPY data/processed/darglobal/chunks.jsonl ./data/processed/darglobal/chunks.jsonl

# Pre-download the embedding + reranker models at build time
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('BAAI/bge-m3', allow_patterns=['onnx/*']); \
    snapshot_download('jinaai/jina-reranker-v2-base-multilingual', \
        allow_patterns=['onnx/model.onnx','onnx/model.onnx_data','tokenizer.json','*.json'])"

# Build the ChromaDB index fresh with the current chromadb version
# (the local chroma_db was created with 0.5.23; 1.x can't read it).
# Takes ~5-8 min at build time but means zero migration pain at runtime.
RUN python embed.py

RUN chmod -R 777 /app

EXPOSE 8080

# Cloud Run sends requests to $PORT (8080 by default)
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
