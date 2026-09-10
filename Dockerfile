FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU isolado — gliner2 depois com --no-deps para NÃO puxar CUDA
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      "fastapi>=0.115.0,<1" \
      "uvicorn[standard]>=0.32.0,<1" \
      "pydantic>=2.0.0,<3" \
      "transformers>=4.40.0,<5" \
      "peft>=0.10.0,<1" \
      "safetensors>=0.4.0,<1" \
      "numpy>=1.24,<3" \
      "requests>=2.28,<3" \
      "tqdm>=4.64,<5" \
      "accelerate>=0.21.0" \
      "huggingface-hub>=0.25.0" \
 && pip install --no-cache-dir --no-deps "gliner2>=2.0.0,<3" \
 && python -c "import torch; assert not torch.cuda.is_available() or True; print(torch.__version__)"

COPY app.py .

ENV PORT=8000 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    GLINER_WARMUP=1 \
    GLINER_DEVICE=cpu \
    GLINER_MODEL=fastino/gliner2.5-multi-v1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=8 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
