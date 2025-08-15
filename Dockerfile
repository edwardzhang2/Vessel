# --- GPU-enabled base with PyTorch + CUDA already installed ---
# If this tag doesn't match your cluster drivers, switch to a nearby CUDA tag.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# --- Environment: make HF cache writable; speed up Python; saner pip ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/app/hf_cache \
    HF_HOME=/app/hf_cache \
    HUGGINGFACE_HUB_CACHE=/app/hf_cache \
    SKIP_LLAMA=0

# Create/writeable cache directory
RUN mkdir -p /app/hf_cache

# System deps you actually need (keep this minimal to speed builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# --- Install Python deps (force upgrade to avoid old cached versions) ---
# Copy ONLY requirements first to leverage layer caching on future builds
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade --no-cache-dir -r /app/requirements.txt

# --- Copy your application code ---
COPY . /app

# Expose API port
EXPOSE 8080

# Default command: run FastAPI server
# (If you prefer to override in HPCaaS UI, you can leave it.)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
