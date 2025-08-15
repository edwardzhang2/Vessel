# Dockerfile (GPU)
# Base: PyTorch with CUDA so torch+CUDA are preinstalled
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# --- Env & caching ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache \
    HUGGINGFACE_HUB_CACHE=/cache

WORKDIR /app

# Install system deps first (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (to leverage Docker layer cache)
COPY requirements.txt /app/requirements.txt

# Install Python deps (pin transformers to >=4.44.* which has AutoModelForVision2Seq)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the app
COPY . /app

# Create writable cache dirs (in case you forget to mount /cache)
RUN mkdir -p /cache && chmod -R 777 /cache

EXPOSE 8080
# Start the API
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
