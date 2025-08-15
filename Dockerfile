# 1. Use a PyTorch CUDA image so torch + CUDA are already installed
#    Pick a CUDA version compatible with your HPCaaS drivers
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

# 2. Env vars to fix Hugging Face cache write errors
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/cache \
    HF_HOME=/cache \
    HUGGINGFACE_HUB_CACHE=/cache

WORKDIR /app

# 3. Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

# 4. Copy only requirements first (keeps dependency layer cached)
COPY requirements.txt /app/requirements.txt

# 5. Install Python packages (skip torch here—it's in the base image)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your code
COPY . /app

# 7. Set entrypoint & expose port
ENTRYPOINT ["/usr/bin/tini","--"]
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
