FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Copy only reqs first so this layer caches
COPY requirements.txt /app/requirements.txt

# 2) Install deps (this becomes a cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir -r requirements.txt

# 3) Copy the rest of your app (changes here don't bust the pip layer)
COPY . /app

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
