# ── Base Image ──────────────────────────────────────────────────────────────
FROM python:3.12.7-slim

# ── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install \
        mlflow==2.19.0 \
        scikit-learn>=1.3.0 \
        pandas>=2.0.0 \
        numpy>=1.24.0 \
        matplotlib>=3.7.0 \
        dagshub>=0.3.0

# ── Copy project files ───────────────────────────────────────────────────────
COPY MLProject/ /app/

# ── Default port untuk mlflow model serve ────────────────────────────────────
EXPOSE 5002

# ── Entrypoint ───────────────────────────────────────────────────────────────
CMD ["python", "modelling.py"]