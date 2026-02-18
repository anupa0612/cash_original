# ─── Stage 1: build dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed by pdfplumber / Pillow / lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpoppler-cpp-dev \
        poppler-utils \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim

# Minimal runtime libs (pdfplumber needs poppler at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Ensure the brokers package is importable
RUN test -f brokers/__init__.py || touch brokers/__init__.py

# Create a writable temp directory for session pickles
RUN mkdir -p /tmp/cash_recon_sessions /tmp/cash_recon_data \
    && chmod 777 /tmp/cash_recon_sessions /tmp/cash_recon_data

# Back4App / most PaaS providers set PORT at runtime
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

# Use gunicorn in production; worker count tuned for low-memory containers
CMD ["sh", "-c", \
     "gunicorn app_with_mongodb:app \
        --bind 0.0.0.0:${PORT} \
        --workers 2 \
        --threads 4 \
        --timeout 120 \
        --access-logfile - \
        --error-logfile -"]
