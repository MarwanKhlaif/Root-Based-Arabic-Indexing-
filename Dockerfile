# ==============================================================================
# Dockerfile - Arabic Full-Text Indexing & Search Web Interface (PostgreSQL 16)
# Multi-system Arabic IR Research Dashboard (S0–S5)
# ==============================================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications for layer caching
COPY requirements.txt pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords data required by ISRI and S1
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Copy application source code, resources, queries, and evaluation results
COPY src/ /app/src/
COPY queries/ /app/queries/
COPY results/ /app/results/
COPY tsearch_data/ /app/tsearch_data/
COPY data/khoja_roots.txt /app/data/khoja_roots.txt

# Expose HTTP port
EXPOSE 8000

# Health check verification
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Launch FastAPI web server with dynamic PORT support
CMD ["sh", "-c", "exec uvicorn src.web.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
