# Phase 5 — not wired to a running service yet (API/dashboard land in
# Phases 2-5). Kept here now so the ingestion pipeline is containerizable
# from day one rather than retrofitted later.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Overridden by docker-compose per-service; this default just seeds the
# sample corpus so a bare `docker build && docker run` does something visible.
CMD ["python", "-m", "scripts.seed", "--corpus", "corpus/throttleguard-docs", "--strategy", "structure"]
