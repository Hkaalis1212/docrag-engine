# Shared base image for all three docker-compose services (ingest, api,
# dashboard) - each overrides CMD via its own `command:` in docker-compose.yml.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Overridden by docker-compose per-service; this default just seeds the
# sample corpus so a bare `docker build && docker run` does something visible.
CMD ["python", "-m", "scripts.seed", "--corpus", "corpus/throttleguard-docs", "--strategy", "structure"]
