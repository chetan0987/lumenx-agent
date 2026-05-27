FROM python:3.11-slim

WORKDIR /app

# Install build tools (needed for scikit-learn on some platforms)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Ensure runtime directories exist (volumes override at run time)
RUN mkdir -p data wiki/wiki_data models

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default port — Railway overrides via PORT env var
EXPOSE 8001

CMD ["python", "entrypoint.py"]
