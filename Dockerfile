FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway sets PORT environment variable
EXPOSE 8000

# Use PORT from environment, default to 8000
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}