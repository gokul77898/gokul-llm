FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Ensure log dirs exist
RUN mkdir -p logs/traces data/rag data/replay

EXPOSE 7860

CMD ["uvicorn", "src.inference.server:app", "--host", "0.0.0.0", "--port", "7860"]
