# Base image
FROM python:3.11-slim

# Prevent debconf prompts (silences warnings during apt-get)
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (quiet mode, no extras)
RUN apt-get update -yqq && \
    apt-get install -yqq --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      libgomp1 \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . /app

# Environment vars for Whisper + FastAPI
ENV CT2_FORCE_CPU=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Expose the port Cloud Run will use
EXPOSE 8080

# Start FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
