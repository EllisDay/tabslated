# ---------- Base ----------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (no network fetch for sonic/qm; we vendor those)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      libsndfile1 ffmpeg ca-certificates tar \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy vendored tarballs ----------
# You already committed these files into third_party/
WORKDIR /tmp
COPY third_party/sonic-annotator-*.tar.gz /tmp/
COPY third_party/*vamp*plugins*linux64*.tar.gz /tmp/

# Install Sonic Annotator (static)
RUN tar -xzf /tmp/sonic-annotator-*.tar.gz -C /tmp && \
    cp -v /tmp/sonic-annotator-*/sonic-annotator /usr/local/bin/sonic-annotator && \
    chmod +x /usr/local/bin/sonic-annotator && \
    rm -rf /tmp/sonic-annotator-*

# Install Vamp plugins (.so) → /usr/local/lib/vamp
RUN mkdir -p /usr/local/lib/vamp && \
    for f in /tmp/*vamp*plugins*linux64*.tar.gz; do tar -xzf "$f" -C /tmp; done && \
    find /tmp -name "*.so" -exec cp -v {} /usr/local/lib/vamp/ \; && \
    rm -rf /tmp/*vamp*plugins* /tmp/*.tar.gz

# Make sure host can find plugins & sonic path
ENV VAMP_PATH=/usr/local/lib/vamp
ENV SONIC=/usr/local/bin/sonic-annotator

# Quick listing (doesn't fail build)
RUN sonic-annotator -l || true

# ---------- Python deps ----------
WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# App code
COPY . .

# FastAPI on uvicorn
ENV PORT=8080
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
