# ---------- Base ----------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (no sonic-annotator here)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      libsndfile1 ffmpeg ca-certificates wget tar \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Sonic Annotator (static binary) ----------
# v1.7 static build for 64-bit Linux
# Source: Sonic Annotator downloads page
# https://code.soundsoftware.ac.uk/projects/sonic-annotator/files
RUN mkdir -p /opt/sonic && cd /opt/sonic && \
    wget -q https://code.soundsoftware.ac.uk/attachments/download/3134/sonic-annotator-1.7.0-linux64-static.tar.gz && \
    tar -xzf sonic-annotator-1.7.0-linux64-static.tar.gz && \
    cp -v sonic-annotator-*/sonic-annotator /usr/local/bin/sonic-annotator && \
    chmod +x /usr/local/bin/sonic-annotator && \
    rm -rf /opt/sonic

# ---------- Install Vamp plugins (Chordino/NNLS-Chroma) ----------
# We take the plugin-only pack and place .so files into /usr/lib/vamp
# Vamp Plugin Pack: https://code.soundsoftware.ac.uk/projects/vamp-plugin-pack/files
RUN mkdir -p /usr/lib/vamp /tmp/vamp && cd /tmp/vamp && \
    wget -q https://code.soundsoftware.ac.uk/attachments/download/2627/vamp-plugin-pack-plugins-2.8-linux64.tar.gz && \
    tar -xzf vamp-plugin-pack-plugins-2.8-linux64.tar.gz && \
    cp -v *.so /usr/lib/vamp/ || true && \
    rm -rf /tmp/vamp

# Make sure the host can find plugins
ENV VAMP_PATH=/usr/lib/vamp:/usr/local/lib/vamp
ENV SONIC=/usr/local/bin/sonic-annotator

# Sanity check (doesn’t fail build)
RUN sonic-annotator -l || true

# ---------- Python deps ----------
WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Run
ENV PORT=8080
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
