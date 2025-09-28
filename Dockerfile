# ---------- Base (Ubuntu 22.04) ----------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Enable 'universe' and install system deps + Sonic Annotator
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common gnupg ca-certificates \
  && add-apt-repository -y universe \
  && apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      libsndfile1 ffmpeg wget \
      sonic-annotator qm-vamp-plugins \
  && rm -rf /var/lib/apt/lists/*

# Ensure plugin path and sonic path are known
ENV VAMP_PATH=/usr/lib/vamp:/usr/local/lib/vamp
ENV SONIC=/usr/bin/sonic-annotator

# Fallback: if nnls-chroma/chordino not present from apt, fetch plugin pack
# (This downloads the pack and copies *.so into /usr/lib/vamp)
RUN set -e; \
  if ! ls /usr/lib/vamp 2>/dev/null | grep -E 'nnls|chordino' >/dev/null; then \
    echo "Chordino not found via apt; fetching Vamp plugin pack..."; \
    mkdir -p /tmp/vamp && cd /tmp/vamp; \
    wget -q https://code.soundsoftware.ac.uk/attachments/download/2627/vamp-plugin-pack-plugins-2.8-linux64.tar.gz; \
    tar -xzf vamp-plugin-pack-plugins-2.8-linux64.tar.gz; \
    mkdir -p /usr/lib/vamp; \
    cp -v *.so /usr/lib/vamp/ || true; \
  fi

# Quick sanity check: list plugins (does not fail build)
RUN sonic-annotator -l || true

# ---------- Python deps ----------
WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip \
 && pip3 install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Run app
ENV PORT=8080
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
