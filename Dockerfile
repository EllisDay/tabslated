# ---------- Base (Ubuntu has sonic-annotator in apt) ----------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps: Python + audio libs + sonic-annotator + wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libsndfile1 ffmpeg ca-certificates \
    sonic-annotator wget \
    && rm -rf /var/lib/apt/lists/*

# Try to install the QM Vamp plugins via apt (package exists on Ubuntu)
# If apt doesn't have it, fall back to downloading the plugin pack tarball.
RUN apt-get update && (apt-get install -y --no-install-recommends qm-vamp-plugins || true) && \
    rm -rf /var/lib/apt/lists/*

# Fallback: fetch the plugin pack tarball and extract to /usr/lib/vamp if qm-vamp-plugins wasn't installed
RUN test -e /usr/lib/vamp || mkdir -p /usr/lib/vamp && \
    if [ ! -f /usr/lib/vamp/qm-vamp-plugins.so ] && [ ! -f /usr/lib/vamp/nnls-chroma.so ]; then \
      echo "QM Vamp plugins not found via apt; fetching plugin pack..." && \
      mkdir -p /tmp/vamp && cd /tmp/vamp && \
      wget -q https://code.soundsoftware.ac.uk/attachments/download/2627/vamp-plugin-pack-plugins-2.8-linux64.tar.gz && \
      tar -xzf vamp-plugin-pack-plugins-2.8-linux64.tar.gz && \
      cp -v *.so /usr/lib/vamp/ || true; \
    fi

# Make sure Sonic can find plugins
ENV VAMP_PATH=/usr/lib/vamp
ENV SONIC=/usr/bin/sonic-annotator

# Upgrade pip (some images ship older pip)
RUN python3 -m pip install --upgrade pip

# ---------- App ----------
WORKDIR /app

# Install Python deps first (for layer caching)
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# (Optional) quick check during build: list available plugins (won't fail the build)
RUN sonic-annotator -l || true

# Expose
ENV PORT=8080
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
