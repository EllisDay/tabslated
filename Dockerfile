# ---------- Base ----------
FROM ubuntu:22.04

ARG CACHE_BUSTER=4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      libsndfile1 ffmpeg ca-certificates tar xz-utils \
      libfftw3-single3 libsamplerate0 libmad0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Sonic Annotator (tar.gz or AppImage → always ends as real ELF) ----------
WORKDIR /tmp/sonic
# Copy whatever you committed (tar.gz, AppImage, or plain binary)
COPY third_party/sonic-annotator* /tmp/sonic/

RUN set -eux; \
    # 1) Expand any tarballs we copied
    for f in /tmp/sonic/*.tar.gz 2>/dev/null; do \
      [ -f "$f" ] && tar -xzf "$f" -C /tmp/sonic || true; \
    done; \
    # 2) Pick a candidate file named sonic-annotator* (first match)
    cand="$(find /tmp/sonic -maxdepth 3 -type f -name 'sonic-annotator*' | head -n1 || true)"; \
    if [ -z "$cand" ]; then echo "No sonic-annotator artifact found" >&2; exit 1; fi; \
    cp -v "$cand" /usr/local/bin/sonic-annotator; chmod +x /usr/local/bin/sonic-annotator; \
    # 3) Try to extract if it's an AppImage; if extraction works, replace with the inner ELF
    mkdir -p /tmp/sonic/extract; cd /tmp/sonic/extract; \
    if /usr/local/bin/sonic-annotator --appimage-extract >/dev/null 2>&1; then \
      if [ -f squashfs-root/usr/bin/sonic-annotator ]; then \
        cp -v squashfs-root/usr/bin/sonic-annotator /usr/local/bin/sonic-annotator; \
      else \
        cp -v squashfs-root/AppRun /usr/local/bin/sonic-annotator; \
      fi; \
      chmod +x /usr/local/bin/sonic-annotator; \
    fi; \
    cd /; rm -rf /tmp/sonic

# Optional libs Sonic may want (harmless if not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libfftw3-single3 libsamplerate0 libmad0 \
    && rm -rf /var/lib/apt/lists/*


# ---------- Vamp plugins ----------
WORKDIR /tmp/vamp
COPY third_party/*vamp*plugins*linux64*.tar.gz /tmp/vamp/

RUN set -eux; \
    mkdir -p /usr/local/lib/vamp; \
    for f in /tmp/vamp/*.tar.gz; do tar -xzf "$f" -C /tmp/vamp; done; \
    find /tmp/vamp -name "*.so" -exec cp -v {} /usr/local/lib/vamp/ \; || true; \
    rm -rf /tmp/vamp

ENV VAMP_PATH=/usr/local/lib/vamp
ENV SONIC=/usr/local/bin/sonic-annotator

# Helpful during build; won’t fail the image if it prints warnings
RUN /usr/local/bin/sonic-annotator -l || true

# ---------- Python deps ----------
WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# App code
COPY . .

ENV PORT=8080
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
