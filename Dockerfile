# ---------- Base ----------
FROM ubuntu:22.04

ARG CACHE_BUSTER=3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      libsndfile1 ffmpeg ca-certificates tar xz-utils \
      libfftw3-single3 libsamplerate0 libmad0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Sonic Annotator (vendor either .tar.gz or .AppImage) ----------
WORKDIR /tmp/sonic
# Copy whatever you committed (pattern must match at least one file)
COPY third_party/sonic-annotator* /tmp/sonic/

RUN set -eux; \
    if ls /tmp/sonic/sonic-annotator-*.tar.gz 1>/dev/null 2>&1; then \
        tar -xzf /tmp/sonic/sonic-annotator-*.tar.gz -C /tmp/sonic; \
        cp -v /tmp/sonic/sonic-annotator-*/sonic-annotator /usr/local/bin/sonic-annotator; \
        chmod +x /usr/local/bin/sonic-annotator; \
    elif ls /tmp/sonic/*.AppImage 1>/dev/null 2>&1 || [ -f /tmp/sonic/sonic-annotator ]; then \
        appimg="$(ls /tmp/sonic/*.AppImage 2>/dev/null | head -n1 || true)"; \
        [ -z "$appimg" ] && appimg="/tmp/sonic/sonic-annotator"; \
        chmod +x "$appimg"; \
        "$appimg" --appimage-extract; \
        if [ -f squashfs-root/usr/bin/sonic-annotator ]; then \
            cp -v squashfs-root/usr/bin/sonic-annotator /usr/local/bin/sonic-annotator; \
        else \
            cp -v squashfs-root/AppRun /usr/local/bin/sonic-annotator; \
        fi; \
        chmod +x /usr/local/bin/sonic-annotator; \
        mkdir -p /usr/local/lib; \
        cp -rv squashfs-root/usr/lib/* /usr/local/lib/ 2>/dev/null || true; \
    else \
        echo "No usable Sonic Annotator archive/AppImage found under /tmp/sonic" >&2; exit 1; \
    fi; \
    rm -rf /tmp/sonic

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
