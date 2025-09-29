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
WORKDIR /tmp/sonic

# Install Sonic Annotator (static)
COPY third_party/sonic-annotator* /tmp/sonic/

# Install Sonic Annotator → /usr/local/bin/sonic-annotator
RUN set -eux; \
    # Case 1: tarball containing a 'sonic-annotator' binary in a folder
    if ls /tmp/sonic/sonic-annotator-*.tar.gz 1>/dev/null 2>&1; then \
        tar -xzf /tmp/sonic/sonic-annotator-*.tar.gz -C /tmp/sonic; \
        cp -v /tmp/sonic/sonic-annotator-*/sonic-annotator /usr/local/bin/sonic-annotator; \
    # Case 2: an AppImage (maybe even inside a tar you already unpacked)
    elif ls /tmp/sonic/*.AppImage 1>/dev/null 2>&1 || ls /tmp/sonic/sonic-annotator 1>/dev/null 2>&1; then \
        appimg="$(ls /tmp/sonic/*.AppImage 2>/dev/null | head -n1 || true)"; \
        # Some bundles name the AppImage just 'sonic-annotator' with no extension
        [ -z "$appimg" ] && appimg="/tmp/sonic/sonic-annotator"; \
        chmod +x "$appimg"; \
        "$appimg" --appimage-extract; \
        # copy the real binary and any libs the AppImage includes
        cp -v squashfs-root/usr/bin/sonic-annotator /usr/local/bin/sonic-annotator || cp -v squashfs-root/AppRun /usr/local/bin/sonic-annotator; \
        chmod +x /usr/local/bin/sonic-annotator; \
        mkdir -p /usr/local/lib; \
        cp -rv squashfs-root/usr/lib/* /usr/local/lib/ 2>/dev/null || true; \
    else \
        echo "Could not find a usable Sonic Annotator archive/AppImage" >&2; exit 1; \
    fi; \
    rm -rf /tmp/sonic

# Install Vamp plugins (.so) → /usr/local/lib/vamp
COPY third_party/*vamp*plugins*linux64*.tar.gz /tmp/

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
