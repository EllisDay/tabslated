FROM python:3.11-slim

# System deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget ca-certificates libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Sonic Annotator + Chordino (Linux)
RUN mkdir -p /opt/sonic && cd /opt/sonic && \
    wget -q https://code.soundsoftware.ac.uk/attachments/download/2625/sonic-annotator_1.4_amd64.deb && \
    apt-get update && apt-get install -y ./sonic-annotator_1.4_amd64.deb && \
    rm -f sonic-annotator_1.4_amd64.deb

# Vamp plugins (Chordino/nnls-chroma)
RUN mkdir -p "/usr/local/lib/vamp" && \
    cd /usr/local/lib/vamp && \
    wget -q https://code.soundsoftware.ac.uk/attachments/download/2627/vamp-plugin-pack-plugins-2.8-linux64.tar.gz && \
    tar -xzf vamp-plugin-pack-plugins-2.8-linux64.tar.gz && \
    rm vamp-plugin-pack-plugins-2.8-linux64.tar.gz

ENV VAMP_PATH=/usr/local/lib/vamp
ENV SONIC=/usr/bin/sonic-annotator

# App
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Uvicorn
ENV PORT=8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
