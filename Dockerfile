# Cloud Run / local container image for STVR Debate Analyzer.
# Heavy ML (Whisper/Pyannote) prefers a GPU image; CPU works but is slow.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py main.py ./
COPY src/ ./src/

RUN mkdir -p data/raw data/audio data/transcripts data/reports data/posts

# Non-secret defaults. Provide GCP_PROJECT_ID (and optionally
# GOOGLE_APPLICATION_CREDENTIALS / HUGGINGFACE_TOKEN) via the Cloud Run job
# env / secrets — never bake project ids or tokens into the image.
ENV PYTHONUNBUFFERED=1 \
    WHISPER_DEVICE=cpu \
    USE_GCS=false \
    GCP_LOCATION=europe-west1 \
    GEMINI_MODEL=gemini-2.5-pro

# Run as a Cloud Run job: the container executes the CLI once and exits.
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
