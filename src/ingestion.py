"""Download debate media via registered sources and extract 16 kHz mono WAV."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from config import Settings, get_settings
from src.gcs import upload_file
from src.sources import resolve

logger = logging.getLogger(__name__)


def extract_audio(video_path: Path, settings: Settings | None = None) -> Path:
    """Extract 16 kHz mono PCM WAV with ffmpeg (required by Whisper/Pyannote)."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    audio_path = settings.audio_dir / f"{video_path.stem}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    logger.info("Extracting audio -> %s", audio_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")

    upload_file(audio_path, object_name=f"audio/{audio_path.name}", settings=settings)
    return audio_path


def ingest(url: str, settings: Settings | None = None) -> tuple[str, Path, Path]:
    """Full ingestion: resolve source + download + audio extract.

    Returns (episode_id, media_path, audio_path).
    """
    settings = settings or get_settings()
    source = resolve(url)
    ep_id = source.episode_id(url)
    media = source.download(url, settings=settings)
    audio = extract_audio(media, settings=settings)
    return ep_id, media, audio
