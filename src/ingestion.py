"""Download STVR archive episodes and extract 16 kHz mono WAV audio."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

import yt_dlp

from config import Settings, get_settings
from src.gcs import upload_file

logger = logging.getLogger(__name__)

_EPISODE_ID_RE = re.compile(r"/archiv/(?:\d+/)?(\d+)/?(?:[#?]|$)")


def episode_id_from_url(url: str) -> str:
    match = _EPISODE_ID_RE.search(url)
    if not match:
        raise ValueError(f"Cannot parse STVR episode id from URL: {url}")
    return match.group(1)


def download_episode(url: str, settings: Settings | None = None) -> Path:
    """Download best audio (or lowest video) via yt-dlp's stvr extractor.

    Prefers audio-only streams — full HD video is unnecessary for transcription
    and can be multi-GB for a 1h episode.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    ep_id = episode_id_from_url(url)
    outtmpl = str(settings.raw_dir / f"{ep_id}.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        # Prefer audio-only; fall back to worst video+audio if no audio stream.
        "format": "bestaudio/worstaudio/worst",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }

    logger.info("Downloading (audio-preferred) %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = Path(ydl.prepare_filename(info))
        if not filepath.exists():
            candidates = [
                p
                for p in settings.raw_dir.glob(f"{ep_id}.*")
                if p.suffix not in {".part", ".ytdl"} and ".part" not in p.name
            ]
            if not candidates:
                raise FileNotFoundError(f"Download finished but no file for episode {ep_id}")
            filepath = max(candidates, key=lambda p: p.stat().st_mtime)

    logger.info("Saved media: %s (%.1f MB)", filepath, filepath.stat().st_size / 1e6)
    upload_file(filepath, object_name=f"raw/{filepath.name}", settings=settings)
    return filepath


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
    """Full ingestion: download + audio extract. Returns (episode_id, video, audio)."""
    settings = settings or get_settings()
    ep_id = episode_id_from_url(url)
    video = download_episode(url, settings=settings)
    audio = extract_audio(video, settings=settings)
    return ep_id, video, audio
