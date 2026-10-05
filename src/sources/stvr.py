"""STVR archive episode source (yt-dlp)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yt_dlp

from config import Settings, get_settings
from src.gcs import upload_file
from src.sources.base import MediaSource, normalize_host

logger = logging.getLogger(__name__)

_EPISODE_ID_RE = re.compile(r"/archiv/(?:\d+/)?(\d+)/?(?:[#?]|$)")


class StvrSource(MediaSource):
    name = "stvr"

    def matches(self, url: str) -> bool:
        return normalize_host(url) == "stvr.sk"

    def episode_id(self, url: str) -> str:
        match = _EPISODE_ID_RE.search(url)
        if not match:
            raise ValueError(f"Cannot parse STVR episode id from URL: {url}")
        return match.group(1)

    def download(self, url: str, settings: Settings | None = None) -> Path:
        """Download best audio (or lowest video) via yt-dlp's stvr extractor."""
        settings = settings or get_settings()
        settings.ensure_dirs()
        ep_id = self.episode_id(url)
        outtmpl = str(settings.raw_dir / f"{ep_id}.%(ext)s")

        ydl_opts = {
            "outtmpl": outtmpl,
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
                    raise FileNotFoundError(
                        f"Download finished but no file for episode {ep_id}"
                    )
                filepath = max(candidates, key=lambda p: p.stat().st_mtime)

        logger.info("Saved media: %s (%.1f MB)", filepath, filepath.stat().st_size / 1e6)
        upload_file(filepath, object_name=f"raw/{filepath.name}", settings=settings)
        return filepath
