"""YouTube video source (yt-dlp)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp

from config import Settings, get_settings
from src.gcs import upload_file
from src.sources.base import MediaSource, normalize_host

logger = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = frozenset({"youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"})
_PATH_MARKERS = ("shorts", "embed", "live", "v")


def video_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = normalize_host(url)
    candidate = ""
    if host == "youtu.be":
        parts = [p for p in parsed.path.split("/") if p]
        candidate = parts[0] if parts else ""
    else:
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if query_id:
            candidate = query_id
        else:
            parts = [p for p in parsed.path.split("/") if p]
            for marker in _PATH_MARKERS:
                if marker in parts:
                    idx = parts.index(marker)
                    if idx + 1 < len(parts):
                        candidate = parts[idx + 1]
                    break
    if not _VIDEO_ID_RE.fullmatch(candidate):
        raise ValueError(f"Cannot parse YouTube video id from URL: {url}")
    return candidate


class YoutubeSource(MediaSource):
    name = "youtube"

    def matches(self, url: str) -> bool:
        return normalize_host(url) in _HOSTS

    def episode_id(self, url: str) -> str:
        return video_id_from_url(url)

    def download(self, url: str, settings: Settings | None = None) -> Path:
        """Download best audio (or lowest video) via yt-dlp."""
        settings = settings or get_settings()
        settings.ensure_dirs()
        video_id = self.episode_id(url)
        outtmpl = str(settings.raw_dir / f"{video_id}.%(ext)s")

        ydl_opts = {
            "outtmpl": outtmpl,
            "format": "bestaudio/worstaudio/worst",
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "socket_timeout": 60,
            # Default android_vr URLs currently 403. mweb returns a fetchable
            # progressive file (https DASH needs a GVS PO token and is skipped).
            # node solves the YouTube n-challenge (deno is absent).
            "js_runtimes": {"node": {}},
            "remote_components": ["ejs:github"],
            "extractor_args": {"youtube": {"player_client": ["mweb"]}},
        }

        logger.info("Downloading (audio-preferred) %s", url)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = Path(ydl.prepare_filename(info))
            if not filepath.exists():
                candidates = [
                    p
                    for p in settings.raw_dir.glob(f"{video_id}.*")
                    if p.suffix not in {".part", ".ytdl"} and ".part" not in p.name
                ]
                if not candidates:
                    raise FileNotFoundError(
                        f"Download finished but no file for video {video_id}"
                    )
                filepath = max(candidates, key=lambda p: p.stat().st_mtime)

        logger.info("Saved media: %s (%.1f MB)", filepath, filepath.stat().st_size / 1e6)
        upload_file(filepath, object_name=f"raw/{filepath.name}", settings=settings)
        return filepath
