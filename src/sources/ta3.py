"""ta3.com podcast articles via Transistor.fm embed."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import requests

from config import Settings, get_settings
from src.gcs import upload_file
from src.sources.base import MediaSource, normalize_host

logger = logging.getLogger(__name__)

_ARTICLE_ID_RE = re.compile(r"/clanok/(\d+)/", re.IGNORECASE)
_TRANSISTOR_EMBED_RE = re.compile(
    r"https?://share\.transistor\.fm/e/[A-Za-z0-9_-]+",
    re.IGNORECASE,
)
_MEDIA_URL_RE = re.compile(
    r"https?://media\.transistor\.fm/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+\.mp3",
    re.IGNORECASE,
)
_TRACKABLE_RE = re.compile(
    r"trackable_media_url\\?&quot;:\\?&quot;(https://media\.transistor\.fm/[^\\&quot;]+\.mp3)",
    re.IGNORECASE,
)


def article_id_from_url(url: str) -> str:
    match = _ARTICLE_ID_RE.search(url)
    if not match:
        raise ValueError(f"Cannot parse ta3 article id from URL: {url}")
    return match.group(1)


def find_transistor_embed_url(html: str) -> str:
    match = _TRANSISTOR_EMBED_RE.search(html)
    if not match:
        raise ValueError(
            "No Transistor.fm podcast embed found on ta3 page "
            "(Livebox/video-only articles are not supported)."
        )
    return match.group(0)


def find_transistor_media_url(embed_html: str) -> str:
    match = _TRACKABLE_RE.search(embed_html) or _MEDIA_URL_RE.search(embed_html)
    if not match:
        raise ValueError("No mp3 media URL found in Transistor embed page.")
    return match.group(1) if match.lastindex else match.group(0)


class Ta3PodcastSource(MediaSource):
    name = "ta3"

    def matches(self, url: str) -> bool:
        return normalize_host(url) == "ta3.com"

    def episode_id(self, url: str) -> str:
        return f"ta3-{article_id_from_url(url)}"

    def download(self, url: str, settings: Settings | None = None) -> Path:
        settings = settings or get_settings()
        settings.ensure_dirs()
        ep_id = self.episode_id(url)
        out_path = settings.raw_dir / f"{ep_id}.mp3"

        logger.info("Fetching ta3 article HTML %s", url)
        article = requests.get(url, timeout=60)
        article.raise_for_status()
        embed_url = find_transistor_embed_url(article.text)

        logger.info("Fetching Transistor embed %s", embed_url)
        embed = requests.get(embed_url, timeout=60)
        embed.raise_for_status()
        media_url = find_transistor_media_url(embed.text)

        logger.info("Downloading mp3 %s -> %s", media_url, out_path)
        with requests.get(media_url, timeout=120, stream=True) as resp:
            resp.raise_for_status()
            with out_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fh.write(chunk)

        logger.info("Saved media: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
        upload_file(out_path, object_name=f"raw/{out_path.name}", settings=settings)
        return out_path
