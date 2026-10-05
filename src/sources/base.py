"""Media source adapters: download debate audio from different broadcasters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

from config import Settings


def normalize_host(url: str) -> str:
    host = urlparse(url).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


class MediaSource(ABC):
    name: str

    @abstractmethod
    def matches(self, url: str) -> bool:
        """True if this source can handle the URL."""

    @abstractmethod
    def episode_id(self, url: str) -> str:
        """Stable id used as the stem for raw/audio/transcript files."""

    @abstractmethod
    def download(self, url: str, settings: Settings) -> Path:
        """Download media into settings.raw_dir and return the local path."""
