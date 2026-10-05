"""Registry of media sources. First match wins."""

from __future__ import annotations

from src.sources.base import MediaSource
from src.sources.stvr import StvrSource

SOURCES: list[MediaSource] = [
    StvrSource(),
]


def supported_hosts_message() -> str:
    names = ", ".join(s.name for s in SOURCES) or "(none registered)"
    return f"Supported sources: {names}"


def resolve(url: str) -> MediaSource:
    for source in SOURCES:
        if source.matches(url):
            return source
    raise ValueError(
        f"Unsupported media URL: {url}. {supported_hosts_message()}"
    )
