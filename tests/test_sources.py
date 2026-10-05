"""Unit tests for multi-source media ingest registry."""

from __future__ import annotations

import pytest

from src.sources.base import MediaSource, normalize_host
from src.sources import resolve, SOURCES


def test_normalize_host_strips_www() -> None:
    assert normalize_host("https://www.ta3.com/clanok/1/x") == "ta3.com"
    assert normalize_host("https://ta3.com/clanok/1/x") == "ta3.com"
    assert normalize_host("https://www.stvr.sk/televizia/archiv/1") == "stvr.sk"


def test_resolve_unknown_host_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported media URL"):
        resolve("https://example.com/episode/1")


def test_sources_registry_is_list() -> None:
    assert isinstance(SOURCES, list)
    assert all(isinstance(s, MediaSource) for s in SOURCES)
