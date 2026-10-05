"""Unit tests for multi-source media ingest registry."""

from __future__ import annotations

import pytest

from src.sources.base import MediaSource, normalize_host
from src.sources import resolve, SOURCES
from src.sources.stvr import StvrSource


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


def test_stvr_matches_and_episode_id() -> None:
    src = StvrSource()
    url = "https://www.stvr.sk/televizia/archiv/14036/621954"
    assert src.matches(url) is True
    assert src.episode_id(url) == "621954"
    assert src.episode_id("https://stvr.sk/televizia/archiv/621954") == "621954"


def test_stvr_rejects_bad_url() -> None:
    src = StvrSource()
    assert src.matches("https://www.ta3.com/clanok/1/x") is False
    with pytest.raises(ValueError, match="Cannot parse STVR episode id"):
        src.episode_id("https://www.stvr.sk/televizia/")


def test_resolve_stvr_url() -> None:
    source = resolve("https://www.stvr.sk/televizia/archiv/14036/621954")
    assert source.name == "stvr"
