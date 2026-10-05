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


from unittest.mock import MagicMock, patch

from src.sources.ta3 import (
    Ta3PodcastSource,
    article_id_from_url,
    find_transistor_embed_url,
    find_transistor_media_url,
)


ARTICLE_HTML = """
<html><body>
<iframe width="100%" height="180"
  src="https://share.transistor.fm/e/5bc6ce9e"></iframe>
</body></html>
"""

EMBED_HTML = """
<div x-data="transistor.audioEmbedPlayer({&quot;episodes&quot;:[{&quot;trackable_media_url&quot;:&quot;https://media.transistor.fm/5bc6ce9e/6720d2ca.mp3&quot;}]})"></div>
"""

ARTICLE_NO_EMBED = "<html><body><p>no player</p></body></html>"


def test_ta3_article_id_and_matches() -> None:
    src = Ta3PodcastSource()
    url = "https://www.ta3.com/clanok/1074913/v-politike-tomas-taraba-vs-michal-simecka"
    assert src.matches(url) is True
    assert article_id_from_url(url) == "1074913"
    assert src.episode_id(url) == "ta3-1074913"


def test_find_transistor_urls() -> None:
    assert (
        find_transistor_embed_url(ARTICLE_HTML)
        == "https://share.transistor.fm/e/5bc6ce9e"
    )
    assert (
        find_transistor_media_url(EMBED_HTML)
        == "https://media.transistor.fm/5bc6ce9e/6720d2ca.mp3"
    )
    with pytest.raises(ValueError, match="Transistor"):
        find_transistor_embed_url(ARTICLE_NO_EMBED)


def test_resolve_ta3_url() -> None:
    source = resolve(
        "https://www.ta3.com/clanok/1074913/v-politike-x"
    )
    assert source.name == "ta3"


def test_ta3_download_writes_mp3(tmp_path, monkeypatch) -> None:
    from config import Settings

    settings = Settings(_env_file=None, data_dir=tmp_path)
    # Settings may not expose data_dir override — if not, patch raw_dir:
    monkeypatch.setattr(settings, "raw_dir", tmp_path / "raw")
    settings.raw_dir.mkdir(parents=True)

    article_url = (
        "https://www.ta3.com/clanok/1074913/v-politike-tomas-taraba-vs-michal-simecka"
    )
    mp3_bytes = b"ID3fake-mp3-content"

    def fake_get(url, timeout=60, stream=False, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        # download() uses `with requests.get(...)`; MagicMock.__enter__
        # would otherwise return a new mock and write zero bytes.
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        if "ta3.com/clanok" in url:
            resp.text = ARTICLE_HTML
            resp.content = ARTICLE_HTML.encode()
        elif "share.transistor.fm" in url:
            resp.text = EMBED_HTML
            resp.content = EMBED_HTML.encode()
        elif "media.transistor.fm" in url:
            resp.iter_content = lambda chunk_size: [mp3_bytes]
            resp.content = mp3_bytes
        else:
            raise AssertionError(f"unexpected url {url}")
        return resp

    src = Ta3PodcastSource()
    with patch("src.sources.ta3.requests.get", side_effect=fake_get):
        with patch("src.sources.ta3.upload_file", return_value=None):
            path = src.download(article_url, settings=settings)

    assert path.name == "ta3-1074913.mp3"
    assert path.read_bytes() == mp3_bytes
