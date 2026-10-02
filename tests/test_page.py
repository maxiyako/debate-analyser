"""Offline tests for page fetching/extraction (src/tools/page.py)."""

from __future__ import annotations

from datetime import date

import requests

from src.tools.page import (
    claim_keywords,
    extract_pub_date,
    extract_text,
    fetch_page,
    parse_iso_date,
    window_around_keywords,
)

HTML = """
<html><head>
<title>Deficit za rok 2025</title>
<meta property="article:published_time" content="2026-03-01T10:00:00+01:00">
</head><body>
<nav>Menu Domov Kontakt</nav>
<article><p>Deficit verejných financií za rok 2025 dosiahol 5,3 % HDP, uviedol Eurostat.</p></article>
<script>var x = 1;</script>
</body></html>
"""


class FakeResponse:
    def __init__(self, status: int = 200, text: str = HTML, url: str = "https://a.sk/x", ctype: str = "text/html"):
        self.status_code = status
        self.text = text
        self.url = url
        self.headers = {"content-type": ctype}


def test_extract_pub_date_from_meta() -> None:
    assert extract_pub_date(HTML).startswith("2026-03-01")


def test_parse_iso_date() -> None:
    assert parse_iso_date("2026-03-01T10:00:00+01:00") == date(2026, 3, 1)
    assert parse_iso_date("") is None
    assert parse_iso_date("yesterday") is None
    assert parse_iso_date("2026-13-45") is None


def test_extract_text_keeps_article_and_drops_script() -> None:
    text = extract_text(HTML)
    assert "5,3 % HDP" in text
    assert "var x" not in text


def test_claim_keywords_keep_numbers_and_long_words() -> None:
    kws = claim_keywords("Deficit je 6 % HDP a dlh dosiahol 59 percent v roku 2025")
    assert "deficit" in kws
    assert "2025" in kws and "59" in kws
    assert "je" not in kws and "6" not in kws


def test_window_prefers_dense_keyword_region() -> None:
    filler = "x " * 2000
    text = filler + "Deficit dosiahol 5,3 percent HDP v roku 2025" + filler
    win = window_around_keywords(text, ["deficit", "2025", "percent"], width=200, max_windows=1)
    assert "Deficit dosiahol" in win
    assert len(win) <= 260


def test_window_without_hits_returns_head() -> None:
    assert window_around_keywords("abc def", ["zzz"], width=4) == "abc "


def test_fetch_page_ok() -> None:
    page = fetch_page("https://a.sk/x", get=lambda *a, **k: FakeResponse())
    assert page.status == "ok"
    assert page.title == "Deficit za rok 2025"
    assert page.published.startswith("2026-03-01")
    assert "5,3 % HDP" in page.text


def test_fetch_page_dead_on_404() -> None:
    page = fetch_page("https://a.sk/x", get=lambda *a, **k: FakeResponse(status=404, text=""))
    assert page.status == "dead"


def test_fetch_page_inconclusive_on_403_and_timeout() -> None:
    assert fetch_page("https://a.sk/x", get=lambda *a, **k: FakeResponse(status=403, text="")).status == "inconclusive"

    def boom(*a, **k):
        raise requests.exceptions.Timeout()

    assert fetch_page("https://a.sk/x", get=boom).status == "inconclusive"


def test_fetch_page_dead_on_connection_error() -> None:
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError()

    assert fetch_page("https://nope.invalid/x", get=boom).status == "dead"


def test_fetch_page_pdf_is_inconclusive_for_now() -> None:
    page = fetch_page("https://a.sk/x.pdf", get=lambda *a, **k: FakeResponse(ctype="application/pdf"))
    assert page.status == "inconclusive"
    assert "PDF" in page.detail
