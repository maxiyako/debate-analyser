"""Page fetching and passage extraction.

Shared by the evidence judge now and by the research loop's `open_page` later.
Three-state fetch semantics mirror `validation.url_reachable`:
  ok            - fetched; text follows
  dead          - 404/410 or DNS/connection failure: fabricated or removed
  inconclusive  - 403, timeout, PDF, other: may just be bot-blocking, NEVER
                  treated as proof the page is fake
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable

import requests

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; STVRDebateAnalyzer/1.0; +fact-check-link-verifier)"
    )
}

_META_DATE_RES = [
    re.compile(
        r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|'
        r'datePublished|date|dc\.date|publish-date|publication_date)["\']'
        r'[^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)='
        r'["\'](?:article:published_time|datePublished|date)["\']',
        re.IGNORECASE,
    ),
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"'),
    re.compile(r'<time[^>]+datetime=["\']([^"\']+)["\']', re.IGNORECASE),
]
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(
    r"<(?:script|style|nav|footer|header)[^>]*>.*?</(?:script|style|nav|footer|header)>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class PageResult:
    status: str  # "ok" | "dead" | "inconclusive"
    url: str
    final_url: str = ""
    title: str = ""
    published: str = ""
    text: str = ""
    detail: str = ""


def extract_pub_date(html: str) -> str:
    for rx in _META_DATE_RES:
        m = rx.search(html)
        if m:
            return m.group(1).strip()[:25]
    return ""


def parse_iso_date(s: str) -> date | None:
    m = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", s or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _html_to_text(html: str, limit: int = 50_000) -> str:
    text = _TAG_RE.sub(" ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def extract_text(html: str) -> str:
    """Main-content text: trafilatura when available, regex strip otherwise."""
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, favor_recall=True)
        if text:
            return re.sub(r"[ \t]+", " ", text).strip()[:50_000]
    except ImportError:
        pass
    return _html_to_text(html)


def claim_keywords(claim: str) -> list[str]:
    """Distinctive tokens of a claim: words of 5+ chars and numbers of 2+ digits."""
    out: list[str] = []
    for tok in re.findall(r"\w+", (claim or "").lower()):
        keep = (tok.isdigit() and len(tok) >= 2) or (not tok.isdigit() and len(tok) >= 5)
        if keep and tok not in out:
            out.append(tok)
    return out[:12]


def window_around_keywords(
    text: str, keywords: list[str], width: int = 700, max_windows: int = 3
) -> str:
    """Return the densest keyword-bearing passages instead of the page head."""
    if not text:
        return ""
    low = text.lower()
    hits = sorted(
        m.start() for kw in keywords if kw for m in re.finditer(re.escape(kw), low)
    )
    if not hits:
        return text[:width]
    half = width // 2
    scored = sorted(
        ((sum(1 for h in hits if abs(h - p) <= half), -p, p) for p in set(hits)),
        reverse=True,
    )
    chosen: list[int] = []
    for _, _, p in scored:
        if all(abs(p - c) > width for c in chosen):
            chosen.append(p)
        if len(chosen) >= max_windows:
            break
    chosen.sort()
    parts = [text[max(0, p - half) : p + half].strip() for p in chosen]
    return "\n…\n".join(parts)


def fetch_page(
    url: str,
    *,
    get: Callable[..., requests.Response] = requests.get,
    timeout: float = 12.0,
) -> PageResult:
    try:
        resp = get(url, timeout=timeout, headers=_UA, allow_redirects=True)
    except requests.exceptions.ConnectionError:
        return PageResult("dead", url, detail="connection failed / domain does not resolve")
    except requests.exceptions.RequestException as exc:
        return PageResult("inconclusive", url, detail=type(exc).__name__)

    final = str(getattr(resp, "url", "") or url)
    if resp.status_code in (404, 410):
        return PageResult("dead", url, final_url=final, detail=f"HTTP {resp.status_code}")
    if resp.status_code >= 400:
        return PageResult(
            "inconclusive", url, final_url=final, detail=f"HTTP {resp.status_code}"
        )
    if "pdf" in (resp.headers.get("content-type") or "").lower():
        return PageResult(
            "inconclusive", url, final_url=final, detail="PDF extraction not supported yet"
        )

    html = resp.text[:600_000]
    title_m = _TITLE_RE.search(html)
    title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""
    return PageResult(
        "ok",
        url,
        final_url=final,
        title=title,
        published=extract_pub_date(html),
        text=extract_text(html),
    )
