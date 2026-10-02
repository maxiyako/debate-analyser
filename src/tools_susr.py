"""ŠÚSR DATAcube search tool for the statistician fact-checker.

Primary path: the public DATAcube REST API (https://data.statistics.sk/api/v2,
JSON-stat) — keyword search over the cube collection, then value download for
the best-matching cube. Fallback: DuckDuckGo restricted to statistics.sk +
Eurostat. All emitted URLs are registered into `seen_urls` so the provenance
guard in validation.py keeps working.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_API_BASE = "https://data.statistics.sk/api/v2"
_COLLECTION_URL = f"{_API_BASE}/collection?lang=sk"
_TIMEOUT = 20.0
_MAX_DATASETS = 5
_MAX_VALUES = 240  # don't dump huge cubes into agent context
_RECENT_YEARS = 4

# DATAcube labels use official terminology, not colloquial terms
# (e.g. inflation cubes are named "Indexy spotrebiteľských cien").
# Keys are stems matched by prefix against query tokens.
_SYNONYMS: dict[str, list[str]] = {
    "inflac": ["spotrebitelskych cien"],
    "potravin": ["coicop", "spotrebitelskych cien"],
    "plat": ["mzda", "mzdy"],
    "hdp": ["hruby domaci produkt"],
}


def _synonyms_for(token: str) -> list[str]:
    out: list[str] = []
    for stem, syns in _SYNONYMS.items():
        if token.startswith(stem):
            out.extend(syns)
    return out

_collection_cache: list[dict] | None = None


def _norm(text: str) -> str:
    """Lowercase + strip diacritics for keyword matching."""
    return (
        unicodedata.normalize("NFD", (text or "").lower())
        .encode("ascii", "ignore")
        .decode()
    )


def _fetch_collection() -> list[dict]:
    global _collection_cache
    if _collection_cache is not None:
        return _collection_cache
    import requests

    resp = requests.get(_COLLECTION_URL, timeout=_TIMEOUT)
    resp.raise_for_status()
    items = resp.json().get("link", {}).get("item", [])
    _collection_cache = [it for it in items if it.get("label") and it.get("href")]
    return _collection_cache


def _score_dataset(label_norm: str, tokens: list[str]) -> int:
    label_words = label_norm.split()
    score = 0
    for tok in tokens:
        # Direct match: a label word shares a prefix with the query token
        # (handles Slovak inflection: mzda/mzdy/mzde, inflacia/inflacie).
        stem = tok[: max(4, len(tok) - 2)]
        if any(w.startswith(stem) for w in label_words):
            score += 2
            continue
        # Synonym match (official terminology) — full phrase substring.
        if any(_norm(s) in label_norm for s in _synonyms_for(tok)):
            score += 1
    return score


def _search_datasets(query: str) -> list[dict]:
    tokens = [t for t in re.findall(r"\w+", _norm(query)) if len(t) >= 3]
    if not tokens:
        return []
    scored = []
    for it in _fetch_collection():
        s = _score_dataset(_norm(it["label"]), tokens)
        if s > 0:
            scored.append((s, it))
    # Highest keyword score first; most recently updated cube breaks ties.
    scored.sort(key=lambda x: (x[0], x[1].get("update", "")), reverse=True)
    return [it for _, it in scored[:_MAX_DATASETS]]


def _dataset_code(href: str) -> tuple[str, list[str]]:
    """Extract cube code + ordered dimension names from a collection href."""
    path = href.split("/api/v2/dataset/", 1)[1].split("?", 1)[0]
    parts = path.split("/")
    return parts[0], parts[1:]


def _is_time_dim(dim_name: str, note: str) -> bool:
    n = dim_name.lower()
    return n.endswith(("_rok", "_year")) or _norm(note) == "rok"


def _fetch_values(item: dict) -> str:
    """Download recent values for one cube, compactly formatted for the agent."""
    import requests

    code, dims = _dataset_code(item["href"])
    dim_meta = item.get("dimension", {})
    selectors = []
    for d in dims:
        note = (dim_meta.get(d) or {}).get("note", "")
        if _is_time_dim(d, note):
            resp = requests.get(
                f"{_API_BASE}/dimension/{code}/{d}?lang=sk", timeout=_TIMEOUT
            )
            resp.raise_for_status()
            years = list(resp.json().get("category", {}).get("index", {}))
            years.sort(reverse=True)
            selectors.append(",".join(years[:_RECENT_YEARS]) or "all")
        elif d.startswith("nuts"):
            # Regional dimension: country total only (SK0 is always first).
            selectors.append("SK0")
        else:
            selectors.append("all")

    url = f"{_API_BASE}/dataset/{code}/" + "/".join(selectors) + "?lang=sk"
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    values = data.get("value") or []
    ids = data.get("id") or []
    sizes = data.get("size") or []
    dim_labels: dict[str, list[str]] = {}
    for did in ids:
        cat = (data.get("dimension", {}).get(did) or {}).get("category", {})
        idx = cat.get("index", {})
        labels = cat.get("label", {})
        ordered = sorted(idx, key=idx.get) if isinstance(idx, dict) else list(idx)
        dim_labels[did] = [str(labels.get(k, k)) for k in ordered]

    if not values or len(values) > _MAX_VALUES:
        return (
            f"(cube too large or empty to inline: {len(values)} values; "
            f"data URL: {url})"
        )

    # JSON-stat row-major order over `size`.
    lines = []
    for flat, val in enumerate(values):
        if val is None:
            continue
        coords = []
        rem = flat
        for did, sz in zip(reversed(ids), reversed(sizes)):
            coords.append(dim_labels[did][rem % sz] if dim_labels[did] else "?")
            rem //= sz
        lines.append("  " + " | ".join(reversed(coords)) + f" = {val}")
    return "\n".join(lines) if lines else "(no non-null values)"


def _ddg_fallback(query: str, seen_urls: set[str] | None) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return "SUSR fallback search unavailable: install `ddgs`."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="sk-sk", max_results=25))
    except Exception as exc:  # noqa: BLE001
        return f"SUSR search error: {exc}"

    allowed = (
        "statistics.sk",
        "ec.europa.eu",  # Eurostat
        "ecb.europa.eu",
        "oecd.org",
    )
    picked = []
    for r in results:
        url = r.get("href") or r.get("url") or ""
        if not any(d in url for d in allowed):
            continue
        if seen_urls is not None:
            seen_urls.add(url)
        picked.append(f"- {r.get('title', '')}\n  URL: {url}\n  {r.get('body', '')}")
        if len(picked) >= 5:
            break
    return "\n".join(picked) or "No results from statistics.sk / Eurostat / ECB / OECD."


def build_susr_data_tool(seen_urls: set[str] | None = None):
    """Official-statistics tool: ŠÚSR DATAcube API (CPI incl. COICOP food,
    wages, unemployment, GDP...), with a statistics.sk/Eurostat web fallback."""
    from crewai.tools import BaseTool

    class SusrDataSearchTool(BaseTool):
        name: str = "susr_data_search"
        description: str = (
            "Look up OFFICIAL Slovak statistics directly from the Statistical "
            "Office (ŠÚSR) DATAcube API: consumer price indices (headline CPI "
            "and COICOP breakdowns incl. food), wages, unemployment, GDP, "
            "demographics. Input: concise Slovak keywords naming the exact "
            "indicator (e.g. 'indexy spotrebiteľských cien potraviny', "
            "'priemerná mzda', 'miera nezamestnanosti'). Returns matching "
            "cubes with update dates and recent values for the best match. "
            "Falls back to statistics.sk / Eurostat / ECB / OECD web search. "
            "This is primary-source data — prefer it over press articles for "
            "any statistical claim."
        )
        cache: bool = True

        def _run(self, query: str) -> str:
            try:
                matches = _search_datasets(query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DATAcube collection fetch failed: %s", exc)
                return _ddg_fallback(query, seen_urls)

            if not matches:
                return _ddg_fallback(query, seen_urls)

            out = []
            for i, it in enumerate(matches):
                href = it["href"]
                if seen_urls is not None:
                    seen_urls.add(href)
                out.append(
                    f"- {it['label']} (aktualizované: {it.get('update', '?')})\n"
                    f"  URL: {href}"
                )
                if i == 0:
                    try:
                        out.append("  RECENT VALUES:\n" + _fetch_values(it))
                    except Exception as exc:  # noqa: BLE001
                        out.append(f"  (value download failed: {exc})")
            return "\n".join(out)

    return SusrDataSearchTool()
