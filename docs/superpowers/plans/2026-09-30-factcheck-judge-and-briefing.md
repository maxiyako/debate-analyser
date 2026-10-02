# Fact-check Judge Baseline + Political Briefing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship migration steps 1–2 of `docs/factcheck-quality-spec.md`: an LLM evidence judge with a measurable baseline over the 11 stored reports, then a per-debate political briefing that drives consequence-based adaptive claim selection.

**Architecture:** A new `src/judge.py` fetches each fact's cited pages, asks Gemini (structured output) to rate five evidence axes, computes the quality score and downgrade decisions in code, and aggregates to a run-level number; `src/eval.py rejudge` runs it offline over stored reports. A new `src/briefing.py` runs grounded research (3 calls + 1 structuring call) before extraction, filters the result in code (provenance, anachronism), and renders it into the extraction prompt; `src/selection.py` replaces the top-N salience cut with checkability filtering, duplicate merging, a per-speaker consequence threshold, a per-speaker floor, and a cost fuse.

**Tech Stack:** Python 3.11, pydantic v2, `google-genai` (Vertex, structured output + Google Search grounding), CrewAI (unchanged for crews), `trafilatura` (new), `click`, `pytest`.

**Out of scope (later plans):** per-claim native research loop, adjudicator, tier map, `register_lookup`/`calc` tools, PDF extraction, numeric calibration, Slovak claim text, confidence gating in `scoring.py`, briefing use by the Facebook publisher. Those are spec migration steps 3–5.

## Global Constraints

- Consequence threshold default `3`; per-speaker floor `2` claims; `max_claims` becomes a cost fuse with default `40` (spec: "cost fuse (default 40)").
- Judge quality axes (each 0–2): support, metric_fidelity, date_fit, coverage, independence. A judge may force a **downgrade only**, always to `Unverified`; it never upgrades a verdict.
- Briefing items (participants, entity_index, timeline, glossary) must carry at least one source URL that was actually returned by a search in this run; nothing dated after the debate date is admitted. Disputes are exempt from the source requirement (transcript-derived).
- The briefing is **background only**: it may shape claim selection, `context`, and queries; it must never be handed to a verdict-producing step as evidence.
- Keep the five verdicts `True / False / Misleading / Unverified / Contested`; `scoring.py` is untouched.
- Claim text stays in its current language in this plan (Slovak claim text is spec step 5).
- **Import-cycle rule:** `src/briefing.py` and `src/llm.py` must NOT import `src.validation` or `src.agents` at module level (`validation` imports `agents`; `agents` will import `briefing`). Use function-level imports.
- Tests run fully offline with injected fakes. Run the suite with `python3 -m pytest -q` from the repo root (baseline: 62 passed).
- Code comments/docstrings/commit messages in English; LLM prompts follow the existing convention (English instructions, Slovak output where the text reaches the report).

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `requirements.txt` | modify | add `trafilatura` |
| `config.py` | modify | `judge_enabled`, `judge_model`, `briefing_enabled`, `consequence_threshold`, `claims_floor_per_speaker`, `max_claims` default 40 |
| `src/tools/__init__.py` | create | package marker |
| `src/tools/page.py` | create | `fetch_page`, `extract_text`, `extract_pub_date`, `parse_iso_date`, `claim_keywords`, `window_around_keywords` |
| `src/llm.py` | create | `generate_json` (structured output) and `grounded_search` (Vertex grounding, no allowlist) |
| `src/judge.py` | create | judge schemas, prompt, `judge_fact`, `judge_facts`, downgrade logic, summary |
| `src/eval.py` | create | `rejudge` CLI over stored reports |
| `src/briefing.py` | create | `DebateBriefing` models, `filter_briefing`, `render_briefing`, `run_briefing` |
| `src/selection.py` | create | `merge_duplicates`, `select_claims`, `priority` |
| `src/agents.py` | modify | `ExtractedClaim` fields, `AnalysisReport.briefing`, extraction prompt, Phase 0 + selection wiring; `_extract_pub_date` re-exported from `src/tools/page.py` |
| `main.py` | modify | persist `debate_date`; optional judge pass |
| `data/debate_dates.json` | create | episode id → broadcast date (input to `rejudge`) |
| `tests/test_page.py`, `tests/test_judge.py`, `tests/test_eval.py`, `tests/test_briefing.py`, `tests/test_selection.py`, `tests/test_extract_prompt.py` | create | offline unit tests |

---

### Task 0: Commit the existing pipeline as a baseline

The repo has a single commit (the spec); all source is untracked. Commit it first so later diffs are reviewable.

**Files:** none modified.

- [ ] **Step 1: Confirm the suite is green**

Run: `python3 -m pytest -q`
Expected: `62 passed`

- [ ] **Step 2: Commit source, tests, and config (not data, models, or .env)**

```bash
git add .gitignore .env.example Dockerfile README.md config.py main.py pytest.ini requirements.txt run_batch.sh src tests docs/verdict-improvement-spec.md
git status --short
```
Expected: `git status --short` still lists `?? data/` (and nothing staged from `data/`, `models/`, `.env`). If `.env` or `models/` appears staged, run `git reset` and fix `.gitignore` before continuing.

```bash
git commit -m "chore: import existing pipeline as baseline"
```

---

### Task 1: Page fetching and passage extraction

The judge needs to read what cited pages actually say. This is the foundation of spec step 3's `open_page`, limited here to HTML.

**Files:**
- Create: `src/tools/__init__.py`, `src/tools/page.py`
- Modify: `src/agents.py` (replace `_META_DATE_RES`, `_TITLE_RE`, `_extract_pub_date` with imports), `requirements.txt`
- Test: `tests/test_page.py`

**Interfaces:**
- Produces:
  - `@dataclass PageResult(status: str, url: str, final_url: str = "", title: str = "", published: str = "", text: str = "", detail: str = "")` — `status` ∈ `"ok" | "dead" | "inconclusive"`.
  - `fetch_page(url: str, *, get=requests.get, timeout: float = 12.0) -> PageResult`
  - `extract_text(html: str) -> str`
  - `extract_pub_date(html: str) -> str`
  - `parse_iso_date(s: str) -> date | None`
  - `claim_keywords(claim: str) -> list[str]`
  - `window_around_keywords(text: str, keywords: list[str], width: int = 700, max_windows: int = 3) -> str`

- [ ] **Step 1: Install the dependency and record it**

```bash
.venv/bin/pip install "trafilatura>=1.12"
```
Edit `requirements.txt`, replacing the `# HTTP utilities` block with:

```
# HTTP utilities
requests>=2.31
trafilatura>=1.12
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_page.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_page.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tools'`

- [ ] **Step 4: Implement**

Create empty `src/tools/__init__.py`.

Create `src/tools/page.py`:

```python
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
```

- [ ] **Step 5: Re-export the shared date helper from `agents.py`**

In `src/agents.py`, delete the `_META_DATE_RES = [...]` list, the `_TITLE_RE = ...` line, and the `def _extract_pub_date(html: str) -> str:` function, and keep `_TAG_RE`, `_html_to_text` as they are. Replace the deleted block with:

```python
from src.tools.page import _TITLE_RE, extract_pub_date as _extract_pub_date  # noqa: E402
```

(`fetch_page` inside `agents.py` keeps using `_TITLE_RE` and `_extract_pub_date`; behaviour is unchanged.)

- [ ] **Step 6: Run the new tests and the full suite**

Run: `python3 -m pytest tests/test_page.py -q && python3 -m pytest -q`
Expected: `11 passed` for `test_page.py`; full suite `73 passed`.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/tools src/agents.py tests/test_page.py
git commit -m "feat: page fetch + passage extraction tool for evidence auditing"
```

---

### Task 2: Judge schemas, scoring, and downgrade rules (pure logic)

**Files:**
- Create: `src/judge.py` (first half)
- Test: `tests/test_judge.py`

**Interfaces:**
- Produces:
  - `JudgeAxes(support, metric_fidelity, date_fit, coverage, independence)` — each `int` 0..2.
  - `JudgeOutput(axes: JudgeAxes, issues: list[str], recommend_downgrade: bool)`
  - `quality_score(axes: JudgeAxes) -> float` (0–100, weights 0.35/0.25/0.15/0.15/0.10)
  - `forced_downgrade(verdict: Verdict, axes: JudgeAxes, recommend: bool) -> Verdict | None`
  - `FactJudgement(fact_index, claim, verdict, assessable, reason, sources_read, quality, axes, issues, downgrade_to)`
  - `JudgeSummary(judgements, judged, skipped, mean_quality, downgrades, mean_quality_by_verdict)`
  - `summarize(judgements: list[FactJudgement]) -> JudgeSummary`
  - `apply_judge_downgrades(facts: list[VerifiedFact], judgements: list[FactJudgement]) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judge.py`:

```python
"""Offline tests for the evidence judge (src/judge.py)."""

from __future__ import annotations

from src.agents import Severity, Verdict, VerifiedFact
from src.judge import (
    FactJudgement,
    JudgeAxes,
    apply_judge_downgrades,
    forced_downgrade,
    quality_score,
    summarize,
)


def axes(s=2, m=2, d=2, c=2, i=2) -> JudgeAxes:
    return JudgeAxes(support=s, metric_fidelity=m, date_fit=d, coverage=c, independence=i)


def make_fact(verdict: Verdict = Verdict.FALSE) -> VerifiedFact:
    return VerifiedFact(
        claim="Deficit je 6 % HDP v roku 2025.",
        speaker="X",
        quote="q",
        verdict=verdict,
        severity=Severity.MATERIAL if verdict in (Verdict.FALSE, Verdict.MISLEADING) else None,
        sources=["https://a.sk/x"],
    )


def test_quality_score_bounds_and_weights() -> None:
    assert quality_score(axes()) == 100.0
    assert quality_score(axes(0, 0, 0, 0, 0)) == 0.0
    assert quality_score(axes(2, 0, 0, 0, 0)) == 35.0
    assert quality_score(axes(0, 2, 0, 0, 0)) == 25.0


def test_forced_downgrade_rules() -> None:
    assert forced_downgrade(Verdict.FALSE, axes(s=0), False) == Verdict.UNVERIFIED
    assert forced_downgrade(Verdict.FALSE, axes(m=0), False) == Verdict.UNVERIFIED
    assert forced_downgrade(Verdict.MISLEADING, axes(d=0), False) == Verdict.UNVERIFIED
    assert forced_downgrade(Verdict.FALSE, axes(), True) == Verdict.UNVERIFIED
    assert forced_downgrade(Verdict.FALSE, axes(), False) is None
    # True is only downgraded when its evidence does not support it at all.
    assert forced_downgrade(Verdict.TRUE, axes(s=0), False) == Verdict.UNVERIFIED
    assert forced_downgrade(Verdict.TRUE, axes(s=1, m=1, d=1, c=1, i=1), False) is None
    # Never touches non-judged verdicts.
    assert forced_downgrade(Verdict.UNVERIFIED, axes(0, 0, 0, 0, 0), True) is None
    assert forced_downgrade(Verdict.CONTESTED, axes(0, 0, 0, 0, 0), True) is None


def _j(idx: int, verdict: Verdict, quality: float | None, down: Verdict | None = None, ok: bool = True):
    return FactJudgement(
        fact_index=idx,
        claim="c",
        verdict=verdict.value,
        assessable=ok,
        quality=quality,
        downgrade_to=down,
    )


def test_summarize_means_counts_and_by_verdict() -> None:
    js = [
        _j(0, Verdict.TRUE, 80.0),
        _j(1, Verdict.FALSE, 40.0, Verdict.UNVERIFIED),
        _j(2, Verdict.FALSE, 60.0),
        _j(3, Verdict.UNVERIFIED, None, ok=False),
    ]
    s = summarize(js)
    assert s.judged == 3
    assert s.skipped == 1
    assert s.mean_quality == 60.0
    assert s.downgrades == 1
    assert s.mean_quality_by_verdict == {"True": 80.0, "False": 50.0}


def test_summarize_empty() -> None:
    s = summarize([])
    assert s.judged == 0 and s.mean_quality is None


def test_apply_downgrades_mutates_and_clears_severity() -> None:
    facts = [make_fact(Verdict.FALSE), make_fact(Verdict.TRUE)]
    notes = apply_judge_downgrades(
        facts, [_j(0, Verdict.FALSE, 20.0, Verdict.UNVERIFIED), _j(1, Verdict.TRUE, 90.0)]
    )
    assert facts[0].verdict == Verdict.UNVERIFIED
    assert facts[0].severity is None
    assert facts[1].verdict == Verdict.TRUE
    assert len(notes) == 1 and "False->Unverified" in notes[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_judge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.judge'`

- [ ] **Step 3: Implement the pure half of `src/judge.py`**

```python
"""Evidence judge: does the evidence cited for a verdict actually justify it?

The judge reads the cited pages, rates five axes (0-2 each), and may force a
DOWNGRADE to Unverified — never an upgrade. Quality score and downgrade
decisions are computed in code from the axes; the model's own opinion of
"overall quality" is never trusted. Aggregated, the per-fact quality becomes a
run-level number stored with the report (the evaluation harness the spec asks
for).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents import Verdict, VerifiedFact

_JUDGED = (Verdict.TRUE, Verdict.FALSE, Verdict.MISLEADING)
_ACCUSATIONS = (Verdict.FALSE, Verdict.MISLEADING)
_WEIGHTS = {
    "support": 0.35,
    "metric_fidelity": 0.25,
    "date_fit": 0.15,
    "coverage": 0.15,
    "independence": 0.10,
}


class JudgeAxes(BaseModel):
    support: int = Field(
        ge=0, le=2,
        description="2 = cited evidence entails the verdict; 1 = partially; 0 = does not support or contradicts.",
    )
    metric_fidelity: int = Field(
        ge=0, le=2,
        description="2 = evidence is about exactly the claimed metric, period, territory; 1 = close; 0 = different quantity.",
    )
    date_fit: int = Field(
        ge=0, le=2,
        description="2 = dated at/before the debate and fresh enough for the claim's time frame; 1 = dated but somewhat stale or undated; 0 = published after the debate or about the wrong period.",
    )
    coverage: int = Field(
        ge=0, le=2,
        description="2 = addresses the claim as a whole; 1 = addresses part of it; 0 = addresses none of it.",
    )
    independence: int = Field(
        ge=0, le=2,
        description="2 = two or more genuinely independent sources agree; 1 = one source, or several copying one origin; 0 = none usable.",
    )


class JudgeOutput(BaseModel):
    axes: JudgeAxes
    issues: list[str] = Field(
        default_factory=list,
        description="Short concrete problems found, in Slovak. Empty if none.",
    )
    recommend_downgrade: bool = Field(
        default=False,
        description="True only if the verdict should be withdrawn to Unverified because the evidence cannot carry it.",
    )


class FactJudgement(BaseModel):
    fact_index: int
    claim: str
    verdict: str
    assessable: bool = True
    reason: str = ""
    sources_read: int = 0
    quality: float | None = None
    axes: JudgeAxes | None = None
    issues: list[str] = Field(default_factory=list)
    downgrade_to: Verdict | None = None


class JudgeSummary(BaseModel):
    judgements: list[FactJudgement] = Field(default_factory=list)
    judged: int = 0
    skipped: int = 0
    mean_quality: float | None = None
    downgrades: int = 0
    mean_quality_by_verdict: dict[str, float] = Field(default_factory=dict)


def quality_score(axes: JudgeAxes) -> float:
    return round(
        100.0 * sum(w * getattr(axes, k) / 2.0 for k, w in _WEIGHTS.items()), 1
    )


def forced_downgrade(
    verdict: Verdict, axes: JudgeAxes, recommend: bool
) -> Verdict | None:
    """Downgrade target (always Unverified) or None. Never upgrades."""
    if verdict in _ACCUSATIONS:
        if recommend or axes.support == 0 or axes.metric_fidelity == 0 or axes.date_fit == 0:
            return Verdict.UNVERIFIED
        return None
    if verdict == Verdict.TRUE:
        return Verdict.UNVERIFIED if axes.support == 0 else None
    return None


def summarize(judgements: list[FactJudgement]) -> JudgeSummary:
    scored = [j for j in judgements if j.assessable and j.quality is not None]
    by_verdict: dict[str, list[float]] = {}
    for j in scored:
        by_verdict.setdefault(j.verdict, []).append(j.quality)  # type: ignore[arg-type]
    return JudgeSummary(
        judgements=judgements,
        judged=len(scored),
        skipped=len(judgements) - len(scored),
        mean_quality=(
            round(sum(j.quality for j in scored) / len(scored), 1) if scored else None  # type: ignore[misc]
        ),
        downgrades=sum(1 for j in judgements if j.downgrade_to is not None),
        mean_quality_by_verdict={
            v: round(sum(q) / len(q), 1) for v, q in by_verdict.items()
        },
    )


def apply_judge_downgrades(
    facts: list[VerifiedFact], judgements: list[FactJudgement]
) -> list[str]:
    """Apply forced downgrades in place. Returns human-readable notes."""
    notes: list[str] = []
    for j in judgements:
        if j.downgrade_to is None or not (0 <= j.fact_index < len(facts)):
            continue
        fact = facts[j.fact_index]
        notes.append(
            f"Judge downgraded {fact.verdict.value}->{j.downgrade_to.value} "
            f'(quality {j.quality}): "{fact.claim[:120]}" — '
            + "; ".join(j.issues[:3])
        )
        fact.verdict = j.downgrade_to
        fact.severity = None
    return notes
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_judge.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/judge.py tests/test_judge.py
git commit -m "feat(judge): evidence axes, quality score, downgrade-only rules"
```

---

### Task 3: Judge execution and the LLM adapter

**Files:**
- Create: `src/llm.py`
- Modify: `src/judge.py` (append), `config.py`
- Test: `tests/test_judge.py` (append)

**Interfaces:**
- Consumes: `PageResult`, `fetch_page`, `claim_keywords`, `window_around_keywords`, `parse_iso_date` (Task 1); `JudgeOutput`, `quality_score`, `forced_downgrade`, `FactJudgement`, `summarize` (Task 2).
- Produces:
  - `src.llm.generate_json(prompt: str, schema: type[T], settings: Settings, *, label: str, model: str | None = None, temperature: float = 0.0) -> T`
  - `src.llm.GroundedAnswer(text: str, urls: list[str])` dataclass
  - `src.llm.grounded_search(prompt: str, settings: Settings, *, label: str) -> GroundedAnswer`
  - `build_judge_prompt(fact: VerifiedFact, debate_date: date | None, pages: list[PageResult]) -> str`
  - `judge_fact(fact, fact_index, debate_date, *, fetch, llm, max_sources=4) -> FactJudgement`
  - `judge_facts(facts, debate_date, *, fetch, llm, max_workers=4) -> JudgeSummary`
  - `default_judge_llm(settings) -> Callable[[str], JudgeOutput]`

- [ ] **Step 1: Add settings**

In `config.py`, change the claim-cap lines and add new settings. Replace:

```python
    max_claims: int = 30  # top-N claims (by salience) forwarded to fact-checkers
```
with:
```python
    # Cost fuse for the adaptive claim selection (was: a hard top-N cut).
    max_claims: int = 40
    # Adaptive selection: research every empirical claim with consequence >=
    # threshold; every speaker always keeps their top `floor` claims.
    consequence_threshold: int = 3
    claims_floor_per_speaker: int = 2
    # Phase 0: political briefing researched before extraction.
    briefing_enabled: bool = True
    # Evidence judge: audits cited pages, may downgrade unsupported verdicts.
    judge_enabled: bool = False
    judge_model: str | None = None  # None = same as gemini_model
```

- [ ] **Step 2: Write the failing tests (append to `tests/test_judge.py`)**

Add these imports at the top of the file (merge with the existing ones):

```python
from datetime import date

from src.judge import JudgeOutput, build_judge_prompt, judge_fact, judge_facts
from src.tools.page import PageResult
```

Append:

```python
DEBATE = date(2026, 4, 12)


def page_ok(url="https://a.sk/x", published="2026-03-01", text="Deficit verejných financií za rok 2025 dosiahol 5,3 % HDP."):
    return PageResult("ok", url, final_url=url, title="t", published=published, text=text)


class RecordingLLM:
    def __init__(self, out: JudgeOutput):
        self.out = out
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> JudgeOutput:
        self.prompts.append(prompt)
        return self.out


def good_out(**kw) -> JudgeOutput:
    return JudgeOutput(axes=axes(**kw), issues=[], recommend_downgrade=False)


def test_prompt_contains_excerpt_dates_and_flags_anachronism() -> None:
    fact = make_fact()
    late = page_ok("https://b.sk/y", published="2026-05-01")
    prompt = build_judge_prompt(fact, DEBATE, [page_ok(), late])
    assert "5,3 % HDP" in prompt
    assert "2026-04-12" in prompt
    assert prompt.count("PUBLISHED AFTER THE DEBATE DATE") == 1
    assert fact.claim in prompt


def test_prompt_marks_unread_sources() -> None:
    prompt = build_judge_prompt(
        make_fact(), DEBATE, [PageResult("inconclusive", "https://c.sk/z", detail="HTTP 403")]
    )
    assert "NOT READ" in prompt and "HTTP 403" in prompt


def test_judge_fact_scores_from_axes_not_from_model() -> None:
    llm = RecordingLLM(good_out(s=2, m=1, d=2, c=2, i=1))
    j = judge_fact(make_fact(), 0, DEBATE, fetch=lambda u: page_ok(u), llm=llm)
    assert j.assessable and j.sources_read == 1
    assert j.quality == quality_score(axes(2, 1, 2, 2, 1))
    assert j.downgrade_to is None
    assert len(llm.prompts) == 1


def test_judge_fact_downgrades_on_zero_support() -> None:
    llm = RecordingLLM(JudgeOutput(axes=axes(s=0), issues=["zdroj tvrdenie nepodporuje"]))
    j = judge_fact(make_fact(Verdict.FALSE), 3, DEBATE, fetch=lambda u: page_ok(u), llm=llm)
    assert j.fact_index == 3
    assert j.downgrade_to == Verdict.UNVERIFIED
    assert j.issues == ["zdroj tvrdenie nepodporuje"]


def test_judge_fact_unassessable_when_nothing_readable_skips_llm() -> None:
    llm = RecordingLLM(good_out())
    j = judge_fact(
        make_fact(),
        0,
        DEBATE,
        fetch=lambda u: PageResult("inconclusive", u, detail="HTTP 403"),
        llm=llm,
    )
    assert not j.assessable
    assert llm.prompts == []


def test_judge_fact_skips_unverified_and_unsourced() -> None:
    llm = RecordingLLM(good_out())
    unv = make_fact(Verdict.UNVERIFIED)
    assert not judge_fact(unv, 0, DEBATE, fetch=lambda u: page_ok(u), llm=llm).assessable
    nosrc = make_fact(Verdict.TRUE)
    nosrc.sources = []
    assert not judge_fact(nosrc, 0, DEBATE, fetch=lambda u: page_ok(u), llm=llm).assessable
    assert llm.prompts == []


def test_judge_facts_keeps_order_and_summarizes() -> None:
    llm = RecordingLLM(good_out())
    facts = [make_fact(Verdict.TRUE), make_fact(Verdict.UNVERIFIED), make_fact(Verdict.FALSE)]
    summary = judge_facts(facts, DEBATE, fetch=lambda u: page_ok(u), llm=llm, max_workers=2)
    assert [j.fact_index for j in summary.judgements] == [0, 1, 2]
    assert summary.judged == 2 and summary.skipped == 1
    assert summary.mean_quality == 100.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_judge.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_judge_prompt' from 'src.judge'`

- [ ] **Step 4: Create the LLM adapter `src/llm.py`**

```python
"""Thin google-genai helpers for structured output and grounded search.

Cycle rule: `src.agents` imports `src.briefing`, which imports this module. So
this file must not import `src.agents` or `src.validation` at module level;
retry and redirect helpers are imported inside the functions that need them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from config import Settings
from src.costs import TRACKER

T = TypeVar("T", bound=BaseModel)


def _client(settings: Settings) -> tuple[Any, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=settings.gcp_project_id,
        location=settings.gcp_location,
        http_options=types.HttpOptions(timeout=int(settings.llm_timeout_s * 1000)),
    )
    return client, types


def generate_json(
    prompt: str,
    schema: type[T],
    settings: Settings,
    *,
    label: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> T:
    """One structured-output call validated into `schema` (with 429 backoff)."""
    from src.agents import _retry_on_rate_limit

    client, types = _client(settings)

    def _call():
        return client.models.generate_content(
            model=model or settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

    resp = _retry_on_rate_limit(_call)
    TRACKER.add_genai(label, getattr(resp, "usage_metadata", None))
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    return schema.model_validate_json(resp.text)


@dataclass
class GroundedAnswer:
    text: str
    urls: list[str]


def grounded_search(prompt: str, settings: Settings, *, label: str) -> GroundedAnswer:
    """Google-Search-grounded answer plus the real cited URLs.

    Citations are redirect-resolved and filtered only by the hard disinfo/party
    blocklist — no positive allowlist (source tiering arrives in a later plan).
    """
    from src.agents import _resolve_redirect, _retry_on_rate_limit
    from src.validation import is_blocked_source

    client, types = _client(settings)

    def _call():
        return client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

    resp = _retry_on_rate_limit(_call)
    TRACKER.add_genai(label, getattr(resp, "usage_metadata", None))
    urls: list[str] = []
    try:
        meta = resp.candidates[0].grounding_metadata
        for chunk in getattr(meta, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None):
                real = _resolve_redirect(web.uri)
                if real not in urls and not is_blocked_source(real):
                    urls.append(real)
    except (AttributeError, IndexError, TypeError):
        pass
    return GroundedAnswer(text=(resp.text or "").strip(), urls=urls)
```

- [ ] **Step 5: Append the execution half to `src/judge.py`**

Add to the imports at the top of `src/judge.py`:

```python
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Callable

from src.tools.page import (
    PageResult,
    claim_keywords,
    fetch_page,
    parse_iso_date,
    window_around_keywords,
)
```

Append to the file:

```python
_JUDGE_INSTRUCTIONS = """\
You are a strict evidence auditor for a fact-checking desk. You do NOT decide
whether the claim is true from your own knowledge. You judge ONLY whether the
evidence below justifies the verdict that was given.

Rate five axes, each 0, 1 or 2:
- support: does the excerpted evidence entail the verdict? (for True: confirm
  the claim; for False: explicitly contradict it; for Misleading: show it
  distorts reality). 0 = it does not, or it points the other way.
- metric_fidelity: is the evidence about exactly the same metric, period and
  territory as the claim? (headline CPI is not food inflation; nominal is not
  real; year-on-year is not cumulative; a different year is not this year.)
- date_fit: is every relied-on source dated at or before the debate date and
  fresh enough for the claim's time frame? A source marked "PUBLISHED AFTER
  THE DEBATE DATE" cannot support anything. Undated sources score at most 1.
- coverage: does the evidence address the claim as a WHOLE, or only a part?
- independence: do two or more genuinely independent sources agree (not one
  wire story republished)? One source = 1.

Set recommend_downgrade=true only if the verdict should be withdrawn to
Unverified because this evidence cannot carry it. A sources marked NOT READ
tell you nothing either way. List concrete problems in `issues`, in Slovak,
one short sentence each. Be strict but fair: do not invent problems.
"""


def build_judge_prompt(
    fact: VerifiedFact, debate_date: date | None, pages: list[PageResult]
) -> str:
    ref = debate_date.isoformat() if debate_date else "unknown"
    kws = claim_keywords(fact.claim)
    blocks: list[str] = []
    for i, p in enumerate(pages, 1):
        if p.status == "ok":
            late = ""
            pub = parse_iso_date(p.published)
            if debate_date and pub and pub > debate_date:
                late = "  !! PUBLISHED AFTER THE DEBATE DATE\n"
            blocks.append(
                f"[{i}] {p.url}\n  title: {p.title or '(none)'}\n"
                f"  published: {p.published or 'unknown'}\n{late}"
                f"  excerpt: {window_around_keywords(p.text, kws)}"
            )
        else:
            blocks.append(f"[{i}] {p.url}\n  NOT READ ({p.status}: {p.detail})")
    return (
        f"{_JUDGE_INSTRUCTIONS}\n"
        f"DEBATE DATE (reference 'now'): {ref}\n\n"
        f"CLAIM: {fact.claim}\n"
        f"SPEAKER QUOTE: {fact.quote}\n"
        f"VERDICT GIVEN: {fact.verdict.value}"
        f"{' / ' + fact.severity.value if fact.severity else ''}\n"
        f"RATIONALE GIVEN: {fact.rationale[:1200]}\n\n"
        "CITED EVIDENCE:\n" + "\n\n".join(blocks)
    )


def judge_fact(
    fact: VerifiedFact,
    fact_index: int,
    debate_date: date | None,
    *,
    fetch: Callable[[str], PageResult] = fetch_page,
    llm: Callable[[str], JudgeOutput],
    max_sources: int = 4,
) -> FactJudgement:
    base = dict(fact_index=fact_index, claim=fact.claim, verdict=fact.verdict.value)
    if fact.verdict not in _JUDGED:
        return FactJudgement(**base, assessable=False, reason="verdict not judged")
    if not fact.sources:
        return FactJudgement(**base, assessable=False, reason="no sources cited")

    pages = [fetch(u) for u in fact.sources[:max_sources]]
    readable = [p for p in pages if p.status == "ok"]
    if not readable:
        return FactJudgement(**base, assessable=False, reason="no cited source could be read")

    out = llm(build_judge_prompt(fact, debate_date, pages))
    return FactJudgement(
        **base,
        assessable=True,
        sources_read=len(readable),
        quality=quality_score(out.axes),
        axes=out.axes,
        issues=list(out.issues),
        downgrade_to=forced_downgrade(fact.verdict, out.axes, out.recommend_downgrade),
    )


def judge_facts(
    facts: list[VerifiedFact],
    debate_date: date | None,
    *,
    fetch: Callable[[str], PageResult] = fetch_page,
    llm: Callable[[str], JudgeOutput],
    max_workers: int = 4,
) -> JudgeSummary:
    def _one(item: tuple[int, VerifiedFact]) -> FactJudgement:
        i, f = item
        return judge_fact(f, i, debate_date, fetch=fetch, llm=llm)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        judgements = list(pool.map(_one, enumerate(facts)))  # map preserves order
    return summarize(judgements)


def default_judge_llm(settings) -> Callable[[str], JudgeOutput]:
    from src.llm import generate_json

    model = settings.judge_model or settings.gemini_model
    return lambda prompt: generate_json(
        prompt, JudgeOutput, settings, label="judge", model=model
    )
```

- [ ] **Step 6: Run tests and full suite**

Run: `python3 -m pytest tests/test_judge.py -q && python3 -m pytest -q`
Expected: `test_judge.py` `12 passed`; full suite green (85 passed).

- [ ] **Step 7: Commit**

```bash
git add config.py src/llm.py src/judge.py tests/test_judge.py
git commit -m "feat(judge): page-reading evidence judge with injectable LLM and fetcher"
```

---

### Task 4: `rejudge` CLI, debate dates, pipeline hook, and the BASELINE

**Files:**
- Create: `src/eval.py`, `data/debate_dates.json`
- Modify: `main.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `judge_facts`, `JudgeSummary`, `default_judge_llm` (Task 3); `VerifiedFact`.
- Produces:
  - `load_dates(path: Path) -> dict[str, str]`
  - `base_id(stem: str) -> str | None` (`"591624_v2"` → `"591624"`; non-report stems → `None`)
  - `report_files(reports_dir: Path) -> list[Path]`
  - `rejudge_reports(reports_dir, dates, *, fetch, llm, only=None) -> dict` with keys `reports`, `skipped_no_date`, `overall`.

- [ ] **Step 1: Dates file (already created by the controller)**

`data/debate_dates.json` already exists and is committed before this task starts. Dates for 591624, 592879, 594102, 595293, 596580, 597879, 599134, 601853 were recovered from `DEBATE DATE:` lines in the run logs. 602993 (`2026-06-14`), 604147 (`2026-06-21`) and 605345 (`2026-06-28`) are INFERRED from the weekly-Sunday cadence and the `2026-06-14` example in the `ExtractedClaim.time_window` docstring; the user has been asked to confirm them. Do not modify this file in this task.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_eval.py`:

```python
"""Offline tests for the rejudge harness (src/eval.py)."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval import base_id, load_dates, rejudge_reports, report_files
from src.judge import JudgeAxes, JudgeOutput
from src.tools.page import PageResult


def _write_report(dirpath: Path, stem: str, facts: list[dict], **extra) -> None:
    payload = {"summary": "s", "facts": facts, **extra}
    (dirpath / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def _fact(verdict: str) -> dict:
    return {
        "claim": "Deficit je 6 % HDP.",
        "speaker": "X",
        "quote": "q",
        "verdict": verdict,
        "severity": "material" if verdict == "False" else None,
        "sources": ["https://a.sk/x"],
    }


def _ok(url: str) -> PageResult:
    return PageResult("ok", url, final_url=url, title="t", published="2026-03-01", text="Deficit 6 % HDP")


def _llm(prompt: str) -> JudgeOutput:
    return JudgeOutput(axes=JudgeAxes(support=2, metric_fidelity=2, date_fit=2, coverage=2, independence=1))


def test_base_id_and_report_files(tmp_path: Path) -> None:
    assert base_id("591624") == "591624"
    assert base_id("591624_v2") == "591624"
    assert base_id("602993.corrected") is None
    assert base_id("605345.baseline") is None
    assert base_id("run_591624") is None
    for stem in ("591624", "591624_v2", "602993.corrected", "run_591624"):
        (tmp_path / f"{stem}.json").write_text("{}")
    (tmp_path / "x.log").write_text("")
    assert [p.stem for p in report_files(tmp_path)] == ["591624", "591624_v2"]


def test_load_dates_missing_file(tmp_path: Path) -> None:
    assert load_dates(tmp_path / "nope.json") == {}


def test_rejudge_uses_dates_file_and_payload_and_lists_skipped(tmp_path: Path) -> None:
    _write_report(tmp_path, "100", [_fact("True"), _fact("False")])
    _write_report(tmp_path, "200_v2", [_fact("True")], debate_date="2026-04-19")
    _write_report(tmp_path, "300", [_fact("True")])  # no date anywhere
    out = rejudge_reports(tmp_path, {"100": "2026-04-12"}, fetch=_ok, llm=_llm)
    assert set(out["reports"]) == {"100", "200_v2"}
    assert out["skipped_no_date"] == ["300"]
    assert out["reports"]["100"]["date"] == "2026-04-12"
    assert out["reports"]["200_v2"]["date"] == "2026-04-19"
    assert out["overall"]["judged"] == 3
    assert out["overall"]["mean_quality"] == 95.0  # axes (2,2,2,2,1) -> 95.0


def test_rejudge_only_filter(tmp_path: Path) -> None:
    _write_report(tmp_path, "100", [_fact("True")])
    _write_report(tmp_path, "101", [_fact("True")])
    out = rejudge_reports(
        tmp_path, {"100": "2026-04-12", "101": "2026-04-12"}, fetch=_ok, llm=_llm, only={"101"}
    )
    assert set(out["reports"]) == {"101"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval'`

- [ ] **Step 4: Implement `src/eval.py`**

```python
"""Offline evaluation harness. Currently: re-judge stored reports.

    python -m src.eval rejudge --out data/judge_baseline.json

Establishes the evidence-quality baseline for the CURRENT pipeline before any
change lands, and scores every later run against it.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Callable

import click

from src.agents import VerifiedFact
from src.judge import JudgeOutput, judge_facts
from src.tools.page import PageResult, fetch_page, parse_iso_date

_STEM = re.compile(r"^(\d+)(?:_v\d+)?$")


def base_id(stem: str) -> str | None:
    m = _STEM.match(stem)
    return m.group(1) if m else None


def load_dates(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def report_files(reports_dir: Path) -> list[Path]:
    return sorted(p for p in reports_dir.glob("*.json") if base_id(p.stem))


def rejudge_reports(
    reports_dir: Path,
    dates: dict[str, str],
    *,
    fetch: Callable[[str], PageResult] = fetch_page,
    llm: Callable[[str], JudgeOutput],
    only: set[str] | None = None,
) -> dict:
    reports: dict[str, dict] = {}
    skipped_no_date: list[str] = []
    total_judged = 0
    weighted = 0.0
    total_downgrades = 0

    for path in report_files(reports_dir):
        stem = path.stem
        if only and stem not in only and base_id(stem) not in only:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_date = payload.get("debate_date") or dates.get(base_id(stem) or "", "")
        debate: date | None = parse_iso_date(raw_date)
        if debate is None:
            skipped_no_date.append(stem)
            continue
        facts = [VerifiedFact.model_validate(f) for f in payload.get("facts", [])]
        summary = judge_facts(facts, debate, fetch=fetch, llm=llm)
        reports[stem] = {"date": debate.isoformat(), "summary": summary.model_dump(mode="json")}
        total_judged += summary.judged
        weighted += (summary.mean_quality or 0.0) * summary.judged
        total_downgrades += summary.downgrades

    return {
        "reports": reports,
        "skipped_no_date": skipped_no_date,
        "overall": {
            "judged": total_judged,
            "mean_quality": round(weighted / total_judged, 1) if total_judged else None,
            "downgrades": total_downgrades,
        },
    }


@click.group()
def cli() -> None:
    """Evaluation tools."""


@cli.command()
@click.option("--reports", "reports_dir", type=click.Path(path_type=Path, exists=True), default=Path("data/reports"))
@click.option("--dates", "dates_path", type=click.Path(path_type=Path), default=Path("data/debate_dates.json"))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path("data/judge_baseline.json"))
@click.option("--only", multiple=True, help="Episode id(s) to judge; default all.")
def rejudge(reports_dir: Path, dates_path: Path, out_path: Path, only: tuple[str, ...]) -> None:
    """Re-judge stored reports and print the run-level evidence quality."""
    from config import get_settings
    from src.costs import TRACKER
    from src.judge import default_judge_llm

    settings = get_settings()
    TRACKER.reset()
    result = rejudge_reports(
        reports_dir,
        load_dates(dates_path),
        llm=default_judge_llm(settings),
        only=set(only) or None,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    click.echo(f"{'report':<14}{'date':<12}{'judged':>7}{'skipped':>8}{'quality':>9}{'down':>6}")
    for stem, r in result["reports"].items():
        s = r["summary"]
        q = "-" if s["mean_quality"] is None else f"{s['mean_quality']:.1f}"
        click.echo(f"{stem:<14}{r['date']:<12}{s['judged']:>7}{s['skipped']:>8}{q:>9}{s['downgrades']:>6}")
    o = result["overall"]
    click.echo(f"OVERALL judged={o['judged']} mean_quality={o['mean_quality']} downgrades={o['downgrades']}")
    if result["skipped_no_date"]:
        click.echo("Skipped (no debate date): " + ", ".join(result["skipped_no_date"]))
    click.echo(TRACKER.summary(settings.judge_model or settings.gemini_model))
    click.echo(f"Written: {out_path}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_eval.py -q`
Expected: `4 passed`

- [ ] **Step 6: Persist `debate_date` and add the optional judge pass in `main.py`**

In `main.py`, replace this block:

```python
    verdict = score_report(report, transcript=corrected.text)
    report_path = settings.report_dir / f"{ep_id}.json"
    payload = report.model_dump(mode="json")
    payload["verdict"] = verdict.model_dump(mode="json")
```
with:
```python
    judge_summary = None
    if settings.judge_enabled:
        from src.judge import apply_judge_downgrades, default_judge_llm, judge_facts

        logger.info("=== Evidence judge ===")
        judge_summary = judge_facts(
            report.facts, debate_day, llm=default_judge_llm(settings)
        )
        judge_notes = apply_judge_downgrades(report.facts, judge_summary.judgements)
        report.critic_notes = [*report.critic_notes, *judge_notes]
        click.echo(
            f"Judge: mean evidence quality {judge_summary.mean_quality} over "
            f"{judge_summary.judged} facts, {judge_summary.downgrades} downgrades"
        )

    verdict = score_report(report, transcript=corrected.text)
    report_path = settings.report_dir / f"{ep_id}.json"
    payload = report.model_dump(mode="json")
    payload["verdict"] = verdict.model_dump(mode="json")
    payload["debate_date"] = debate_day.isoformat() if debate_day else None
    if judge_summary is not None:
        payload["judge"] = judge_summary.model_dump(mode="json")
```

- [ ] **Step 7: Run the full suite, then commit**

Run: `python3 -m pytest -q`
Expected: all pass.

```bash
git add src/eval.py main.py tests/test_eval.py
git commit -m "feat(eval): rejudge harness, persisted debate_date, optional judge pass"
```

- [ ] **Step 8: Produce the BASELINE (manual, uses Vertex + network)**

```bash
.venv/bin/python -m src.eval rejudge --out data/judge_baseline.json
```
Expected: a table with one row each for 591624 and 592879, an `OVERALL mean_quality=<number>` line, and `Skipped (no debate date): …` listing the other reports. Record the overall number and the per-verdict means (`mean_quality_by_verdict` in the JSON) — this is the baseline every later step is compared against.

If the user later corrects an inferred date in `data/debate_dates.json`, re-run with `--only <id>` for that report.

```bash
git status --short data/debate_dates.json   # should be clean
```

(`data/judge_baseline.json` is left uncommitted on purpose — it lives under `data/`, which is local working data; copy the overall numbers into the commit message or `docs/` if a permanent record is wanted.)

---

### Task 5: Briefing models, provenance/anachronism filter, prompt rendering

**Files:**
- Create: `src/briefing.py` (first half)
- Test: `tests/test_briefing.py`

**Interfaces:**
- Produces:
  - Models: `Side` (str enum `coalition|opposition|other`), `Participant`, `EntityEntry`, `Dispute`, `TimelineEvent`, `GlossaryEntry`, `DebateBriefing`.
  - `filter_briefing(briefing: DebateBriefing, allowed_urls: set[str], debate_date: date) -> tuple[DebateBriefing, list[str]]`
  - `render_briefing(briefing: DebateBriefing, max_chars: int = 7000) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_briefing.py`:

```python
"""Offline tests for the political briefing (src/briefing.py)."""

from __future__ import annotations

from datetime import date

from src.briefing import (
    DebateBriefing,
    Dispute,
    EntityEntry,
    GlossaryEntry,
    Participant,
    Side,
    TimelineEvent,
    filter_briefing,
    render_briefing,
)

DEBATE = date(2026, 4, 12)
GOOD = "https://www.sme.sk/c/1/clanok.html"
OTHER = "https://dennikn.sk/2/clanok/"
FAKE = "https://www.sme.sk/c/999/vymysleny.html"
PARTY = "https://www.smer.sk/clanok"


def make_briefing() -> DebateBriefing:
    return DebateBriefing(
        debate_date="2026-04-12",
        topic="Maďarské voľby",
        participants=[
            Participant(name="Ján Novák", party="Smer", office="poslanec", side=Side.COALITION, sources=[GOOD]),
            Participant(name="Fiktívny Hosť", party="X", sources=[FAKE]),  # only fabricated source
        ],
        entity_index=[
            EntityEntry(reference="predseda vlády", person="Robert Fico", valid_on="2026-04-12", sources=[GOOD, PARTY]),
        ],
        live_disputes=[
            Dispute(title="Eurofondy", factual_question="Boli fondy pozastavené?", stakes="Dopad na rozpočet"),
        ],
        timeline=[
            TimelineEvent(date="2026-03-01", event="Komisia zmrazila fondy", sources=[OTHER]),
            TimelineEvent(date="2026-05-01", event="Udalosť po debate", sources=[OTHER]),
            TimelineEvent(date="2026-02-01", event="Bez zdroja", sources=[]),
        ],
        glossary=[GlossaryEntry(term="konsolidácia", definition="Znižovanie deficitu", sources=[GOOD])],
    )


def test_filter_drops_anachronistic_unsourced_and_fabricated() -> None:
    out, notes = filter_briefing(make_briefing(), {GOOD, OTHER}, DEBATE)
    assert [p.name for p in out.participants] == ["Ján Novák"]
    assert [t.event for t in out.timeline] == ["Komisia zmrazila fondy"]
    assert out.entity_index[0].sources == [GOOD]  # party URL stripped
    assert len(out.glossary) == 1
    assert len(out.live_disputes) == 1  # disputes are exempt from the source rule
    joined = " | ".join(notes)
    assert "after debate date" in joined
    assert "Fiktívny Hosť" in joined


def test_filter_keeps_undated_timeline_out() -> None:
    b = make_briefing()
    b.timeline = [TimelineEvent(date="", event="Neurčený dátum", sources=[GOOD])]
    out, _ = filter_briefing(b, {GOOD}, DEBATE)
    assert out.timeline == []  # a dateless event cannot be shown to precede the debate


def test_render_contains_sections_and_is_background_labelled() -> None:
    out, _ = filter_briefing(make_briefing(), {GOOD, OTHER}, DEBATE)
    text = render_briefing(out)
    assert "podklad" in text.lower()
    assert "Ján Novák" in text and "koalícia" in text
    assert '"predseda vlády" = Robert Fico' in text
    assert "Eurofondy" in text and "Boli fondy pozastavené?" in text
    assert "2026-03-01" in text
    assert "konsolidácia" in text


def test_render_respects_max_chars_on_line_boundary() -> None:
    b = make_briefing()
    b.timeline = [
        TimelineEvent(date="2026-03-01", event="E" * 100 + str(i), sources=[OTHER]) for i in range(100)
    ]
    text = render_briefing(b, max_chars=1000)
    assert len(text) <= 1000
    assert not text.endswith("E" * 3 + "\n\n")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_briefing.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.briefing'`

- [ ] **Step 3: Implement the models, filter, and renderer**

Create `src/briefing.py`:

```python
"""Phase 0: political desk briefing researched before analysis.

The briefing gives the pipeline what a desk journalist knows before watching a
debate: who the speakers are, what role references mean on the debate date,
what the live disputes are and what is at stake, and what happened recently.

It is BACKGROUND ONLY. It shapes claim selection, claim context and query
construction; it must never be handed to a verdict-producing step as evidence.

Cycle rule: `src.agents` imports this module, and `src.validation` imports
`src.agents`. Do NOT import `src.validation` (or `src.agents`) at module level
here; import inside functions.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Side(str, Enum):
    COALITION = "coalition"
    OPPOSITION = "opposition"
    OTHER = "other"


class Participant(BaseModel):
    name: str
    party: str = ""
    office: str = Field(default="", description="Office held ON THE DEBATE DATE.")
    side: Side = Side.OTHER
    prior_roles: list[str] = Field(default_factory=list)
    dossiers: list[str] = Field(default_factory=list, description="Policy areas the person owns.")
    sources: list[str] = Field(default_factory=list)


class EntityEntry(BaseModel):
    reference: str = Field(description='Role reference as spoken, e.g. "prezident Poľska".')
    person: str
    valid_on: str = Field(default="", description="ISO date the mapping is valid for.")
    sources: list[str] = Field(default_factory=list)


class Dispute(BaseModel):
    title: str
    factual_question: str = Field(description="The checkable question underneath the dispute.")
    stakes: str = Field(
        default="",
        description="What a claim about this would mean for a voter's judgement (descriptive, not evaluative).",
    )
    sources: list[str] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    date: str = Field(description="ISO date YYYY-MM-DD.")
    event: str
    sources: list[str] = Field(default_factory=list)


class GlossaryEntry(BaseModel):
    term: str
    definition: str
    status: str = Field(default="", description="Status of the matter on the debate date.")
    sources: list[str] = Field(default_factory=list)


class DebateBriefing(BaseModel):
    debate_date: str = ""
    topic: str = ""
    participants: list[Participant] = Field(default_factory=list)
    entity_index: list[EntityEntry] = Field(default_factory=list)
    live_disputes: list[Dispute] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)


def filter_briefing(
    briefing: DebateBriefing, allowed_urls: set[str], debate_date
) -> tuple[DebateBriefing, list[str]]:
    """Enforce provenance and anachronism in code (the model's output is untrusted).

    - URLs not returned by a search this run, or on the disinfo/party blocklist,
      are stripped.
    - participants / entity_index / timeline / glossary items left with no
      source are dropped. Disputes are transcript-derived and exempt.
    - timeline events dated after the debate (or undated) are dropped.
    """
    from src.validation import _normalize_url, is_blocked_source

    allowed = {_normalize_url(u) for u in allowed_urls}
    notes: list[str] = []

    def clean(urls: list[str]) -> list[str]:
        out: list[str] = []
        for u in urls:
            if is_blocked_source(u) or _normalize_url(u) not in allowed or u in out:
                continue
            out.append(u)
        return out

    def keep(items, label_of, kind: str):
        kept = []
        for it in items:
            it = it.model_copy(update={"sources": clean(it.sources)})
            if not it.sources:
                notes.append(f"Briefing: dropped unsourced {kind} \"{label_of(it)}\"")
                continue
            kept.append(it)
        return kept

    participants = keep(briefing.participants, lambda p: p.name, "participant")
    entities = keep(briefing.entity_index, lambda e: e.reference, "entity mapping")
    glossary = keep(briefing.glossary, lambda g: g.term, "glossary term")

    from src.tools.page import parse_iso_date

    timeline = []
    for ev in keep(briefing.timeline, lambda t: t.event[:60], "timeline event"):
        d = parse_iso_date(ev.date)
        if d is None:
            notes.append(f'Briefing: dropped undated timeline event "{ev.event[:60]}"')
            continue
        if d > debate_date:
            notes.append(
                f'Briefing: dropped timeline event after debate date ({ev.date}): "{ev.event[:60]}"'
            )
            continue
        timeline.append(ev)

    disputes = [
        d.model_copy(update={"sources": clean(d.sources)}) for d in briefing.live_disputes
    ]
    return (
        briefing.model_copy(
            update={
                "participants": participants,
                "entity_index": entities,
                "glossary": glossary,
                "timeline": sorted(timeline, key=lambda t: t.date),
                "live_disputes": disputes,
            }
        ),
        notes,
    )


_SIDE_SK = {Side.COALITION: "koalícia", Side.OPPOSITION: "opozícia", Side.OTHER: "iné"}


def render_briefing(briefing: DebateBriefing, max_chars: int = 7000) -> str:
    """Compact Slovak rendering for the extraction prompt."""
    lines = [
        "POLITICKÝ KONTEXT (podklad na výber a pochopenie tvrdení — NIE dôkaz; "
        "nepoužívaj ho ako zdroj verdiktu):",
        f"Dátum debaty: {briefing.debate_date}. Téma: {briefing.topic}",
        "",
        "ÚČASTNÍCI:",
    ]
    for p in briefing.participants:
        bits = ", ".join(b for b in (p.party, p.office, _SIDE_SK[p.side]) if b)
        dossiers = f" — agendy: {', '.join(p.dossiers)}" if p.dossiers else ""
        lines.append(f"- {p.name} ({bits}){dossiers}")
    lines += ["", f"FUNKCIE k dátumu {briefing.debate_date}:"]
    for e in briefing.entity_index:
        lines.append(f'- "{e.reference}" = {e.person}')
    lines += ["", "SPORNÉ TÉMY (z nich odvodzuj dôležitosť tvrdení):"]
    for i, d in enumerate(briefing.live_disputes, 1):
        stakes = f"; v stávke: {d.stakes}" if d.stakes else ""
        lines.append(f"{i}. {d.title} — otázka: {d.factual_question}{stakes}")
    lines += ["", "UDALOSTI PRED DEBATOU:"]
    for t in briefing.timeline:
        lines.append(f"- {t.date}: {t.event}")
    lines += ["", "POJMY:"]
    for g in briefing.glossary:
        status = f" (stav: {g.status})" if g.status else ""
        lines.append(f"- {g.term}: {g.definition}{status}")

    out: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) + 1 > max_chars:
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_briefing.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/briefing.py tests/test_briefing.py
git commit -m "feat(briefing): models, provenance/anachronism filter, prompt rendering"
```

---

### Task 6: Briefing research pass

**Files:**
- Modify: `src/briefing.py` (append)
- Test: `tests/test_briefing.py` (append)

**Interfaces:**
- Consumes: `DebateBriefing`, `filter_briefing` (Task 5); `GroundedAnswer`, `grounded_search`, `generate_json` (Task 3).
- Produces: `run_briefing(transcript: str, debate_date: date, settings=None, *, search: Callable[[str], GroundedAnswer] | None = None, structure: Callable[[str], DebateBriefing] | None = None) -> tuple[DebateBriefing, list[str]]`

- [ ] **Step 1: Write the failing tests (append to `tests/test_briefing.py`)**

Add to the imports: `from src.briefing import run_briefing` and `from src.llm import GroundedAnswer`. Append:

```python
def test_run_briefing_unions_urls_filters_and_builds_three_research_prompts() -> None:
    searched: list[str] = []

    def search(prompt: str) -> GroundedAnswer:
        searched.append(prompt)
        return GroundedAnswer(text=f"poznámky {len(searched)}", urls=[GOOD] if len(searched) == 1 else [OTHER])

    seen_structure: list[str] = []

    def structure(prompt: str) -> DebateBriefing:
        seen_structure.append(prompt)
        return make_briefing()

    briefing, notes = run_briefing(
        "Moderátor [00:01]: Dobrý večer.", DEBATE, search=search, structure=structure
    )
    assert len(searched) == 3
    assert all("2026-04-12" in p and "Dobrý večer" in p for p in searched)
    assert len(seen_structure) == 1
    assert GOOD in seen_structure[0] and OTHER in seen_structure[0]
    assert "poznámky 1" in seen_structure[0] and "poznámky 3" in seen_structure[0]
    assert briefing.debate_date == "2026-04-12"
    assert [t.event for t in briefing.timeline] == ["Komisia zmrazila fondy"]  # FAKE/late/unsourced removed
    assert any("after debate date" in n for n in notes)


def test_run_briefing_survives_one_failed_research_call() -> None:
    calls = {"n": 0}

    def search(prompt: str) -> GroundedAnswer:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return GroundedAnswer(text="ok", urls=[GOOD])

    briefing, notes = run_briefing(
        "t", DEBATE, search=search, structure=lambda p: make_briefing()
    )
    assert any("research call" in n and "failed" in n for n in notes)
    assert briefing.participants
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_briefing.py -q`
Expected: FAIL — `ImportError: cannot import name 'run_briefing'`

- [ ] **Step 3: Implement**

Add to the imports at the top of `src/briefing.py`:

```python
import logging
from datetime import date
from typing import Callable

from src.llm import GroundedAnswer

logger = logging.getLogger(__name__)
```

Append to `src/briefing.py`:

```python
_RESEARCH_HEADER = (
    "Si politický redaktor, ktorý pripravuje podklady pred spracovaním televíznej "
    "debaty. DÁTUM DEBATY (referenčné 'teraz'): {ref}. Používaj iba informácie a "
    "zdroje datované do {ref}; nič neskoršie neuvádzaj. Píš vecne a opisne — "
    "nehodnoť, kto má pravdu, žiadne názory ani hodnotiace prívlastky. U každého "
    "údaja uveď dátum a vyhľadaj ho cez Google Search.\n\nPREPIS DEBATY:\n{transcript}\n\n"
)

_RESEARCH_TASKS = (
    # A: who is who
    "ÚLOHA: Identifikuj každého hosťa debaty (meno, strana, funkcia k dátumu "
    "debaty, či ide o koalíciu alebo opozíciu, predchádzajúce funkcie, agendy, "
    "ktoré drží). Potom nájdi VŠETKY odkazy na funkcie, ktoré v prepise zaznievajú "
    "('predseda vlády', 'prezident Poľska', 'minister financií', 'šéf Európskej "
    "komisie' …) a pre každý uveď, kto túto funkciu zastával k dátumu debaty.",
    # B: disputes and stakes
    "ÚLOHA: Urči 3 až 6 vecných sporov, o ktoré sa debata v skutočnosti opiera. "
    "Pri každom uveď: názov, overiteľnú faktickú otázku, ktorá je pod ním, a čo je "
    "v stávke pre voliča pri posudzovaní rečníka (peniaze, zodpovednosť, záznam, "
    "politika) — opisne, bez hodnotenia. Použi aktuálne spravodajstvo k týmto témam.",
    # C: timeline and glossary
    "ÚLOHA: Zostav časovú os udalostí z posledných ~90 dní pred dátumom debaty, "
    "ktoré súvisia so sporovými témami debaty (každá s presným dátumom YYYY-MM-DD). "
    "Potom vysvetli odborné pojmy a kauzy, ktoré v debate zaznievajú (neutrálna "
    "jednovetová definícia a stav veci k dátumu debaty).",
)

_STRUCTURE_PROMPT = (
    "Z nasledujúcich poznámok zostav štruktúrovaný politický brief debaty.\n"
    "PRAVIDLÁ:\n"
    "- Použi IBA fakty z poznámok; nič nedopĺňaj z vlastných znalostí.\n"
    "- Pole `sources` smie obsahovať IBA URL zo zoznamu ALLOWED_URLS nižšie, a to "
    "len tie, ktoré danú položku skutočne podporujú. Nevymýšľaj URL.\n"
    "- Dátumy vo formáte YYYY-MM-DD. Nič po dátume debaty ({ref}).\n"
    "- Opisuj, nehodnoť. Text píš po slovensky.\n"
    "- debate_date = {ref}.\n\n"
    "ALLOWED_URLS:\n{urls}\n\n"
    "POZNÁMKY A (účastníci, funkcie):\n{a}\n\n"
    "POZNÁMKY B (spory, čo je v stávke):\n{b}\n\n"
    "POZNÁMKY C (časová os, pojmy):\n{c}\n"
)


def run_briefing(
    transcript: str,
    debate_date: date,
    settings=None,
    *,
    search: Callable[[str], GroundedAnswer] | None = None,
    structure: Callable[[str], DebateBriefing] | None = None,
) -> tuple[DebateBriefing, list[str]]:
    """Research + structure + filter. `search`/`structure` are injectable for tests."""
    if search is None or structure is None:
        from config import get_settings
        from src.llm import generate_json, grounded_search

        settings = settings or get_settings()
        search = search or (
            lambda p: grounded_search(p, settings, label="briefing_research")
        )
        structure = structure or (
            lambda p: generate_json(p, DebateBriefing, settings, label="briefing_structure")
        )

    ref = debate_date.isoformat()
    notes: list[str] = []
    answers: list[GroundedAnswer | None] = []
    for i, task in enumerate(_RESEARCH_TASKS, 1):
        prompt = _RESEARCH_HEADER.format(ref=ref, transcript=transcript) + task
        try:
            answers.append(search(prompt))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Briefing research call %d failed: %s", i, exc)
            notes.append(f"Briefing: research call {i} failed ({exc}); continuing without it.")
            answers.append(None)

    urls: list[str] = []
    for a in answers:
        for u in (a.urls if a else []):
            if u not in urls:
                urls.append(u)
    text = [(a.text if a else "(výskum zlyhal)") for a in answers]

    briefing = structure(
        _STRUCTURE_PROMPT.format(
            ref=ref,
            urls="\n".join(urls) or "(žiadne)",
            a=text[0],
            b=text[1],
            c=text[2],
        )
    )
    briefing = briefing.model_copy(update={"debate_date": ref})
    filtered, filter_notes = filter_briefing(briefing, set(urls), debate_date)
    notes.extend(filter_notes)
    return filtered, notes
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_briefing.py -q && python3 -m pytest -q`
Expected: `6 passed` in `test_briefing.py`; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/briefing.py tests/test_briefing.py
git commit -m "feat(briefing): grounded research + structuring pass with in-code filtering"
```

---

### Task 7: Consequence-based adaptive claim selection

**Files:**
- Modify: `src/agents.py` (`ExtractedClaim` only, plus the `Checkability` enum)
- Create: `src/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Produces:
  - `Checkability` (str enum `empirical|opinion|prediction|definitional`) in `src/agents.py`
  - New `ExtractedClaim` fields: `consequence: int = 3` (1–5), `checkability: Checkability = EMPIRICAL`, `consequence_reason: str = ""`, `repeats: int = 1`
  - `priority(c: ExtractedClaim) -> float`
  - `merge_duplicates(claims: list[ExtractedClaim], min_jaccard: float = 0.7) -> tuple[list[ExtractedClaim], list[str]]`
  - `select_claims(claims: list[ExtractedClaim], *, threshold: int, floor_per_speaker: int, fuse: int) -> tuple[list[ExtractedClaim], list[str]]` — returned claims have `salience` set to `consequence`.

- [ ] **Step 1: Extend `ExtractedClaim` in `src/agents.py`**

Directly above `class ExtractedClaim(BaseModel):` add:

```python
class Checkability(str, Enum):
    """Only empirical claims enter fact-checking and scoring."""

    EMPIRICAL = "empirical"  # verifiable against records, data, or reporting
    OPINION = "opinion"  # value judgement, evaluation, rhetoric
    PREDICTION = "prediction"  # about the future
    DEFINITIONAL = "definitional"  # about meaning of a term, translation, semantics


```

Inside `ExtractedClaim`, directly after the `usage_reason` field block (the `Field(default="", description="One short sentence justifying the usage/salience rating.")`), add:

```python
    consequence: int = Field(
        default=3,
        ge=1,
        le=5,
        description=(
            "Political consequence of the claim for how a voter judges the speaker, "
            "measured against the briefing's disputes and stakes: 5 = decides a "
            "central dispute (money, responsibility, record); 1 = trivia."
        ),
    )
    consequence_reason: str = Field(
        default="",
        description="One short sentence naming the dispute/stake the rating rests on.",
    )
    checkability: Checkability = Field(
        default=Checkability.EMPIRICAL,
        description="empirical / opinion / prediction / definitional.",
    )
    repeats: int = Field(
        default=1,
        ge=1,
        description="How many times the speaker made this claim (set in code when merging duplicates).",
    )
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_selection.py`:

```python
"""Offline tests for adaptive claim selection (src/selection.py)."""

from __future__ import annotations

from src.agents import Checkability, ClaimUsage, ExtractedClaim
from src.selection import merge_duplicates, priority, select_claims


def claim(i: int, speaker: str = "A", consequence: int = 3, check=Checkability.EMPIRICAL, text: str | None = None, usage=ClaimUsage.OTHER) -> ExtractedClaim:
    return ExtractedClaim(
        id=i,
        claim=text or f"unikátne tvrdenie číslo {i} o rozpočte {i * 7}",
        speaker=speaker,
        quote="q",
        consequence=consequence,
        checkability=check,
        usage=usage,
        salience=1,
    )


def ids(claims) -> list[int]:
    return [c.id for c in claims]


def test_priority_role_and_repeats_are_secondary_multipliers() -> None:
    plain = claim(1, consequence=4)
    attack = claim(2, consequence=4, usage=ClaimUsage.ATTACK)
    repeated = claim(3, consequence=4).model_copy(update={"repeats": 3})
    assert priority(attack) > priority(plain)
    assert priority(repeated) > priority(plain)
    assert priority(claim(4, consequence=5)) > priority(attack)  # consequence dominates


def test_non_empirical_claims_excluded_and_reported() -> None:
    cs = [claim(1), claim(2, check=Checkability.DEFINITIONAL), claim(3, check=Checkability.OPINION)]
    kept, notes = select_claims(cs, threshold=3, floor_per_speaker=2, fuse=40)
    assert ids(kept) == [1]
    assert any("non-empirical" in n and "#2" in n and "#3" in n for n in notes)


def test_threshold_and_floor() -> None:
    cs = [
        claim(1, "A", 5), claim(2, "A", 4),
        claim(3, "B", 2), claim(4, "B", 1), claim(5, "B", 1),
    ]
    kept, notes = select_claims(cs, threshold=3, floor_per_speaker=2, fuse=40)
    # A: both >= 3. B: nothing reaches 3, but the floor keeps their top two (3 and 4 by id order on ties).
    assert ids(kept) == [1, 2, 3, 4]
    assert any("threshold" in n and "#5" in n for n in notes)


def test_floor_never_admits_non_empirical() -> None:
    cs = [claim(1, "B", 1, check=Checkability.OPINION), claim(2, "B", 2)]
    kept, _ = select_claims(cs, threshold=3, floor_per_speaker=2, fuse=40)
    assert ids(kept) == [2]


def test_merge_duplicates_same_speaker_only() -> None:
    a = "deficit verejných financií dosiahol šesť percent HDP"
    b = "deficit verejných financií dosiahol šesť percent HDP minulý rok"
    cs = [claim(1, "A", 3, text=a), claim(2, "A", 4, text=b), claim(3, "B", 3, text=a)]
    merged, notes = merge_duplicates(cs)
    assert ids(merged) == [2, 3]  # higher consequence survives; other speaker untouched
    assert merged[0].repeats == 2
    assert notes and "#1" in notes[0]


def test_fuse_cuts_lowest_priority_but_protects_floor() -> None:
    cs = [claim(i, "A", 5) for i in range(1, 5)] + [claim(i, "B", 3) for i in range(5, 9)]
    cs[3] = claim(4, "A", 4)
    kept, notes = select_claims(cs, threshold=3, floor_per_speaker=1, fuse=4)
    assert len(kept) == 4
    speakers = {c.speaker for c in kept}
    assert speakers == {"A", "B"}  # floor keeps B alive even though A outranks
    assert any("cap" in n.lower() for n in notes)


def test_selected_claims_get_salience_from_consequence() -> None:
    kept, _ = select_claims([claim(1, "A", 5)], threshold=3, floor_per_speaker=1, fuse=40)
    assert kept[0].salience == 5
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_selection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.selection'`

- [ ] **Step 4: Implement `src/selection.py`**

```python
"""Consequence-based adaptive claim selection (replaces the top-N salience cut).

Order of operations:
  1. drop non-empirical claims (opinion / prediction / definitional) — reported;
  2. merge near-duplicate claims by the same speaker (repeats recorded);
  3. keep claims with consequence >= threshold, plus every speaker's top
     `floor_per_speaker` claims so no speaker reaches scoring on a single check;
  4. if still above the cost fuse, cut lowest priority — never below the floor.

`salience` on returned claims is set to `consequence`, so scoring (which weights
errors by salience/3) keeps working unchanged.

Imported lazily from `run_analysis` (this module imports `src.agents`).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from src.agents import Checkability, ClaimUsage, ExtractedClaim
from src.validation import normalize

_ROLE_MULT = {
    ClaimUsage.ATTACK: 1.15,
    ClaimUsage.COUNTERARGUMENT: 1.15,
    ClaimUsage.DEFLECTION: 1.10,
    ClaimUsage.SUPPORTING: 1.0,
    ClaimUsage.OTHER: 1.0,
}


def priority(c: ExtractedClaim) -> float:
    """Consequence first; rhetorical role and repetition are secondary multipliers."""
    return c.consequence * _ROLE_MULT.get(c.usage, 1.0) * (1.0 + 0.1 * (c.repeats - 1))


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(normalize(a).split()), set(normalize(b).split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def merge_duplicates(
    claims: list[ExtractedClaim], min_jaccard: float = 0.7
) -> tuple[list[ExtractedClaim], list[str]]:
    notes: list[str] = []
    survivors: list[ExtractedClaim] = []
    for c in sorted(claims, key=lambda c: -priority(c)):
        twin = next(
            (
                s
                for s in survivors
                if s.speaker == c.speaker and _jaccard(s.claim, c.claim) >= min_jaccard
            ),
            None,
        )
        if twin is None:
            survivors.append(c)
            continue
        idx = survivors.index(twin)
        survivors[idx] = twin.model_copy(update={"repeats": twin.repeats + c.repeats})
        notes.append(
            f'Merged duplicate claim #{c.id} into #{twin.id} ({c.speaker}): "{c.claim[:60]}"'
        )
    return sorted(survivors, key=lambda c: c.id), notes


def select_claims(
    claims: list[ExtractedClaim],
    *,
    threshold: int,
    floor_per_speaker: int,
    fuse: int,
) -> tuple[list[ExtractedClaim], list[str]]:
    notes: list[str] = []

    excluded = [c for c in claims if c.checkability != Checkability.EMPIRICAL]
    if excluded:
        notes.append(
            "Excluded non-empirical claims (not fact-checked, not scored): "
            + "; ".join(
                f'#{c.id} ({c.checkability.value}) "{c.claim[:60]}"' for c in excluded
            )
        )
    empirical = [c for c in claims if c.checkability == Checkability.EMPIRICAL]

    merged, merge_notes = merge_duplicates(empirical)
    notes.extend(merge_notes)

    by_speaker: dict[str, list[ExtractedClaim]] = defaultdict(list)
    for c in merged:
        by_speaker[c.speaker].append(c)
    protected: set[int] = set()
    for lst in by_speaker.values():
        ranked = sorted(lst, key=lambda c: (-priority(c), c.id))
        protected.update(c.id for c in ranked[:floor_per_speaker])

    kept = [c for c in merged if c.consequence >= threshold or c.id in protected]
    kept_ids = {c.id for c in kept}
    below = [c for c in merged if c.id not in kept_ids]
    if below:
        notes.append(
            f"Below consequence threshold {threshold} (not checked): "
            + "; ".join(f'#{c.id} (c{c.consequence}) "{c.claim[:60]}"' for c in below)
        )

    if len(kept) > fuse:
        removable = sorted(
            (c for c in kept if c.id not in protected), key=lambda c: (priority(c), -c.id)
        )
        cut: list[ExtractedClaim] = []
        for c in removable:
            if len(kept) - len(cut) <= fuse:
                break
            cut.append(c)
        cut_ids = {c.id for c in cut}
        kept = [c for c in kept if c.id not in cut_ids]
        notes.append(
            f"Claim cap (cost fuse {fuse}): dropped {len(cut)} lowest-priority claims: "
            + "; ".join(f'#{c.id} (c{c.consequence}) "{c.claim[:60]}"' for c in cut)
        )
        per_speaker = Counter(c.speaker for c in cut)
        notes.append(
            "Unchecked claims per speaker (dropped by cap): "
            + ", ".join(f"{s}: {n}" for s, n in per_speaker.most_common())
        )

    out = [c.model_copy(update={"salience": max(1, min(5, c.consequence))}) for c in kept]
    return sorted(out, key=lambda c: c.id), notes
```

- [ ] **Step 5: Run tests and the full suite**

Run: `python3 -m pytest tests/test_selection.py -q && python3 -m pytest -q`
Expected: `7 passed`; full suite green. (`ExtractedClaim` changes are additive with defaults, so `test_manager.py` and `test_scoring.py` are unaffected.)

- [ ] **Step 6: Commit**

```bash
git add src/agents.py src/selection.py tests/test_selection.py
git commit -m "feat(selection): consequence threshold, per-speaker floor, dedupe, cost fuse"
```

---

### Task 8: Wire the briefing into extraction and `run_analysis`

**Files:**
- Modify: `src/agents.py`
- Test: `tests/test_extract_prompt.py`

**Interfaces:**
- Consumes: `DebateBriefing`, `run_briefing`, `render_briefing` (Tasks 5–6); `select_claims` (Task 7); settings from Task 3.
- Produces:
  - `AnalysisReport.briefing: DebateBriefing | None`
  - `_importance_block(briefing_text: str) -> str`
  - `build_extract_crew(transcript, settings=None, debate_date=None, briefing_text: str = "")`

- [ ] **Step 1: Write the failing test**

Create `tests/test_extract_prompt.py`:

```python
"""The extraction prompt switches from rhetoric-based salience to briefing-based consequence."""

from __future__ import annotations

from src.agents import AnalysisReport, _importance_block


def test_legacy_block_when_no_briefing() -> None:
    block = _importance_block("")
    assert "salience" in block
    assert "consequence" not in block.lower()


def test_consequence_block_embeds_briefing_and_rules() -> None:
    block = _importance_block("POLITICKÝ KONTEXT: Eurofondy")
    assert "POLITICKÝ KONTEXT: Eurofondy" in block
    assert "consequence" in block
    assert "checkability" in block
    assert "definitional" in block
    assert "entity" in block.lower()  # role references resolved via the briefing


def test_report_briefing_defaults_to_none() -> None:
    assert AnalysisReport().briefing is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_extract_prompt.py -q`
Expected: FAIL — `ImportError: cannot import name '_importance_block'`

- [ ] **Step 3: Add the report field and import**

In `src/agents.py`, below `from src.costs import TRACKER` add:

```python
from src.briefing import DebateBriefing
```

In `class AnalysisReport(BaseModel):`, add as the last field:

```python
    briefing: DebateBriefing | None = Field(
        default=None,
        description="Phase 0 political briefing (background only, never evidence).",
    )
```

- [ ] **Step 4: Add `_importance_block` above `build_extract_crew`**

Insert immediately above `def build_extract_crew(`:

```python
_IMPORTANCE_LEGACY = (
    "IMPORTANCE (salience 1-5): rate each claim by its ROLE IN THE DEBATE, "
    "not by how easy it is to verify. Use the Behavioral Analyst's findings "
    "(attacks, manipulation, deflections) from the prior task as evidence:\n"
    "- 5 = the claim is used as a COUNTERARGUMENT to an opponent, an ATTACK "
    "on an opponent, or a DEFLECTION/smokescreen to dodge a question.\n"
    "- 4 = repeated multiple times or central to the debate's main topic.\n"
    "- 3 = relevant supporting fact for the speaker's argument.\n"
    "- 2-1 = side mention, common knowledge, or trivial detail — these are "
    "NOT worth fact-checking; be strict and rate them low.\n"
    "Set `usage` (counterargument/attack/deflection/supporting/other) and a "
    "one-sentence `usage_reason`.\n\n"
)

_IMPORTANCE_CONSEQUENCE = (
    "CONSEQUENCE (consequence 1-5): rate each claim by its POLITICAL CONSEQUENCE, "
    "judged against the SPORNÉ TÉMY (disputes and stakes) in the briefing above — "
    "not by rhetorical flourish and not by how easy it is to verify:\n"
    "- 5 = decides a central dispute: money, responsibility, or a politician's "
    "record that changes how a voter judges the speaker.\n"
    "- 4 = bears directly on a live dispute or on a named actor's conduct.\n"
    "- 3 = relevant supporting fact for an argument about a live dispute.\n"
    "- 2-1 = side mention, common knowledge, translation/semantics, trivia "
    "unrelated to any dispute — be strict and rate these low.\n"
    "Write a one-sentence `consequence_reason` naming the dispute/stake. Set "
    "`usage` (counterargument/attack/deflection/supporting/other) and "
    "`usage_reason` as before; usage is only a secondary signal.\n"
    "CHECKABILITY: set `checkability` to empirical (verifiable against records, "
    "data or reporting), opinion (value judgement or evaluation), prediction "
    "(about the future) or definitional (meaning of a term, translation, "
    "semantics). Only empirical claims will be fact-checked.\n"
    "ENTITY RESOLUTION: in `context`, resolve vague role references "
    "('prezident Poľska', 'minister financií', 'bývalá vláda') to the concrete "
    "person or government using the briefing's FUNKCIE list, valid on the debate "
    "date. The briefing is BACKGROUND for understanding what the claim refers to; "
    "it is not evidence for or against the claim.\n\n"
)


def _importance_block(briefing_text: str) -> str:
    if not briefing_text:
        return _IMPORTANCE_LEGACY
    return f"{briefing_text}\n\n{_IMPORTANCE_CONSEQUENCE}"
```

- [ ] **Step 5: Use it in `build_extract_crew`**

Change the signature:

```python
def build_extract_crew(
    transcript: str,
    settings: Settings | None = None,
    debate_date: "date | None" = None,
    briefing_text: str = "",
):
```

In `t_extract`'s description, replace the whole legacy importance block — everything from the string literal `"IMPORTANCE (salience 1-5): rate each claim by its ROLE IN THE DEBATE, "` through `"one-sentence `usage_reason`.\n\n"` — with the single expression:

```python
            f"{_importance_block(briefing_text)}"
```

(The surrounding `"CATEGORY: classify each claim..."` string literal that follows stays as is.)

Also update `expected_output` of `t_extract` to:

```python
        expected_output=(
            "ExtractedClaims JSON: list of {id, claim, speaker, quote, salience, "
            "usage, usage_reason, consequence, consequence_reason, checkability, "
            "category, context, time_reference, time_window} with sequential ids."
        ),
```

- [ ] **Step 6: Wire Phase 0 and selection into `run_analysis`**

Directly before the `# Phase A: behavioral + moderator + context-aware extraction.` comment, insert:

```python
    # Phase 0: political briefing (background only — never evidence).
    briefing: DebateBriefing | None = None
    briefing_text = ""
    if settings.briefing_enabled and debate_date is not None:
        from src.briefing import render_briefing, run_briefing

        try:
            briefing, briefing_notes = run_briefing(working, debate_date, settings)
            briefing_text = render_briefing(briefing)
            pipeline_notes.extend(briefing_notes)
            logger.info(
                "Briefing: %d participants, %d disputes, %d timeline events",
                len(briefing.participants),
                len(briefing.live_disputes),
                len(briefing.timeline),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Briefing failed; continuing with legacy extraction")
            pipeline_notes.append(
                f"Briefing failed ({exc}); used legacy salience-based selection."
            )
            briefing = None
    elif settings.briefing_enabled:
        pipeline_notes.append("Briefing skipped: no --debate-date given.")
```

Change the extract-crew call:

```python
    extract_crew, a_tasks = build_extract_crew(
        working,
        settings=settings,
        debate_date=debate_date,
        briefing_text=briefing_text,
    )
```

Replace the top-N selection block:

```python
    if extracted and extracted.claims:
        kept_claims, cap_notes = select_top_claims(extracted, settings.max_claims)
```
with:
```python
    if extracted and extracted.claims:
        if briefing is not None:
            from src.selection import select_claims

            kept_claims, cap_notes = select_claims(
                extracted.claims,
                threshold=settings.consequence_threshold,
                floor_per_speaker=settings.claims_floor_per_speaker,
                fuse=settings.max_claims,
            )
        else:
            kept_claims, cap_notes = select_top_claims(extracted, min(settings.max_claims, 30))
```

Finally, directly after `report.facts = facts`, add:

```python
    report.briefing = briefing
```

- [ ] **Step 7: Run tests and a syntax/import smoke check**

Run: `python3 -m pytest -q && python3 -c "import src.agents, src.briefing, src.selection, src.judge, src.eval; print('imports ok')"`
Expected: all pass; `imports ok` (this also proves there is no import cycle).

- [ ] **Step 8: Commit**

```bash
git add src/agents.py tests/test_extract_prompt.py
git commit -m "feat: run political briefing before extraction; adaptive consequence-based selection"
```

---

### Task 9: Acceptance run and comparison (manual, Vertex + network)

Re-run one debate that already has a baseline, end to end, and compare. Uses the stored transcript, so no download or ASR.

- [ ] **Step 1: Run the new pipeline on 591624 with the judge on**

```bash
JUDGE_ENABLED=true .venv/bin/python main.py \
  --url https://www.stvr.sk/televizia/archiv/14036/591624 \
  --transcript data/transcripts/591624.txt \
  --episode-id 591624_v2 \
  --debate-date 2026-04-12 2>&1 | tee data/reports/run_591624_v2.log | tail -40
```
Expected: log lines `Briefing: N participants, M disputes, K timeline events`, `Judge: mean evidence quality …`, a cost summary with `briefing_research`, `briefing_structure`, `judge` rows, and `Report written: data/reports/591624_v2.json`.

- [ ] **Step 2: Check the spec's acceptance criteria against the output**

```bash
python3 - <<'EOF'
import json
d = json.load(open("data/reports/591624_v2.json"))
print("briefing participants:", [p["name"] for p in d["briefing"]["participants"]])
print("disputes:", [x["title"] for x in d["briefing"]["live_disputes"]])
print("facts:", len(d["facts"]))
for f in d["facts"]:
    print(f'[{f["verdict"]}] s{f.get("salience")} {f["speaker"]}: {f["claim"][:110]}')
print("---- selection notes")
for n in d["critic_notes"]:
    if n.startswith(("Excluded", "Below", "Merged", "Claim cap", "Briefing")):
        print(n[:300])
print("judge:", d["judge"]["mean_quality"], "judged", d["judge"]["judged"], "downgrades", d["judge"]["downgrades"])
EOF
```
Check by eye against the spec's definition of done:
- The briefing names the real participants and the EU-funds / Beneš-style disputes of that debate, with dated, sourced timeline entries and no entry after 2026-04-12.
- "Juden Mord translates to murder of Jews" (and similar definitional/trivia claims) no longer occupies a fact-check slot; it appears under `Excluded non-empirical claims` or `Below consequence threshold`.
- Every speaker has at least 2 checked facts.
- `judge.mean_quality` for `591624_v2` is at least the baseline for `591624` from Task 4 Step 8. If it is lower, do not proceed: inspect `data/reports/591624_v2.json` → `judge.judgements[*].issues` to see what the judge objects to, and report findings before starting the next plan.

- [ ] **Step 3: Record the result and commit**

Write the baseline vs new numbers (overall mean quality, by-verdict means, claim counts, what the selection dropped) into a short section appended to `docs/factcheck-quality-spec.md` under a new heading `## Results: steps 1–2`, then:

```bash
git add docs/factcheck-quality-spec.md
git commit -m "docs: record baseline vs briefing+selection results"
```

---

## Self-Review

**1. Spec coverage (steps 1–2 of the spec's migration order):**
- Judge with five axes, downgrade-only, code-applied, run-level score stored with report → Tasks 2, 3, 4 (`payload["judge"]`).
- `--rejudge` baseline over stored reports → Task 4 (`python -m src.eval rejudge`, date seeding and gap-filling).
- Briefing model fields (participants, entity_index, live_disputes, timeline, glossary, stakes) → Task 5. `stakes_map` is folded into `Dispute.stakes` (one record per dispute instead of a parallel map); same information, noted here rather than silently dropped.
- Briefing sourcing/anachronism constraints, descriptive-only → Task 5 filter + Task 6 prompts; "no background as evidence" → render header plus Global Constraints; not passed to any checker (checkers are untouched; only extraction's `context` uses it).
- `consequence`, `checkability`, dedupe/repetition multiplier, adaptive threshold, per-speaker floor, cost fuse default 40 with reporting → Tasks 3 (config), 7, 8.
- Explicitly deferred (stated under Out of scope): research loop, adjudicator, tier map, tools, calibration, Slovak claim text, publisher using the briefing, confidence gating, `unverified_reason`.

**2. Placeholder scan:** no TBD/"handle edge cases"; every code step has full code. The only manual-data gap is the nine missing debate dates, handled by an explicit instruction to ask the user and a `skipped_no_date` path.

**3. Type consistency:** `PageResult` fields and `fetch_page` signature match across Tasks 1/3/4; `JudgeOutput`/`JudgeAxes`/`FactJudgement`/`JudgeSummary` match between Tasks 2, 3, 4; `judge_fact(fact, fact_index, debate_date, *, fetch, llm, max_sources)` is called identically in `judge_facts` and the tests; `run_briefing(..., search=, structure=)` and `GroundedAnswer` match between Tasks 3 and 6; `select_claims(..., threshold=, floor_per_speaker=, fuse=)` matches Task 8's call and the `Settings` names (`consequence_threshold`, `claims_floor_per_speaker`, `max_claims`). `ExtractedClaim` new fields (`consequence`, `consequence_reason`, `checkability`, `repeats`) are used consistently in Tasks 7–8.

**Known risk flagged for the implementer:** Gemini structured output with `response_schema=DebateBriefing` (nested models, enums) is supported by `google-genai` 1.49, but if Vertex rejects a schema construct, `generate_json` will raise; `run_analysis` catches this and falls back to the legacy path with a `critic_notes` entry, so a schema problem degrades gracefully rather than failing the run.
