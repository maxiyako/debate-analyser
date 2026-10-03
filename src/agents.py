"""CrewAI multi-agent analysis on Vertex AI Gemini with context caching."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from enum import Enum
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field

from config import Settings, get_settings
from src.costs import TRACKER
from src.briefing import DebateBriefing
from src.report_models import (
    ClaimFunnel,
    EditResult,
    QuestionItem,
    SpeakerMap,
    TranscriptQuality,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Every human-readable field in the final report must be Slovak.
_SK_OUTPUT = (
    "\n\nJAZYK VÝSTUPU: Všetok voľný text píš po SLOVENSKY (zhrnutia, opisy "
    "taktík, manipulácií, faulov, otázkových únikov, poznámky, findings, "
    "interruption_frequency, zdôvodnenia/rationale, subfacts, intent_signals). "
    "Doslovné citácie z prepisu (evidence quotes) ponechaj v pôvodnom znení. "
    "Hodnoty verdict (True/False/Misleading/Unverified/Contested) a severity "
    "ponechaj v angličtine — sú to enum hodnoty."
)

# Vertex 429 RESOURCE_EXHAUSTED — CrewAI re-raises litellm errors with no backoff.
_RATE_LIMIT_ATTEMPTS = 8
_RATE_LIMIT_BASE_DELAY_S = 8.0
_RATE_LIMIT_MAX_DELAY_S = 120.0


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        s in text
        for s in (
            "ratelimit",
            "rate limit",
            "resource_exhausted",
            "resource exhausted",
            "429",
        )
    )


def _is_transient_llm_error(exc: BaseException) -> bool:
    """True for known-flaky Vertex/CrewAI failures worth a blind retry.

    Besides 429s, Gemini's native tool-calling occasionally returns an empty
    response for no discernible reason (CrewAI then raises this ValueError).
    Retrying the same call almost always succeeds.
    """
    if _is_rate_limit_error(exc):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        s in text
        for s in (
            "invalid response from llm call",
            "timeout",
            "timed out",
            "deadline exceeded",
        )
    )


def _retry_on_rate_limit(fn: Callable[[], T]) -> T:
    """Retry Vertex/LiteLLM 429s and other transient failures with backoff + jitter."""
    last: BaseException | None = None
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_transient_llm_error(exc) or attempt >= _RATE_LIMIT_ATTEMPTS - 1:
                raise
            delay = min(
                _RATE_LIMIT_MAX_DELAY_S,
                _RATE_LIMIT_BASE_DELAY_S * (2**attempt),
            )
            delay += random.uniform(0, min(3.0, delay * 0.15))
            logger.warning(
                "Transient Vertex/LLM error (attempt %d/%d): %s — sleeping %.1fs",
                attempt + 1,
                _RATE_LIMIT_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last is not None
    raise last


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    TRUE = "True"
    FALSE = "False"
    MISLEADING = "Misleading"  # evidence shows the claim distorts reality
    UNVERIFIED = "Unverified"  # no usable evidence either way
    CONTESTED = "Contested"  # assigned in code when sourced checkers disagree


class Severity(str, Enum):
    """How far a non-true claim is from reality (not speaker intent)."""

    TRIVIAL = "trivial"
    MATERIAL = "material"
    FABRICATION = "fabrication"


class Civility(BaseModel):
    """Sociologically grounded civility, not naive politeness.

    Assertiveness, directness, and stating uncomfortable truths are NOT
    incivility. Penalize only genuine norm violations: personal insults,
    contempt/dehumanization, bad-faith interrupting, shouting over others,
    lies weaponized to demean, condescension aimed at silencing.
    """

    score: float = Field(
        default=5.0,
        description="0 (hostile/contemptuous) to 10 (respectful, still assertive)",
    )
    incivility: list[str] = Field(
        default_factory=list,
        description="Genuine norm violations with short evidence quotes.",
    )
    fair_conduct: list[str] = Field(
        default_factory=list,
        description="Respectful-yet-assertive moments (credit where due).",
    )
    notes: str = ""


class SpeakerTactics(BaseModel):
    speaker: str
    speech_tactics: list[str] = Field(default_factory=list)
    manipulation: list[str] = Field(default_factory=list)
    logical_fallacies: list[str] = Field(default_factory=list)
    question_dodging: list[str] = Field(default_factory=list)
    civility: Civility = Field(default_factory=Civility)
    notes: str = ""


class BehavioralAnalysis(BaseModel):
    speakers: list[SpeakerTactics] = Field(default_factory=list)
    summary: str = ""


class TimeShare(BaseModel):
    speaker: str
    approximate_share_percent: float = 0.0
    turns: int = 0


class ModeratorAudit(BaseModel):
    neutrality_score: float = Field(
        default=5.0,
        description="0 (heavily biased) to 10 (fully neutral)",
    )
    interruption_frequency: str = ""
    equal_time_distribution: list[TimeShare] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    summary: str = ""


class VerifiedFact(BaseModel):
    claim: str
    speaker: str
    quote: str = Field(
        default="",
        description=(
            "A short verbatim excerpt COPIED from the transcript that the claim is "
            "based on. Must be exact transcript text, not a paraphrase — this is "
            "used to catch hallucinated/misattributed claims."
        ),
    )
    verdict: Verdict
    severity: Severity | None = Field(
        default=None,
        description=(
            "Required when verdict is False or Misleading; omit for "
            "True/Unverified/Contested."
        ),
    )
    intent_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence-based signals only (e.g. overlaps with manipulation, "
            "publicly debunked narrative, repeated after correction). Never assert intent."
        ),
    )
    sources: list[str] = Field(default_factory=list)
    salience: int = Field(
        default=3,
        ge=1,
        le=5,
        description=(
            "Rhetorical weight copied from extraction. 5 = attack, counter, or "
            "deflection; 1 = trivia. Scoring weights errors by salience/3."
        ),
    )
    rationale: str = ""
    quote_raw: str = Field(
        default="",
        description="The matching window of the original ASR transcript (before correction).",
    )
    timestamp: str = ""
    transcript_edits: list[int] = Field(
        default_factory=list,
        description="Ids of applied transcript edits on the quoted line(s).",
    )


class AnalysisReport(BaseModel):
    summary: str = ""
    behavioral_analysis: BehavioralAnalysis = Field(default_factory=BehavioralAnalysis)
    moderator_audit: ModeratorAudit = Field(default_factory=ModeratorAudit)
    facts: list[VerifiedFact] = Field(default_factory=list)
    critic_notes: list[str] = Field(default_factory=list)
    briefing: DebateBriefing | None = Field(
        default=None,
        description="Phase 0 political briefing (background only, never evidence).",
    )
    speaker_map: SpeakerMap | None = None
    question_audit: list[QuestionItem] = Field(default_factory=list)
    claim_funnel: list[ClaimFunnel] = Field(default_factory=list)
    transcript_quality: TranscriptQuality | None = None


class ClaimHighlight(BaseModel):
    quote: str
    speaker: str = ""
    verdict: Verdict
    severity: Severity | None = None
    source_url: str = ""


class FacebookPost(BaseModel):
    headline: str = ""
    body: str = ""
    top_claims: list[ClaimHighlight] = Field(default_factory=list)
    disclaimer: str = ""


class CorrectedTranscript(BaseModel):
    text: str = Field(
        description=(
            "Full corrected transcript in the same line format: "
            "'Speaker X [MM:SS]: utterance'. Preserve timestamps; fix speaker labels."
        )
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Short notes on speaker reassignments and splits performed.",
    )
    log: list[EditResult] = Field(
        default_factory=list,
        description="Every proposed transcript edit with its guard decision.",
    )


class ClaimUsage(str, Enum):
    """How the speaker used the claim in the debate."""

    COUNTERARGUMENT = "counterargument"
    ATTACK = "attack"
    DEFLECTION = "deflection"
    SUPPORTING = "supporting"
    OTHER = "other"


class ClaimCategory(str, Enum):
    """Routing category for specialist fact-checkers."""

    ECONOMY_FINANCE = "economy_finance"
    STATISTICS = "statistics"
    HISTORY_POLITICS = "history_politics"
    CURRENT_EVENTS = "current_events"


class Checkability(str, Enum):
    """Only empirical claims enter fact-checking and scoring."""

    EMPIRICAL = "empirical"  # verifiable against records, data, or reporting
    OPINION = "opinion"  # value judgement, evaluation, rhetoric
    PREDICTION = "prediction"  # about the future
    DEFINITIONAL = "definitional"  # about meaning of a term, translation, semantics


class ExtractedClaim(BaseModel):
    id: int = Field(description="Stable 1-based claim id for cross-agent matching.")
    claim: str
    speaker: str
    quote: str = Field(
        description="Verbatim transcript excerpt the claim is based on."
    )
    salience: int = Field(
        default=3,
        ge=1,
        le=5,
        description=(
            "Importance in the debate: 5 = used as a counterargument, attack or "
            "deflection; 4 = repeated or central to the debate topic; 3 = relevant "
            "supporting fact; 2-1 = side mention, common knowledge, or triviality."
        ),
    )
    usage: ClaimUsage = Field(
        default=ClaimUsage.OTHER,
        description="Rhetorical role of the claim: counterargument / attack / "
        "deflection / supporting / other.",
    )
    usage_reason: str = Field(
        default="",
        description="One short sentence justifying the usage/salience rating.",
    )
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
    category: ClaimCategory = Field(
        default=ClaimCategory.CURRENT_EVENTS,
        description=(
            "economy_finance = budget, deficit, debt, taxes, salaries; "
            "statistics = inflation, wages, unemployment, demographics (official "
            "statistical indicators); history_politics = votes, laws, past "
            "governments, past statements; current_events = recent events, "
            "quote attributions, media affairs."
        ),
    )
    context: str = Field(
        default="",
        description=(
            "One sentence: what in the debate the claim was reacting to (needed "
            "for precise verification, e.g. that 'inflation' meant food inflation)."
        ),
    )
    time_reference: str = Field(
        default="",
        description=(
            "Verbatim time expression the speaker used, if any "
            "(e.g. 'pred dvomi mesiacmi', 'minulý rok', 'v roku 2014'). Empty if "
            "the claim carries no explicit time reference."
        ),
    )
    time_window: str = Field(
        default="",
        description=(
            "The time_reference resolved to an absolute window RELATIVE TO THE "
            "DEBATE DATE (e.g. debate 2026-06-14 + 'pred dvomi mesiacmi' -> "
            "'~2026-04'; 'v roku 2014' -> '2014'). Empty if no time reference."
        ),
    )


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


class ClaimCheck(BaseModel):
    claim_id: int = Field(description="Must match ExtractedClaim.id from the extractor.")
    verdict: Verdict
    severity: Severity | None = None
    sources: list[str] = Field(default_factory=list)
    intent_signals: list[str] = Field(default_factory=list)
    rationale: str = ""


class ClaimCheckList(BaseModel):
    checks: list[ClaimCheck] = Field(default_factory=list)


class RemovedSource(BaseModel):
    url: str
    reason: str = Field(description="Why the source was dropped (dead, anachronistic, off-topic...).")


class ManagerReview(BaseModel):
    """Second-pass review of one claim by the fact-check manager."""

    claim_id: int = Field(description="Must match ExtractedClaim.id.")
    verdict: Verdict
    severity: Severity | None = None
    sources: list[str] = Field(
        default_factory=list,
        description="FINAL source list for the claim — only URLs returned by tools in this run.",
    )
    removed_sources: list[RemovedSource] = Field(
        default_factory=list,
        description="First-pass sources dropped during the audit, each with a reason.",
    )
    subfacts: list[str] = Field(
        default_factory=list,
        description="Atomic sub-facts checked, each with its one-line result.",
    )
    escalate: bool = Field(
        default=False,
        description=(
            "True only when the claim is still Unverified/Contested AND a concrete "
            "lead was found that one more focused round could confirm."
        ),
    )
    rationale: str = ""


class ManagerReviewList(BaseModel):
    reviews: list[ManagerReview] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fact-checking search tools (CrewAI BaseTool)
# ---------------------------------------------------------------------------


def _resolve_redirect(url: str, timeout: float = 8.0) -> str:
    """Resolve a Vertex grounding redirect to the real publisher URL.

    Grounding returns opaque `vertexaisearch.cloud.google.com/...redirect/...`
    URIs; without resolving them the agent only sees a redirect + bare domain
    and tends to fabricate a plausible full URL. Falls back to the input on error.
    """
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    # Only follow Vertex's own grounding-redirect host — never an arbitrary
    # model-supplied URL (SSRF guard).
    if not host.endswith("vertexaisearch.cloud.google.com"):
        return url
    import requests

    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        return resp.url or url
    except Exception:  # noqa: BLE001
        return url


def _build_grounded_search_tool(
    settings: Settings,
    seen_urls: set[str] | None = None,
    debate_date: "date | None" = None,
):
    """GCP-native tool: Vertex Gemini Grounding with Google Search.

    Uses the Vertex AI project we already run on — no separate API key.
    Returns a grounded answer plus the REAL source URLs Google cited (redirects
    resolved), restricted to the same FCRI allowlist as `sk_source_search` —
    unlike that tool, native Google grounding has no built-in domain control,
    so without this filter it can (and did) surface disinfo sites or a
    political party's own site as "evidence". Non-allowlisted citations are
    dropped before the agent ever sees them. Records emitted (allowed) URLs in
    `seen_urls` for provenance enforcement.
    """
    from crewai.tools import BaseTool

    class VertexGroundedSearchTool(BaseTool):
        name: str = "vertex_grounded_search"
        description: str = (
            "Verify a factual claim via Google Search grounded through Vertex AI, "
            "restricted to the same allowlist as sk_source_search: FCRI-ranked "
            "Slovak sources (official registers, agencies, high-FCRI press, "
            "Demagog/Konšpirátori; STVR/TA3 cross-check only) plus European/"
            "international official bodies (Eurostat/EC, ECB, EUR-Lex, "
            "Europarl, OECD, IMF, World Bank) and major international outlets "
            "(Reuters, AP, AFP, BBC, DW, Politico Europe, Euractiv). "
            "Input: a concise factual query in Slovak or English. "
            "Returns a grounded summary and the real source URLs Google cited. "
            "Cite ONLY URLs from the SOURCES list; if there is no SOURCES list, "
            "treat the claim as unverified and do not invent a URL."
        )
        cache: bool = True

        def _run(self, query: str) -> str:
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                return "Grounded search unavailable: google-genai not installed."

            try:
                client = genai.Client(
                    vertexai=True,
                    project=settings.gcp_project_id,
                    location=settings.gcp_location,
                    # Hard timeout so a stalled response raises (and gets
                    # retried by _retry_on_rate_limit) instead of hanging.
                    http_options=types.HttpOptions(
                        timeout=int(settings.llm_timeout_s * 1000)
                    ),
                )
                from datetime import date as _date

                ref = (debate_date or _date.today()).isoformat()
                dated_query = (
                    f"Dátum debaty (referenčné 'teraz'): {ref}. Výroky rečníka "
                    f"sa vzťahujú k tomuto dátumu. Udalosti a zdroje s dátumom do "
                    f"{ref} ber ako reálne. Udalosť alebo zdroj s dátumom PO {ref} "
                    "je anachronická (vznikla až po debate): NEPOUŽÍVAJ ju ako "
                    "dôkaz a na jej základe netvrď, že výrok je nepravdivý.\n\n"
                    f"{query}"
                )
                def _generate():
                    return client.models.generate_content(
                        model=settings.gemini_model,
                        contents=dated_query,
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            tools=[types.Tool(google_search=types.GoogleSearch())],
                        ),
                    )

                resp = _retry_on_rate_limit(_generate)
                TRACKER.add_genai(
                    "grounded_search", getattr(resp, "usage_metadata", None)
                )
            except Exception as exc:  # noqa: BLE001
                return f"Grounded search error: {exc}"

            answer = (resp.text or "").strip()
            urls: list[str] = []
            tagged: list[str] = []
            try:
                meta = resp.candidates[0].grounding_metadata
                for chunk in getattr(meta, "grounding_chunks", None) or []:
                    web = getattr(chunk, "web", None)
                    if web and getattr(web, "uri", None):
                        real = _resolve_redirect(web.uri)
                        dom = _match_sk_domain(real)
                        if not dom:
                            continue  # not on the allowlist — drop, never cite
                        if real in urls:
                            continue
                        if seen_urls is not None:
                            seen_urls.add(real)
                        urls.append(real)
                        tag = " [cross-check]" if dom in _SK_CROSSCHECK_DOMAINS else ""
                        tagged.append(f"{real}{tag}")
            except (AttributeError, IndexError, TypeError):
                pass

            # No allowlisted citations => either the model answered from memory
            # (not search) or every cited site failed the allowlist. Either way,
            # do not hand this back as verifiable (it invites fabricated sources
            # or citing an untrusted/partisan site).
            if not urls:
                return "No grounded results from allowlisted sources."
            lines = [f"  {u}" for u in tagged]
            out = [answer] if answer else []
            out.append("SOURCES:\n" + "\n".join(lines))
            return "\n\n".join(out)

    return VertexGroundedSearchTool()


# Slovak sources ranked by FCRI (Fact-Checking Relevance Index).
# Allowlist = low R_risk mainstream + official + fact-check. EXCLUDED: high-risk /
# gray-zone (hlavnespravy, infovojna, zemavek, badatel, ereport, standard, …).
_SK_OFFICIAL_DOMAINS = [
    "nrsr.sk",
    "slov-lex.sk",
    "susr.statistics.sk",
    "statistics.sk",
    "nbs.sk",
    "mfsr.sk",
    "rozpoctovarada.sk",  # RRZ
    "rozpocet.sk",  # MF SR budget portal
    "rokovania.gov.sk",
    "gov.sk",
]
# Agencies + high-FCRI / low-risk press (prefer these for verification).
_SK_NEWS_DOMAINS = [
    "tasr.sk",
    "teraz.sk",
    "sita.sk",
    "tvnoviny.sk",  # Markíza — FCRI 100
    "aktuality.sk",  # FCRI 97
    "sme.sk",  # FCRI 95
    "dennikn.sk",  # FCRI 89
    "pravda.sk",  # FCRI 85
    "noviny.sk",  # JOJ — FCRI 85
    "hnonline.sk",  # FCRI 80 (economy/stats)
    "postoj.sk",  # FCRI 62, low risk
]
# Usable but cross-check (elevated plurality / capture risk per RMS/IPI).
_SK_CROSSCHECK_DOMAINS = [
    "stvr.sk",  # FCRI 96, R_risk 4.0
    "ta3.com",  # FCRI 86, R_risk 3.0
]
_SK_FACTCHECK_DOMAINS = [
    "demagog.sk",
    "konspiratori.sk",
]
# European/international official + statistical bodies (primary sources for
# claims about EU, foreign affairs, and cross-country statistics).
_INTL_OFFICIAL_DOMAINS = [
    "ec.europa.eu",  # European Commission + Eurostat
    "ecb.europa.eu",
    "eur-lex.europa.eu",
    "europarl.europa.eu",
    "consilium.europa.eu",
    "oecd.org",
    "imf.org",
    "worldbank.org",
]
# Major international wire services / broadcasters with strong fact-checking.
_INTL_NEWS_DOMAINS = [
    "reuters.com",
    "apnews.com",
    "afp.com",
    "bbc.com",
    "bbc.co.uk",
    "dw.com",
    "politico.eu",
    "euractiv.com",
]

_SK_SOURCE_DOMAINS = (
    _SK_OFFICIAL_DOMAINS
    + _SK_NEWS_DOMAINS
    + _SK_CROSSCHECK_DOMAINS
    + _SK_FACTCHECK_DOMAINS
    + _INTL_OFFICIAL_DOMAINS
    + _INTL_NEWS_DOMAINS
)

# Prefer official → fact-check → news → cross-check when ranking hits.
_SK_DOMAIN_PRIORITY = {
    **{d: 0 for d in _SK_OFFICIAL_DOMAINS},
    **{d: 0 for d in _INTL_OFFICIAL_DOMAINS},
    **{d: 1 for d in _SK_FACTCHECK_DOMAINS},
    **{d: 2 for d in _SK_NEWS_DOMAINS},
    **{d: 2 for d in _INTL_NEWS_DOMAINS},
    **{d: 3 for d in _SK_CROSSCHECK_DOMAINS},
}


def _match_sk_domain(url: str) -> str | None:
    """Return the allowlisted domain a URL belongs to, or None if disallowed.

    Shared by both search tools (`sk_source_search` and `vertex_grounded_search`)
    so they enforce the exact same allowlist: verified Slovak sources plus
    European/international official bodies and major international outlets —
    an allowlist, since we can't maintain an exhaustive blocklist of every
    partisan/alternative site out there.
    """
    for d in _SK_SOURCE_DOMAINS:
        if d in url:
            return d
    return None


def _build_sk_source_search_tool(
    seen_urls: set[str] | None = None,
    debate_date: "date | None" = None,
):
    """Free/public tool: search restricted to FCRI-relevant Slovak sources.

    Uses DuckDuckGo (ddgs) with a domain allowlist: official registers, high-FCRI
    low-risk news, agencies, Demagog/Konšpirátori. Prefers diversity + priority
    ranking; never fabricates sources. Records emitted URLs in `seen_urls` for
    provenance enforcement.
    """
    from crewai.tools import BaseTool

    class SlovakSourceSearchTool(BaseTool):
        name: str = "sk_source_search"
        description: str = (
            "Verify a factual claim using ONLY FCRI-relevant Slovak sources: "
            "official (nrsr.sk, slov-lex.sk, ŠÚSR, NBS, gov.sk), news agencies "
            "(TASR/teraz.sk, SITA), high-FCRI press (tvnoviny.sk, Aktuality, SME, "
            "Denník N, Pravda, noviny.sk, HN, Postoj), STVR/TA3 (cross-check only), "
            "fact-checks (demagog.sk, konspiratori.sk), plus European/international "
            "official bodies (Eurostat/EC, ECB, EUR-Lex, Europarl, OECD, IMF, "
            "World Bank) and international outlets (Reuters, AP, AFP, BBC, DW, "
            "Politico Europe, Euractiv). "
            "Never use disinfo/gray-zone sites. Input: concise factual query "
            "(Slovak, or English for international claims). "
            "Returns titles, URLs and snippets."
        )
        cache: bool = True

        def _run(self, query: str) -> str:
            try:
                from ddgs import DDGS
            except ImportError:
                try:
                    from duckduckgo_search import DDGS  # type: ignore
                except ImportError:
                    return "SK source search unavailable: install `ddgs`."

            # Plain query + client-side allowlist filtering (below). A giant
            # `site:` OR filter (~26 domains) makes the query long and triggers
            # engine rate-limits/captchas; a plain query is far more reliable and
            # actually returns more allowlist hits.
            import time

            results: list[dict] = []
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    with DDGS() as ddgs:
                        results = list(
                            ddgs.text(query, region="sk-sk", max_results=30)
                        )
                    if results:
                        break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                time.sleep(1.5 * (attempt + 1))
            if not results and last_exc is not None:
                return f"SK source search error: {last_exc}"

            # Rank by FCRI priority, then keep diversity (one per domain first).
            ranked: list[tuple[int, dict, str]] = []
            for r in results:
                url = r.get("href") or r.get("url") or ""
                dom = _match_sk_domain(url)
                if not dom:
                    continue
                ranked.append((_SK_DOMAIN_PRIORITY.get(dom, 9), r, dom))
            ranked.sort(key=lambda x: x[0])

            picked: list[str] = []
            seen_domains: set[str] = set()
            for _, r, dom in ranked:
                if dom in seen_domains:
                    continue
                seen_domains.add(dom)
                url = r.get("href") or r.get("url") or ""
                if seen_urls is not None and url:
                    seen_urls.add(url)
                tag = " [cross-check]" if dom in _SK_CROSSCHECK_DOMAINS else ""
                picked.append(
                    f"- {r.get('title', '')}{tag}\n  URL: {url}\n  {r.get('body', '')}"
                )
                if len(picked) >= 5:
                    break
            if len(picked) < 5:
                for _, r, dom in ranked:
                    url = r.get("href") or r.get("url") or ""
                    tag = " [cross-check]" if dom in _SK_CROSSCHECK_DOMAINS else ""
                    line = (
                        f"- {r.get('title', '')}{tag}\n  URL: {url}\n  {r.get('body', '')}"
                    )
                    if line not in picked:
                        if seen_urls is not None and url:
                            seen_urls.add(url)
                        picked.append(line)
                    if len(picked) >= 5:
                        break

            if not picked:
                return "No results from verified Slovak sources."
            return "\n".join(picked)

    return SlovakSourceSearchTool()


from src.tools.page import _TITLE_RE, extract_pub_date as _extract_pub_date  # noqa: E402

_TAG_RE = re.compile(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", re.IGNORECASE | re.DOTALL)


def _html_to_text(html: str, limit: int = 1800) -> str:
    text = _TAG_RE.sub(" ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _search_snippet_for_url(url: str) -> str:
    """Retrieve a web-search snippet (title + body) for one exact URL.

    Fallback path when a direct fetch is bot-blocked/timing out: the search
    engine has usually already crawled the page, so its cached snippet lets the
    manager still judge relevance and (often) the date — exactly the "agent
    can't reach it but web search can" case. Returns "" if nothing matches.
    """
    from urllib.parse import urlparse

    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return ""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path_tokens = [t for t in re.split(r"[/\-_]", parsed.path) if len(t) > 3][:8]
    query = f"site:{host} " + " ".join(path_tokens) if host else url
    target = _normalize_search_url(url)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="sk-sk", max_results=15))
    except Exception:  # noqa: BLE001
        return ""

    for r in results:
        cand = r.get("href") or r.get("url") or ""
        if _normalize_search_url(cand) == target:
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            return f"TITLE: {title}\nSNIPPET: {body}".strip()
    return ""


def _normalize_search_url(url: str) -> str:
    u = (url or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.rstrip("/")


def _build_fetch_page_tool(
    seen_urls: set[str] | None = None,
    debate_date: "date | None" = None,
):
    """Page-audit tool for the fact-check manager.

    Fetches a URL (allowlisted domain, or one already returned by a search
    tool this run) and reports HTTP status, publication date, title and a
    text excerpt, so the agent can verify a citation really exists, really
    supports the claim, and is not anachronistic.

    Three-state semantics mirror validation.url_reachable():
    * OK           - page fetched; content follows.
    * DEAD         - 404/410 or DNS failure: fabricated/dead, safe to remove.
    * INCONCLUSIVE - 403/timeout/other: the site may just block bots while
      being perfectly reachable through web search. NEVER treated as dead.
    """
    from crewai.tools import BaseTool

    class FetchPageTool(BaseTool):
        name: str = "fetch_page"
        description: str = (
            "Fetch a source URL and return its HTTP status, publication date, "
            "title and a text excerpt. Use it to AUDIT citations: confirm the "
            "page exists, its content actually supports the claim, and its "
            "date fits (not published after the debate; fresh enough for the "
            "claim). Input: a full URL that either belongs to the trusted "
            "allowlist or was returned by a search tool in this run. "
            "Result starts with OK / DEAD / INCONCLUSIVE. DEAD = fabricated "
            "or removed page (drop the source). INCONCLUSIVE = the site may "
            "merely block automated access while still being a real, "
            "reachable page — do NOT drop a source only because the fetch "
            "was inconclusive; in that case a web-search snippet for the same "
            "URL is appended when available so you can still judge relevance "
            "and date."
        )
        cache: bool = True

        def _run(self, url: str) -> str:
            import requests

            url = (url or "").strip()
            dom = _match_sk_domain(url)
            in_provenance = bool(seen_urls) and url in seen_urls
            if not dom and not in_provenance:
                return (
                    "REFUSED: URL is neither on the trusted-source allowlist "
                    "nor returned by any search tool in this run."
                )
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; STVRDebateAnalyzer/1.0; "
                    "+fact-check-link-verifier)"
                )
            }
            try:
                resp = requests.get(
                    url, timeout=10, headers=headers, allow_redirects=True
                )
            except requests.exceptions.ConnectionError:
                return (
                    "DEAD: domain does not resolve / connection refused — "
                    "the URL is fabricated or gone."
                )
            except requests.exceptions.RequestException as exc:
                return self._inconclusive(
                    url, f"fetch failed ({type(exc).__name__})"
                )
            if resp.status_code in (404, 410):
                return f"DEAD: HTTP {resp.status_code} — page does not exist."
            if resp.status_code >= 400:
                return self._inconclusive(url, f"HTTP {resp.status_code}")

            html = resp.text[:400_000]
            title_m = _TITLE_RE.search(html)
            title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""
            pub_date = _extract_pub_date(html)
            # Successful fetch of an allowlisted page counts as provenance —
            # the manager may cite it (validation would drop it otherwise).
            if seen_urls is not None and dom:
                seen_urls.add(url)
                if resp.url and resp.url != url:
                    seen_urls.add(str(resp.url))
            ref = debate_date.isoformat() if debate_date else None
            lines = [
                f"OK: HTTP {resp.status_code}",
                f"FINAL URL: {resp.url}",
                f"TITLE: {title or '(none)'}",
                f"PUBLISHED: {pub_date or 'unknown'}",
            ]
            if ref:
                lines.append(
                    f"DEBATE DATE: {ref} — if PUBLISHED is after this date, "
                    "the source is anachronistic and cannot support a verdict."
                )
            lines.append(f"EXCERPT: {_html_to_text(html)}")
            return "\n".join(lines)

        def _inconclusive(self, url: str, why: str) -> str:
            """Direct fetch blocked — fall back to a web-search snippet so the
            relevance/date of the source can still be judged."""
            snippet = _search_snippet_for_url(url)
            head = (
                f"INCONCLUSIVE: direct fetch blocked ({why}) — likely bot "
                "blocking, NOT proof the page is fake; do NOT drop the source "
                "for this reason alone."
            )
            if not snippet:
                return (
                    head + " No web-search snippet available either; judge "
                    "relevance from the first-pass search result instead."
                )
            ref = debate_date.isoformat() if debate_date else None
            tail = (
                "\nWeb search DID reach this URL — use the snippet below to "
                "judge whether it addresses the claim (same metric/period/"
                "territory) and whether its date fits."
            )
            if ref:
                tail += f" Debate date: {ref} (anything published after it is anachronistic)."
            return f"{head}{tail}\n{snippet}"

    return FetchPageTool()


# ---------------------------------------------------------------------------
# Gemini context cache
# ---------------------------------------------------------------------------


def build_llm(settings: Settings):
    """CrewAI LLM pointing at Vertex Gemini."""
    import os

    from crewai import LLM

    class RateLimitRetryLLM(LLM):
        """Retries Vertex 429s — CrewAI otherwise aborts the whole crew."""

        def call(self, *args: Any, **kwargs: Any) -> Any:
            # Bind before lambda — bare super() inside a lambda has no __class__ cell.
            parent_call = super().call
            return _retry_on_rate_limit(lambda: parent_call(*args, **kwargs))

        async def acall(self, *args: Any, **kwargs: Any) -> Any:
            import asyncio

            last: BaseException | None = None
            for attempt in range(_RATE_LIMIT_ATTEMPTS):
                try:
                    return await super().acall(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    if (
                        not _is_transient_llm_error(exc)
                        or attempt >= _RATE_LIMIT_ATTEMPTS - 1
                    ):
                        raise
                    delay = min(
                        _RATE_LIMIT_MAX_DELAY_S,
                        _RATE_LIMIT_BASE_DELAY_S * (2**attempt),
                    )
                    delay += random.uniform(0, min(3.0, delay * 0.15))
                    logger.warning(
                        "Transient Vertex/LLM error async (attempt %d/%d): %s — "
                        "sleeping %.1fs",
                        attempt + 1,
                        _RATE_LIMIT_ATTEMPTS,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
            assert last is not None
            raise last

    # Force LiteLLM/Vertex to the configured GCP project (ignore leftover gcloud defaults).
    os.environ["VERTEXAI_PROJECT"] = settings.gcp_project_id
    os.environ["VERTEXAI_LOCATION"] = settings.gcp_location
    os.environ["GOOGLE_CLOUD_PROJECT"] = settings.gcp_project_id

    return RateLimitRetryLLM(
        model=f"vertex_ai/{settings.gemini_model}",
        temperature=0.2,
        vertex_project=settings.gcp_project_id,
        vertex_location=settings.gcp_location,
        # Without this, a stalled Vertex response hangs the whole run forever
        # instead of raising — see _is_transient_llm_error / _retry_on_rate_limit.
        timeout=settings.llm_timeout_s,
    )


# ---------------------------------------------------------------------------
# Crew assembly
# ---------------------------------------------------------------------------


def _transcript_block(transcript: str) -> str:
    return f"DEBATE TRANSCRIPT:\n\n{transcript}"


def _allowed_names(smap: SpeakerMap, briefing: "DebateBriefing | None") -> set[str]:
    """Proper nouns the transcript corrector may write: roster, parties, briefing people.

    People and parties only. Glossary terms are ordinary words ("konsolidácia",
    "deficit"), and the PROPER_NOUN guard allows any edit whose new words are all
    in this set — so a glossary term here would license rewriting a content word
    of a claim into debate jargon.
    """
    names = {e.name for e in smap.entries}
    if briefing is not None:
        for p in briefing.participants:
            names.update([p.name, p.party])
        names.update(e.person for e in briefing.entity_index)
    return {n for n in names if n}


def _canonicalize_speakers(report: AnalysisReport, names: list[str]) -> None:
    """Rewrite agent-written speaker names to the speaker-map roster."""
    from src.speakers import canonical_speaker

    if not names:
        return
    for f in report.facts:
        f.speaker = canonical_speaker(f.speaker, names) or f.speaker
    for s in report.behavioral_analysis.speakers:
        s.speaker = canonical_speaker(s.speaker, names) or s.speaker


def _claims_json(claims: list[ExtractedClaim]) -> str:
    data = [c.model_dump(mode="json") for c in claims]
    return json.dumps(data, ensure_ascii=False, indent=1)


def select_top_claims(
    extracted: ExtractedClaims, max_claims: int
) -> tuple[list[ExtractedClaim], list[str]]:
    """Keep the top-N claims by salience (stable order within ties)."""
    notes: list[str] = []
    claims = list(extracted.claims)
    if len(claims) <= max_claims:
        return claims, notes
    ranked = sorted(claims, key=lambda c: -c.salience)
    kept_ids = {c.id for c in ranked[:max_claims]}
    kept = [c for c in claims if c.id in kept_ids]
    dropped = [c for c in claims if c.id not in kept_ids]
    notes.append(
        f"Claim cap: kept top {len(kept)} of {len(claims)} claims by salience; "
        "dropped low-salience claims: "
        + "; ".join(f'#{c.id} (s{c.salience}) "{c.claim[:60]}"' for c in dropped)
    )
    # Transparency guard against Gish gallop: report per speaker how many
    # claims were left UNCHECKED by the cap, so a flood of small claims
    # doesn't silently escape scrutiny.
    from collections import Counter

    unchecked = Counter(c.speaker for c in dropped)
    if unchecked:
        notes.append(
            "Unchecked claims per speaker (dropped by cap): "
            + ", ".join(f"{s}: {n}" for s, n in unchecked.most_common())
        )
    return kept, notes


def kickoff_with_retry(make: "Callable[[], tuple[Any, Any]]", label: str, attempts: int = 3):
    """Build and run a crew, retrying on transient failures (Gemini sometimes returns an
    empty response that crewai surfaces as ValueError after its own retries).
    `make` returns (crew, tasks); a fresh crew is built per attempt. Returns (result, tasks)."""
    last: Exception | None = None
    for n in range(1, attempts + 1):
        crew, tasks = make()
        try:
            return crew.kickoff(), tasks
        except Exception as exc:  # noqa: BLE001
            last = exc
            logger.warning("%s crew failed (attempt %d/%d): %s", label, n, attempts, exc)
    assert last is not None
    raise last


def retry_empty_extraction(
    first: "ExtractedClaims | None",
    rerun: "Callable[[], ExtractedClaims | None]",
    attempts: int = 2,
) -> tuple["ExtractedClaims | None", list[str]]:
    """Re-run extraction when it came back empty (seen in production: the full
    crew returned {"claims":[]} for a 78-minute debate, standalone got 30)."""
    notes: list[str] = []
    result = first
    for n in range(1, attempts + 1):
        if result is not None and result.claims:
            break
        logger.warning("Extraction returned no claims; standalone retry %d/%d", n, attempts)
        try:
            result = rerun()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Extraction retry failed")
            notes.append(f"Extraction retry {n} failed ({exc}).")
            continue
        if result is not None and result.claims:
            notes.append(f"Extraction was empty in the crew run; recovered {len(result.claims)} claims on standalone retry {n}.")
    return result, notes


def _extract_standalone(working, settings, debate_date, briefing_text):
    """Run only the extraction task, without the behavioral-analysis context."""
    from crewai import Crew, Process

    from src.reconcile import parse_task_output

    _crew, tasks = build_extract_crew(
        working, settings=settings, debate_date=debate_date, briefing_text=briefing_text
    )
    task = tasks["extract"]
    task.context = []
    result = Crew(
        agents=[task.agent], tasks=[task], process=Process.sequential, verbose=False
    ).kickoff()
    TRACKER.add_crew("analysis_extract_retry", result)
    return parse_task_output(task, ExtractedClaims)


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


def build_extract_crew(
    transcript: str,
    settings: Settings | None = None,
    debate_date: "date | None" = None,
    briefing_text: str = "",
):
    """Phase A: behavioral + moderator analysis and context-aware claim extraction."""
    from crewai import Agent, Crew, Process, Task

    settings = settings or get_settings()
    llm = build_llm(settings)
    tr_block = _transcript_block(transcript)
    debate_date_note = (
        f"DEBATE DATE: {debate_date.isoformat()}. Resolve every relative time "
        "expression against this date.\n\n"
        if debate_date
        else ""
    )

    behavioral = Agent(
        role="Behavioral Analyst",
        goal=(
            "Identify speech tactics, manipulation, logical fallacies, "
            "question-dodging, and civility/correctness in a political debate transcript."
        ),
        backstory=(
            "You are a rigorous communication scientist specializing in political rhetoric "
            "and the sociology of deliberation. You cite concrete transcript evidence and "
            "avoid partisan framing. You judge civility by discourse norms, not naive "
            "politeness: assertiveness is not rudeness, stating an uncomfortable truth is "
            "not insolence, and disagreement is not disrespect. You flag only genuine "
            "norm violations (personal insults, contempt, dehumanization, bad-faith "
            "interrupting, condescension meant to silence)."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    moderator = Agent(
        role="Moderator Bias Auditor",
        goal=(
            "Evaluate moderator neutrality, interruption patterns, "
            "and whether speaking time was distributed fairly."
        ),
        backstory=(
            "You audit broadcast debate moderation for fairness. "
            "You quantify interruptions and time balance from the transcript."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    fact_extractor = Agent(
        role="Fact Extractor",
        goal=(
            "Extract verifiable factual claims and rate how important each one is "
            "to the debate based on the value of the claim in the debate and also how the speaker used it (counterargument, "
            "attack, deflection); ignore opinions and value judgments."
        ),
        backstory=(
            "You are a meticulous claims analyst. You list atomic, checkable statements "
            "with the speaker who made them. No opinions, metaphors, or predictions. "
            "You judge a claim's importance by its rhetorical role in the discussion — "
            "claims deployed as counterarguments, attacks, or smokescreens matter most; "
            "trivia and common knowledge matter least."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    t_behavioral = Task(
        description=(
            f"{tr_block}\n\n"
            "Analyze speech tactics, manipulation, logical fallacies, and question dodging "
            "per speaker. Return structured findings with short evidence quotes.\n\n"
            "Also rate CIVILITY/CORRECTNESS per speaker (civility field):\n"
            "- score 0-10 by discourse norms, NOT naive politeness.\n"
            "- Do NOT penalize: assertiveness, directness, strong disagreement, "
            "stating uncomfortable but true facts, pointed but fair questions.\n"
            "- Penalize only genuine violations: personal insults, contempt or "
            "dehumanization, bad-faith interrupting / talking over others, lies used "
            "to demean, condescension aimed at silencing an opponent.\n"
            "- incivility: list violations with short evidence quotes.\n"
            "- fair_conduct: note respectful-yet-assertive moments.\n"
            "- Be symmetric across coalition and opposition; avoid partisan bias.\n"
            "- Score ONLY invited debate guests. Do not include the moderator, "
            "correspondents, video-clip voices, or recap narrators as speakers."
            + _SK_OUTPUT
        ),
        expected_output="JSON-like structured behavioral analysis per speaker with evidence.",
        agent=behavioral,
    )

    t_moderator = Task(
        description=(
            f"{tr_block}\n\n"
            "Evaluate the moderator: neutrality (0-10), interruptions, and approximate "
            "speaking-time share per participant. Be evidence-based."
            + _SK_OUTPUT
        ),
        expected_output="Structured moderator audit with neutrality score and findings.",
        agent=moderator,
    )

    t_extract = Task(
        description=(
            f"{tr_block}\n\n"
            f"{debate_date_note}"
            "Extract verifiable factual claims only (numbers, dates, events, "
            "attributions). For each claim assign a stable sequential `id` starting at 1, "
            "plus claim text + speaker + a short VERBATIM quote copied exactly from the "
            "transcript that the claim is based on (do not paraphrase or reconstruct — "
            "copy the exact words). If you cannot find an exact supporting excerpt, do "
            "not include the claim. Ignore opinions.\n\n"
            f"{_importance_block(briefing_text)}"
            "CATEGORY: classify each claim for specialist routing:\n"
            "- economy_finance: budget, deficit, debt, consolidation, taxes, salaries "
            "of officials, public spending.\n"
            "- statistics: official statistical indicators — inflation/CPI, wages, "
            "unemployment, GDP, demographics.\n"
            "- history_politics: parliamentary votes, laws, past governments, past "
            "statements or positions.\n"
            "- current_events: recent events, quote attributions, media affairs.\n\n"
            "CONTEXT: for each claim write one sentence describing what in the debate "
            "the claim was reacting to and what EXACTLY it refers to — be precise "
            "about the metric, period, and territory (e.g. 'reacts to criticism of "
            "food price growth; refers to FOOD inflation in Slovakia in 2025, not "
            "headline inflation'). Verifiers rely on this to avoid checking the "
            "wrong quantity.\n\n"
            "TIME: if the speaker anchors the claim in time, copy the verbatim "
            "expression into `time_reference` (e.g. 'pred dvomi mesiacmi', 'v roku "
            "2014') and resolve it to an absolute `time_window` RELATIVE TO THE "
            "DEBATE DATE above (e.g. debate 2026-06-14 + 'pred dvomi mesiacmi' -> "
            "'~2026-04'). Leave both empty if there is no explicit time reference. "
            "This lets verifiers reject anachronistic (post-debate) sources."
        ),
        expected_output=(
            "ExtractedClaims JSON: list of {id, claim, speaker, quote, salience, "
            "usage, usage_reason, consequence, consequence_reason, checkability, "
            "category, context, time_reference, time_window} with sequential ids."
        ),
        agent=fact_extractor,
        context=[t_behavioral],
        output_pydantic=ExtractedClaims,
    )

    crew = Crew(
        agents=[behavioral, moderator, fact_extractor],
        tasks=[t_behavioral, t_moderator, t_extract],
        process=Process.sequential,
        verbose=True,
    )
    return crew, {
        "behavioral": t_behavioral,
        "moderator": t_moderator,
        "extract": t_extract,
    }

_CHECK_RULES = (
    "When you call a search tool, build the query from the claim text, its "
    "verbatim quote, and its `context` field (the context tells you what the "
    "claim EXACTLY refers to — e.g. food inflation vs headline inflation). "
    "NEVER add names, people, dates or specifics that are not present in the "
    "claim/quote/context — do not guess which official/person is meant from "
    "your own knowledge. If the claim says 'the President of Poland', search "
    "exactly that; do NOT substitute a specific name (e.g. Duda/Nawrocki). "
    "Injecting a wrong entity poisons the verdict.\n"
    "METRIC FIDELITY: a verdict may rest ONLY on evidence about exactly the "
    "same metric, period, and territory as the claim. Evidence about a "
    "different — even closely related — metric is unusable: headline CPI "
    "inflation does NOT verify a claim about FOOD inflation; nominal wages do "
    "NOT verify a claim about REAL wages; year-on-year change does NOT verify "
    "a cumulative change. If you only found data for a different metric or "
    "period, the claim stays unverified — use Unverified, never "
    "a verdict derived from the wrong number.\n"
    "NO STITCHING: a False or True verdict requires at least one source that "
    "addresses the claim AS A WHOLE. You must NOT stitch together partial "
    "snippets from different websites into one conclusion. If the claim is "
    "compound, verify each part separately and say so explicitly in the "
    "rationale; if a verified part is False the verdict is False/Misleading, "
    "otherwise if any part remains unverified the overall verdict is "
    "Unverified.\n"
    "FRESHNESS: the speaker talks about the PRESENT unless the claim says "
    "otherwise. Facts change: rates, prices, laws, officeholders, statistics "
    "get updated. Check the publication/update date of every source; prefer "
    "the most recent one. A source that was accurate in the past but has "
    "since been superseded CANNOT support a verdict — an old article saying "
    "'inflation is 12%' does not verify or refute a claim about today's "
    "inflation. If you cannot find evidence current enough for the claim's "
    "time frame, treat the claim as Unverified, and never mark "
    "a claim False based solely on outdated data.\n"
    "TIME FRAME & ANACHRONISM: the DEBATE DATE given above is your reference "
    "'now'. Events/sources dated AFTER the debate date did not exist when the "
    "speaker talked — they are anachronistic and CANNOT support any verdict, "
    "least of all False. When the claim has a `time_reference`/`time_window`, "
    "resolve it relative to the debate date and verify the event in that window; "
    "include the resolved period/date in your search query. If a real matching "
    "event exists but with a different figure or slightly different date than the "
    "claim, that is Misleading (imprecise), NOT False — do NOT rule 'no such "
    "event' when an event in the window plausibly matches. Only use False when a "
    "source dated at/before the debate date explicitly contradicts the claim.\n"
    "For each claim provide a ClaimCheck with:\n"
    "- claim_id: MUST match the ExtractedClaim.id exactly.\n"
    "- verdict: True / False / Misleading / Unverified. The verdict MUST follow "
    "the search results, NOT your prior knowledge. A False verdict REQUIRES at "
    "least one source (from the tool output) whose content explicitly "
    "contradicts the claim. Misleading means the evidence shows the claim "
    "DISTORTS reality (technically true but deceptively framed, cherry-picked, "
    "or partially false) — it is NOT a fallback for missing evidence. If the "
    "tool output confirms the claim, the verdict is True even if "
    "the claim resembles a known past hoax — recent real events can mirror old "
    "debunked stories, so trust the dated search results over memory. If the "
    "search returned nothing usable, use Unverified, NOT False or Misleading.\n"
    "- severity: when verdict is False or Misleading, set trivial "
    "(minor rounding/imprecision), material (changes the argument), or "
    "fabrication (no factual basis / publicly debunked). Omit severity for "
    "True and Unverified.\n"
    "- intent_signals: short evidence-based notes only "
    "(e.g. 'publicly debunked narrative', 'easily verifiable figure'). "
    "Do NOT claim the speaker intended to lie.\n"
    "- sources: copy ONLY URLs that appear verbatim in THIS run's tool output "
    "for this claim. NEVER write a URL from memory — no guessed article slugs, "
    "no demagog.sk/vyrok/<number>, no konspiratori.sk IDs, no nrsr/slov-lex IDs "
    "unless they were literally returned by the search. If no usable URL was "
    "returned, leave sources empty (do NOT invent one to satisfy this field).\n"
    "- a short rationale that references what the cited sources actually say."
    + _SK_OUTPUT
)

_CHECK_EXPECTED_OUTPUT = (
    "ClaimCheckList JSON: one ClaimCheck per claim_id with verdict, severity "
    "(if False/Misleading), intent_signals, source URLs, and rationale."
)


def _check_task_description(
    header: str,
    claims: list[ExtractedClaim],
    debate_date: "date | None" = None,
) -> str:
    from datetime import date as _date

    ref = (debate_date or _date.today()).isoformat()
    return (
        f"DEBATE DATE (reference 'now'): {ref}. The speakers' statements refer to "
        f"this date; judge source freshness against it. A source or event dated "
        f"AFTER {ref} is anachronistic (it happened after the debate): it CANNOT "
        "be used as evidence and MUST NOT make a claim False. When a claim carries "
        "a time_reference/time_window, verify the event WITHIN that window relative "
        f"to {ref}; if the closest matching event has a different date or number, "
        "the verdict is Misleading (imprecise), NOT False for 'no such event'.\n\n"
        f"{header}\n{_CHECK_RULES}\n\n"
        "CLAIMS TO VERIFY (ExtractedClaims JSON):\n"
        f"{_claims_json(claims)}"
    )


# Specialist definitions per claim category: role, backstory, task header.
_SPECIALISTS: dict[ClaimCategory, dict[str, str]] = {
    ClaimCategory.ECONOMY_FINANCE: {
        "role": "Financial Analyst Fact-Checker",
        "goal": (
            "Verify claims about public finances — budget, deficit, debt, "
            "consolidation, taxes, public salaries and spending — against "
            "official fiscal data and economic press."
        ),
        "backstory": (
            "You are a public-finance analyst. Source priority: (1) official — "
            "MF SR (mfsr.sk, rozpocet.sk), Rada pre rozpočtovú zodpovednosť "
            "(rozpoctovarada.sk), NBS, ŠÚSR, slov-lex.sk, and for EU/cross-"
            "country fiscal claims Eurostat (ec.europa.eu), ECB, European "
            "Commission, IMF; (2) economic press — hnonline.sk; (3) high-FCRI "
            "mainstream press. You know fiscal figures are revised and "
            "re-projected during the year — always use the newest available "
            "figure and note its vintage. You never invent sources; if "
            "evidence is insufficient, use Unverified."
        ),
        "header": (
            "Verify each claim below (public finances). Use sk_source_search "
            "and vertex_grounded_search. Prefer MF SR / RRZ / NBS / ŠÚSR data; "
            "for EU comparisons use Eurostat / EC / ECB / IMF; "
            "distinguish budgeted vs actual figures, deficit in % of GDP vs "
            "absolute EUR, and gross vs net debt."
        ),
    },
    ClaimCategory.STATISTICS: {
        "role": "Statistician Fact-Checker",
        "goal": (
            "Verify claims about statistical indicators (inflation, wages, "
            "unemployment, GDP, demographics) directly against Statistical "
            "Office data."
        ),
        "backstory": (
            "You are a statistician with direct access to ŠÚSR DATAcube via "
            "susr_data_search — that is primary-source data and beats any "
            "press article. For EU/cross-country indicators use Eurostat "
            "(ec.europa.eu), ECB and OECD data via sk_source_search. You are "
            "pedantic about metric definitions: "
            "headline CPI vs food inflation (COICOP class), HICP vs national "
            "CPI, nominal vs real wages, year-on-year vs month-on-month vs "
            "cumulative change, registered vs survey (VZPS) unemployment. A "
            "claim about one metric is NEVER verified with a different one. "
            "You never invent sources; if the exact metric/period is not "
            "available, use Unverified."
        ),
        "header": (
            "Verify each claim below (statistical indicators). FIRST query "
            "susr_data_search for the exact official indicator; use "
            "sk_source_search as a complement (it also covers Eurostat / ECB "
            "/ OECD for EU comparisons). Match metric, period and "
            "territory exactly (see METRIC FIDELITY)."
        ),
    },
    ClaimCategory.HISTORY_POLITICS: {
        "role": "Historian-Politologist Fact-Checker",
        "goal": (
            "Verify claims about parliamentary votes, laws, past governments "
            "and past statements against official records and fact-check "
            "archives."
        ),
        "backstory": (
            "You are a political historian. Source priority: (1) official "
            "records — nrsr.sk (votes, sessions), slov-lex.sk (laws and their "
            "amendments), rokovania.gov.sk, and for EU legislation/votes "
            "eur-lex.europa.eu, europarl.europa.eu, consilium.europa.eu; "
            "(2) demagog.sk fact-checks; "
            "(3) high-FCRI press archives. You are careful about temporal "
            "validity: laws get amended and repealed, positions change — "
            "check that the legal state or quote you cite matches the period "
            "the claim refers to. You never invent sources; if evidence is "
            "insufficient, use Unverified."
        ),
        "header": (
            "Verify each claim below (votes, laws, past governments, past "
            "statements). Use sk_source_search; prefer nrsr.sk / slov-lex.sk "
            "/ demagog.sk, and EUR-Lex / Europarl for EU law. Verify the "
            "claim against the legal/political state "
            "valid in the period the claim refers to."
        ),
    },
    ClaimCategory.CURRENT_EVENTS: {
        "role": "Journalist Fact-Checker",
        "goal": (
            "Verify claims about recent events, quote attributions and media "
            "affairs against current reporting from trusted Slovak sources."
        ),
        "backstory": (
            "You are a news-desk fact-checker using sk_source_search with an "
            "FCRI allowlist. Priority: official sources → Demagog.SK / "
            "Konšpirátori.sk → high-FCRI low-risk press (tvnoviny.sk, "
            "Aktuality, SME, Denník N, Pravda, noviny.sk, HN, Postoj, "
            "TASR/SITA); for foreign/international events use Reuters, AP, "
            "AFP, BBC, DW, Politico Europe, Euractiv (query in English); "
            "STVR/TA3 only as cross-check, never sole authority. "
            "Never cite disinfo/gray-zone domains. Events develop quickly — "
            "always prefer the newest reporting and check article dates. You "
            "never invent sources; if evidence is insufficient, use "
            "Unverified."
        ),
        "header": (
            "Verify each claim below (current events, attributions, media "
            "affairs). Use sk_source_search (international wires included for "
            "foreign events). STVR/TA3 results marked "
            "[cross-check] must not be the sole basis for a verdict — require "
            "a second independent source."
        ),
    },
}


def build_check_crew(
    claims: list[ExtractedClaim],
    settings: Settings,
    seen_urls: set[str],
    debate_date: "date | None" = None,
):
    """Phase B: grounded checker over all claims + one specialist per category.

    Returns (crew, {"grounded": task, "specialists": [tasks...]}).
    """
    from crewai import Agent, Crew, Process, Task

    from src.tools_susr import build_susr_data_tool

    llm = build_llm(settings)
    grounded_tool = _build_grounded_search_tool(
        settings, seen_urls=seen_urls, debate_date=debate_date
    )
    sk_source_tool = _build_sk_source_search_tool(
        seen_urls=seen_urls, debate_date=debate_date
    )
    susr_tool = build_susr_data_tool(seen_urls=seen_urls)

    grounded_checker = Agent(
        role="Grounded Fact-Checker",
        goal=(
            "Independently verify extracted factual claims using Google Search grounded "
            "through Vertex AI, and label each True, False, Misleading, or Unverified "
            "with sources, severity, and evidence-based intent_signals (never assert "
            "speaker intent)."
        ),
        backstory=(
            "You are an independent fact-checker who verifies claims against the live web "
            "via vertex_grounded_search. Prefer primary/official data and high-FCRI "
            "Slovak mainstream (tvnoviny.sk, Aktuality, SME, Denník N, Pravda, HN, "
            "TASR) over gray-zone or disinfo sites. Never treat Hlavné správy, Infovojna, "
            "Zem a Vek, Bádateľ, eReport or Štandard as authoritative. You never invent "
            "sources. If evidence is insufficient, use Unverified. Distinguish "
            "trivial errors from material misstatements and fabrications."
        ),
        llm=llm,
        tools=[grounded_tool],
        verbose=True,
        allow_delegation=False,
        max_rpm=15,  # avoid Vertex burst 429s on many claims
    )

    t_grounded = Task(
        description=_check_task_description(
            "Verify EVERY claim below with the vertex_grounded_search tool "
            "(Google Search grounded via Vertex AI). Prefer official data and "
            "high-FCRI Slovak mainstream sources in citations; do not rely on "
            "disinfo/gray-zone sites as proof.",
            claims,
            debate_date,
        ),
        expected_output=_CHECK_EXPECTED_OUTPUT,
        agent=grounded_checker,
        output_pydantic=ClaimCheckList,
    )

    specialist_tools = {
        ClaimCategory.ECONOMY_FINANCE: [sk_source_tool, grounded_tool],
        ClaimCategory.STATISTICS: [susr_tool, sk_source_tool],
        ClaimCategory.HISTORY_POLITICS: [sk_source_tool],
        ClaimCategory.CURRENT_EVENTS: [sk_source_tool],
    }

    agents = [grounded_checker]
    tasks = [t_grounded]
    specialist_tasks: list[Task] = []
    for category, spec in _SPECIALISTS.items():
        cat_claims = [c for c in claims if c.category == category]
        if not cat_claims:
            continue
        agent = Agent(
            role=spec["role"],
            goal=spec["goal"],
            backstory=spec["backstory"],
            llm=llm,
            tools=specialist_tools[category],
            verbose=True,
            allow_delegation=False,
            max_rpm=15,
        )
        task = Task(
            description=_check_task_description(
                spec["header"], cat_claims, debate_date
            ),
            expected_output=_CHECK_EXPECTED_OUTPUT,
            agent=agent,
            output_pydantic=ClaimCheckList,
        )
        agents.append(agent)
        tasks.append(task)
        specialist_tasks.append(task)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
    return crew, {"grounded": t_grounded, "specialists": specialist_tasks}


_MANAGER_RULES = (
    "You are the second (audit + deepening) pass over first-pass fact-check "
    "results. For EVERY claim below produce one ManagerReview. Work claim by "
    "claim:\n\n"
    "1) SOURCE AUDIT — for each first-pass source URL call fetch_page and check:\n"
    "   a. existence: DEAD (404/DNS) => the URL is fabricated or gone — remove "
    "it and record it in removed_sources with reason. An INCONCLUSIVE fetch "
    "(403, timeout, bot-block) is NOT proof of anything: the page may be "
    "perfectly real and merely block automated access — KEEP such a source "
    "unless something else disqualifies it.\n"
    "   b. date: if PUBLISHED is after the debate date, the source is "
    "anachronistic — remove it. If the claim has a time_window, the source "
    "must cover that window; a source about a clearly different period "
    "cannot support the verdict.\n"
    "   c. relevance: the excerpt must actually address the claim (same "
    "metric, period, territory — see METRIC FIDELITY below). Remove "
    "off-topic sources.\n"
    "2) SUB-FACT DECOMPOSITION — split compound claims into atomic sub-facts "
    "and list them in `subfacts` with a one-line result each ('confirmed by "
    "<url>', 'refuted by <url>', 'no evidence found'). For every sub-fact "
    "lacking evidence run targeted searches (susr_data_search for statistics, "
    "sk_source_search / vertex_grounded_search otherwise) with a QUERY BUILT "
    "ONLY from the claim/quote/context — never inject entities from memory.\n"
    "3) PROACTIVE RE-VERIFICATION — when a False/Misleading verdict rests on "
    "a single source, look for a second independent source. When the verdict "
    "is Unverified, actively try new angles: reformulated queries, English "
    "queries for international topics, the official register (NRSR/slov-lex/"
    "ŠÚSR/Eurostat) behind the number.\n"
    "4) FINAL VERDICT — set verdict/severity per the same rules as the first "
    "pass (below). Downgrading to Unverified is always allowed and is the "
    "correct move when the audit destroyed the supporting evidence. A "
    "False/Misleading verdict still REQUIRES at least one surviving source "
    "that contradicts the claim as a whole.\n"
    "5) ESCALATE — set escalate=true ONLY if the claim ends Unverified or "
    "Contested AND you found a concrete lead (a named dataset, register, or "
    "event) that one more focused round could confirm. Mention the lead in "
    "the rationale. Otherwise escalate=false.\n\n"
    "sources: the FINAL list for the claim — only URLs that appeared in THIS "
    "run's tool output (search results or a fetch_page OK). Never write a URL "
    "from memory.\n\n"
    "FIRST-PASS RULES THAT STILL BIND YOU:\n"
)


def build_manager_crew(
    claims: list[ExtractedClaim],
    prelim_facts_json: str,
    settings: Settings,
    seen_urls: set[str],
    debate_date: "date | None" = None,
    round_no: int = 1,
):
    """Phase B2: agentic fact-check manager — audits sources, hunts sub-facts.

    Returns (crew, task). Run repeatedly (cyclic) on escalated claims.
    """
    from crewai import Agent, Crew, Process, Task

    from src.tools_susr import build_susr_data_tool

    llm = build_llm(settings)
    tools = [
        _build_fetch_page_tool(seen_urls=seen_urls, debate_date=debate_date),
        _build_grounded_search_tool(settings, seen_urls=seen_urls, debate_date=debate_date),
        _build_sk_source_search_tool(seen_urls=seen_urls, debate_date=debate_date),
        build_susr_data_tool(seen_urls=seen_urls),
    ]

    manager = Agent(
        role="Fact-Check Manager",
        goal=(
            "Audit and deepen first-pass fact-check results: verify every cited "
            "source really exists, matches the claim and its time frame; hunt "
            "down missing sub-facts with targeted searches; and deliver a final, "
            "well-founded verdict per claim."
        ),
        backstory=(
            "You lead a fact-checking desk. You trust nothing that isn't "
            "audited: a verdict is only as good as its sources, so you open "
            "each cited page, check its date and whether it addresses the "
            "claim as a whole. You know sites often block bots — an "
            "unreachable page is not automatically a fake one; only 404/DNS "
            "failures are. You decompose compound claims into sub-facts and "
            "close evidence gaps with precise queries against official "
            "registers and high-trust press. You never invent sources and "
            "you'd rather ship Unverified than an unsupported accusation."
        ),
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False,
        max_rpm=15,
    )

    task = Task(
        description=(
            _check_task_description(
                f"MANAGER REVIEW — ROUND {round_no}.\n{_MANAGER_RULES}",
                claims,
                debate_date,
            )
            + "\n\nFIRST-PASS RESULTS (verdict, sources, rationale per claim_id):\n"
            + prelim_facts_json
        ),
        expected_output=(
            "ManagerReviewList JSON: one ManagerReview per claim_id with final "
            "verdict, severity (if False/Misleading), final sources, "
            "removed_sources (each with reason), subfacts, escalate, rationale."
        ),
        agent=manager,
        output_pydantic=ManagerReviewList,
    )

    crew = Crew(
        agents=[manager],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    return crew, task


def build_critic_crew(
    transcript: str,
    behavioral_output: str,
    moderator_output: str,
    settings: Settings,
):
    """Phase C: final editorial review of behavioral/moderator sections."""
    from crewai import Agent, Crew, Process, Task

    llm = build_llm(settings)
    tr_block = _transcript_block(transcript)

    critic = Agent(
        role="Chief Critic",
        goal=(
            "Review all prior agent outputs for objectivity; remove hallucinations "
            "and biased conclusions; produce the final structured report."
        ),
        backstory=(
            "You are the final editorial gate. You demand evidence, cut speculation, "
            "and ensure the report is balanced and grounded in the transcript."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    t_critic = Task(
        description=(
            f"{tr_block}\n\n"
            f"BEHAVIORAL ANALYST OUTPUT:\n{behavioral_output}\n\n"
            f"MODERATOR AUDITOR OUTPUT:\n{moderator_output}\n\n"
            "Review the Behavioral Analyst and Moderator Bias Auditor outputs above. "
            "Do NOT produce or rewrite the facts list — facts are reconciled "
            "deterministically in code from the fact-checkers; leave facts empty. "
            "Remove unsupported claims, hallucinations, and partisan language from "
            "behavioral/moderator sections. "
            "Preserve each speaker's civility assessment (score, incivility, "
            "fair_conduct); ensure it follows discourse norms, not naive politeness "
            "(assertiveness/truth-telling is not incivility) and is symmetric across "
            "coalition and opposition. "
            "Produce the final analysis report with: summary, behavioral_analysis "
            "(per speaker: speech_tactics, manipulation, logical_fallacies, "
            "question_dodging, civility, notes), moderator_audit, critic_notes. "
            "Set facts to an empty list."
            + _SK_OUTPUT
            + " Ak sú vstupy od predošlých agentov po anglicky, prelož ich do "
            "slovenčiny; doslovné citácie z prepisu ponechaj nezmenené."
        ),
        expected_output=(
            "Final AnalysisReport JSON with summary, behavioral_analysis, "
            "moderator_audit, critic_notes; facts empty."
        ),
        agent=critic,
        output_pydantic=AnalysisReport,
    )

    crew = Crew(
        agents=[critic],
        tasks=[t_critic],
        process=Process.sequential,
        verbose=True,
    )
    return crew


def run_analysis(
    transcript: str,
    settings: Settings | None = None,
    debate_date: "date | None" = None,
    guests: list[str] | None = None,
    moderator: str | None = None,
) -> tuple[AnalysisReport, CorrectedTranscript]:
    """Map speakers, correct the transcript by audited edits, then analyze.

    Phases: 0) briefing -> speaker map -> edit-based correction -> A) behavioral/
    moderator/extraction + question audit -> selection -> B) grounded checker +
    specialists -> B2) manager -> C) critic -> deterministic reconcile, validation,
    accusation guard, claim funnel. Returns (report, corrected).
    """
    settings = settings or get_settings()
    if not settings.vertex_ready():
        raise RuntimeError("GCP_PROJECT_ID is not configured")

    from src.correction import correct_transcript_edits, default_correction_llm
    from src.reconcile import merge_checklists, parse_task_output, reconcile_checks
    from src.speakers import CLIP_NAME, apply_speaker_map, default_speaker_llm, map_speakers
    from src.transcript_lines import parse_lines

    pipeline_notes: list[str] = []

    # Phase 0: political briefing (background only — never evidence). Labels are
    # irrelevant to it, so it runs on the raw transcript and its participants
    # then help the speaker mapper.
    briefing: DebateBriefing | None = None
    briefing_text = ""
    if settings.briefing_enabled and debate_date is not None:
        from src.briefing import render_briefing, run_briefing

        try:
            briefing, briefing_notes = run_briefing(transcript, debate_date, settings)
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
            briefing_text = ""
    elif settings.briefing_enabled:
        pipeline_notes.append("Briefing skipped: no --debate-date given.")

    logger.info("Mapping diarization labels to speakers")
    smap = map_speakers(
        transcript,
        guests=guests,
        moderator=moderator,
        llm=default_speaker_llm(settings),
        briefing_text=briefing_text,
    )
    pipeline_notes.append(
        f"Speaker map: status {smap.status}, source {smap.source}, "
        f"debate start {smap.debate_start or 'unknown'}: "
        + ", ".join(f"{e.label}={e.name}" for e in smap.entries)
    )
    pipeline_notes.extend(smap.notes)
    named = apply_speaker_map(transcript, smap)

    logger.info("Correcting transcript by guarded edits")
    outcome = correct_transcript_edits(
        named,
        llm=default_correction_llm(settings),
        allowed_names=_allowed_names(smap, briefing),
        speaker_names={e.name for e in smap.entries},
        tail_owners={smap.moderator() or "Moderátor", CLIP_NAME},
    )
    pipeline_notes.extend(outcome.notes)
    # `outcome.lines` carries edit ids and the raw line each line came from;
    # re-parsing `working` would lose both, so the lines are passed on as-is.
    use_edits = bool(outcome.text.strip())
    working = outcome.text if use_edits else named
    working_lines = outcome.lines if use_edits else parse_lines(named)
    corrected = CorrectedTranscript(
        text=working,
        notes=[
            f"Transcript edits applied: {outcome.quality.applied}; rejected by "
            f"rule: {outcome.quality.rejected_by_rule}"
        ],
        log=outcome.log,
    )

    # Phase A: behavioral + moderator + context-aware extraction.
    extract_crew, a_tasks = build_extract_crew(
        working,
        settings=settings,
        debate_date=debate_date,
        briefing_text=briefing_text,
    )
    a_result = extract_crew.kickoff()
    TRACKER.add_crew("analysis_extract", a_result)

    extracted = parse_task_output(a_tasks["extract"], ExtractedClaims)
    extracted, retry_notes = retry_empty_extraction(
        extracted,
        lambda: _extract_standalone(working, settings, debate_date, briefing_text),
    )
    pipeline_notes.extend(retry_notes)
    behavioral_raw = getattr(a_tasks["behavioral"].output, "raw", "") or ""
    moderator_raw = getattr(a_tasks["moderator"].output, "raw", "") or ""

    from src.questions import default_question_llm, run_question_audit

    question_audit, q_notes = run_question_audit(
        working_lines, smap, llm=default_question_llm(settings)
    )
    pipeline_notes.extend(q_notes)

    # Claim selection (deterministic, in code): consequence-based when a briefing
    # exists, otherwise the legacy salience top-N.
    kept_claims: list[ExtractedClaim] = []
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
        pipeline_notes.extend(cap_notes)
        logger.info(
            "Extracted %d claims, keeping %d (max_claims=%d)",
            len(extracted.claims),
            len(kept_claims),
            settings.max_claims,
        )
    else:
        pipeline_notes.append("Extraction produced no claims; facts will be empty.")

    # Phase B: grounded checker (all claims) + per-category specialists.
    seen_urls: set[str] = set()
    grounded = None
    specialist_checks = None
    if kept_claims:
        b_result, b_tasks = kickoff_with_retry(
            lambda: build_check_crew(
                kept_claims, settings, seen_urls, debate_date=debate_date
            ),
            "check",
        )
        TRACKER.add_crew("analysis_check", b_result)
        grounded = parse_task_output(b_tasks["grounded"], ClaimCheckList)
        specialist_checks = merge_checklists(
            [parse_task_output(t, ClaimCheckList) for t in b_tasks["specialists"]]
        )

    # Phase C: critic review of behavioral/moderator sections.
    c_result, _ = kickoff_with_retry(
        lambda: (build_critic_crew(working, behavioral_raw, moderator_raw, settings), None),
        "critic",
    )
    TRACKER.add_crew("analysis_critic", c_result)

    if isinstance(c_result.pydantic, AnalysisReport):
        report = c_result.pydantic
    else:
        raw = c_result.raw if hasattr(c_result, "raw") else str(c_result)
        try:
            report = AnalysisReport.model_validate(json.loads(raw))
        except Exception:  # noqa: BLE001
            logger.warning("Could not parse structured output; wrapping raw text")
            report = AnalysisReport(
                summary=raw[:4000], critic_notes=["Unstructured model output"]
            )

    kept_extracted = ExtractedClaims(claims=kept_claims)
    facts, reconcile_notes = reconcile_checks(
        kept_extracted, grounded, specialist_checks, seen_urls
    )

    # Phase B2: cyclic fact-check manager — audits first-pass sources
    # (existence, date, relevance), hunts missing sub-facts, and re-verifies
    # weakly supported verdicts. Escalated claims get another focused round.
    manager_notes: list[str] = []
    if facts and settings.factcheck_manager_rounds > 0:
        from src.reconcile import apply_manager_reviews, prelim_facts_json

        # reconcile_checks emits one fact per claim, in claim order.
        facts_by_id = {c.id: f for c, f in zip(kept_claims, facts)}
        claim_by_id = {c.id: c for c in kept_claims}
        review_ids = [c.id for c in kept_claims]
        for round_no in range(1, settings.factcheck_manager_rounds + 1):
            round_claims = [claim_by_id[i] for i in review_ids]
            m_result, m_task = kickoff_with_retry(
                lambda: build_manager_crew(
                    round_claims,
                    prelim_facts_json(round_claims, facts_by_id),
                    settings,
                    seen_urls,
                    debate_date=debate_date,
                    round_no=round_no,
                ),
                f"manager r{round_no}",
            )
            TRACKER.add_crew(f"analysis_manager_r{round_no}", m_result)
            reviews = parse_task_output(m_task, ManagerReviewList)
            escalated, m_notes = apply_manager_reviews(
                round_claims, facts_by_id, reviews, seen_urls
            )
            manager_notes.extend(f"[round {round_no}] {n}" for n in m_notes)
            logger.info(
                "Manager round %d: reviewed %d claims, escalated %d",
                round_no,
                len(round_claims),
                len(escalated),
            )
            review_ids = escalated
            if not review_ids:
                break

    # Code-owned report sections: the LLM sees these fields in the output
    # schema, so they are always overwritten here, even when empty.
    report.facts = facts
    report.briefing = briefing
    report.speaker_map = smap
    report.question_audit = question_audit
    report.transcript_quality = outcome.quality
    report.moderator_audit.equal_time_distribution = []
    guest_names = smap.guests()
    _canonicalize_speakers(report, guest_names)

    from src.questions import question_balance_finding, render_dodges

    for s in report.behavioral_analysis.speakers:
        s.question_dodging = render_dodges(
            [q for q in question_audit if q.addressee == s.speaker]
        )

    notes = [*pipeline_notes, *reconcile_notes, *manager_notes]
    if notes:
        report.critic_notes = [*report.critic_notes, *notes]
    logger.info(
        "Reconciled %d facts from extract=%s grounded=%s specialists=%s (seen_urls=%d)",
        len(facts),
        len(kept_claims),
        len(grounded.checks) if grounded else 0,
        len(specialist_checks.checks) if specialist_checks else 0,
        len(seen_urls),
    )

    # Deterministic moderator metrics: time shares (word counts) and an
    # interruption proxy computed from the transcript beat LLM estimates.
    from src.scoring import apply_deterministic_moderator_metrics

    moderator_notes = apply_deterministic_moderator_metrics(report, working)
    if moderator_notes:
        report.critic_notes = [*report.critic_notes, *moderator_notes]
    balance = question_balance_finding(question_audit, guest_names)
    if balance:
        report.moderator_audit.findings.append(balance)

    from src.selection import build_claim_funnel
    from src.validation import enforce_accusation_support, name_words, validate_report

    report = validate_report(report, working, allowed_urls=seen_urls)
    # Speakers are already canonical here, which the guard needs: it matches
    # a fact's speaker against the transcript line labels exactly.
    accusation_notes = enforce_accusation_support(
        report.facts,
        working_lines,
        parse_lines(named),
        outcome.log,
        risky_words=name_words(_allowed_names(smap, briefing)),
    )
    report.critic_notes = [*report.critic_notes, *accusation_notes]
    report.claim_funnel = build_claim_funnel(
        extracted.claims if extracted else [], kept_claims, report.facts, guest_names
    )
    return report, corrected


_PUBLISHER_EXCLUDE = {
    "briefing",  # background only, never evidence
    "speaker_map",  # plumbing
    "transcript_quality",  # plumbing
    # Derived data: the post takes question counts from DebateVerdict and the
    # dodge texts from behavioral_analysis.question_dodging.
    "question_audit",
}


def _publisher_report_json(report: AnalysisReport) -> str:
    """The report as the publisher sees it: no code-made plumbing, no duplicates."""
    return json.dumps(
        report.model_dump(mode="json", exclude=_PUBLISHER_EXCLUDE),
        ensure_ascii=False,
        indent=2,
    )


def generate_facebook_post(
    report: AnalysisReport,
    verdict: Any,
    settings: Settings | None = None,
) -> FacebookPost:
    """Generate a Slovak Facebook post from report + deterministic DebateVerdict."""
    from crewai import Agent, Crew, Process, Task

    settings = settings or get_settings()
    if not settings.vertex_ready():
        raise RuntimeError("GCP_PROJECT_ID is not configured")

    llm = build_llm(settings)
    report_json = _publisher_report_json(report)
    verdict_json = json.dumps(
        verdict.model_dump(mode="json") if hasattr(verdict, "model_dump") else verdict,
        ensure_ascii=False,
        indent=2,
    )

    publisher = Agent(
        role="Social Media Publisher",
        goal=(
            "Write a clear, fair, slightly gamified Slovak Facebook post that helps "
            "readers orient themselves in a political debate using only provided data."
        ),
        backstory=(
            "You are an editorial social writer for a transparency project. "
            "You never invent facts, winners, or scores. You write about claims, "
            "not about people being liars. Tone: factual, scannable, non-partisan."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Napíš slovenský Facebook post podľa dodaných dát.\n\n"
            "PRAVIDLÁ:\n"
            "- Víťaza, red flag a skóre ber VÝLUČNE z DebateVerdict — neurčuj ich sám.\n"
            "- Ak DebateVerdict.scoring_status nie je 'ok', víťaza nevyhlasuj a "
            "jednou vetou uveď, že rečníkov sa nepodarilo spoľahlivo priradiť "
            "k prepisu.\n"
            "- Pri rečníkoch môžeš uviesť podiel slov medzi hosťami "
            "(word_share_percent), počet prehovorov (turns), podstatné otázky a "
            "vyhnutia (challenging_questions, questions_dodged) — iba z DebateVerdict.\n"
            "- Ak je winner prázdny, výsledok je nerozhodný. Nepíš, že niekto "
            "vyhral debatu. Uveď rozdiel (margin) a že ide o tesný fair-play výsledok.\n"
            "- Zobraz prehľadný scoreboard podľa disciplín z DebateVerdict "
            "(pravdivosť, manipulácia, vecnosť, slušnosť): hodnoty skóre per rečník "
            "a víťaza každej disciplíny (discipline_winners). Ak disciplína v "
            "discipline_winners chýba, napíš pri nej nerozhodné, nevymýšľaj víťaza. "
            "Finálneho víťaza uveď osobitne s celkovým skóre, iba ak winner nie je prázdny.\n"
            "- Pomenuj to ako fair-play hodnotenie (presnosť a fauly), nie ako "
            "to, kto presvedčil diváka.\n"
            "- Verdikt 'Unverified' prezentuj ako 'neoverené' a 'Contested' ako "
            "'sporné (zdroje sa rozchádzajú)' — NIKDY nie ako klamstvo.\n"
            "- O nepravdách píš ako o tvrdeniach, nie o ľuďoch "
            "('toto tvrdenie je nepravdivé', nie 'X klamal').\n"
            "- Pri top 3 problematických tvrdeniach uveď verdikt, severity a source URL.\n"
            "- Férový, vecný tón; krátke odseky; mierna gamifikácia (scoreboard).\n"
            "- headline: jedna silná veta (hook).\n"
            "- body: hotový text postu pripravený na skopírovanie.\n"
            "- top_claims: max 3 položky (quote, speaker, verdict, severity, source_url).\n"
            "- disclaimer: že ide o AI analýzu a zdroje treba overiť.\n\n"
            f"DEBATEVERDICT:\n{verdict_json}\n\n"
            f"ANALYSISREPORT:\n{report_json}"
        ),
        expected_output="FacebookPost JSON: headline, body, top_claims, disclaimer.",
        agent=publisher,
        output_pydantic=FacebookPost,
    )

    crew = Crew(
        agents=[publisher],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()
    TRACKER.add_crew("facebook", result)

    if isinstance(result.pydantic, FacebookPost):
        return result.pydantic

    raw = result.raw if hasattr(result, "raw") else str(result)
    try:
        return FacebookPost.model_validate(json.loads(raw))
    except Exception:  # noqa: BLE001
        logger.warning("Could not parse FacebookPost; wrapping raw text as body")
        return FacebookPost(
            headline="",
            body=raw[:6000],
            disclaimer="AI analýza — over si zdroje v plnom reporte.",
        )
