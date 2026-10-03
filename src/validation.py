"""Post-hoc grounding checks that catch fact-checker hallucinations.

The fact-checking agents sometimes extract or verify a claim that does not
actually correspond to anything said in the transcript (e.g. mixing up who
said what, or fabricating a claim that resembles a *different*, unrelated
real-world event). Since each ``VerifiedFact`` now carries a verbatim
``quote`` copied from the transcript, we can mechanically verify that the
quote really occurs in the transcript and drop facts that fail this check
instead of shipping a hallucinated claim in the final report.

The agents also sometimes invent plausible-looking source URLs (a real
domain with a fabricated article slug) that return 404 / don't resolve at
all. ``validate_fact_sources`` does a best-effort live check and strips
those dead links so the report never presents a fake citation as evidence.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Callable

from src.agents import AnalysisReport, Verdict, VerifiedFact
from src.correction import negation_flip, number_tokens
from src.report_models import EditResult, EditType
from src.transcript_lines import Line

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_QUOTE_RE = re.compile(r"[\"'„“”‚‘’«»]([^\"'„“”‚‘’«»]{6,})[\"'„“”‚‘’«»]")
_ELLIPSIS_RE = re.compile(r"\s*(?:\.{2,}|…)\s*")

# Domains that must NEVER back a verdict, even if they resolve and were
# really returned by a search — dropped exactly like a fabricated URL.
#
# `vertex_grounded_search` (unlike `sk_source_search`) has no domain
# allowlist: it's raw Google Search grounding, so it can and does return
# anything, including a political party's own website reporting on a rival.
# A party site is a primary interested party in the very claims it's being
# cited for, not an independent fact-checking source, so it's blocked here
# regardless of political leaning (symmetric across coalition/opposition).
_SK_DISINFO_DOMAINS = (
    "hlavnespravy.sk",
    "infovojna",
    "zemavek",
    "badatel.net",
    "ereport.sk",
    "standard.sk",
)
_SK_PARTY_DOMAINS = (
    # Governing coalition (2023-2027 term)
    "smer.sk",
    "hlas-sd.sk",
    "sns.sk",
    # Opposition
    "progresivne.sk",
    "kdh.sk",
    "sas.sk",
    "demokrati.sk",
    "hnutie-slovensko.sk",
    "hnutierepublika.sk",
)
BLOCKED_SOURCE_DOMAINS = _SK_DISINFO_DOMAINS + _SK_PARTY_DOMAINS
_UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; STVRDebateAnalyzer/1.0; +fact-check-link-verifier)"
    )
}

MIN_QUOTE_LEN = 8
DEFAULT_MIN_RATIO = 0.6
URL_CHECK_TIMEOUT = 6.0
RAW_MIN_SIMILARITY = 0.85
_ACCUSATION = (Verdict.FALSE, Verdict.MISLEADING)


def normalize(text: str) -> str:
    """Lowercase + collapse to whitespace-separated word tokens for fuzzy matching."""
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _span_grounded(quote_norm: str, transcript_norm: str, min_ratio: float) -> bool:
    """Grounding test for a single already-normalized span."""
    if quote_norm in transcript_norm:
        return True
    matcher = difflib.SequenceMatcher(None, quote_norm, transcript_norm, autojunk=True)
    match = matcher.find_longest_match(0, len(quote_norm), 0, len(transcript_norm))
    return (match.size / len(quote_norm)) >= min_ratio


def quote_grounded(
    quote: str, transcript: str, min_ratio: float = DEFAULT_MIN_RATIO
) -> bool:
    """True if `quote` is substantially present in `transcript`.

    Exact (normalized) substring match is tried first; if that fails, we fall
    back to a fuzzy longest-common-match ratio so minor ASR noise / punctuation
    differences don't cause false positives. A fabricated quote that has no
    real basis in the transcript will score far below `min_ratio`.

    LLM evidence frequently stitches two real excerpts together with an ellipsis
    ("A... B"); such a quote never matches as one contiguous span. We therefore
    also accept it when every substantial ellipsis-separated part is grounded.
    """
    quote_norm = normalize(quote)
    if len(quote_norm) < MIN_QUOTE_LEN:
        # Too short to meaningfully verify either way; don't block on it.
        return True

    transcript_norm = normalize(transcript)
    if _span_grounded(quote_norm, transcript_norm, min_ratio):
        return True

    # Ellipsis-joined quote: verify each part separately.
    parts = [normalize(p) for p in _ELLIPSIS_RE.split(quote)]
    checkable = [p for p in parts if len(p) >= MIN_QUOTE_LEN]
    if len(checkable) >= 2:
        return all(
            _span_grounded(p, transcript_norm, min_ratio) for p in checkable
        )
    return False


def extract_quoted_spans(text: str) -> list[str]:
    """Pull out quoted substrings (e.g. from behavioral-analysis evidence strings)."""
    return _QUOTE_RE.findall(text or "")


def validate_facts(
    facts: list[VerifiedFact], transcript: str
) -> tuple[list[VerifiedFact], list[str]]:
    """Drop facts whose supporting quote cannot be found in the transcript.

    Facts without a `quote` at all are kept (older/legacy data) but flagged so
    they're visibly weaker evidence, without discarding useful reports.
    """
    kept: list[VerifiedFact] = []
    notes: list[str] = []
    for fact in facts:
        quote = (fact.quote or "").strip()
        if not quote:
            notes.append(
                f"Fact has no supporting quote (unverifiable): \"{fact.claim[:120]}\""
            )
            kept.append(fact)
            continue
        if quote_grounded(quote, transcript):
            kept.append(fact)
        else:
            notes.append(
                "Removed ungrounded/hallucinated claim (quote not found in "
                f'transcript): "{fact.claim[:160]}"'
            )
    return kept, notes


def validate_evidence_list(
    entries: list[str], transcript: str
) -> tuple[list[str], list[str]]:
    """Drop behavioral-analysis evidence entries whose quoted excerpt is fabricated.

    Entries without any quoted span are kept as-is (free-text observations).
    """
    kept: list[str] = []
    notes: list[str] = []
    for entry in entries:
        spans = extract_quoted_spans(entry)
        if not spans:
            kept.append(entry)
            continue
        if any(quote_grounded(span, transcript) for span in spans):
            kept.append(entry)
        else:
            notes.append(f"Removed ungrounded evidence quote: \"{entry[:160]}\"")
    return kept, notes


def url_reachable(url: str, timeout: float = URL_CHECK_TIMEOUT) -> bool | None:
    """Best-effort check whether a source URL resolves to real content.

    Returns:
        True  - looks reachable (2xx/3xx).
        False - clearly broken/nonexistent (404, or the domain doesn't
                resolve/connect at all) — a strong signal the URL was
                fabricated by the model.
        None  - inconclusive (timeout, anti-bot block, etc.); callers should
                treat this as "keep, unverified" so legitimate sources that
                reject scrapers (403/999) aren't punished.
    """
    import requests

    try:
        resp = requests.head(
            url, timeout=timeout, allow_redirects=True, headers=_UA_HEADERS
        )
        if resp.status_code == 405:
            resp = requests.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers=_UA_HEADERS,
                stream=True,
            )
        if resp.status_code == 404:
            return False
        if resp.status_code < 400:
            return True
        return None
    except requests.exceptions.ConnectionError:
        # DNS/connection failure — the domain or host genuinely doesn't exist.
        return False
    except requests.exceptions.RequestException:
        return None


def _normalize_url(url: str) -> str:
    """Loosely normalize a URL for provenance matching (scheme/www/trailing slash)."""
    u = (url or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.rstrip("/")


def is_blocked_source(url: str) -> bool:
    """True if the URL is on a disinfo or partisan-primary-source domain."""
    u = _normalize_url(url)
    return any(d in u for d in BLOCKED_SOURCE_DOMAINS)


def validate_fact_sources(
    facts: list[VerifiedFact],
    url_checker: Callable[[str], bool | None] = url_reachable,
    allowed_urls: set[str] | None = None,
) -> tuple[list[VerifiedFact], list[str]]:
    """Strip fabricated source URLs from facts in-place.

    Two guards, strongest first:
    * Provenance: if `allowed_urls` is given (the exact URLs the search tools
      returned during the run), any URL not among them was invented by the model
      and is dropped — this catches fabricated links that still return HTTP 200.
    * Reachability: remaining URLs that 404 / don't resolve are dropped too.
    """
    allowed_norm = {_normalize_url(u) for u in allowed_urls} if allowed_urls else None
    notes: list[str] = []
    for fact in facts:
        if not fact.sources:
            continue
        kept_sources: list[str] = []
        for url in fact.sources:
            if is_blocked_source(url):
                notes.append(
                    f'Removed disinfo/partisan-primary-source URL "{url}" cited '
                    f'for claim: "{fact.claim[:120]}"'
                )
                continue
            if allowed_norm is not None and _normalize_url(url) not in allowed_norm:
                notes.append(
                    f'Removed fabricated source URL "{url}" (not returned by any '
                    f'search) cited for claim: "{fact.claim[:120]}"'
                )
                continue
            if url_checker(url) is False:
                notes.append(
                    f'Removed dead/fabricated source URL "{url}" cited for claim: '
                    f'"{fact.claim[:120]}"'
                )
            else:
                kept_sources.append(url)
        if fact.sources and not kept_sources:
            notes.append(
                f'All sources for claim "{fact.claim[:120]}" were dead/fabricated; '
                "verdict is now uncited."
            )
        fact.sources = kept_sources
    return facts, notes


def enforce_verdict_support(
    facts: list[VerifiedFact],
) -> tuple[list[VerifiedFact], list[str]]:
    """Downgrade False/Misleading verdicts that have no verifiable source left.

    After fabricated/dead URLs are stripped, a `False`/`Misleading` verdict with
    zero sources rests only on the model's prior belief — exactly the failure
    mode where the fact-checker invented a "known disinformation" ruling. Such
    verdicts are downgraded to `Unverified` so the report never ships an
    unsupported accusation as hard fact (and scoring never penalizes it).
    """
    notes: list[str] = []
    for fact in facts:
        if fact.verdict in (Verdict.FALSE, Verdict.MISLEADING) and not fact.sources:
            notes.append(
                f'Downgraded {fact.verdict.value}->Unverified (no verifiable source '
                f'after removing fabricated/dead links): "{fact.claim[:120]}"'
            )
            fact.verdict = Verdict.UNVERIFIED
            fact.severity = None
    return facts, notes


def validate_report(
    report: AnalysisReport,
    transcript: str,
    url_checker: Callable[[str], bool | None] = url_reachable,
    allowed_urls: set[str] | None = None,
) -> AnalysisReport:
    """Strip hallucinated facts/evidence/sources from `report` before it's shipped."""
    kept_facts, fact_notes = validate_facts(report.facts, transcript)
    report.facts = kept_facts

    _, source_notes = validate_fact_sources(
        report.facts, url_checker=url_checker, allowed_urls=allowed_urls
    )
    _, verdict_notes = enforce_verdict_support(report.facts)

    all_notes = [*fact_notes, *source_notes, *verdict_notes]
    for speaker in report.behavioral_analysis.speakers:
        # Validate every evidence list that feeds scoring — the heaviest
        # penalties (manipulation, fallacies, dodging) must not rest on
        # fabricated quotes any more than civility does.
        for attr in ("speech_tactics", "manipulation", "logical_fallacies",
                     "question_dodging"):
            kept_list, list_notes = validate_evidence_list(
                getattr(speaker, attr), transcript
            )
            setattr(speaker, attr, kept_list)
            all_notes.extend(list_notes)

        kept_inciv, inciv_notes = validate_evidence_list(
            speaker.civility.incivility, transcript
        )
        speaker.civility.incivility = kept_inciv
        all_notes.extend(inciv_notes)

        kept_fair, fair_notes = validate_evidence_list(
            speaker.civility.fair_conduct, transcript
        )
        speaker.civility.fair_conduct = kept_fair
        all_notes.extend(fair_notes)

    if all_notes:
        report.critic_notes = [*report.critic_notes, *all_notes]
    return report


def best_window(quote: str, text: str) -> tuple[float, str]:
    """Most similar run of words in `text`, as long as the quote (ratio, original words)."""
    q = normalize(quote)
    toks = (text or "").split()
    if not q or not toks:
        return 0.0, ""
    n = len(q.split())
    norm = [normalize(t) for t in toks]
    q_chars = Counter(q)
    # Upper bound on the ratio of every window (SequenceMatcher.quick_ratio logic,
    # without building a matcher). Windows are then scored best-bound-first and the
    # scan stops once no remaining bound can reach the best ratio, so the result
    # (highest ratio, earliest window on ties) equals scoring every window in order.
    cands: list[tuple[float, int, str]] = []
    for i in range(max(1, len(toks) - n + 1)):
        cand = " ".join(w for w in norm[i : i + n] if w)
        c_chars = Counter(cand)
        shared = sum(min(v, c_chars[c]) for c, v in q_chars.items())
        cands.append((2.0 * shared / (len(q) + len(cand)), i, cand))
    cands.sort(key=lambda c: (-c[0], c[1]))
    best_ratio, best_i = 0.0, -1
    for bound, i, cand in cands:
        if bound < best_ratio:
            break
        ratio = difflib.SequenceMatcher(None, q, cand).ratio()
        if ratio > best_ratio or (ratio == best_ratio and ratio > 0 and i < best_i):
            best_ratio, best_i = ratio, i
    if best_i < 0:
        return 0.0, ""
    return best_ratio, " ".join(toks[best_i : best_i + n])


def raw_quote_support(quote: str, raw_text: str) -> tuple[float, str]:
    """Similarity of a (possibly ellipsis-joined) quote to the raw ASR text."""
    parts = [p for p in _ELLIPSIS_RE.split(quote) if len(normalize(p)) >= MIN_QUOTE_LEN]
    scored = [best_window(p, raw_text) for p in (parts or [quote])]
    return min(s for s, _ in scored), " … ".join(w for _, w in scored)


def _contains_run(hay: list[str], needle: list[str]) -> bool:
    """True when `needle` occurs in `hay` as a contiguous in-order run (empty -> True)."""
    n = len(needle)
    return n == 0 or any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def name_words(names: set[str] | list[str]) -> set[str]:
    """Casefolded words of every name the transcript corrector was allowed to write."""
    return {w.casefold() for name in names for w in _WORD_RE.findall(name or "")}


def _capitalised(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text or "") if w[:1].isupper()}


def _edit_is_risky(result: EditResult, quote_norm: str, risky_words: set[str]) -> bool:
    """True when the accusation could stand or fall on this applied edit.

    A split decides who said the quote. Any other edit matters only inside the
    quoted span, and then whenever it touches a name: the declared edit type is
    the model's own word for it, so a 'spelling' fix of Fico to Fica is judged
    by what it rewrites, not by its label.
    """
    edit = result.edit
    if edit.type == EditType.SPLIT_TURN:
        return True
    if normalize(edit.after) not in quote_norm:
        return False
    if edit.type == EditType.PROPER_NOUN:
        return True
    words = {w.casefold() for w in _WORD_RE.findall(f"{edit.before} {edit.after}")}
    return bool(words & risky_words) or _capitalised(edit.before) != _capitalised(edit.after)


def _downgrade(fact: VerifiedFact, why: str, notes: list[str]) -> None:
    notes.append(f'Downgraded {fact.verdict.value}->Unverified ({why}): "{fact.claim[:120]}"')
    fact.verdict = Verdict.UNVERIFIED
    fact.severity = None
    fact.rationale = f"{fact.rationale} [{why}]".strip()


def enforce_accusation_support(
    facts: list[VerifiedFact],
    corrected_lines: list[Line],
    raw_lines: list[Line],
    log: list[EditResult],
    risky_words: set[str] | None = None,
) -> list[str]:
    """Annotate every fact with its raw ASR window; downgrade False/Misleading
    verdicts whose quote is misattributed, cross-talk only, loosely quoted, or
    dependent on a transcript correction."""
    notes: list[str] = []
    applied = {r.id: r for r in log if r.applied}
    for fact in facts:
        # These fields are code-owned; never trust values the LLM put in its output.
        fact.quote_raw = ""
        fact.timestamp = ""
        fact.transcript_edits = []
        accusation = fact.verdict in _ACCUSATION
        if len(normalize(fact.quote)) < MIN_QUOTE_LEN:
            # Too short to attribute to a line; an accusation needs a substantive quote.
            if accusation:
                _downgrade(fact, "citácia je príliš krátka na overenie", notes)
            continue
        reliable = [ln for ln in corrected_lines if ln.speaker == fact.speaker and not ln.crosstalk]
        parts = [p for p in _ELLIPSIS_RE.split(fact.quote or "") if p.strip()] or [fact.quote or ""]
        hits = [ln for ln in reliable if any(quote_grounded(p, ln.text) for p in parts)]
        if not hits or not quote_grounded(fact.quote or "", "\n".join(ln.text for ln in hits)):
            if accusation:
                _downgrade(fact, "nepotvrdené priradenie rečníka", notes)
            continue
        fact.timestamp = hits[0].ts
        raw_text = "\n".join(
            raw_lines[ln.raw_no].text
            for ln in hits
            if ln.raw_no is not None and ln.raw_no < len(raw_lines)
        )
        similarity, window = raw_quote_support(fact.quote, raw_text)
        fact.quote_raw = window
        fact.transcript_edits = sorted({i for ln in hits for i in ln.edit_ids if i in applied})
        if not accusation:
            continue
        quote_norm = normalize(fact.quote)
        risky = [
            applied[i]
            for i in fact.transcript_edits
            if _edit_is_risky(applied[i], quote_norm, risky_words or set())
        ]
        if similarity < RAW_MIN_SIMILARITY:
            _downgrade(fact, "citácia sa nezhoduje s pôvodným prepisom", notes)
        elif risky:
            _downgrade(fact, "verdikt závisí od opravy prepisu", notes)
        elif not _contains_run(number_tokens(window), number_tokens(fact.quote)) or (
            negation_flip(fact.quote, window)
        ):
            _downgrade(fact, "číslo alebo zápor v citácii chýba v pôvodnom prepise", notes)
    return notes
