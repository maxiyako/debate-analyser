"""Deterministic reconciliation of two independent fact-checker outputs.

Facts (verdict + sources) are authored here in code, not by the LLM critic.
Each checker's sources are provenance-filtered against the URLs actually returned
by search tools in this run; unsupported False/Misleading sides abstain.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents import (
    ClaimCheck,
    ClaimCheckList,
    ExtractedClaim,
    ExtractedClaims,
    ManagerReviewList,
    Severity,
    Verdict,
    VerifiedFact,
)
from src.validation import _normalize_url, is_blocked_source


@dataclass
class _Side:
    check: ClaimCheck | None
    sources: list[str]
    supported: bool


def _filter_sources(sources: list[str], seen_urls: set[str]) -> list[str]:
    """Keep only real (in-provenance) sources; drop disinfo/gray-zone domains."""
    allowed = {_normalize_url(u) for u in seen_urls}
    kept: list[str] = []
    for url in sources or []:
        if is_blocked_source(url):
            continue
        if _normalize_url(url) in allowed and url not in kept:
            kept.append(url)
    return kept


def _is_supported(check: ClaimCheck | None, sources: list[str]) -> bool:
    """True if this side's verdict can stand on real evidence.

    True with zero sources is allowed (search confirmed but citations missing).
    Unverified is an explicit abstention.
    False / Misleading without real sources = abstain (prior belief only).
    """
    if check is None:
        return False
    if check.verdict == Verdict.UNVERIFIED:
        return False
    if check.verdict == Verdict.TRUE:
        return True
    return bool(sources)


def merge_checklists(checklists: list[ClaimCheckList | None]) -> ClaimCheckList | None:
    """Merge specialist ClaimCheckLists into one (claim_ids are globally unique).

    If the same claim_id somehow appears in multiple lists, the first wins.
    Returns None when no checklist produced any checks.
    """
    merged: list[ClaimCheck] = []
    seen_ids: set[int] = set()
    for cl in checklists:
        if not cl:
            continue
        for check in cl.checks:
            if check.claim_id in seen_ids:
                continue
            seen_ids.add(check.claim_id)
            merged.append(check)
    if not merged:
        return None
    return ClaimCheckList(checks=merged)


def _index_by_id(checks: ClaimCheckList | None) -> dict[int, ClaimCheck]:
    if not checks:
        return {}
    return {c.claim_id: c for c in checks.checks if c.claim_id is not None}


def reconcile_checks(
    extracted: ExtractedClaims | None,
    grounded: ClaimCheckList | None,
    sk: ClaimCheckList | None,
    seen_urls: set[str],
) -> tuple[list[VerifiedFact], list[str]]:
    """Merge extractor + two checkers into VerifiedFact list.

    Rules (per claim):
    - Provenance: keep only sources in seen_urls.
    - Unverified, or False/Misleading with no real source → that side abstains.
    - Both agree (effective) → that verdict, merge real sources.
    - One supported, other abstains → take the supported side.
    - Both supported but disagree → Contested + merge sources + note.
    - Both abstain → Unverified, no severity.
    """
    notes: list[str] = []
    if not extracted or not extracted.claims:
        notes.append("Reconcile: no extracted claims; facts empty.")
        return [], notes

    g_by_id = _index_by_id(grounded)
    s_by_id = _index_by_id(sk)
    facts: list[VerifiedFact] = []

    for claim in extracted.claims:
        g_raw = g_by_id.get(claim.id)
        s_raw = s_by_id.get(claim.id)

        g_sources = _filter_sources(g_raw.sources if g_raw else [], seen_urls)
        s_sources = _filter_sources(s_raw.sources if s_raw else [], seen_urls)

        g = _Side(check=g_raw, sources=g_sources, supported=_is_supported(g_raw, g_sources))
        s = _Side(check=s_raw, sources=s_sources, supported=_is_supported(s_raw, s_sources))

        dropped_g = (len(g_raw.sources) if g_raw else 0) - len(g_sources)
        dropped_s = (len(s_raw.sources) if s_raw else 0) - len(s_sources)
        if dropped_g > 0:
            notes.append(
                f"Reconcile claim {claim.id}: dropped {dropped_g} fabricated "
                f'URL(s) from grounded checker for "{claim.claim[:80]}"'
            )
        if dropped_s > 0:
            notes.append(
                f"Reconcile claim {claim.id}: dropped {dropped_s} fabricated "
                f'URL(s) from SK checker for "{claim.claim[:80]}"'
            )

        fact = _merge_sides(claim, g, s, notes)
        facts.append(fact)

    return facts, notes


def _merge_sides(
    claim: ExtractedClaim, g: _Side, s: _Side, notes: list[str]
) -> VerifiedFact:
    label = f'claim {claim.id} "{claim.claim[:80]}"'

    if g.supported and s.supported:
        assert g.check is not None and s.check is not None
        if g.check.verdict == s.check.verdict:
            sources = _dedupe(g.sources + s.sources)
            return _build_fact(
                claim,
                g.check.verdict,
                _pick_severity(g.check, s.check),
                sources,
                _merge_signals(g.check, s.check),
                _merge_rationale(g.check, s.check),
            )
        # Real conflict — both have evidence, disagree. Contested carries no
        # severity and no speaker penalty: our checkers disagreeing is not
        # the speaker's fault.
        sources = _dedupe(g.sources + s.sources)
        notes.append(
            f"Reconcile {label}: grounded={g.check.verdict.value} vs "
            f"sk={s.check.verdict.value} (both sourced) → Contested"
        )
        return _build_fact(
            claim,
            Verdict.CONTESTED,
            None,
            sources,
            _merge_signals(g.check, s.check),
            (
                f"Checkers disagree (grounded={g.check.verdict.value}, "
                f"sk={s.check.verdict.value}). "
                f"Grounded: {g.check.rationale} | SK: {s.check.rationale}"
            ),
        )

    if g.supported and not s.supported:
        assert g.check is not None
        if s.check is not None and not s.supported:
            notes.append(
                f"Reconcile {label}: took grounded={g.check.verdict.value} "
                f"(SK abstained/unsupported)"
            )
        return _build_fact(
            claim,
            g.check.verdict,
            g.check.severity if g.check.verdict != Verdict.TRUE else None,
            g.sources,
            list(g.check.intent_signals),
            g.check.rationale,
        )

    if s.supported and not g.supported:
        assert s.check is not None
        if g.check is not None and not g.supported:
            notes.append(
                f"Reconcile {label}: took sk={s.check.verdict.value} "
                f"(grounded abstained/unsupported)"
            )
        return _build_fact(
            claim,
            s.check.verdict,
            s.check.severity if s.check.verdict != Verdict.TRUE else None,
            s.sources,
            list(s.check.intent_signals),
            s.check.rationale,
        )

    # Both abstain / missing
    notes.append(f"Reconcile {label}: both checkers unsupported → Unverified")
    rationale_parts = []
    if g.check and g.check.rationale:
        rationale_parts.append(f"Grounded: {g.check.rationale}")
    if s.check and s.check.rationale:
        rationale_parts.append(f"SK: {s.check.rationale}")
    return _build_fact(
        claim,
        Verdict.UNVERIFIED,
        None,
        [],
        [],
        " | ".join(rationale_parts) or "Unverified: no usable sources from either checker.",
    )


def prelim_facts_json(claims: list[ExtractedClaim], facts_by_id: dict[int, VerifiedFact]) -> str:
    """Compact first-pass results JSON handed to the fact-check manager."""
    import json

    rows = []
    for c in claims:
        f = facts_by_id.get(c.id)
        if f is None:
            continue
        rows.append(
            {
                "claim_id": c.id,
                "verdict": f.verdict.value,
                "severity": f.severity.value if f.severity else None,
                "sources": f.sources,
                "rationale": f.rationale,
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=1)


def apply_manager_reviews(
    claims: list[ExtractedClaim],
    facts_by_id: dict[int, VerifiedFact],
    reviews: ManagerReviewList | None,
    seen_urls: set[str],
) -> tuple[list[int], list[str]]:
    """Apply manager second-pass reviews onto reconciled facts, in place.

    Deterministic guards (the manager's output is NOT trusted blindly):
    - sources are provenance-filtered against seen_urls + blocklist;
    - False/Misleading with zero surviving sources is downgraded to Unverified;
    - severity is stripped for True/Unverified/Contested.

    Returns (claim_ids escalated for another round, notes).
    """
    notes: list[str] = []
    escalated: list[int] = []
    if not reviews or not reviews.reviews:
        notes.append("Manager review: no reviews produced; first-pass facts kept.")
        return escalated, notes

    claim_by_id = {c.id: c for c in claims}
    for review in reviews.reviews:
        claim = claim_by_id.get(review.claim_id)
        fact = facts_by_id.get(review.claim_id)
        if claim is None or fact is None:
            continue
        label = f'claim {review.claim_id} "{claim.claim[:80]}"'

        sources = _filter_sources(review.sources, seen_urls)
        dropped = len(review.sources or []) - len(sources)
        if dropped > 0:
            notes.append(
                f"Manager {label}: dropped {dropped} out-of-provenance URL(s) "
                "from manager output"
            )
        for rem in review.removed_sources:
            notes.append(
                f'Manager {label}: removed source "{rem.url}" — {rem.reason}'
            )

        verdict = review.verdict
        severity = review.severity
        if verdict in (Verdict.FALSE, Verdict.MISLEADING) and not sources:
            notes.append(
                f"Manager {label}: {verdict.value} had no surviving source → "
                "Unverified"
            )
            verdict = Verdict.UNVERIFIED
            severity = None
        if verdict in (Verdict.TRUE, Verdict.UNVERIFIED, Verdict.CONTESTED):
            severity = None

        if verdict != fact.verdict:
            notes.append(
                f"Manager {label}: verdict {fact.verdict.value} → {verdict.value}"
            )
        fact.verdict = verdict
        fact.severity = severity
        fact.sources = sources
        rationale = (review.rationale or "").strip()
        if review.subfacts:
            rationale += (
                ("\n" if rationale else "")
                + "Sub-facts: "
                + " | ".join(review.subfacts)
            )
        if rationale:
            fact.rationale = rationale

        if review.escalate and verdict in (Verdict.UNVERIFIED, Verdict.CONTESTED):
            escalated.append(review.claim_id)

    return escalated, notes


def _build_fact(
    claim: ExtractedClaim,
    verdict: Verdict,
    severity: Severity | None,
    sources: list[str],
    intent_signals: list[str],
    rationale: str,
) -> VerifiedFact:
    if verdict in (Verdict.TRUE, Verdict.UNVERIFIED, Verdict.CONTESTED):
        severity = None
    return VerifiedFact(
        claim=claim.claim,
        speaker=claim.speaker,
        quote=claim.quote,
        verdict=verdict,
        severity=severity,
        intent_signals=intent_signals,
        sources=sources,
        salience=max(1, min(5, int(claim.salience or 3))),
        rationale=rationale,
    )


def _dedupe(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        key = _normalize_url(u)
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def _pick_severity(a: ClaimCheck, b: ClaimCheck) -> Severity | None:
    if a.verdict == Verdict.TRUE:
        return None
    # Prefer the more severe of the two when both agree on False/Misleading.
    order = {
        None: 0,
        Severity.TRIVIAL: 1,
        Severity.MATERIAL: 2,
        Severity.FABRICATION: 3,
    }
    sa, sb = a.severity, b.severity
    if order.get(sa, 0) >= order.get(sb, 0):
        return sa
    return sb


def _merge_signals(a: ClaimCheck, b: ClaimCheck) -> list[str]:
    out: list[str] = []
    for s in [*a.intent_signals, *b.intent_signals]:
        if s and s not in out:
            out.append(s)
    return out


def _merge_rationale(a: ClaimCheck, b: ClaimCheck) -> str:
    parts = [p for p in (a.rationale, b.rationale) if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if parts[0] == parts[1]:
        return parts[0]
    return f"{parts[0]} | {parts[1]}"


def parse_task_output(task, model_cls):
    """Read structured output from a CrewAI Task after kickoff (pydantic or raw JSON)."""
    import json

    out = getattr(task, "output", None)
    if out is None:
        return None
    py = getattr(out, "pydantic", None)
    if isinstance(py, model_cls):
        return py
    raw = getattr(out, "raw", None) or ""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return model_cls.model_validate(data)
    except Exception:  # noqa: BLE001
        return None
