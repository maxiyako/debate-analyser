"""Tests for the fact-check manager apply logic (src/reconcile.py).

The manager agent's output is applied deterministically with the same
distrust as the first-pass checkers: out-of-provenance URLs are dropped,
and a False/Misleading verdict that loses all its sources is downgraded.
"""

from __future__ import annotations

from src.agents import (
    ExtractedClaim,
    ManagerReview,
    ManagerReviewList,
    RemovedSource,
    Verdict,
    VerifiedFact,
)
from src.reconcile import apply_manager_reviews, prelim_facts_json

REAL_URL = "https://www.sme.sk/c/123/clanok.html"
FAKE_URL = "https://dennikn.sk/999/vymysleny-clanok/"


def make_claim(cid: int = 1) -> ExtractedClaim:
    return ExtractedClaim(id=cid, claim="Deficit je 6 % HDP.", speaker="X", quote="q")


def make_fact(verdict: Verdict = Verdict.FALSE, sources: list[str] | None = None) -> VerifiedFact:
    return VerifiedFact(
        claim="Deficit je 6 % HDP.",
        speaker="X",
        quote="q",
        verdict=verdict,
        sources=sources or [],
    )


def test_manager_verdict_with_provenance_source_is_applied() -> None:
    claim, fact = make_claim(), make_fact(Verdict.UNVERIFIED)
    reviews = ManagerReviewList(
        reviews=[
            ManagerReview(
                claim_id=1,
                verdict=Verdict.FALSE,
                severity="material",
                sources=[REAL_URL],
                rationale="RRZ data contradict the figure.",
            )
        ]
    )
    escalated, notes = apply_manager_reviews([claim], {1: fact}, reviews, {REAL_URL})
    assert fact.verdict == Verdict.FALSE
    assert fact.sources == [REAL_URL]
    assert fact.rationale == "RRZ data contradict the figure."
    assert escalated == []


def test_out_of_provenance_source_is_dropped_and_verdict_downgraded() -> None:
    claim, fact = make_claim(), make_fact(Verdict.UNVERIFIED)
    reviews = ManagerReviewList(
        reviews=[
            ManagerReview(claim_id=1, verdict=Verdict.FALSE, sources=[FAKE_URL])
        ]
    )
    escalated, notes = apply_manager_reviews([claim], {1: fact}, reviews, {REAL_URL})
    assert fact.verdict == Verdict.UNVERIFIED
    assert fact.sources == []
    assert any("no surviving source" in n for n in notes)


def test_manager_downgrade_after_removing_dead_source() -> None:
    claim = make_claim()
    fact = make_fact(Verdict.FALSE, sources=[FAKE_URL])
    reviews = ManagerReviewList(
        reviews=[
            ManagerReview(
                claim_id=1,
                verdict=Verdict.UNVERIFIED,
                sources=[],
                removed_sources=[RemovedSource(url=FAKE_URL, reason="404 dead link")],
            )
        ]
    )
    escalated, notes = apply_manager_reviews([claim], {1: fact}, reviews, {FAKE_URL})
    assert fact.verdict == Verdict.UNVERIFIED
    assert fact.severity is None
    assert fact.sources == []
    assert any("404 dead link" in n for n in notes)


def test_escalate_only_for_unresolved_claims() -> None:
    claims = [make_claim(1), make_claim(2)]
    facts = {1: make_fact(Verdict.UNVERIFIED), 2: make_fact(Verdict.UNVERIFIED)}
    reviews = ManagerReviewList(
        reviews=[
            ManagerReview(claim_id=1, verdict=Verdict.UNVERIFIED, escalate=True),
            # escalate=True but verdict resolved to True -> no escalation
            ManagerReview(claim_id=2, verdict=Verdict.TRUE, escalate=True),
        ]
    )
    escalated, _ = apply_manager_reviews(claims, facts, reviews, set())
    assert escalated == [1]


def test_missing_reviews_keep_first_pass_facts() -> None:
    claim = make_claim()
    fact = make_fact(Verdict.TRUE, sources=[REAL_URL])
    escalated, notes = apply_manager_reviews([claim], {1: fact}, None, {REAL_URL})
    assert fact.verdict == Verdict.TRUE
    assert fact.sources == [REAL_URL]
    assert escalated == []


def test_prelim_facts_json_round_trips_ids() -> None:
    import json

    claim = make_claim(7)
    fact = make_fact(Verdict.MISLEADING, sources=[REAL_URL])
    fact.severity = None
    rows = json.loads(prelim_facts_json([claim], {7: fact}))
    assert rows[0]["claim_id"] == 7
    assert rows[0]["verdict"] == "Misleading"
    assert rows[0]["sources"] == [REAL_URL]
