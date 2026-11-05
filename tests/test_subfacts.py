"""An accusation must rest on a sub-fact that a cited source refutes.

Seen in stored reports: verdicts like "the source does not mention the figure,
so Misleading" or "no vote record was found, so False". Both checkers and the
manager now report each sub-fact with a status, and code allows False only
with a refuted sub-fact and Misleading only with a refuted or distorted one,
each backed by a URL among the claim's surviving sources.
"""

from __future__ import annotations

from src.agents import (
    ClaimCheck,
    ClaimCheckList,
    ExtractedClaim,
    ExtractedClaims,
    ManagerReview,
    ManagerReviewList,
    SubFact,
    SubFactStatus,
    Verdict,
    VerifiedFact,
)
from src.reconcile import accusation_backed, apply_manager_reviews, reconcile_checks

URL = "https://www.sme.sk/c/1/clanok.html"
OTHER = "https://www.aktuality.sk/clanok/2/"


def sf(status: SubFactStatus, source: str = URL, text: str = "časť") -> SubFact:
    return SubFact(text=text, status=status, source=source)


def claim(cid: int = 1) -> ExtractedClaim:
    return ExtractedClaim(id=cid, claim="Deficit je 6 % HDP.", speaker="X", quote="q")


def check(verdict: Verdict, subfacts: list[SubFact], sources=(URL,)) -> ClaimCheck:
    return ClaimCheck(
        claim_id=1,
        verdict=verdict,
        severity="material" if verdict in (Verdict.FALSE, Verdict.MISLEADING) else None,
        sources=list(sources),
        subfacts=subfacts,
        rationale="r",
    )


def reconcile_one(c: ClaimCheck) -> VerifiedFact:
    facts, _ = reconcile_checks(
        ExtractedClaims(claims=[claim()]), ClaimCheckList(checks=[c]), None, {URL, OTHER}
    )
    return facts[0]


def test_false_needs_a_refuted_subfact() -> None:
    assert accusation_backed(Verdict.FALSE, [sf(SubFactStatus.REFUTED)], [URL])
    assert not accusation_backed(Verdict.FALSE, [sf(SubFactStatus.DISTORTED)], [URL])
    assert not accusation_backed(Verdict.FALSE, [sf(SubFactStatus.NOT_FOUND)], [URL])
    assert not accusation_backed(Verdict.FALSE, [], [URL])


def test_misleading_needs_a_refuted_or_distorted_subfact() -> None:
    assert accusation_backed(Verdict.MISLEADING, [sf(SubFactStatus.DISTORTED)], [URL])
    assert accusation_backed(Verdict.MISLEADING, [sf(SubFactStatus.REFUTED)], [URL])
    assert not accusation_backed(
        Verdict.MISLEADING,
        [sf(SubFactStatus.CONFIRMED), sf(SubFactStatus.NOT_FOUND)],
        [URL],
    )


def test_backing_subfact_must_cite_a_surviving_source() -> None:
    assert not accusation_backed(Verdict.FALSE, [sf(SubFactStatus.REFUTED, OTHER)], [URL])
    assert not accusation_backed(Verdict.FALSE, [sf(SubFactStatus.REFUTED, "")], [URL])
    # URL normalisation: trailing slash / scheme case do not matter.
    assert accusation_backed(Verdict.FALSE, [sf(SubFactStatus.REFUTED, URL + "/")], [URL])


def test_non_accusations_need_no_backing() -> None:
    assert accusation_backed(Verdict.TRUE, [], [])
    assert accusation_backed(Verdict.UNVERIFIED, [], [])


def test_reconcile_keeps_backed_false() -> None:
    fact = reconcile_one(check(Verdict.FALSE, [sf(SubFactStatus.REFUTED)]))
    assert fact.verdict == Verdict.FALSE


def test_reconcile_abstains_on_absence_only_misleading() -> None:
    fact = reconcile_one(
        check(
            Verdict.MISLEADING,
            [sf(SubFactStatus.CONFIRMED), sf(SubFactStatus.NOT_FOUND, "")],
        )
    )
    assert fact.verdict == Verdict.UNVERIFIED
    assert fact.severity is None


def test_manager_cannot_turn_missing_vote_record_into_false() -> None:
    fact = VerifiedFact(claim="c", speaker="X", quote="q", verdict=Verdict.UNVERIFIED)
    reviews = ManagerReviewList(
        reviews=[
            ManagerReview(
                claim_id=1,
                verdict=Verdict.FALSE,
                severity="material",
                sources=[URL],
                subfacts=[sf(SubFactStatus.NOT_FOUND, "", "Hlasovanie sa neuskutočnilo")],
                rationale="Záznam o hlasovaní sa nenašiel.",
            )
        ]
    )
    _, notes = apply_manager_reviews([claim()], {1: fact}, reviews, {URL})
    assert fact.verdict == Verdict.UNVERIFIED
    assert fact.severity is None
    assert any("refuted sub-fact" in n for n in notes)


def test_check_and_manager_prompts_require_subfact_statuses() -> None:
    from src.agents import _CHECK_EXPECTED_OUTPUT, _CHECK_RULES, _MANAGER_RULES

    for text in (_CHECK_RULES, _MANAGER_RULES):
        for status in SubFactStatus:
            assert status.value in text
    assert "subfacts" in _CHECK_EXPECTED_OUTPUT


def test_manager_rationale_lists_subfact_statuses() -> None:
    fact = VerifiedFact(claim="c", speaker="X", quote="q", verdict=Verdict.UNVERIFIED)
    reviews = ManagerReviewList(
        reviews=[
            ManagerReview(
                claim_id=1,
                verdict=Verdict.FALSE,
                severity="material",
                sources=[URL],
                subfacts=[sf(SubFactStatus.REFUTED, URL, "Deficit bol 4,9 %")],
                rationale="RRZ uvádza 4,9 %.",
            )
        ]
    )
    apply_manager_reviews([claim()], {1: fact}, reviews, {URL})
    assert fact.verdict == Verdict.FALSE
    assert f"Deficit bol 4,9 %: refuted ({URL})" in fact.rationale
