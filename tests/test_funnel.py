"""Offline tests for the per-speaker claim funnel (src/selection.py)."""

from __future__ import annotations

from src.agents import Checkability, ExtractedClaim, Verdict, VerifiedFact
from src.selection import build_claim_funnel, refresh_funnel_verdicts

NAMES = ["Erik Tomáš", "Marián Viskupič"]


def claim(i: int, speaker: str, check: Checkability = Checkability.EMPIRICAL) -> ExtractedClaim:
    return ExtractedClaim(id=i, claim=f"tvrdenie {i}", speaker=speaker, quote="q", checkability=check)


def test_funnel_counts_and_invariants() -> None:
    claims = [
        claim(1, "Erik Tomáš"),
        claim(2, "Tomáš"),
        claim(3, "Erik Tomáš", Checkability.OPINION),
        claim(4, "Marián Viskupič"),
        claim(5, "Marián Viskupič"),
    ]
    kept = [claims[0], claims[1], claims[3]]
    facts = [
        VerifiedFact(claim="a", speaker="Erik Tomáš", verdict=Verdict.TRUE),
        VerifiedFact(claim="b", speaker="Marián Viskupič", verdict=Verdict.UNVERIFIED),
    ]
    rows = {r.speaker: r for r in build_claim_funnel(claims, kept, facts, NAMES)}
    erik, marian = rows["Erik Tomáš"], rows["Marián Viskupič"]
    assert (erik.extracted, erik.non_empirical, erik.selected_for_check, erik.dropped_by_selection) == (3, 1, 2, 0)
    assert (erik.final_facts, erik.removed_ungrounded, erik.checked) == (1, 1, 1)
    assert (marian.extracted, marian.selected_for_check, marian.dropped_by_selection) == (2, 1, 1)
    assert (marian.unverified, marian.checked) == (1, 0)
    for r in rows.values():
        assert r.extracted == r.non_empirical + r.dropped_by_selection + r.selected_for_check
        assert r.selected_for_check == r.removed_ungrounded + r.final_facts


def test_refresh_reflects_later_downgrades() -> None:
    facts = [VerifiedFact(claim="a", speaker="Erik Tomáš", verdict=Verdict.FALSE)]
    rows = build_claim_funnel([claim(1, "Erik Tomáš")], [claim(1, "Erik Tomáš")], facts, NAMES)
    facts[0].verdict = Verdict.UNVERIFIED
    refresh_funnel_verdicts(rows, facts, NAMES)
    assert (rows[0].checked, rows[0].unverified) == (0, 1)
