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


def _invariants(rows) -> None:
    for r in rows:
        assert r.extracted >= r.selected_for_check >= r.final_facts
        assert r.extracted == r.non_empirical + r.dropped_by_selection + r.selected_for_check
        assert r.selected_for_check == r.removed_ungrounded + r.final_facts


def test_refresh_with_different_roster_keeps_facts() -> None:
    facts = [VerifiedFact(claim="a", speaker="Tomáš", verdict=Verdict.TRUE)]
    c = claim(1, "Erik Tomáš")
    rows = build_claim_funnel([c], [c], facts, NAMES)
    assert rows[0].checked == 1
    for roster in (["Tomáš"], []):
        refresh_funnel_verdicts(rows, facts, roster)
        assert (rows[0].speaker, rows[0].checked) == ("Erik Tomáš", 1)


def test_unmatched_fact_speakers_create_no_rows() -> None:
    c = claim(1, "Erik Tomáš")
    facts = [
        VerifiedFact(claim="a", speaker="Erik Tomáš", verdict=Verdict.TRUE),
        VerifiedFact(claim="b", speaker="", verdict=Verdict.FALSE),
        VerifiedFact(claim="c", speaker="Neznamy Hlas", verdict=Verdict.FALSE),
        VerifiedFact(claim="d", speaker="Marián Viskupič", verdict=Verdict.FALSE),  # no extracted claims
    ]
    rows = build_claim_funnel([c], [c], facts, NAMES)
    assert [r.speaker for r in rows] == ["Erik Tomáš"]
    assert (rows[0].final_facts, rows[0].checked) == (1, 1)
    _invariants(rows)
    refresh_funnel_verdicts(rows, facts, NAMES)
    assert [r.speaker for r in rows] == ["Erik Tomáš"]
    assert (rows[0].final_facts, rows[0].checked) == (1, 1)


def test_empty_inputs() -> None:
    assert build_claim_funnel([], [], [], []) == []
    rows = build_claim_funnel([], [], [VerifiedFact(claim="a", speaker="X", verdict=Verdict.TRUE)], NAMES)
    assert rows == []
    refresh_funnel_verdicts([], [], [])


def test_cap_cut_and_merged_duplicates_are_dropped_by_selection() -> None:
    claims = [claim(1, "Erik Tomáš"), claim(2, "Erik Tomáš"), claim(3, "Erik Tomáš")]
    # #2 merged into #1, #3 cut by the cost fuse: neither is in `kept`.
    rows = build_claim_funnel(claims, [claims[0]], [], NAMES)
    r = rows[0]
    assert (r.extracted, r.dropped_by_selection, r.selected_for_check) == (3, 2, 1)
    assert r.non_empirical == 0
    _invariants(rows)


def test_non_empirical_kept_claim_counts_only_as_non_empirical() -> None:
    c = claim(1, "Erik Tomáš", Checkability.PREDICTION)
    rows = build_claim_funnel([c], [c], [], NAMES)
    r = rows[0]
    assert (r.extracted, r.non_empirical, r.selected_for_check, r.dropped_by_selection) == (1, 1, 0, 0)
    _invariants(rows)
