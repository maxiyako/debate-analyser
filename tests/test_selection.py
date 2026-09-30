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
