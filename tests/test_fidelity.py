"""Offline tests for the accusation fidelity gate (src/fidelity.py).

Stored reports refuted claims the speaker never made: a launched model
instead of a won allocation, "90 votes are not required" against "it got 90
votes", a missing vote record against an earlier first-reading vote. The gate
re-reads each False/Misleading verdict against the transcript around the quote
and the debate's other verdicts, and may only downgrade.
"""

from __future__ import annotations

from src.agents import Severity, Verdict, VerifiedFact
from src.fidelity import (
    FidelityOutput,
    Reading,
    audit_accusations,
    build_fidelity_prompt,
    transcript_window,
)
from src.transcript_lines import parse_lines

TRANSCRIPT = """\
Moderátor [11:50]: Čo ste presadili?
Erik Tomáš [11:59]: Presadili sme v prvom čítaní aj s ústavnou väčšinou odpustenie daní matkám.
Moderátor [12:30]: A opozícia?
Marián Viskupič [12:40]: To je kampaň.
Moderátor [17:50]: Kto to podporil?
Erik Tomáš [18:04]: Prešlo to ústavnou väčšinou. Áno, KDH to podporilo.
Moderátor [18:30]: Ďakujem.
"""
LINES = parse_lines(TRANSCRIPT)


def fact(verdict=Verdict.FALSE, ts="18:04", claim="KDH podporilo návrh.") -> VerifiedFact:
    return VerifiedFact(
        claim=claim,
        speaker="Erik Tomáš",
        quote="Prešlo to ústavnou väčšinou. Áno, KDH to podporilo.",
        verdict=verdict,
        severity=Severity.MATERIAL if verdict in (Verdict.FALSE, Verdict.MISLEADING) else None,
        sources=["https://www.nrsr.sk/x"],
        rationale="K dátumu debaty sa nehlasovalo.",
        timestamp=ts,
    )


def out(reading: Reading) -> FidelityOutput:
    return FidelityOutput(
        asserted="návrh prešiel prvým čítaním a KDH zaň hlasovalo",
        refuted="neuskutočnilo sa záverečné hlasovanie",
        reading=reading,
        issue="vyvracia iné hlasovanie",
    )


def test_window_surrounds_the_quoted_line() -> None:
    w = transcript_window(LINES, fact(), before=2, after=1)
    assert "Erik Tomáš [18:04]" in w
    assert "Moderátor [17:50]" in w and "Marián Viskupič [12:40]" in w
    assert "Moderátor [18:30]" in w
    assert "[11:59]" not in w


def test_window_is_empty_without_a_matching_line() -> None:
    assert transcript_window(LINES, fact(ts="")) == ""
    assert transcript_window(LINES, fact(ts="99:99")) == ""


def test_prompt_carries_quote_context_siblings_and_rationale() -> None:
    target = fact()
    sibling = fact(
        Verdict.MISLEADING, ts="11:59", claim="Návrh prešiel prvým čítaním ústavnou väčšinou."
    )
    sibling.rationale = "Návrh bol posunutý do druhého čítania."
    p = build_fidelity_prompt(target, transcript_window(LINES, target), [target, sibling])
    assert target.quote in p
    assert "Erik Tomáš [18:04]" in p
    assert target.rationale in p
    assert "posunutý do druhého čítania" in p
    assert p.count("KDH podporilo návrh.") == 1  # the fact itself is not its own sibling


def test_only_accusations_are_audited_and_mismatches_downgraded() -> None:
    facts = [fact(Verdict.FALSE), fact(Verdict.TRUE), fact(Verdict.MISLEADING)]
    readings = iter([Reading.DIFFERENT, Reading.SAME])
    prompts: list[str] = []

    def llm(prompt: str) -> FidelityOutput:
        prompts.append(prompt)
        return out(next(readings))

    notes = audit_accusations(facts, LINES, llm=llm)
    assert len(prompts) == 2
    assert facts[0].verdict == Verdict.UNVERIFIED and facts[0].severity is None
    assert "vyvracia iné hlasovanie" in facts[0].rationale
    assert facts[1].verdict == Verdict.TRUE
    assert facts[2].verdict == Verdict.MISLEADING
    assert len(notes) == 1 and "different_referent" in notes[0]


def test_every_non_same_reading_downgrades() -> None:
    for reading in Reading:
        f = fact()
        audit_accusations([f], LINES, llm=lambda _p, r=reading: out(r))
        expected = Verdict.FALSE if reading == Reading.SAME else Verdict.UNVERIFIED
        assert f.verdict == expected, reading


def test_llm_failure_keeps_the_verdict_and_notes_it() -> None:
    f = fact()

    def boom(_prompt: str) -> FidelityOutput:
        raise RuntimeError("quota")

    notes = audit_accusations([f], LINES, llm=boom)
    assert f.verdict == Verdict.FALSE
    assert any("quota" in n for n in notes)


def test_prompt_carries_recorded_reading_when_present() -> None:
    f = fact()
    f.interpretation = "Návrh prešiel prvým čítaním a KDH zaň hlasovalo."
    f.not_asserted = ["návrh bol definitívne schválený"]
    p = build_fidelity_prompt(f, "", [f])
    assert "READING RECORDED BEFORE CHECKING (may be wrong)" in p
    assert f.interpretation in p and "návrh bol definitívne schválený" in p
    # The instructions mention the reading; only the block carries this header.
    assert "READING RECORDED BEFORE CHECKING (may be wrong)" not in build_fidelity_prompt(fact(), "", [])


def test_siblings_can_differ_from_audited_facts() -> None:
    target = fact()
    other = fact(Verdict.TRUE, ts="11:59", claim="Iné tvrdenie o daniach.")
    prompts: list[str] = []

    def llm(prompt: str) -> FidelityOutput:
        prompts.append(prompt)
        return out(Reading.SAME)

    audit_accusations([target], LINES, llm=llm, siblings=[target, other])
    assert len(prompts) == 1 and "Iné tvrdenie o daniach." in prompts[0]
