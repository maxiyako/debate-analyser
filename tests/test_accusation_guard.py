"""Offline tests: an accusation never rests on a corrected or misattributed quote."""

from __future__ import annotations

from src.agents import Severity, Verdict, VerifiedFact
from src.correction import apply_corrections
from src.report_models import EditType, TranscriptEdit
from src.transcript_lines import parse_lines
from src.validation import enforce_accusation_support

RAW = parse_lines(
    "Moderátor [00:10]: Podporili ste to?\n"
    "Erik Tomáš [00:12]: Strana HLas presadila zvýšenie výživného zo 40 na 135 eur pre ľudi.\n"
    "Marián Viskupič [00:20]: (cez seba) Opozícia nepodporila zvýšenie príspevku pri narodení.\n"
    "Marián Viskupič [00:25]: Vláda zobrala každej rodine 858 eur cez konsolidáciu.\n"
)
CORRECTED, LOG = apply_corrections(
    RAW,
    [
        TranscriptEdit(line_no=1, type=EditType.PROPER_NOUN, before="HLas", after="HLAS"),
        TranscriptEdit(line_no=1, type=EditType.SPELLING, before="ľudi", after="ľudí"),
    ],
    {"HLAS"},
    {"Moderátor", "Erik Tomáš", "Marián Viskupič"},
)


def _fact(speaker: str, quote: str, verdict: Verdict = Verdict.FALSE) -> VerifiedFact:
    return VerifiedFact(
        claim="c", speaker=speaker, quote=quote, verdict=verdict,
        severity=Severity.MATERIAL, sources=["https://nrsr.sk/x"],
    )


def _run(fact: VerifiedFact) -> list[str]:
    return enforce_accusation_support([fact], CORRECTED, RAW, LOG)


def test_quote_from_another_speaker_is_downgraded() -> None:
    fact = _fact("Marián Viskupič", "zvýšenie výživného zo 40 na 135 eur")
    notes = _run(fact)
    assert fact.verdict == Verdict.UNVERIFIED and fact.severity is None
    assert "priradenie" in notes[0]


def test_crosstalk_only_quote_is_downgraded() -> None:
    fact = _fact("Marián Viskupič", "Opozícia nepodporila zvýšenie príspevku pri narodení")
    _run(fact)
    assert fact.verdict == Verdict.UNVERIFIED


def test_quote_depending_on_proper_noun_edit_is_downgraded() -> None:
    fact = _fact("Erik Tomáš", "Strana HLAS presadila zvýšenie výživného")
    notes = _run(fact)
    assert fact.verdict == Verdict.UNVERIFIED
    assert "opravy prepisu" in notes[0]


def test_spelling_only_quote_keeps_verdict_and_gets_raw_window() -> None:
    fact = _fact("Erik Tomáš", "zo 40 na 135 eur pre ľudí")
    notes = _run(fact)
    assert notes == []
    assert fact.verdict == Verdict.FALSE
    assert fact.quote_raw == "zo 40 na 135 eur pre ľudi."
    assert fact.timestamp == "00:12"
    assert fact.transcript_edits == [0, 1]


def test_loose_quote_below_raw_similarity_is_downgraded() -> None:
    fact = _fact(
        "Marián Viskupič",
        "Vláda zobrala každej rodine 858 eur cez konsolidáciu a ešte oveľa viac peňazí",
    )
    notes = _run(fact)
    assert fact.verdict == Verdict.UNVERIFIED
    assert "nezhoduje" in notes[0]


def test_true_verdicts_are_annotated_not_downgraded() -> None:
    fact = _fact("Marián Viskupič", "zvýšenie výživného zo 40 na 135 eur", Verdict.TRUE)
    assert _run(fact) == []
    assert fact.verdict == Verdict.TRUE


def test_llm_invented_code_fields_are_overwritten() -> None:
    # Misattributed quote: nothing grounded, so invented values must not survive.
    fact = _fact("Marián Viskupič", "zvýšenie výživného zo 40 na 135 eur")
    fact.quote_raw, fact.timestamp, fact.transcript_edits = "invented", "99:99", [7]
    _run(fact)
    assert (fact.quote_raw, fact.timestamp, fact.transcript_edits) == ("", "", [])
    # Grounded quote: replaced with code-derived values.
    ok = _fact("Erik Tomáš", "zo 40 na 135 eur pre ľudí")
    ok.quote_raw, ok.timestamp, ok.transcript_edits = "invented", "99:99", [7]
    _run(ok)
    assert ok.quote_raw == "zo 40 na 135 eur pre ľudi."
    assert ok.timestamp == "00:12"
    assert ok.transcript_edits == [0, 1]
