"""Offline tests for run_analysis helpers and the CLI output helpers."""

from __future__ import annotations

import pytest

from src.agents import (
    AnalysisReport,
    BehavioralAnalysis,
    Severity,
    SpeakerTactics,
    Verdict,
    VerifiedFact,
    _allowed_names,
    _canonicalize_speakers,
)
from src.briefing import DebateBriefing, EntityEntry, GlossaryEntry, Participant
from src.report_models import SpeakerMap, SpeakerMapEntry, SpeakerRole
from src.scoring import DebateVerdict
from src.transcript_lines import parse_lines
from src.validation import enforce_accusation_support

LINE = "Erik Tomáš [00:10]: Minuli sme na to tristo miliónov eur z rozpočtu.\n"
QUOTE = "Minuli sme na to tristo miliónov eur z rozpočtu."


def test_allowed_names_merge_map_and_briefing() -> None:
    smap = SpeakerMap(entries=[SpeakerMapEntry(label="H", name="Erik Tomáš", role=SpeakerRole.GUEST)])
    briefing = DebateBriefing(
        participants=[Participant(name="Marián Viskupič", party="SaS")],
        entity_index=[EntityEntry(reference="minister financií", person="Ladislav Kamenický")],
        glossary=[GlossaryEntry(term="konsolidácia", definition="zníženie deficitu")],
    )
    names = _allowed_names(smap, briefing)
    assert {"Erik Tomáš", "Marián Viskupič", "SaS", "Ladislav Kamenický"} <= names
    # Glossary terms are ordinary words: as allowed proper nouns they would let
    # the corrector replace any word of a claim with debate jargon.
    assert "konsolidácia" not in names
    assert _allowed_names(smap, None) == {"Erik Tomáš"}


def test_allowed_names_never_let_a_content_word_be_rewritten() -> None:
    from src.correction import check_edit
    from src.report_models import EditType, TranscriptEdit

    smap = SpeakerMap(entries=[SpeakerMapEntry(label="H", name="Erik Tomáš", role=SpeakerRole.GUEST)])
    briefing = DebateBriefing(
        glossary=[
            GlossaryEntry(term="konsolidácia", definition="d"),
            GlossaryEntry(term="deficit", definition="d"),
        ]
    )
    names = _allowed_names(smap, briefing)
    line = "Minuli sme to z rozpočtu."
    for before, after in (("rozpočtu", "deficit"), ("Minuli", "konsolidácia")):
        edit = TranscriptEdit(line_no=0, type=EditType.PROPER_NOUN, before=before, after=after)
        assert check_edit(edit, line, names, {"Erik Tomáš"}) == "proper_noun"


def test_allowed_names_become_the_guard_risky_word_set() -> None:
    from src.validation import name_words

    smap = SpeakerMap(entries=[SpeakerMapEntry(label="H", name="Erik Tomáš", role=SpeakerRole.GUEST)])
    briefing = DebateBriefing(participants=[Participant(name="Robert Fico", party="SMER")])
    assert name_words(_allowed_names(smap, briefing)) == {
        "erik", "tomáš", "robert", "fico", "smer",
    }


def test_canonicalize_speakers_in_facts_and_behavior() -> None:
    report = AnalysisReport(
        behavioral_analysis=BehavioralAnalysis(speakers=[SpeakerTactics(speaker="Tomáš")]),
        facts=[VerifiedFact(claim="c", speaker="Erika Tomáša", verdict=Verdict.TRUE)],
    )
    _canonicalize_speakers(report, ["Erik Tomáš", "Marián Viskupič"])
    assert report.facts[0].speaker == "Erik Tomáš"
    assert report.behavioral_analysis.speakers[0].speaker == "Erik Tomáš"


def test_canonicalize_speakers_keeps_unknown_name() -> None:
    report = AnalysisReport(facts=[VerifiedFact(claim="c", speaker="Speaker B", verdict=Verdict.TRUE)])
    _canonicalize_speakers(report, ["Erik Tomáš"])
    assert report.facts[0].speaker == "Speaker B"


def _accusation() -> VerifiedFact:
    return VerifiedFact(
        claim="c",
        speaker="Erika Tomáša",
        verdict=Verdict.FALSE,
        severity=Severity.MATERIAL,
        quote=QUOTE,
    )


def test_accusation_guard_needs_canonical_speakers_first() -> None:
    """The guard matches `ln.speaker == fact.speaker`, so canonicalization must run before it."""
    raw_fact = _accusation()
    enforce_accusation_support(
        [raw_fact], parse_lines(LINE), parse_lines(LINE), []
    )
    assert raw_fact.verdict == Verdict.UNVERIFIED
    assert raw_fact.timestamp == ""

    report = AnalysisReport(facts=[_accusation()])
    _canonicalize_speakers(report, ["Erik Tomáš"])
    enforce_accusation_support(
        report.facts, parse_lines(LINE), parse_lines(LINE), []
    )
    assert report.facts[0].verdict == Verdict.FALSE
    assert report.facts[0].timestamp == "00:10"
    assert report.facts[0].quote_raw


def _main():
    """`main` pulls in the download stack; skip where it is not installed."""
    pytest.importorskip("yt_dlp")
    import main

    return main


def test_split_guests() -> None:
    _split_guests = _main()._split_guests
    assert _split_guests("Erik Tomáš; Marián Viskupič ") == ["Erik Tomáš", "Marián Viskupič"]
    assert _split_guests(" ; ") is None
    assert _split_guests(None) is None


def test_verdict_line_degraded_names_no_winner() -> None:
    line = _main()._verdict_line(DebateVerdict(scoring_status="degraded", margin=1.2, winner=""))
    assert "degrad" in line
    assert "scoring_status=degraded" in line
    assert "náskok" not in line


def test_verdict_line_tie_and_winner() -> None:
    _verdict_line = _main()._verdict_line
    assert "nerozhodn" in _verdict_line(DebateVerdict(scoring_status="ok", margin=1.2))
    won = _verdict_line(DebateVerdict(scoring_status="ok", winner="Erik Tomáš", margin=9.0))
    assert "Erik Tomáš" in won and "9.0" in won


def test_run_analysis_takes_the_roster_and_has_no_llm_rewrite() -> None:
    import inspect

    from src import agents

    params = inspect.signature(agents.run_analysis).parameters
    assert params["guests"].default is None
    assert params["moderator"].default is None
    assert not hasattr(agents, "correct_transcript")


def test_cli_has_guests_and_moderator_options() -> None:
    opts = {p.name for p in _main().main.params}
    assert {"guests", "moderator", "debate_date", "url"} <= opts
