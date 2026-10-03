"""Offline tests for report section models."""

from __future__ import annotations

from src.agents import AnalysisReport, CorrectedTranscript, TimeShare, Verdict, VerifiedFact
from src.report_models import QuestionItem, SpeakerMap, SpeakerMapEntry, SpeakerRole


def _map() -> SpeakerMap:
    return SpeakerMap(
        status="ok",
        source="cli",
        entries=[
            SpeakerMapEntry(label="Speaker A", name="Moderátor", role=SpeakerRole.MODERATOR, confidence=1),
            SpeakerMapEntry(label="Speaker H", name="Erik Tomáš", role=SpeakerRole.GUEST, confidence=0.9),
            SpeakerMapEntry(label="Speaker D", name="Marián Viskupič", role=SpeakerRole.GUEST, confidence=0.9),
            SpeakerMapEntry(label="Speaker B", name="Záznam", role=SpeakerRole.CLIP, confidence=1),
        ],
    )


def test_speaker_map_helpers() -> None:
    smap = _map()
    assert smap.name_for("Speaker H") == "Erik Tomáš"
    assert smap.name_for("Speaker Z") is None
    assert smap.guests() == ["Erik Tomáš", "Marián Viskupič"]
    assert smap.moderator() == "Moderátor"


def test_analysis_report_carries_new_sections() -> None:
    dumped = AnalysisReport().model_dump(mode="json")
    assert dumped["speaker_map"] is None
    assert dumped["question_audit"] == []
    assert dumped["claim_funnel"] == []
    assert dumped["transcript_quality"] is None


def test_question_answer_text_not_serialized() -> None:
    q = QuestionItem(id=1, timestamp="01:00", addressee="A", question="Q?", answer_text="long")
    assert "answer_text" not in q.model_dump()


def test_new_fact_and_timeshare_fields_default() -> None:
    fact = VerifiedFact(claim="c", speaker="s", verdict=Verdict.TRUE)
    assert fact.quote_raw == "" and fact.timestamp == "" and fact.transcript_edits == []
    assert TimeShare(speaker="A").turns == 0
    assert CorrectedTranscript(text="x").log == []
