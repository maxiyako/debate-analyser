"""Unit tests for deterministic fair-play scoring (src/scoring.py)."""

from __future__ import annotations

from src.agents import (
    AnalysisReport,
    BehavioralAnalysis,
    Severity,
    SpeakerTactics,
    Verdict,
    VerifiedFact,
)
from src.scoring import score_report, transcript_speaker_stats


def _report(
    speakers: list[SpeakerTactics | str], facts: list[VerifiedFact] | None = None
) -> AnalysisReport:
    rows = [
        s if isinstance(s, SpeakerTactics) else SpeakerTactics(speaker=s)
        for s in speakers
    ]
    return AnalysisReport(
        behavioral_analysis=BehavioralAnalysis(speakers=rows),
        facts=facts or [],
    )


def _row(verdict, name: str):
    for r in verdict.scoreboard:
        if r.speaker == name:
            return r
    raise AssertionError(f"{name} not in scoreboard")


def _discipline(row, name: str):
    for d in row.disciplines:
        if d.discipline == name:
            return d
    raise AssertionError(f"discipline {name} missing")


def _fact(speaker: str, verdict: Verdict, severity=None, sources=None) -> VerifiedFact:
    return VerifiedFact(
        claim="c", speaker=speaker, quote="q",
        verdict=verdict, severity=severity, sources=sources or [],
    )


def test_unverified_is_not_penalized() -> None:
    facts = [_fact("Andrej Danko", Verdict.UNVERIFIED)]
    verdict = score_report(_report(["Andrej Danko", "Jan Novak"], facts))
    assert _row(verdict, "Andrej Danko").score == _row(verdict, "Jan Novak").score
    assert _discipline(_row(verdict, "Andrej Danko"), "truthfulness").score is None
    assert _row(verdict, "Andrej Danko").unverified_count == 1


def test_contested_is_not_penalized() -> None:
    facts = [_fact("Andrej Danko", Verdict.CONTESTED, sources=["https://nrsr.sk/x"])]
    verdict = score_report(_report(["Andrej Danko", "Jan Novak"], facts))
    assert _row(verdict, "Andrej Danko").score == _row(verdict, "Jan Novak").score
    assert _row(verdict, "Andrej Danko").contested_count == 1


def test_false_claim_lowers_truthfulness() -> None:
    facts = [
        _fact("Andrej Danko", Verdict.FALSE, Severity.MATERIAL, ["https://nrsr.sk/x"]),
        _fact("Jan Novak", Verdict.TRUE),
    ]
    verdict = score_report(_report(["Andrej Danko", "Jan Novak"], facts))
    danko = _discipline(_row(verdict, "Andrej Danko"), "truthfulness")
    novak = _discipline(_row(verdict, "Jan Novak"), "truthfulness")
    assert danko.score < novak.score < 100.0
    assert verdict.discipline_winners["truthfulness"] == "Jan Novak"
    assert _row(verdict, "Andrej Danko").score < _row(verdict, "Jan Novak").score


def test_truthfulness_is_rate_not_count() -> None:
    # 1 false of 5 checked beats 1 false of 1 checked — accuracy per claim.
    facts = [
        _fact("A", Verdict.FALSE, Severity.MATERIAL, ["https://nrsr.sk/x"]),
        *[_fact("A", Verdict.TRUE) for _ in range(4)],
        _fact("B", Verdict.FALSE, Severity.MATERIAL, ["https://nrsr.sk/x"]),
    ]
    verdict = score_report(_report(["A", "B"], facts))
    a = _discipline(_row(verdict, "A"), "truthfulness")
    b = _discipline(_row(verdict, "B"), "truthfulness")
    assert a.score > b.score


def test_behavioral_fouls_normalized_per_words() -> None:
    # Same 2 manipulations; A spoke 4x more words than B -> A scores higher.
    transcript = "\n".join(
        [*["A [00:01]: " + "slovo " * 600] * 4, "B [00:05]: " + "slovo " * 600]
    )
    speakers = [
        SpeakerTactics(speaker="A", manipulation=["m1", "m2"]),
        SpeakerTactics(speaker="B", manipulation=["m1", "m2"]),
    ]
    verdict = score_report(_report(speakers), transcript=transcript)
    a = _discipline(_row(verdict, "A"), "manipulation")
    b = _discipline(_row(verdict, "B"), "manipulation")
    assert a.score > b.score


def test_surname_only_fact_speaker_still_penalized() -> None:
    fact = _fact("Danko", Verdict.FALSE, Severity.MATERIAL, ["https://nrsr.sk/x"])
    verdict = score_report(_report(["Andrej Danko", "Jan Novak"], [fact]))
    assert _row(verdict, "Andrej Danko").false_count == 1
    assert _row(verdict, "Andrej Danko").score < _row(verdict, "Jan Novak").score


def test_speaker_label_style_does_not_over_merge() -> None:
    fact = _fact("Speaker B", Verdict.FALSE, Severity.MATERIAL, ["https://nrsr.sk/x"])
    verdict = score_report(_report(["Speaker A"], [fact]))
    assert _row(verdict, "Speaker A").false_count == 0
    assert _discipline(_row(verdict, "Speaker A"), "truthfulness").score is None


def test_transcript_speaker_stats_words_and_interruptions() -> None:
    transcript = (
        "Moderator [00:00]: Dobrý večer, vitajte v diskusii.\n"
        "A [00:05]: Chcem povedať, že rozpočet\n"
        "B [00:08]: To nie je pravda!\n"
        "A [00:10]: Nechajte ma dohovoriť.\n"
    )
    stats = transcript_speaker_stats(transcript)
    key_a = "a"
    assert stats[key_a].words == 4 + 3
    assert stats["b"].interruptions_caused == 1


def test_single_true_claim_is_not_perfect_truthfulness() -> None:
    facts = [_fact("A", Verdict.TRUE), _fact("B", Verdict.TRUE)]
    verdict = score_report(_report(["A", "B"], facts))
    score = _discipline(_row(verdict, "A"), "truthfulness").score
    assert score is not None and score < 95


def test_moderator_cannot_win() -> None:
    speakers = [
        SpeakerTactics(speaker="Moderátor", civility={"score": 10}),
        SpeakerTactics(speaker="Martin Dubéci", civility={"score": 6}),
        SpeakerTactics(speaker="Andrej Danko", civility={"score": 1}),
    ]
    verdict = score_report(_report(speakers))
    names = [r.speaker for r in verdict.scoreboard]
    assert "Moderátor" not in names
    assert verdict.winner == "Martin Dubéci"


def test_tied_discipline_has_no_winner() -> None:
    # Both clean: manipulation is 100 for each. Naming one winner was an artifact
    # of max() following overall scoreboard order.
    verdict = score_report(_report(["A", "B"]))
    assert "manipulation" not in verdict.discipline_winners
    assert verdict.winner == ""
    assert verdict.margin == 0.0


def test_close_scores_are_not_a_win() -> None:
    # Civility 6 vs 5 is a 10-point discipline gap but only ~3 final points
    # (weight 0.20). Under the 5-point margin that is not a debate winner.
    speakers = [
        SpeakerTactics(speaker="A", civility={"score": 6}),
        SpeakerTactics(speaker="B", civility={"score": 5}),
    ]
    verdict = score_report(_report(speakers))
    assert verdict.margin is not None and verdict.margin < 4
    assert verdict.winner == ""


def test_central_false_claim_hurts_more_than_trivia() -> None:
    facts = [
        VerifiedFact(
            claim="central", speaker="A", quote="q",
            verdict=Verdict.FALSE, severity=Severity.MATERIAL,
            sources=["https://nrsr.sk/x"], salience=5,
        ),
        VerifiedFact(
            claim="trivia", speaker="B", quote="q",
            verdict=Verdict.FALSE, severity=Severity.MATERIAL,
            sources=["https://nrsr.sk/x"], salience=1,
        ),
    ]
    verdict = score_report(_report(["A", "B"], facts))
    a = _discipline(_row(verdict, "A"), "truthfulness").score
    b = _discipline(_row(verdict, "B"), "truthfulness").score
    assert a is not None and b is not None and a < b


def test_evidence_and_details_present_in_disciplines() -> None:
    facts = [
        _fact("A", Verdict.FALSE, Severity.FABRICATION, ["https://nrsr.sk/x"]),
    ]
    speakers = [SpeakerTactics(speaker="A", manipulation=["strach: 'citát'"])]
    verdict = score_report(_report(speakers, facts))
    row = _row(verdict, "A")
    truth = _discipline(row, "truthfulness")
    manip = _discipline(row, "manipulation")
    assert truth.evidence and "False" in truth.evidence[0]
    assert manip.evidence == ["strach: 'citát'"]
    assert "nepravda: 1" in truth.detail
    assert verdict.method_notes
