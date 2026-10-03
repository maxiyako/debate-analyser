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
from src.report_models import (
    ClaimFunnel,
    QuestionItem,
    QuestionKind,
    QuestionOutcome,
    SpeakerMap,
)
from src.scoring import (
    apply_deterministic_moderator_metrics,
    score_report,
    transcript_speaker_stats,
)


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
    # Transcript given so both speakers have manipulation data (no 1000-word
    # default any more); otherwise the comparison mixes discipline sets.
    transcript = (
        "Andrej Danko [00:01]: " + "slovo " * 600 + "\n"
        "Jan Novak [00:05]: " + "slovo " * 600 + "\n"
    )
    verdict = score_report(
        _report(["Andrej Danko", "Jan Novak"], [fact]), transcript=transcript
    )
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
    transcript = (
        "Moderátor [00:00]: " + "slovo " * 100 + "\n"
        "Martin Dubéci [00:10]: " + "slovo " * 600 + "\n"
        "Andrej Danko [00:20]: " + "slovo " * 600 + "\n"
    )
    verdict = score_report(_report(speakers), transcript=transcript)
    names = [r.speaker for r in verdict.scoreboard]
    assert "Moderátor" not in names
    assert verdict.winner == "Martin Dubéci"


def test_tied_discipline_has_no_winner() -> None:
    # Both clean: manipulation is 100 for each. Naming one winner was an artifact
    # of max() following overall scoreboard order.
    transcript = "A [00:01]: " + "slovo " * 600 + "\nB [00:05]: " + "slovo " * 600 + "\n"
    verdict = score_report(_report(["A", "B"]), transcript=transcript)
    assert _discipline(_row(verdict, "A"), "manipulation").score == 100.0
    assert "manipulation" not in verdict.discipline_winners
    assert verdict.winner == ""
    assert verdict.margin == 0.0


def test_close_scores_are_not_a_win() -> None:
    # Civility 6 vs 5 is a 10-point discipline gap but under 5 final points
    # once clean manipulation (100 each) is weighed in. Not a debate winner.
    transcript = "A [00:01]: " + "slovo " * 1000 + "\nB [00:05]: " + "slovo " * 1000 + "\n"
    speakers = [
        SpeakerTactics(speaker="A", civility={"score": 6}),
        SpeakerTactics(speaker="B", civility={"score": 5}),
    ]
    verdict = score_report(_report(speakers), transcript=transcript)
    assert verdict.margin is not None and verdict.margin < 5
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


def test_stats_ignore_markers_and_count_turn_types() -> None:
    transcript = (
        "M [00:00]: Otázka?\n"
        "A [00:05]: (cez seba) " + "slovo " * 20 + "\n"
        "M [00:10]: Áno.\n"
        "A [00:12]: Nie nie nie.\n"
    )
    stats = transcript_speaker_stats(transcript)
    assert stats["a"].words == 23
    assert stats["a"].turns == 2
    assert stats["a"].substantive_turns == 1
    assert stats["a"].interjections == 1
    assert stats["m"].interjections == 2


def test_stats_skip_lines_before_debate_start() -> None:
    transcript = "A [00:10]: zostrih slová\nA [02:00]: jedna dva tri\n"
    assert transcript_speaker_stats(transcript, debate_start="01:58")["a"].words == 3


def test_unmatched_transcript_is_degraded_not_assumed_1000() -> None:
    transcript = (
        "Speaker H [02:00]: " + "slovo " * 50 + "\n"
        "Speaker D [02:10]: " + "slovo " * 50 + "\n"
    )
    speakers = [
        SpeakerTactics(speaker="Erik Tomáš", manipulation=["m"], civility={"score": 9}),
        SpeakerTactics(speaker="Marián Viskupič", civility={"score": 2}),
    ]
    verdict = score_report(_report(speakers), transcript=transcript)
    assert verdict.scoring_status == "degraded"
    assert verdict.winner == ""
    row = _row(verdict, "Erik Tomáš")
    assert row.words == 0
    assert _discipline(row, "manipulation").score is None
    assert "manipulation" not in verdict.discipline_winners
    assert verdict.discipline_winners["civility"] == "Erik Tomáš"
    assert not any("1000 words" in n for n in verdict.method_notes)


def test_partial_speaker_map_is_degraded() -> None:
    report = _report(["A", "B"])
    report.speaker_map = SpeakerMap(status="partial", source="heuristic")
    transcript = "A [00:01]: " + "slovo " * 600 + "\nB [00:05]: " + "slovo " * 600 + "\n"
    assert score_report(report, transcript=transcript).scoring_status == "degraded"


def test_manipulation_rate_is_reproducible_from_fields() -> None:
    transcript = "A [00:01]: " + "slovo " * 2000 + "\n"
    speakers = [SpeakerTactics(speaker="A", manipulation=["m1", "m2"], logical_fallacies=["f"])]
    verdict = score_report(_report(speakers), transcript=transcript)
    d = _discipline(_row(verdict, "A"), "manipulation")
    assert d.inputs == {"manipulation": 2, "fallacies": 1}
    assert d.weights == {"manipulation": 8.0, "fallacies": 6.0}
    assert d.penalty_points == 22.0
    assert d.normalization_words == 2000
    assert d.rate_per_1000 == 11.0
    assert d.score == 89.0
    assert verdict.scoring_status == "ok"


def test_word_share_among_guests_and_turns() -> None:
    transcript = (
        "Moderátor [00:00]: " + "a " * 100 + "\n"
        "A [00:10]: " + "b " * 300 + "\n"
        "B [00:20]: " + "c " * 100 + "\n"
    )
    verdict = score_report(_report(["Moderátor", "A", "B"]), transcript=transcript)
    a = _row(verdict, "A")
    assert a.words == 300
    assert a.word_share_percent == 75.0
    assert a.transcript_share_percent == 60.0
    assert (a.turns, a.substantive_turns) == (1, 1)
    assert sum(r.word_share_percent for r in verdict.scoreboard) == 100.0


def _q(i: int, outcome, kind=QuestionKind.CHALLENGING) -> QuestionItem:
    return QuestionItem(id=i, timestamp="01:00", addressee="A", question=f"otázka {i}?",
                        kind=kind, outcome=outcome, reason="r")


def test_responsiveness_per_challenging_question() -> None:
    report = _report(["A"])
    report.question_audit = [
        _q(1, QuestionOutcome.INTERRUPTED),
        _q(2, QuestionOutcome.DODGED), _q(3, QuestionOutcome.DODGED),
        _q(4, QuestionOutcome.PARTIAL),
        *[_q(i, QuestionOutcome.ANSWERED) for i in range(5, 9)],
        _q(9, None, QuestionKind.OPEN),
    ]
    verdict = score_report(report, transcript="A [00:01]: " + "slovo " * 600 + "\n")
    row = _row(verdict, "A")
    d = _discipline(row, "responsiveness")
    assert d.inputs == {"challenging": 8, "interrupted": 1, "dodged": 2, "partial": 1}
    assert d.rate == 0.311
    assert d.score == 68.9  # 100 * (1 - 0.311), from the published rate
    assert (row.questions_received, row.challenging_questions, row.questions_dodged) == (9, 8, 2)
    assert "2x vyhýbanie sa otázke" in row.badges
    assert len(d.evidence) == 3


def test_responsiveness_none_without_challenging_questions() -> None:
    report = _report(["A"])
    report.question_audit = [_q(1, None, QuestionKind.OPEN)]
    verdict = score_report(report, transcript="A [00:01]: " + "slovo " * 600 + "\n")
    assert _discipline(_row(verdict, "A"), "responsiveness").score is None


def test_responsiveness_none_when_audit_is_empty() -> None:
    # Task 8: an empty audit means "no data" (LLM failed), not perfect answers.
    verdict = score_report(
        _report(["A"]), transcript="A [00:01]: " + "slovo " * 600 + "\n"
    )
    d = _discipline(_row(verdict, "A"), "responsiveness")
    assert d.score is None and d.rate is None


def test_claim_funnel_is_copied_and_refreshed() -> None:
    report = _report(["A"], [_fact("A", Verdict.TRUE)])
    report.claim_funnel = [ClaimFunnel(speaker="A", extracted=5, selected_for_check=3, final_facts=1)]
    verdict = score_report(report, transcript="A [00:01]: " + "slovo " * 600 + "\n")
    row = _row(verdict, "A")
    assert (row.claims_extracted, row.claims_selected, row.checked_claims) == (5, 3, 1)
    assert report.claim_funnel[0].checked == 1
    assert "z 3 vybraných (extrahovaných 5)" in _discipline(row, "truthfulness").detail


def test_moderator_time_distribution_uses_names_turns_and_start() -> None:
    report = _report(["Erik Tomáš"])
    report.speaker_map = SpeakerMap(status="ok", source="cli", debate_start="01:00")
    transcript = (
        "Záznam [00:10]: " + "x " * 50 + "\n"
        "Moderátor [01:00]: Otázka?\n"
        "Erik Tomáš [01:05]: " + "y " * 20 + "\n"
        "Moderátor [01:30]: Ďalej?\n"
    )
    apply_deterministic_moderator_metrics(report, transcript)
    dist = {t.speaker: t for t in report.moderator_audit.equal_time_distribution}
    assert "Záznam" not in dist
    assert dist["Moderátor"].turns == 2
    assert dist["Erik Tomáš"].approximate_share_percent == 90.9


def test_no_transcript_suppresses_winner_like_degraded() -> None:
    speakers = [
        SpeakerTactics(speaker="A", manipulation=["m"], civility={"score": 9}),
        SpeakerTactics(speaker="B", civility={"score": 1}),
    ]
    verdict = score_report(_report(speakers))
    assert verdict.scoring_status == "no_transcript"
    assert verdict.winner == ""
    assert "manipulation" not in verdict.discipline_winners
    assert "responsiveness" not in verdict.discipline_winners
    assert verdict.discipline_winners["civility"] == "A"
    assert any("could not be counted" in n for n in verdict.method_notes)
    assert _discipline(_row(verdict, "A"), "manipulation").score is None


def test_red_flag_suppressed_when_speaker_map_not_ok() -> None:
    report = _report([SpeakerTactics(speaker="A", manipulation=["m1", "m2"]), "B"])
    report.speaker_map = SpeakerMap(status="partial", source="heuristic")
    transcript = "A [00:01]: " + "slovo " * 600 + "\nB [00:05]: " + "slovo " * 600 + "\n"
    verdict = score_report(report, transcript=transcript)
    assert verdict.scoring_status == "degraded"
    assert verdict.red_flag_speaker == ""


def test_red_flag_kept_when_degraded_only_by_unmatched_words() -> None:
    report = _report([SpeakerTactics(speaker="A", manipulation=["m1", "m2"]), "B", "C"])
    report.speaker_map = SpeakerMap(status="ok", source="cli")
    transcript = "A [00:01]: " + "slovo " * 600 + "\nB [00:05]: " + "slovo " * 600 + "\n"
    verdict = score_report(report, transcript=transcript)
    assert verdict.scoring_status == "degraded"
    assert verdict.red_flag_speaker == "A"


def test_published_scores_match_published_rates() -> None:
    # 8 points / 700 words -> unrounded rate 11.428571...
    transcript = "A [00:01]: " + "slovo " * 700 + "\n"
    verdict = score_report(
        _report([SpeakerTactics(speaker="A", manipulation=["m"])]), transcript=transcript
    )
    d = _discipline(_row(verdict, "A"), "manipulation")
    assert d.rate_per_1000 == 11.43
    assert d.score == round(100.0 - d.rate_per_1000, 2) == 88.57

    report = _report(["A"])
    report.question_audit = [_q(1, QuestionOutcome.DODGED), _q(2, QuestionOutcome.ANSWERED)]
    verdict = score_report(report, transcript="A [00:01]: " + "slovo " * 600 + "\n")
    r = _discipline(_row(verdict, "A"), "responsiveness")
    assert r.rate == 0.325
    assert r.score == round(100.0 * (1.0 - r.rate), 2) == 67.5


def test_word_floor_applies_below_500_words_only() -> None:
    speakers = [SpeakerTactics(speaker="A", manipulation=["m"]), "B"]
    transcript = "A [00:01]: " + "slovo " * 100 + "\nB [00:05]: " + "slovo " * 600 + "\n"
    verdict = score_report(_report(speakers), transcript=transcript)
    row = _row(verdict, "A")
    assert row.words == 100
    assert _discipline(row, "manipulation").normalization_words == 500
    assert any("Word floor 500" in n for n in verdict.method_notes)

    transcript = "A [00:01]: " + "slovo " * 501 + "\nB [00:05]: " + "slovo " * 600 + "\n"
    verdict = score_report(_report(speakers), transcript=transcript)
    assert _discipline(_row(verdict, "A"), "manipulation").normalization_words == 501
    assert not any("Word floor" in n for n in verdict.method_notes)


def test_empty_scoreboard_has_no_winner() -> None:
    verdict = score_report(_report([]), transcript="A [00:01]: " + "slovo " * 600 + "\n")
    assert verdict.scoreboard == []
    assert verdict.winner == ""
    assert verdict.margin is None


def test_dodge_prior_mass_uses_decimal_comma() -> None:
    report = _report(["A"])
    report.question_audit = [_q(1, QuestionOutcome.DODGED), _q(2, QuestionOutcome.ANSWERED)]
    verdict = score_report(report, transcript="A [00:01]: " + "slovo " * 600 + "\n")
    detail = _discipline(_row(verdict, "A"), "responsiveness").detail
    assert "+ 0,3)" in detail and "+ 0.3)" not in detail
