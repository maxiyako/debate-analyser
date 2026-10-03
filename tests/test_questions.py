"""Offline tests for the question audit (src/questions.py)."""

from __future__ import annotations

from src.questions import extract_question_candidates
from src.transcript_lines import parse_lines

GUESTS = ["Erik Tomáš", "Marián Viskupič"]
LINES = parse_lines(
    "Moderátor [00:10]: Dobrý deň.\n"
    "Moderátor [00:12]: Pán Viskupič, zvýšite dane, áno alebo nie?\n"
    "Erik Tomáš [00:15]: (cez seba) To nie je pravda.\n"
    "Marián Viskupič [00:18]: "
    + "Dane nezvýšime, to vám garantujem úplne jasne a zreteľne " * 2 + "\n"
    "Moderátor [00:30]: Áno?\n"
    "Marián Viskupič [00:31]: A ešte jedna vec k tomu.\n"
    "Moderátor [00:40]: A vy, koľko to bude stáť?\n"
    "Erik Tomáš [00:45]: Uvidíme.\n"
    "Moderátor [00:50]: Ďalšia téma, prečo ste hlasovali proti?\n"
    "Marián Viskupič [00:55]: " + "slovo " * 20 + "\n"
)


def test_candidates_addressee_and_answer_span() -> None:
    items = extract_question_candidates(LINES, "Moderátor", GUESTS)
    assert [(i.id, i.timestamp, i.addressee) for i in items] == [
        (1, "00:12", "Marián Viskupič"),
        (2, "00:40", "Erik Tomáš"),
        (3, "00:50", "Marián Viskupič"),
    ]
    # The short "Áno?" does not end the answer span.
    assert items[0].answer_words == 24
    assert items[1].answer_words == 1
    assert items[0].question == "Pán Viskupič, zvýšite dane, áno alebo nie?"


def test_candidates_respect_debate_start() -> None:
    items = extract_question_candidates(LINES, "Moderátor", GUESTS, start_s=45)
    assert [i.timestamp for i in items] == ["00:50"]


def test_mixed_crosstalk_line_counts_real_speech() -> None:
    lines = parse_lines(
        "Moderátor [00:10]: Pán Viskupič, zvýšite dane, áno alebo nie?\n"
        "Marián Viskupič [00:15]: (hovorenie cez seba) Nie, nezvyšíme vôbec nič.\n"
        "Marián Viskupič [00:20]: (hovorenie cez seba)\n"
    )
    items = extract_question_candidates(lines, "Moderátor", GUESTS)
    assert [i.answer_words for i in items] == [4]


def test_fallback_addressee_stops_at_next_moderator_question() -> None:
    lines = parse_lines(
        "Moderátor [00:10]: Koľko to bude stáť podľa vás?\n"
        "Moderátor [00:12]: Pán Viskupič, zvýšite dane, áno alebo nie?\n"
        "Marián Viskupič [00:15]: Nie.\n"
    )
    # Consecutive lines merge into one turn addressed by the vocative.
    assert [(i.timestamp, i.addressee) for i in
            extract_question_candidates(lines, "Moderátor", GUESTS)] == [
        ("00:10", "Marián Viskupič")
    ]
    lines = parse_lines(
        "Moderátor [00:10]: Koľko to bude stáť podľa vás?\n"
        "Erik Tomáš [00:11]: Neviem.\n"
        "Moderátor [00:12]: Pán Viskupič, zvýšite dane, áno alebo nie?\n"
        "Marián Viskupič [00:15]: Nie.\n"
    )
    assert [(i.timestamp, i.addressee) for i in
            extract_question_candidates(lines, "Moderátor", GUESTS)] == [
        ("00:10", "Erik Tomáš"),
        ("00:12", "Marián Viskupič"),
    ]


def test_question_without_guest_before_next_question_is_skipped() -> None:
    lines = parse_lines(
        "Moderátor [00:10]: Koľko to bude stáť podľa vás?\n"
        "Moderátor [00:11]: Dobre.\n"
        "Erik Tomáš [00:20]: Neviem.\n"
    )
    # Same turn here, so the guest after the turn answers.
    assert len(extract_question_candidates(lines, "Moderátor", GUESTS)) == 1
    lines = parse_lines(
        "Moderátor [00:10]: Koľko to bude stáť podľa vás?\n"
        "Záznam [00:11]: Reklama.\n"
        "Moderátor [00:12]: Koľko to bude stáť podľa vás?\n"
        "Erik Tomáš [00:20]: Neviem.\n"
    )
    items = extract_question_candidates(lines, "Moderátor", GUESTS)
    assert [(i.timestamp, i.addressee) for i in items] == [("00:12", "Erik Tomáš")]


def test_consecutive_moderator_question_lines_form_one_candidate() -> None:
    lines = parse_lines(
        "Moderátor [00:10]: Koľko to bude stáť podľa vás?\n"
        "Moderátor [00:12]: Pán Viskupič, zvýšite dane, áno alebo nie?\n"
        "Marián Viskupič [00:15]: " + "slovo " * 20 + "\n"
    )
    items = extract_question_candidates(lines, "Moderátor", GUESTS)
    assert len(items) == 1
    assert items[0].id == 1
    assert items[0].addressee == "Marián Viskupič"
    assert items[0].answer_words == 20
    assert items[0].question == "Pán Viskupič, zvýšite dane, áno alebo nie?"


def test_turn_without_question_mark_yields_no_candidate() -> None:
    lines = parse_lines(
        "Moderátor [00:10]: Povedzte nám, ako zvýšite dane.\n"
        "Erik Tomáš [00:15]: Neviem.\n"
    )
    assert extract_question_candidates(lines, "Moderátor", GUESTS) == []


def test_question_under_four_words_yields_no_candidate() -> None:
    lines = parse_lines(
        "Moderátor [00:10]: Dane áno alebo?\n"
        "Erik Tomáš [00:15]: Neviem.\n"
    )
    assert extract_question_candidates(lines, "Moderátor", GUESTS) == []


from src.questions import (  # noqa: E402
    ClassifiedQuestion,
    QuestionClassification,
    classify_questions,
    question_balance_finding,
    question_counts,
    render_dodges,
    run_question_audit,
)
from src.report_models import (  # noqa: E402
    QuestionItem,
    QuestionKind,
    QuestionOutcome,
    SpeakerMap,
    SpeakerMapEntry,
    SpeakerRole,
)

CH, OPEN = QuestionKind.CHALLENGING, QuestionKind.OPEN
DODGED, ANSWERED, INTERRUPTED = (
    QuestionOutcome.DODGED,
    QuestionOutcome.ANSWERED,
    QuestionOutcome.INTERRUPTED,
)


def _all_dodged(prompt: str) -> QuestionClassification:
    return QuestionClassification(items=[
        ClassifiedQuestion(id=1, kind=CH, outcome=DODGED, reason="odbočil",
                           evidence=["[00:18] Dane nezvýšime, to vám garantujem"]),
        ClassifiedQuestion(id=2, kind=CH, outcome=DODGED, evidence=["Uvidíme"]),
        ClassifiedQuestion(id=3, kind=CH, outcome=DODGED,
                           evidence=["Toto sa v odpovedi nikde nenachádza vôbec"]),
    ])


def test_classification_is_validated_against_the_answer() -> None:
    items = extract_question_candidates(LINES, "Moderátor", GUESTS)
    notes = classify_questions(items, llm=_all_dodged)
    assert [i.outcome for i in items] == [DODGED, INTERRUPTED, ANSWERED]
    assert items[0].evidence[0].startswith("[00:12] otázka:")
    assert "garantujem" in items[0].evidence[1]
    assert len(notes) == 1 and "#3" in notes[0]
    assert question_counts(items) == {
        "received": 3, "challenging": 3, "interrupted": 1, "dodged": 1, "partial": 0,
    }
    assert render_dodges(items) == [
        "[00:12] Pán Viskupič, zvýšite dane, áno alebo nie? → dodged: odbočil"
    ]


def test_non_challenging_and_unclassified_are_not_counted() -> None:
    items = extract_question_candidates(LINES, "Moderátor", GUESTS)

    def fake(prompt: str) -> QuestionClassification:
        return QuestionClassification(items=[
            ClassifiedQuestion(id=1, kind=OPEN, outcome=DODGED, evidence=["x"]),
        ])

    notes = classify_questions(items, llm=fake)
    assert items[0].kind == OPEN and items[0].outcome is None
    assert items[1].kind == OPEN and items[2].kind == OPEN
    assert len(notes) == 2
    assert question_counts(items)["challenging"] == 0


def test_question_balance_finding() -> None:
    items = [
        QuestionItem(id=i, timestamp="00:00", addressee="A" if i < 6 else "B", question="q", kind=CH)
        for i in range(7)
    ]
    finding = question_balance_finding(items, ["A", "B"])
    assert finding is not None and "A: 6" in finding and "B: 1" in finding
    assert question_balance_finding(items[:2] + items[6:], ["A", "B"]) is None


def _smap() -> SpeakerMap:
    return SpeakerMap(status="ok", entries=[
        SpeakerMapEntry(label="a", name="Moderátor", role=SpeakerRole.MODERATOR),
        SpeakerMapEntry(label="b", name="Erik Tomáš", role=SpeakerRole.GUEST),
        SpeakerMapEntry(label="c", name="Marián Viskupič", role=SpeakerRole.GUEST),
    ])


def test_run_question_audit_and_failure() -> None:
    items, notes = run_question_audit(LINES, _smap(), llm=_all_dodged)
    assert len(items) == 3 and len(notes) == 1

    def boom(prompt: str) -> QuestionClassification:
        raise RuntimeError("vertex down")

    items, notes = run_question_audit(LINES, _smap(), llm=boom)
    assert items == [] and "failed" in notes[0]


def test_duplicate_classification_ids_are_ignored() -> None:
    items = extract_question_candidates(LINES, "Moderátor", GUESTS)

    def dup(prompt: str) -> QuestionClassification:
        return QuestionClassification(items=[
            ClassifiedQuestion(id=1, kind=CH, outcome=ANSWERED),
            ClassifiedQuestion(id=1, kind=CH, outcome=DODGED, reason="odbočil",
                               evidence=["Dane nezvýšime, to vám garantujem"]),
            ClassifiedQuestion(id=2, kind=CH, outcome=ANSWERED),
            ClassifiedQuestion(id=3, kind=CH, outcome=ANSWERED),
        ])

    notes = classify_questions(items, llm=dup)
    assert question_counts(items)["dodged"] == 0
    assert items[0].kind == OPEN and items[0].outcome is None
    assert "Question audit: duplicate classification for question 1 ignored" in notes
    assert items[2].outcome == ANSWERED


def test_grounded_partial_is_counted_and_rendered() -> None:
    items = extract_question_candidates(LINES, "Moderátor", GUESTS)

    def partial(prompt: str) -> QuestionClassification:
        return QuestionClassification(items=[
            ClassifiedQuestion(id=1, kind=CH, outcome=QuestionOutcome.PARTIAL, reason="čiastočne",
                               evidence=["Dane nezvýšime, to vám garantujem"]),
        ])

    classify_questions(items, llm=partial)
    assert items[0].outcome == QuestionOutcome.PARTIAL
    assert question_counts(items)["partial"] == 1
    assert render_dodges(items) == [
        "[00:12] Pán Viskupič, zvýšite dane, áno alebo nie? → partial: čiastočne"
    ]
