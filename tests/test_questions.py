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
