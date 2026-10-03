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
