"""Offline tests for guarded transcript correction (src/correction.py)."""

from __future__ import annotations

import pytest

from src.correction import apply_corrections, check_edit
from src.report_models import EditType, TranscriptEdit
from src.transcript_lines import parse_lines

NAMES = {"Erik Tomáš", "Marián Viskupič", "Milan Majerský", "HLAS"}
SPEAKERS = {"Moderátor", "Erik Tomáš", "Marián Viskupič"}
S, WB, PN, P = EditType.SPELLING, EditType.WORD_BOUNDARY, EditType.PROPER_NOUN, EditType.PUNCTUATION


def e(kind: EditType, before: str, after: str = "") -> TranscriptEdit:
    return TranscriptEdit(line_no=0, type=kind, before=before, after=after)


@pytest.mark.parametrize(
    "edit,line,rule",
    [
        (e(S, "ľudi", "ľudí"), "pre ľudi", ""),
        (e(WB, "kpointe", "k pointe"), "príde aj kpointe", ""),
        (e(WB, "napl nela", "naplnila"), "HLas napl nela", ""),
        (e(S, "čí", "či"), "čí áno", ""),
        (e(PN, "HLas", "HLAS"), "strana HLas", ""),
        (e(P, "áno ale", "áno, ale"), "áno ale", ""),
        (e(S, "40", "140"), "zo 40 na 135", "number"),
        (e(S, "tri", "štyri"), "tri roky", "number"),
        (e(S, "podporili", "nepodporili"), "oni podporili", "negation"),
        (e(S, "neni", "je"), "to neni pravda", "negation"),
        (e(S, "Armádsky", "Pán Majerský"), "Armádsky, povedal", "similarity"),
        (e(PN, "Armádsky", "Pán Majerský"), "Armádsky, povedal", "proper_noun"),
        (e(S, "Ako to vidíte vy,", ""), "Ako to vidíte vy, pán", "word_count"),
        (e(P, "spoločnost", "spoločnosť."), "spoločnost", "punctuation"),
        (e(S, "xyz", "xy"), "abc", "not_found"),
    ],
)
def test_check_edit(edit: TranscriptEdit, line: str, rule: str) -> None:
    assert check_edit(edit, line, NAMES, SPEAKERS) == rule


def test_apply_records_log_and_edit_ids() -> None:
    lines = parse_lines("Erik Tomáš [00:12]: Strana HLas presadila zo 40 na 135 eur pre ľudi.\n")
    edits = [
        TranscriptEdit(line_no=0, type=PN, before="HLas", after="HLAS"),
        TranscriptEdit(line_no=0, type=S, before="40", after="140"),
        TranscriptEdit(line_no=0, type=S, before="ľudi", after="ľudí"),
        TranscriptEdit(line_no=7, type=S, before="a", after="b"),
    ]
    out, log = apply_corrections(lines, edits, NAMES, SPEAKERS)
    assert out[0].text == "Strana HLAS presadila zo 40 na 135 eur pre ľudí."
    assert [(r.id, r.applied, r.rule) for r in log] == [
        (0, True, ""), (1, False, "number"), (2, True, ""), (3, False, "line"),
    ]
    assert out[0].edit_ids == [0, 2]
    assert lines[0].text.startswith("Strana HLas")  # input is not mutated


def test_split_turn_preserves_text_and_alternates_speakers() -> None:
    text = (
        "Moderátor [01:58]: Vítam Erika Tomáša. Ďakujem za pozvanie a všetkým "
        "prajem peknú nedeľu. A rovnako vítam Mariána Viskupiča.\n"
    )
    lines = parse_lines(text)
    edits = [
        TranscriptEdit(line_no=0, type=EditType.SPLIT_TURN, before="A rovnako", new_speaker="Moderátor"),
        TranscriptEdit(line_no=0, type=EditType.SPLIT_TURN, before="Ďakujem za pozvanie", new_speaker="Erik Tomáš"),
    ]
    out, log = apply_corrections(lines, edits, NAMES, SPEAKERS)
    assert [ln.speaker for ln in out] == ["Moderátor", "Erik Tomáš", "Moderátor"]
    assert " ".join(ln.text for ln in out) == lines[0].text
    assert all(ln.raw_no == 0 and ln.ts == "01:58" for ln in out)
    assert [ln.no for ln in out] == [0, 1, 2]
    assert out[1].edit_ids == [1] and out[0].edit_ids == []
    assert all(r.applied for r in log)


def test_split_rejected_for_unknown_speaker_or_line_start() -> None:
    lines = parse_lines("Moderátor [01:58]: Dobrý deň, vitajte.\n")
    edits = [
        TranscriptEdit(line_no=0, type=EditType.SPLIT_TURN, before="vitajte", new_speaker="Robert Fico"),
        TranscriptEdit(line_no=0, type=EditType.SPLIT_TURN, before="Dobrý deň", new_speaker="Erik Tomáš"),
    ]
    out, log = apply_corrections(lines, edits, NAMES, SPEAKERS)
    assert len(out) == 1
    assert [r.rule for r in log] == ["split", "split"]
