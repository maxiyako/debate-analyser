"""Offline tests for guarded transcript correction (src/correction.py)."""

from __future__ import annotations

import pytest

from src.correction import apply_corrections, check_edit, number_tokens
from src.report_models import EditType, TranscriptEdit
from src.transcript_lines import format_lines, parse_lines

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
    # Both cuts shaped every segment, so every segment carries both edit ids.
    assert [ln.edit_ids for ln in out] == [[0, 1], [0, 1], [0, 1]]
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


# --- fix round 1 -----------------------------------------------------------


@pytest.mark.parametrize(
    "before,after",
    [
        ("trinásť", "štrnásť"),
        ("dvoch", "troch"),
        ("dvaja", "traja"),
        ("tristo", "štyristo"),
        ("pätnásť", "šestnásť"),
        ("štyria", "piati"),
        ("dvadsiatich", "tridsiatich"),
    ],
)
def test_numeral_stems_any_inflection_rejected(before: str, after: str) -> None:
    line = f"Prišlo {before} ľudí."
    assert check_edit(e(S, before, after), line, NAMES, SPEAKERS) == "number"


@pytest.mark.parametrize(
    "edit,line",
    [
        (e(S, "čí", "či"), "Keď počítame, čí áno."),
        (e(S, "ministerka", "ministérka"), "Nová ministerka prišla."),
        (e(S, "dvere", "dvére"), "Zavrel dvere."),
        (e(S, "trh", "trhu"), "Na trh prišli."),
        (e(S, "tisícročie", "tisícročia"), "Za tisícročie sa to nezmení."),
        (e(S, "desiata", "desiatá"), "Bola desiata hodina."),
    ],
)
def test_ordinary_words_near_numeral_stems_still_apply(edit: TranscriptEdit, line: str) -> None:
    assert check_edit(edit, line, NAMES, SPEAKERS) == ""


def test_whole_word_replacement_hits_standalone_word() -> None:
    lines = parse_lines("Erik Tomáš [00:12]: Keď počítame, čí áno.\n")
    edits = [TranscriptEdit(line_no=0, type=S, before="čí", after="či")]
    out, log = apply_corrections(lines, edits, NAMES, SPEAKERS)
    assert out[0].text == "Keď počítame, či áno."
    assert log[0].applied


def test_before_only_inside_another_word_is_not_found() -> None:
    assert check_edit(e(S, "čí", "či"), "Keď počítame.", NAMES, SPEAKERS) == "not_found"
    assert check_edit(e(S, "ľud", "ľuď"), "pre ľudí", NAMES, SPEAKERS) == "not_found"


def test_whole_word_punctuation_edge_needle() -> None:
    lines = parse_lines("Erik Tomáš [00:12]: Ako vidíte vy, pán Tomáš.\n")
    edits = [TranscriptEdit(line_no=0, type=P, before="vy, pán", after="vy pán")]
    out, log = apply_corrections(lines, edits, NAMES, SPEAKERS)
    assert out[0].text == "Ako vidíte vy pán Tomáš."


def test_split_anchor_must_be_whole_word() -> None:
    edit = TranscriptEdit(line_no=0, type=EditType.SPLIT_TURN, before="deň", new_speaker="Erik Tomáš")
    assert check_edit(edit, "Dobrý večer, deň, vitajte.", NAMES, SPEAKERS) == ""
    assert check_edit(edit, "Dobrý predeň vitajte.", NAMES, SPEAKERS) == "split"


def test_split_works_when_line_numbers_do_not_start_at_zero() -> None:
    lines = parse_lines(
        "Moderátor [01:58]: Dobrý deň. Ďakujem za pozvanie.\n"
        "Erik Tomáš [02:10]: Nech sa páči.\n"
    )
    for k, ln in enumerate(lines):
        ln.no = 100 + k
    edits = [
        TranscriptEdit(line_no=0, type=EditType.SPLIT_TURN, before="Ďakujem", new_speaker="Erik Tomáš")
    ]
    out, log = apply_corrections(lines, edits, NAMES, SPEAKERS)
    assert log[0].applied
    assert [ln.speaker for ln in out] == ["Moderátor", "Erik Tomáš", "Erik Tomáš"]
    assert out[1].text == "Ďakujem za pozvanie."


def test_proper_noun_cannot_swap_allowed_name_for_another() -> None:
    names = {"Igor Matovič", "Milan Majerský", "Fico"}
    assert check_edit(e(PN, "Matovič", "Majerský"), "Povedal Matovič.", names, SPEAKERS) == "proper_noun"


def test_proper_noun_legit_fix_still_works() -> None:
    names = {"Robert Fico"}
    assert check_edit(e(PN, "Fica", "Fico"), "Povedal Fica.", names, SPEAKERS) == ""


# --- fix round 2 -----------------------------------------------------------


@pytest.mark.parametrize(
    "before,after",
    [
        ("dvadsaťpäť", "dvadsaťšesť"),
        ("tridsaťdva", "tridsaťtri"),
        ("stopäťdesiat", "stošesťdesiat"),
        ("milióne", "miliónu"),
        ("milióne", "miliónami"),
        ("tisícami", "tisíci"),
        ("tretine", "štvrtine"),
        ("polovici", "tretine"),
        ("dvestom", "tristom"),
        ("stom", "dvestom"),
        ("štyrnásť", "šestnásť"),
        ("šesťnásť", "sedemnásť"),
        ("jedni", "dvaja"),
        ("jedny", "dve"),
        ("jedni", "jedny"),
    ],
)
def test_numeralish_layer_rejects_quantity_changes(before: str, after: str) -> None:
    line = f"Ide o {before} ľudí."
    assert check_edit(e(S, before, after), line, NAMES, SPEAKERS) == "number"


def test_digit_tokens_compared_in_order() -> None:
    edit = e(S, "zo 40 na 135", "zo 135 na 40")
    assert check_edit(edit, "Išlo to zo 40 na 135 eur.", NAMES, SPEAKERS) == "number"


def test_digit_and_numeral_word_order_matters() -> None:
    edit = e(S, "40 tri", "tri 40")
    assert check_edit(edit, "Bolo 40 tri ľudí.", NAMES, SPEAKERS) == "number"


@pytest.mark.parametrize(
    "word",
    [
        "dvere", "trh", "trojka", "tretí", "desiata", "stôl", "stopa", "piatok",
        "polovičný", "trochu", "naštartovať", "nástroj", "prestížny", "rozhodnutie",
        "vláda", "ministerka", "premiér", "tisícročie", "trinásobok", "počítame",
        "stojí", "storočie", "dvojka", "tretia", "štvrtok",
    ],
)
def test_ordinary_words_are_not_numeralish(word: str) -> None:
    assert number_tokens(f"Povedal {word} včera.") == []


# --- fix round 3 -----------------------------------------------------------


@pytest.mark.parametrize(
    "before,after",
    [
        ("stopäť", "stošesť"),
        ("stošesť", "stodeväť"),
        ("stosedem", "stoosem"),
        ("stojeden", "stosedem"),
        ("stodesať", "stopäť"),
        ("dvestopäť", "dvestošesť"),
        ("dvoje", "troje"),
        ("troje", "štvoro"),
        ("štvoro", "pätoro"),
        ("dvojica", "trojica"),
    ],
)
def test_sto_compounds_and_collectives_rejected(before: str, after: str) -> None:
    line = f"Bolo ich {before} naraz."
    assert check_edit(e(S, before, after), line, NAMES, SPEAKERS) == "number"


@pytest.mark.parametrize(
    "word",
    [
        "stolica", "stojan", "stoka", "stonoha", "štvorec", "stojí", "stodola", "stovka",
        "trojuholník", "dvojka", "trojka", "stopár", "stoličku", "štvorcový", "dvojitý",
    ],
)
def test_more_ordinary_words_are_not_numeralish(word: str) -> None:
    assert number_tokens(f"Povedal {word} včera.") == []


from src.correction import (  # noqa: E402
    CorrectionBatch,
    correct_transcript_edits,
    propose_corrections,
)


def _many(n: int) -> str:
    return "".join(f"Moderátor [{i // 60:02d}:{i % 60:02d}]: riadok {i} pre ľudi\n" for i in range(n))


def test_propose_chunks_and_drops_out_of_range_edits() -> None:
    lines = parse_lines(_many(85))
    prompts: list[str] = []

    def fake(prompt: str) -> CorrectionBatch:
        prompts.append(prompt)
        n = len(prompts) - 1
        return CorrectionBatch(edits=[
            TranscriptEdit(line_no=n * 40, type=S, before="ľudi", after="ľudí"),
            TranscriptEdit(line_no=999, type=S, before="x", after="y"),
        ])

    edits, notes = propose_corrections(lines, llm=fake, allowed_names=set(), speaker_names={"Moderátor"})
    assert len(prompts) == 3
    assert [x.line_no for x in edits] == [0, 40, 80]
    assert "Edit only lines #40..#79" in prompts[1]
    assert "#35 Moderátor" in prompts[1]  # 5 context lines before the chunk
    assert notes == [
        "Correction chunk #0-#39: 1 edit(s) outside the chunk ignored",
        "Correction chunk #40-#79: 1 edit(s) outside the chunk ignored",
        "Correction chunk #80-#84: 1 edit(s) outside the chunk ignored",
    ]


def test_failed_chunk_is_noted_and_skipped() -> None:
    lines = parse_lines(_many(85))
    calls = {"n": 0}

    def fake(prompt: str) -> CorrectionBatch:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("timeout")
        return CorrectionBatch(edits=[])

    _, notes = propose_corrections(lines, llm=fake, allowed_names=set(), speaker_names=set())
    assert calls["n"] == 3
    assert len(notes) == 1 and "#40-#79" in notes[0]


def test_correct_transcript_edits_end_to_end() -> None:
    named = (
        "Moderátor [00:10]: Vitajte pre ľudi.\n"
        "Erik Tomáš [00:12]: Strana HLas presadila zo 40 na 135 eur.\n"
    )

    def fake(prompt: str) -> CorrectionBatch:
        return CorrectionBatch(edits=[
            TranscriptEdit(line_no=0, type=S, before="ľudi", after="ľudí"),
            TranscriptEdit(line_no=1, type=PN, before="HLas", after="HLAS"),
            TranscriptEdit(line_no=1, type=S, before="40", after="140"),
        ])

    out = correct_transcript_edits(
        named, llm=fake, allowed_names={"HLAS"}, speaker_names={"Moderátor", "Erik Tomáš"}
    )
    assert out.text == (
        "Moderátor [00:10]: Vitajte pre ľudí.\n"
        "Erik Tomáš [00:12]: Strana HLAS presadila zo 40 na 135 eur.\n"
    )
    assert out.quality.applied == 2
    assert out.quality.rejected_by_rule == {"number": 1}
    assert [ln.raw_no for ln in out.lines] == [0, 1]
    assert out.lines[1].edit_ids == [1]


def test_chunk_edits_use_global_indices_with_split_in_later_chunk() -> None:
    # A split in chunk 2 shifts later lines; the edit must still hit the right line.
    lines = parse_lines(_many(45))
    lines[41].text = "pravda Marián Viskupič: dobre"
    calls = {"n": 0}

    def fake(prompt: str) -> CorrectionBatch:
        calls["n"] += 1
        if calls["n"] == 1:
            return CorrectionBatch(edits=[])
        return CorrectionBatch(edits=[
            TranscriptEdit(line_no=41, type=EditType.SPLIT_TURN, before="Marián",
                           new_speaker="Marián Viskupič"),
            TranscriptEdit(line_no=43, type=S, before="ľudi", after="ľudí"),
        ])

    out = correct_transcript_edits(
        format_lines(lines), llm=fake, allowed_names=set(),
        speaker_names={"Moderátor", "Marián Viskupič"},
    )
    assert out.quality.applied == 2
    assert out.lines[41].text == "pravda" and out.lines[42].speaker == "Marián Viskupič"
    assert out.lines[44].text == "riadok 43 pre ľudí"
