"""Auditable transcript correction: the LLM proposes small edits, code decides.

The raw ASR transcript stays the source of truth. An edit is applied only if
it cannot change what was said: numbers, negation, and content words are
invariant; only spelling, word boundaries, known proper nouns, punctuation,
and turn splits are allowed. Every decision is logged.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import replace

from src.report_models import EditResult, EditType, TranscriptEdit
from src.transcript_lines import Line

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
NUMERAL_WORDS = frozenset({
    "jeden", "jedna", "jedno", "dva", "dve", "tri", "štyri", "päť", "šesť",
    "sedem", "osem", "deväť", "desať", "jedenásť", "dvanásť", "dvadsať",
    "tridsať", "štyridsať", "päťdesiat", "šesťdesiat", "sedemdesiat",
    "osemdesiat", "deväťdesiat", "sto", "dvesto", "tisíc", "milión",
    "milióny", "miliónov", "miliarda", "miliardy", "miliárd", "polovica",
    "tretina", "štvrtina",
})
NEGATION_WORDS = frozenset({
    "nie", "nikdy", "nič", "ani", "nikto", "nikde", "nijako",
    "žiaden", "žiadny", "žiadna", "žiadne", "žiadnu", "žiadneho",
})
_NEG_ALIASES = {"neni": "nie"}
_ORTHO = (EditType.SPELLING, EditType.WORD_BOUNDARY)
_SIMILARITY_MIN = 0.6


def _words(s: str) -> list[str]:
    return [_NEG_ALIASES.get(w, w) for w in _WORD.findall((s or "").casefold())]


def number_tokens(s: str) -> list[str]:
    return sorted(_NUM.findall(s or "")) + sorted(w for w in _words(s) if w in NUMERAL_WORDS)


def negation_tokens(s: str) -> set[str]:
    return {w for w in _words(s) if w in NEGATION_WORDS}


def _negation_flip(before: str, after: str) -> bool:
    if negation_tokens(before) != negation_tokens(after):
        return True
    wb, wa = set(_words(before)), set(_words(after))
    return any("ne" + w in wa for w in wb) or any("ne" + w in wb for w in wa)


def _similar(before: str, after: str) -> bool:
    b, a = before.casefold(), after.casefold()
    if len(b) == len(a) and sum(x != y for x, y in zip(b, a)) <= 1:
        return True
    return difflib.SequenceMatcher(None, b, a).ratio() >= _SIMILARITY_MIN


def check_edit(
    edit: TranscriptEdit, line_text: str, allowed_names: set[str], speaker_names: set[str]
) -> str:
    """'' when the edit is safe to apply, else the name of the guard it breaks."""
    if edit.type == EditType.SPLIT_TURN:
        pos = line_text.find(edit.before) if edit.before else -1
        if edit.new_speaker not in speaker_names or pos <= 0:
            return "split"
        return ""
    if not edit.before or edit.before not in line_text:
        return "not_found"
    if edit.before == edit.after:
        return "noop"
    if number_tokens(edit.before) != number_tokens(edit.after):
        return "number"
    if _negation_flip(edit.before, edit.after):
        return "negation"
    if edit.type == EditType.PUNCTUATION:
        return "" if _words(edit.before) == _words(edit.after) else "punctuation"
    if abs(len(edit.before.split()) - len(edit.after.split())) > 1:
        return "word_count"
    if edit.type in _ORTHO and not _similar(edit.before, edit.after):
        return "similarity"
    if edit.type == EditType.PROPER_NOUN:
        allowed = {w for name in allowed_names for w in _words(name)}
        before_words = set(_words(edit.before))
        if any(w not in allowed for w in _words(edit.after) if w not in before_words):
            return "proper_noun"
    return ""


def apply_corrections(
    lines: list[Line],
    edits: list[TranscriptEdit],
    allowed_names: set[str],
    speaker_names: set[str],
) -> tuple[list[Line], list[EditResult]]:
    """Apply guarded edits; returns new lines (input untouched) and the full log."""
    work = [replace(ln, edit_ids=list(ln.edit_ids)) for ln in lines]
    log: list[EditResult] = []
    splits: dict[int, list[int]] = {}
    for edit in edits:
        eid = len(log)
        if not 0 <= edit.line_no < len(work):
            log.append(EditResult(id=eid, edit=edit, applied=False, rule="line"))
            continue
        ln = work[edit.line_no]
        rule = check_edit(edit, ln.text, allowed_names, speaker_names)
        log.append(EditResult(id=eid, edit=edit, applied=not rule, rule=rule))
        if rule:
            continue
        ln.edit_ids.append(eid)
        if edit.type == EditType.SPLIT_TURN:
            splits.setdefault(ln.no, []).append(eid)
        else:
            ln.text = ln.text.replace(edit.before, edit.after, 1)

    out: list[Line] = []
    for ln in work:
        cuts: list[tuple[int, int]] = []
        for eid in splits.get(ln.no, []):
            pos = ln.text.find(log[eid].edit.before)
            if pos <= 0 or any(p == pos for p, _ in cuts):
                log[eid] = log[eid].model_copy(update={"applied": False, "rule": "split"})
                ln.edit_ids.remove(eid)
            else:
                cuts.append((pos, eid))
        if not cuts:
            out.append(ln)
            continue
        cuts.sort()
        split_ids = {eid for _, eid in cuts}
        base_ids = [i for i in ln.edit_ids if i not in split_ids]
        bounds = [0, *(p for p, _ in cuts), len(ln.text)]
        speakers = [ln.speaker, *(log[eid].edit.new_speaker for _, eid in cuts)]
        own_ids: list[list[int]] = [[], *([eid] for _, eid in cuts)]
        for j, speaker in enumerate(speakers):
            out.append(
                replace(
                    ln,
                    speaker=speaker,
                    text=ln.text[bounds[j] : bounds[j + 1]].strip(),
                    edit_ids=[*base_ids, *own_ids[j]],
                )
            )
    for i, ln in enumerate(out):
        ln.no = i
    return out, log
