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
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable

from pydantic import BaseModel, Field

from src.report_models import EditResult, EditType, TranscriptEdit, TranscriptQuality
from src.transcript_lines import Line, format_lines, parse_lines

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
# Cardinal numerals are matched by stem + closed set of inflection endings (fullmatch
# on a whole word), not by prefix, so "dvere", "trh", "tisícročie" or "desiata" are
# NOT numerals. Ordinals (piaty, desiaty) are deliberately out of scope.
_TEENS = "jede|dva|tri|štr|pät|šest|sedem|osem|devät"  # + "násť" (štrnásť, pätnásť, ...)
_UNITS = "päť|šesť|sedem|osem|deväť"
NUMERAL_RE = re.compile(
    r"(?:"
    r"jed(?:en|n(?:a|o|u|ej|é|ého|ému|om|ým|ou|í|y|i|ých|ými))"  # jeden/jedna/jedno/...
    r"|dv(?:a|e|aja|och|om|oma|omi)"  # dva/dve/dvaja/dvoch/dvom/...
    r"|tr(?:i|aja|och|om|oma|omi)"  # tri/traja/troch/trom/...
    r"|štyr(?:i|ia|och|om|mi|oma)"  # štyri/štyria/štyroch/štyrom/...
    r"|päť|piati(?:ch|m|mi)?"
    r"|šesť|šiesti(?:ch|m|mi)?"
    r"|sedem|siedm(?:i|ich|im|imi)"
    r"|osem|[oô]sm(?:i|ich|im|imi)"
    r"|deväť|devia(?:ti|tich|tim|timi)"
    r"|desať|desia(?:ti|tich|tim|timi)"
    rf"|(?:{_TEENS})nás(?:ť|tich|tim|timi|ti)"  # 11-19
    r"|(?:dva|tri|štyri)ds(?:ať|iati|iatich|iatim|iatimi)"  # 20, 30, 40
    rf"|(?:{_UNITS})desiat(?:ich|im|imi)?"  # 50-90
    rf"|(?:(?:dve|tri|štyri|{_UNITS}))?sto(?:ch|m|ma|mi)?"  # sto, dvesto, tristo, štyristo, ...
    r"|tisíc(?:e|ov|om|och|mi)?"
    r"|milión(?:a|y|ov|om|och|mi)?"
    r"|miliard(?:a|y|u|ou|e|ách|ám|ami)|miliárd(?:y|ov|am|ach|ami)?"
    r"|polovic(?:a|e|u|ou)|tretin(?:a|y|u|ou)|štvrtin(?:a|y|u|ou)"
    # collectives: dvoje, troje, štvoro, pätoro, ... and dvojica/trojica/štvorica
    r"|(?:dv|tr)oj(?:e|ich|im|iach)|štvoro|(?:pät|šest|sedm|osm|devät|desat)oro"
    r"|(?:dvoj|troj|štvor)ic(?:a|e|u|ou|iam|iach|ami)"
    r")"
)
# Second, looser "numeral-ish" layer: a word containing one of these long stems is
# treated as a quantity word (catches compounds like "dvadsaťpäť", "stopäťdesiat" and
# inflections missing from NUMERAL_RE). Over-matching is safe (the edit is merely
# rejected); every stem is long enough not to occur in ordinary words.
NUMERALISH_RE = re.compile(
    r"dvadsa[ťt]|dvadsiat|tridsa[ťt]|tridsiat|štyridsa[ťt]|štyridsiat"
    r"|päťdesiat|šesťdesiat|sedemdesiat|osemdesiat|deväťdesiat"
    # teens: stem + "násť"/"nástich"... ("trinásobok", "nástroj" do not match)
    r"|(?:jede|dva|tri|štr|štyr|pät|päť|šest|šesť|šiest|sedem|osem|devät|deväť)nás(?:ť|ti)"
    r"|tisíc(?!roč)|milión|miliar|miliár"
    r"|polovic|tretin|štvrtin|štvrť"
    # 101-109, 110: "sto" + unit ("stopäť", "stojeden"); 2xx-9xx hundreds are caught by
    # the dvest/tristo/... stems above.
    r"|\bsto(?:jeden|dva|tri|štyri|päť|šesť|sedem|osem|deväť|desať)"
    r"|dvest|tristo|štyristo|päťsto|šesťsto|sedemsto|osemsto|deväťsto"
)
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
    """Digit numbers and numeral(-ish) words, casefolded, in order of appearance."""
    text = (s or "").casefold()
    found = [(m.start(), m.group()) for m in _NUM.finditer(text)]
    found += [
        (m.start(), m.group())
        for m in _WORD.finditer(text)
        if NUMERAL_RE.fullmatch(m.group()) or NUMERALISH_RE.search(m.group())
    ]
    return [tok for _, tok in sorted(found)]


def negation_tokens(s: str) -> set[str]:
    return {w for w in _words(s) if w in NEGATION_WORDS}


def negation_flip(before: str, after: str) -> bool:
    """True when `after` adds/drops a negation relative to `before`.

    Only words that differ between the two sides count, so a verbatim text that
    contains both a word and its ne- form (e.g. "vie a nevie") is not a flip.
    """
    if negation_tokens(before) != negation_tokens(after):
        return True
    cb, ca = Counter(_words(before)), Counter(_words(after))
    differing = set((cb - ca) + (ca - cb))
    union = set(cb) | set(ca)
    return any(
        "ne" + w in union or (w.startswith("ne") and w[2:] in union) for w in differing
    )


def _find_word(text: str, needle: str) -> int:
    """Start of `needle` as a whole word/phrase in `text`, else -1.

    Word-boundary checks apply only on sides where the needle edge is a word char,
    so punctuation-edged needles ("vy,") still match.
    """
    if not needle:
        return -1
    pre = r"(?<!\w)" if re.match(r"\w", needle[0]) else ""
    post = r"(?!\w)" if re.match(r"\w", needle[-1]) else ""
    m = re.search(pre + re.escape(needle) + post, text)
    return m.start() if m else -1


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
        pos = _find_word(line_text, edit.before)
        if edit.new_speaker not in speaker_names or pos <= 0:
            return "split"
        return ""
    if _find_word(line_text, edit.before) < 0:
        return "not_found"
    if edit.before == edit.after:
        return "noop"
    if number_tokens(edit.before) != number_tokens(edit.after):
        return "number"
    if negation_flip(edit.before, edit.after):
        return "negation"
    if edit.type == EditType.PUNCTUATION:
        return "" if _words(edit.before) == _words(edit.after) else "punctuation"
    if abs(len(edit.before.split()) - len(edit.after.split())) > 1:
        return "word_count"
    if edit.type in _ORTHO and not _similar(edit.before, edit.after):
        return "similarity"
    if edit.type == EditType.PROPER_NOUN:
        allowed = {w for name in allowed_names for w in _words(name)}
        # Case-sensitive: "HLas" is a misspelling of "HLAS", not itself a correct name.
        exact = {w for name in allowed_names for w in _WORD.findall(name)}
        before_exact = _WORD.findall(edit.before)
        if before_exact and all(w in exact for w in before_exact):
            return "proper_noun"
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
            splits.setdefault(edit.line_no, []).append(eid)
        else:
            pos = _find_word(ln.text, edit.before)
            ln.text = ln.text[:pos] + edit.after + ln.text[pos + len(edit.before) :]

    out: list[Line] = []
    for idx, ln in enumerate(work):
        cuts: list[tuple[int, int]] = []
        for eid in splits.get(idx, []):
            pos = _find_word(ln.text, log[eid].edit.before)
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


CHUNK_LINES = 40
CONTEXT_LINES = 5

_RULES = (
    "You fix ASR errors in a Slovak TV debate transcript. Return ONLY a list "
    "of small edits; never rewrite lines.\n"
    "Allowed edit types:\n"
    "- spelling: a misspelled word or missing diacritics ('ľudi' -> 'ľudí').\n"
    "- word_boundary: wrongly split or merged words ('kpointe' -> 'k pointe').\n"
    "- proper_noun: a misheard name, ONLY to a name from NAMES.\n"
    "- punctuation: commas and sentence ends; words stay identical.\n"
    "- split_turn: one line contains two speakers. `before` = the first words "
    "spoken by the second speaker, copied exactly; `new_speaker` from SPEAKERS. "
    "Use one split_turn per speaker change.\n"
    "Never: fix grammar or style of spoken language, paraphrase, add missing "
    "words, delete repetitions or filler words, change numbers or negation. "
    "If unsure, propose nothing.\n"
    "`before` must be copied exactly from the line and kept short (the wrong "
    "word with at most one neighbour). `line_no` is the number after '#'.\n"
)


class CorrectionBatch(BaseModel):
    edits: list[TranscriptEdit] = Field(default_factory=list)


def build_correction_prompt(
    lines: list[Line],
    start: int,
    end: int,
    allowed_names: set[str],
    speaker_names: set[str],
) -> str:
    # `#N` is the list index: apply_corrections addresses lines by index, so the
    # prompt numbers must match it even if `Line.no` was renumbered upstream.
    lo, hi = max(0, start - CONTEXT_LINES), min(len(lines), end + CONTEXT_LINES)
    body = "\n".join(
        f"#{i} {lines[i].speaker} [{lines[i].ts}]: {lines[i].text}" for i in range(lo, hi)
    )
    return (
        f"{_RULES}\n"
        f"NAMES: {', '.join(sorted(allowed_names)) or '(none)'}\n"
        f"SPEAKERS: {', '.join(sorted(speaker_names)) or '(none)'}\n\n"
        f"Edit only lines #{start}..#{end - 1}; other lines are context.\n\n{body}"
    )


def propose_corrections(
    lines: list[Line],
    *,
    llm: Callable[[str], CorrectionBatch],
    allowed_names: set[str],
    speaker_names: set[str],
) -> tuple[list[TranscriptEdit], list[str]]:
    """Ask the LLM for edits chunk by chunk; edits keep global list indices.

    Prompts show global indices and the whole `lines` list is later passed to
    `apply_corrections`, so no index translation is needed; edits outside the
    chunk's own range are dropped.
    """
    edits: list[TranscriptEdit] = []
    notes: list[str] = []
    for start in range(0, len(lines), CHUNK_LINES):
        end = min(len(lines), start + CHUNK_LINES)
        try:
            batch = llm(build_correction_prompt(lines, start, end, allowed_names, speaker_names))
        except Exception as exc:  # noqa: BLE001 - a failed chunk keeps its raw text
            logger.warning("Correction chunk %d-%d failed: %s", start, end - 1, exc)
            notes.append(f"Transcript correction chunk #{start}-#{end - 1} failed: {exc}"[:300])
            continue
        edits.extend(x for x in batch.edits if start <= x.line_no < end)
    return edits, notes


@dataclass
class CorrectionOutcome:
    text: str
    lines: list[Line]
    log: list[EditResult]
    quality: TranscriptQuality
    notes: list[str]


def correct_transcript_edits(
    named_text: str,
    *,
    llm: Callable[[str], CorrectionBatch],
    allowed_names: set[str],
    speaker_names: set[str],
) -> CorrectionOutcome:
    lines = parse_lines(named_text)
    edits, notes = propose_corrections(
        lines, llm=llm, allowed_names=allowed_names, speaker_names=speaker_names
    )
    out, log = apply_corrections(lines, edits, allowed_names, speaker_names)
    quality = TranscriptQuality(
        applied=sum(1 for r in log if r.applied),
        rejected_by_rule=dict(Counter(r.rule for r in log if not r.applied)),
    )
    return CorrectionOutcome(
        text=format_lines(out), lines=out, log=log, quality=quality, notes=notes
    )


def default_correction_llm(settings) -> Callable[[str], CorrectionBatch]:
    from src.llm import generate_json

    return lambda prompt: generate_json(
        prompt, CorrectionBatch, settings, label="transcript_correction"
    )
