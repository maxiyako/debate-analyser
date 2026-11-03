"""Pull one highlight out of a report via a JSON path.

Path grammar (dots, indexes, and `key="value"` filters):

    behavioral_analysis.speakers[speaker="Erik Tomáš"].question_dodging[1]
    behavioral_analysis.speakers[speaker="Erik Tomáš"].logical_fallacies[0]
    behavioral_analysis.speakers[speaker="Erik Tomáš"].civility.incivility[0]
    facts[speaker="Erik Tomáš"][verdict="False"][0]
    question_audit[addressee="Erik Tomáš"][outcome="dodged"][0]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reel_generator.transcript import (
    TranscriptLine,
    format_timestamp,
    load_transcript,
    parse_timestamp,
    speaker_match,
)

_TOKEN = re.compile(
    r"""
    \.?(?P<key>[A-Za-z_][A-Za-z0-9_]*)
    | \[(?P<index>\d+)\]
    | \[
        (?P<fkey>[A-Za-z_][A-Za-z0-9_]*)
        \s*=\s*
        (?P<fval>"[^"]*"|'[^']*'|[^\]\s]+)
      \]
    """,
    re.VERBOSE,
)

_DODGE_RE = re.compile(
    r"^\[(?P<ts>(?:\d{1,3}:)?\d{1,2}:\d{2})\]\s*"
    r"(?P<quote>.*?)\s*(?:→|->)\s*"
    r"(?P<outcome>dodged|partial|answered|interrupted)\s*:\s*"
    r"(?P<reason>.*)$",
    re.DOTALL,
)
_QUOTED_RE = re.compile(
    r"""['"„“]\s*\[(?P<ts>(?:\d{1,3}:)?\d{1,2}:\d{2})\]\s*(?P<quote>.*?)['"“”]""",
    re.DOTALL,
)
_PAREN_RE = re.compile(r"\(([^)]{1,40})\)")

_VERDICT_STAMP = {
    "False": ("LOŽ", "loz"),
    "Misleading": ("ZAVÁDZAJÚCE", "misleading"),
    "Unverified": ("NEOVERENÉ", "unverified"),
    "True": ("PRAVDA", "true"),
    "Contested": ("SPORNÉ", "misleading"),
}
_OUTCOME_STAMP = {
    "dodged": ("UHNUTIE", "dodge"),
    "partial": ("ČIASTOČNE", "partial"),
    "interrupted": ("PRERUŠENÉ", "partial"),
    "answered": ("ODPOVEĎ", "fair"),
}

BEHAVIOUR_LISTS = ("manipulation", "logical_fallacies", "question_dodging")
CIVILITY_LISTS = ("incivility", "fair_conduct")


@dataclass(frozen=True)
class Highlight:
    discussion_id: str
    speaker: str
    kind: str
    index: int | None
    source_path: str
    timestamp_s: float
    timestamp_label: str
    quote: str
    detail: str
    outcome: str
    stamp_text: str
    stamp_key: str

    @property
    def slug(self) -> str:
        index = "x" if self.index is None else str(self.index)
        who = _ascii_speaker(self.speaker)
        if self.kind == "fact" and self.outcome:
            who = f"{_ascii_speaker(self.outcome)}_{who}"
        return f"{self.discussion_id}_{self.kind}_{index}_{who}"


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_highlight_path(
    speaker: str,
    kind: str,
    index: int,
    verdict: str | None = None,
    outcome: str | None = None,
) -> str:
    if kind in BEHAVIOUR_LISTS:
        return f'behavioral_analysis.speakers[speaker="{speaker}"].{kind}[{index}]'
    if kind in CIVILITY_LISTS:
        return f'behavioral_analysis.speakers[speaker="{speaker}"].civility.{kind}[{index}]'
    if kind == "facts":
        path = f'facts[speaker="{speaker}"]'
        if verdict:
            path += f'[verdict="{verdict}"]'
        return path + f"[{index}]"
    if kind == "question_audit":
        path = f'question_audit[addressee="{speaker}"]'
        if outcome:
            path += f'[outcome="{outcome}"]'
        return path + f"[{index}]"
    known = ", ".join((*BEHAVIOUR_LISTS, *CIVILITY_LISTS, "facts", "question_audit"))
    raise ValueError(f"unknown highlight type {kind!r}. expected one of: {known}")


def resolve_path(data: Any, path: str) -> Any:
    text = path.strip()
    tokens = list(_TOKEN.finditer(text))
    if not tokens:
        raise ValueError(f"empty highlight path: {path!r}")
    consumed = tokens[-1].end()
    leftover = text[consumed:].strip()
    if leftover:
        raise ValueError(f"cannot parse highlight path near {leftover!r} in {path!r}")
    cursor: Any = data
    for index, token in enumerate(tokens):
        key = token.group("key")
        if key:
            if not isinstance(cursor, dict) or key not in cursor:
                raise KeyError(f"{path}: missing key {key!r}")
            cursor = cursor[key]
            continue
        if token.group("index") is not None:
            item_index = int(token.group("index"))
            if not isinstance(cursor, list):
                raise TypeError(f"{path}: [{item_index}] used on a non-list")
            if item_index < 0 or item_index >= len(cursor):
                raise IndexError(
                    f"{path}: index {item_index} out of range (length {len(cursor)})"
                )
            cursor = cursor[item_index]
            continue
        field_name = token.group("fkey")
        field_value = _unquote(token.group("fval"))
        if not isinstance(cursor, list):
            raise TypeError(f"{path}: filter [{field_name}=...] used on a non-list")
        matched = [
            item
            for item in cursor
            if isinstance(item, dict) and str(item.get(field_name, "")) == field_value
        ]
        if not matched:
            seen = sorted(
                {
                    str(item.get(field_name))
                    for item in cursor
                    if isinstance(item, dict) and field_name in item
                }
            )
            raise KeyError(
                f"{path}: no {field_name}={field_value!r}. values present: {seen}"
            )
        nxt = tokens[index + 1] if index + 1 < len(tokens) else None
        keep_list = nxt is not None and (
            nxt.group("index") is not None or nxt.group("fkey")
        )
        if keep_list:
            cursor = matched
        elif len(matched) == 1:
            cursor = matched[0]
        else:
            cursor = matched
    return cursor


def highlight_from_report(
    report: dict[str, Any],
    path: str,
    discussion_id: str,
    lines: list[TranscriptLine] | None = None,
) -> Highlight:
    value = resolve_path(report, path)
    if isinstance(value, str):
        return _from_string(value, path, discussion_id, lines)
    if isinstance(value, dict):
        return _from_dict(value, path, discussion_id)
    raise TypeError(f"{path} resolved to {type(value).__name__}, expected a string or object")


def _from_string(
    text: str,
    path: str,
    discussion_id: str,
    lines: list[TranscriptLine] | None = None,
) -> Highlight:
    kind = _kind_from_path(path)
    speaker = _speaker_from_path(path)
    index = _index_from_path(path)
    dodge = _DODGE_RE.match(text.strip())
    if dodge:
        outcome = dodge.group("outcome")
        stamp_text, stamp_key = _OUTCOME_STAMP.get(outcome, ("UHNUTIE", "dodge"))
        ts = dodge.group("ts")
        return Highlight(
            discussion_id=discussion_id,
            speaker=speaker,
            kind=kind,
            index=index,
            source_path=path,
            timestamp_s=parse_timestamp(ts),
            timestamp_label=ts,
            quote=dodge.group("quote").strip(),
            detail=dodge.group("reason").strip(),
            outcome=outcome,
            stamp_text=stamp_text,
            stamp_key=stamp_key,
        )
    quoted = _QUOTED_RE.search(text)
    short = _short_name(text)
    if kind == "logical_fallacies":
        stamp_text, stamp_key = f"FAUL: {short}", "fallacy"
    elif kind == "manipulation":
        stamp_text, stamp_key = f"MANIPULÁCIA: {short}", "manipulation"
    elif kind == "incivility":
        stamp_text, stamp_key = "ÚTOK", "incivility"
    elif kind == "fair_conduct":
        stamp_text, stamp_key = "FÉR", "fair"
    else:
        stamp_text, stamp_key = short or "POZNÁMKA", "misleading"
    if quoted:
        ts = quoted.group("ts")
        quote = quoted.group("quote").strip()
    else:
        bare = re.search(r"\[((?:\d{1,3}:)?\d{1,2}:\d{2})\]", text)
        if bare:
            ts = bare.group(1)
            quote = text
        else:
            moment, quote = _time_from_transcript(text, speaker, discussion_id, lines)
            return Highlight(
                discussion_id=discussion_id,
                speaker=speaker,
                kind=kind,
                index=index,
                source_path=path,
                timestamp_s=moment,
                timestamp_label=format_timestamp(moment),
                quote=quote,
                detail=text.strip(),
                outcome="",
                stamp_text=stamp_text,
                stamp_key=stamp_key,
            )
    return Highlight(
        discussion_id=discussion_id,
        speaker=speaker,
        kind=kind,
        index=index,
        source_path=path,
        timestamp_s=parse_timestamp(ts),
        timestamp_label=ts,
        quote=quote,
        detail=text.strip(),
        outcome="",
        stamp_text=stamp_text,
        stamp_key=stamp_key,
    )


def _time_from_transcript(
    text: str,
    speaker: str,
    discussion_id: str,
    lines: list[TranscriptLine] | None,
) -> tuple[float, str]:
    """Place a report quote that has no [mm:ss] onto the transcript line that says it."""
    pieces = _evidence_pieces(text)
    if not pieces:
        raise ValueError("highlight text has no quote to look up in the transcript")
    if lines is None:
        from reel_generator.paths import DEFAULT_DATA_DIR, find_transcript

        lines = load_transcript(find_transcript(DEFAULT_DATA_DIR, discussion_id))
    best: tuple[int, float, TranscriptLine, str] | None = None
    for line in lines:
        if speaker and not speaker_match(line.speaker, speaker):
            continue
        hay = _norm_tokens(line.text)
        if not hay:
            continue
        for piece in pieces:
            needle = _norm_tokens(piece)
            if len(needle) < 3:
                continue
            matched, start = _contig_at(hay, needle)
            if matched < 4 and not (matched >= 3 and matched == len(needle)):
                continue
            if best is None or matched > best[0]:
                best = (matched, start / len(hay), line, piece)
    if best is None:
        sample = pieces[0][:80]
        raise ValueError(
            f"no transcript line for {speaker!r} contains the quote {sample!r}"
        )
    _matched, fraction, line, piece = best
    span = max(0.0, line.end - line.start)
    return line.start + fraction * span, piece.strip()


def _evidence_pieces(text: str) -> list[str]:
    quoted = re.findall(r"['\"„“]([^'\"“”]{6,}?)['\"“”]", text)
    blobs = quoted or [text]
    pieces: list[str] = []
    for blob in blobs:
        for part in re.split(r"\.{2,}|…", blob):
            part = part.strip(" .,;:")
            if len(_norm_tokens(part)) >= 3:
                pieces.append(part)
    return pieces


def _contig_at(hay: list[str], needle: list[str]) -> tuple[int, int]:
    best = (0, 0)
    for index in range(len(hay)):
        matched = 0
        while (
            index + matched < len(hay)
            and matched < len(needle)
            and hay[index + matched] == needle[matched]
        ):
            matched += 1
        if matched > best[0]:
            best = (matched, index)
    return best


def _norm_tokens(text: str) -> list[str]:
    import unicodedata

    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = re.sub(r"[^a-z0-9]+", " ", folded.casefold())
    return [token for token in folded.split() if token]


def _from_dict(item: dict[str, Any], path: str, discussion_id: str) -> Highlight:
    kind = _kind_from_path(path)
    index = _index_from_path(path)
    if kind == "question_audit" or "question" in item and "outcome" in item:
        outcome = str(item.get("outcome") or "")
        stamp_text, stamp_key = _OUTCOME_STAMP.get(outcome, ("OTÁZKA", "partial"))
        ts = str(item.get("timestamp") or "")
        return Highlight(
            discussion_id=discussion_id,
            speaker=str(item.get("addressee") or _speaker_from_path(path)),
            kind="question_audit",
            index=index if index is not None else item.get("id"),
            source_path=path,
            timestamp_s=parse_timestamp(ts),
            timestamp_label=ts,
            quote=str(item.get("question") or ""),
            detail=str(item.get("reason") or ""),
            outcome=outcome,
            stamp_text=stamp_text,
            stamp_key=stamp_key,
        )
    verdict = str(item.get("verdict") or "")
    stamp_text, stamp_key = _VERDICT_STAMP.get(verdict, (verdict.upper() or "TVRDENIE", "misleading"))
    ts = str(item.get("timestamp") or "")
    if not ts:
        raise ValueError(f"{path}: fact has no timestamp")
    return Highlight(
        discussion_id=discussion_id,
        speaker=str(item.get("speaker") or _speaker_from_path(path)),
        kind="fact",
        index=index,
        source_path=path,
        timestamp_s=parse_timestamp(ts),
        timestamp_label=ts,
        quote=str(item.get("quote") or item.get("claim") or ""),
        detail=str(item.get("claim") or ""),
        outcome=verdict,
        stamp_text=stamp_text,
        stamp_key=stamp_key,
    )


def _ascii_speaker(name: str) -> str:
    import unicodedata

    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = re.sub(r"[^a-z0-9]+", "_", folded.casefold()).strip("_")
    return folded or "speaker"


def _kind_from_path(path: str) -> str:
    if path.startswith("facts"):
        return "fact"
    if path.startswith("question_audit"):
        return "question_audit"
    for name in (*BEHAVIOUR_LISTS, *CIVILITY_LISTS):
        if name in path:
            return name
    return "highlight"


def _speaker_from_path(path: str) -> str:
    match = re.search(r'\[(?:speaker|addressee)="([^"]+)"\]', path)
    if match:
        return match.group(1)
    match = re.search(r"\[(?:speaker|addressee)='([^']+)'\]", path)
    return match.group(1) if match else ""


def _index_from_path(path: str) -> int | None:
    indexes = re.findall(r"\[(\d+)\]", path)
    if not indexes:
        return None
    return int(indexes[-1])


def _short_name(text: str) -> str:
    head = text.split(":")[0].strip()
    paren = _PAREN_RE.search(head)
    name = paren.group(1).strip() if paren else head
    name = re.sub(r"\s+", " ", name).strip(" .")
    if len(name) > 32:
        name = name[:31].rstrip() + "…"
    return name.upper()


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
