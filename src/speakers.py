"""Map diarization labels (Speaker A..) to real people.

The mapping is decided once (heuristics + one small LLM call, validated in
code) and applied to the transcript in code, so names never depend on an LLM
rewriting the transcript.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.report_models import SpeakerMap, SpeakerMapEntry, SpeakerRole
from src.transcript_lines import LINE_RE, Line

logger = logging.getLogger(__name__)

CLIP_NAME = "Záznam"
DEFAULT_MODERATOR = "Moderátor"
_CLIP_SHARE = 0.015
_MAIN_SHARE = 0.10
_START_WINDOW = 10
_SPEAKER_ID = re.compile(r"\bspeaker\s+[a-z0-9]+\b", re.IGNORECASE)
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class LabelProfile:
    label: str
    words: int = 0
    turns: int = 0
    first_no: int = 0
    share: float = 0.0

    @property
    def words_per_turn(self) -> float:
        return self.words / self.turns if self.turns else 0.0


def label_profiles(lines: list[Line]) -> dict[str, LabelProfile]:
    profiles: dict[str, LabelProfile] = {}
    prev: str | None = None
    for ln in lines:
        p = profiles.get(ln.speaker)
        if p is None:
            p = profiles[ln.speaker] = LabelProfile(label=ln.speaker, first_no=ln.no)
        p.words += ln.words
        if ln.speaker != prev:
            p.turns += 1
        prev = ln.speaker
    total = sum(p.words for p in profiles.values()) or 1
    for p in profiles.values():
        p.share = p.words / total
    return profiles


def heuristic_roles(lines: list[Line], n_main: int | None = None) -> dict[str, SpeakerRole]:
    """Moderator = main label with the fewest words per turn; other main labels are guests.

    Share of questions does not work: guests ask more questions than the
    moderator in some debates (617000, 605345).
    """
    profiles = label_profiles(lines)
    ranked = sorted(profiles.values(), key=lambda p: -p.words)
    if n_main:
        main = [p for p in ranked[:n_main] if p.share >= _CLIP_SHARE]
    else:
        main = [p for p in ranked if p.share >= _MAIN_SHARE]
    roles = {
        p.label: SpeakerRole.CLIP if p.share < _CLIP_SHARE else SpeakerRole.UNKNOWN
        for p in profiles.values()
    }
    if main:
        moderator = min(main, key=lambda p: (p.words_per_turn, p.first_no))
        for p in main:
            roles[p.label] = SpeakerRole.MODERATOR if p is moderator else SpeakerRole.GUEST
    return roles


def detect_debate_start(lines: list[Line], roles: dict[str, SpeakerRole]) -> str | None:
    """First line after which the opening recap (many short clip voices) is over."""
    main = {
        label
        for label, role in roles.items()
        if role in (SpeakerRole.MODERATOR, SpeakerRole.GUEST)
    }
    for i in range(len(lines) - _START_WINDOW + 1):
        if all(ln.speaker in main for ln in lines[i : i + _START_WINDOW]):
            return lines[i].ts
    return None


def apply_speaker_map(transcript: str, smap: SpeakerMap) -> str:
    """Replace line labels with mapped names; the spoken text is untouched."""
    out: list[str] = []
    for raw in (transcript or "").splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            out.append(raw)
            continue
        label = m.group("speaker").strip()
        name = smap.name_for(label) or label
        out.append(f"{name} [{m.group('ts')}]: {m.group('text').strip()}")
    return "\n".join(out) + ("\n" if out else "")


def _tokens(name: str) -> list[str]:
    return [t.casefold() for t in _TOKEN.findall(name or "")]


def canonical_speaker(name: str, names: list[str]) -> str | None:
    """Resolve an agent-written speaker name to exactly one roster name.

    Exact (case-insensitive) match, then a unique surname match, then a unique
    5-letter surname stem (Slovak inflection: 'Tomáša' -> 'Tomáš'). 'Speaker X'
    labels only ever match exactly.
    """
    if not name:
        return None
    folded = name.casefold().strip()
    for n in names:
        if n.casefold() == folded:
            return n
    if _SPEAKER_ID.search(name):
        return None
    toks = _tokens(name)
    if not toks:
        return None
    surname = toks[-1]
    hits = [n for n in names if surname in _tokens(n)]
    if len(hits) == 1:
        return hits[0]
    stem = surname[:5]
    if len(stem) < 4:
        return None
    hits = [n for n in names if any(t.startswith(stem) for t in _tokens(n))]
    return hits[0] if len(hits) == 1 else None
