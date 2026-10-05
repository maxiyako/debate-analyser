"""Map diarization labels (Speaker A..) to real people.

The mapping is decided once (heuristics + one small LLM call, validated in
code) and applied to the transcript in code, so names never depend on an LLM
rewriting the transcript.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field

from src.report_models import SpeakerMap, SpeakerMapEntry, SpeakerRole
from src.transcript_lines import LINE_RE, Line, parse_lines, ts_seconds

logger = logging.getLogger(__name__)

CLIP_NAME = "Záznam"
DEFAULT_MODERATOR = "Moderátor"
_CLIP_SHARE = 0.015
_MAIN_SHARE = 0.10
_START_WINDOW = 10
_SPEAKER_ID = re.compile(r"\bspeaker\s+[a-z0-9]+\b", re.IGNORECASE)
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
_MIN_CONFIDENCE = 0.7
_MIN_EVIDENCE_CHARS = 15
_MIN_STEM = 4
_EVIDENCE_MIN_RATIO = 0.85
_SELF_ID = re.compile(
    r"\b(ja ako|my v|za stranu|naša strana|moja strana|ako minister|ako poslan)",
    re.IGNORECASE,
)
_TS_PREFIX = re.compile(r"^\s*\[[\d:]+\]\s*")


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


def is_valid_speaker_name(name: str) -> bool:
    """True when the name can label a transcript line (LINE_RE speaker group).

    A name that breaks the grammar would make every line it labels unparsable,
    so word counts, the question audit and the accusation guard would silently
    see an empty transcript.
    """
    clean = (name or "").strip()
    if not clean or "\n" in clean or "\r" in clean:
        return False
    m = LINE_RE.match(f"{clean} [00:00]: text")
    return m is not None and m.group("speaker").strip() == clean


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


def _fold(text: str) -> str:
    """Casefold and drop diacritics: ASR writes 'Tomas' as often as 'Tomáš'."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _tokens(name: str) -> list[str]:
    return [_fold(t) for t in _TOKEN.findall(name or "")]


def canonical_speaker(name: str, names: list[str]) -> str | None:
    """Resolve an agent-written speaker name to exactly one roster name.

    Exact match, then a unique surname match, then a unique 5-letter surname
    stem (Slovak inflection: 'Tomáša' -> 'Tomáš'). All comparisons ignore case
    and diacritics ('Tomas' is 'Tomáš'); the roster spelling is returned.
    'Speaker X' labels only ever match exactly.
    """
    if not name:
        return None
    folded = _fold(name).strip()
    for n in names:
        if _fold(n).strip() == folded:
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


class LabelAssignment(BaseModel):
    label: str
    name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class SpeakerAssignment(BaseModel):
    guests_from_intro: list[str] = Field(default_factory=list)
    assignments: list[LabelAssignment] = Field(default_factory=list)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "…"


def label_context(lines: list[Line], label: str, moderator_label: str | None) -> list[Line]:
    """The label's lines plus the moderator line right before each of its turns."""
    ctx: list[Line] = []
    seen: set[int] = set()
    for i, ln in enumerate(lines):
        if ln.speaker != label:
            continue
        prev = lines[i - 1] if i else None
        if prev is not None and moderator_label and prev.speaker == moderator_label and prev.no not in seen:
            ctx.append(prev)
            seen.add(prev.no)
        if ln.no not in seen:
            ctx.append(ln)
            seen.add(ln.no)
    return ctx


def _moderator_label(roles: dict[str, SpeakerRole]) -> str | None:
    return next((lbl for lbl, r in roles.items() if r == SpeakerRole.MODERATOR), None)


def _main_labels(roles: dict[str, SpeakerRole]) -> list[str]:
    return [lbl for lbl, r in roles.items() if r in (SpeakerRole.MODERATOR, SpeakerRole.GUEST)]


def build_speaker_prompt(
    intro: list[Line],
    body: list[Line],
    roles: dict[str, SpeakerRole],
    roster: list[str],
    moderator_name: str,
    briefing_text: str = "",
) -> str:
    profiles = label_profiles(body)
    mod_label = _moderator_label(roles)
    roster_line = (
        ", ".join(roster)
        if roster
        else "(empty: first list in guests_from_intro the invited guests the "
        "moderator introduces, full names in nominative)"
    )
    parts = [
        "You map anonymous diarization labels of a Slovak TV debate to real people.",
        f"GUEST ROSTER: {roster_line}",
        f"MODERATOR NAME: {moderator_name}",
        "Return one assignment per LABEL below: a guest from the roster, or the "
        "moderator name. Use only: self-identification (party, office, 'ja ako "
        "minister'), the moderator addressing someone right before the label "
        "speaks (the addressed person usually answers next), and greetings after "
        "being introduced ('ďakujem za pozvanie').",
        "evidence: 1-3 excerpts copied verbatim from the lines shown for that "
        "label, each prefixed with its [MM:SS]. confidence: 0-1.",
    ]
    if briefing_text:
        parts.append(f"BACKGROUND (who is who; not evidence):\n{_clip(briefing_text, 2500)}")
    parts.append(
        "INTRO:\n" + "\n".join(f"{ln.speaker} [{ln.ts}]: {_clip(ln.text, 400)}" for ln in intro[:25])
    )
    for label in _main_labels(roles):
        p = profiles.get(label)
        if p is None:
            continue
        ctx = label_context(body, label, None if label == mod_label else mod_label)
        own = [ln for ln in ctx if ln.speaker == label]
        picked = own[:8] + [ln for ln in own[8:] if _SELF_ID.search(ln.text)][:5]
        before = [ln for ln in ctx if ln.speaker != label][:5]
        block = [
            f"LABEL {label} ({100 * p.share:.1f}% words, {p.turns} turns, "
            f"candidate role: {roles[label].value})"
        ]
        block += [f"  [{ln.ts}] {_clip(ln.text, 300)}" for ln in picked]
        block += [f"  moderator before it [{ln.ts}]: {_clip(ln.text, 200)}" for ln in before]
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def _roster_from_intro(names: list[str], lines: list[Line]) -> list[str]:
    """Keep only intro names whose surname stem actually occurs in the opening lines."""
    head = _fold(" ".join(ln.text for ln in lines[:40]))
    out: list[str] = []
    for n in names:
        toks = _tokens(n)
        if toks and toks[-1][:5] in head and n not in out:
            out.append(n)
    return out


def validate_assignment(
    assign: SpeakerAssignment,
    lines: list[Line],
    roles: dict[str, SpeakerRole],
    roster: list[str],
    moderator_name: str,
) -> tuple[list[SpeakerMapEntry], list[str], bool]:
    """Validate the LLM assignment; evidence is grounded against the whole transcript
    (intro greetings included), restricted to the label's own context."""
    from src.validation import normalize, quote_grounded

    notes: list[str] = []
    ok = True
    mod_label = _moderator_label(roles)
    main = set(_main_labels(roles))
    entries: dict[str, SpeakerMapEntry] = {}
    seen: set[str] = set()
    for a in assign.assignments:
        if a.label not in main:
            notes.append(f"Speaker map: ignored assignment for non-main label {a.label}")
            continue
        if a.label in seen:
            notes.append(f"Speaker map: duplicate assignment for {a.label}")
            ok = False
            continue
        seen.add(a.label)
        name = a.name
        if a.label == mod_label and name not in roster:
            # The turn-taking heuristic owns the moderator label. Whatever person
            # the LLM reads out of the intro, the moderator is published under the
            # known name, so the label is never left unnamed (which would silently
            # disable the question audit). A roster guest here is a real
            # contradiction and is handled by the branches below.
            if name != moderator_name:
                notes.append(
                    f"Speaker map: moderator label {a.label} named '{name}' by LLM; "
                    f"using '{moderator_name}'"
                )
            name = moderator_name
            role = SpeakerRole.MODERATOR
        elif name == moderator_name:
            role = SpeakerRole.MODERATOR
        elif name in roster:
            role = SpeakerRole.GUEST
        else:
            notes.append(f"Speaker map: {a.label} -> '{name}' is not in the roster")
            ok = False
            continue
        if (role == SpeakerRole.MODERATOR) != (a.label == mod_label):
            notes.append(f"Speaker map: LLM role for {a.label} contradicts the turn-taking heuristic")
            ok = False
        ctx = label_context(lines, a.label, None if a.label == mod_label else mod_label)
        context = "\n".join(ln.text for ln in ctx)
        grounded = []
        for ev in a.evidence:
            text = _TS_PREFIX.sub("", ev)
            if len(normalize(text)) >= _MIN_EVIDENCE_CHARS and quote_grounded(
                text, context, min_ratio=_EVIDENCE_MIN_RATIO
            ):
                grounded.append(ev)
        confidence = a.confidence if grounded else 0.0
        if not grounded:
            notes.append(f"Speaker map: no grounded evidence for {a.label} -> {a.name}")
        if confidence < _MIN_CONFIDENCE:
            ok = False
        entries[a.label] = SpeakerMapEntry(
            label=a.label, name=name, role=role, confidence=confidence, evidence=grounded[:3]
        )
    guest_names = [e.name for e in entries.values() if e.role == SpeakerRole.GUEST]
    dup = sorted({n for n in guest_names if guest_names.count(n) > 1})
    if dup:
        notes.append(f"Speaker map: guest assigned to several main labels: {', '.join(dup)}")
        ok = False
    missing = [n for n in roster if n not in guest_names]
    if missing:
        notes.append(f"Speaker map: roster guests without a label: {', '.join(missing)}")
        ok = False
    unassigned = sorted(main - set(entries))
    if unassigned:
        notes.append(f"Speaker map: main labels left unnamed: {', '.join(unassigned)}")
        ok = False
    return list(entries.values()), notes, ok


def _surname_stems(name: str) -> list[str]:
    """Folded surname plus its 1-2 character shorter stems (Slovak case endings)."""
    toks = [_fold(t) for t in _TOKEN.findall(name or "")]
    if not toks:
        return []
    surname = toks[-1]
    return [surname[: len(surname) - k] for k in range(3) if len(surname) - k >= _MIN_STEM]


def _name_hits(lines: list[Line], name: str) -> int:
    stems = _surname_stems(name)
    if not stems:
        return 0
    return sum(1 for ln in lines if any(s in _fold(ln.text) for s in stems))


def _evidence_assignment(
    body: list[Line], guests: list[str], mod_label: str | None, roster: list[str]
) -> dict[str, str]:
    """Match guest labels to roster names by the transcript, or not at all.

    Primary evidence: the moderator addressing a surname right before the
    label's turn ("Viete si, pán Viskupič, predstaviť..."). Own lines are only a
    tiebreak — in a duel each guest says the rival's name more often than their
    own. A pair is used only when it beats every rival of both its label and its
    name; anything ambiguous is left unassigned so the map degrades visibly
    instead of publishing a swapped scoreboard.
    """
    scores = {
        (label, name): (
            _name_hits([ln for ln in ctx if ln.speaker != label], name),
            _name_hits([ln for ln in ctx if ln.speaker == label], name),
        )
        for label in guests
        for ctx in [label_context(body, label, mod_label)]
        for name in roster
    }
    assigned: dict[str, str] = {}
    labels, names = list(guests), list(roster)
    while labels and names:
        label, name = max(
            ((lbl, nm) for lbl in labels for nm in names), key=lambda pair: scores[pair]
        )
        best = scores[(label, name)]
        rivals = [scores[(label, nm)] for nm in names if nm != name]
        rivals += [scores[(lbl, name)] for lbl in labels if lbl != label]
        if best == (0, 0) or any(rival >= best for rival in rivals):
            break
        assigned[label] = name
        labels.remove(label)
        names.remove(name)
    return assigned


def _heuristic_entries(
    body: list[Line], roles: dict[str, SpeakerRole], roster: list[str], moderator_name: str
) -> list[SpeakerMapEntry]:
    """Fallback: the moderator comes from turn taking, guests from name evidence."""
    first: dict[str, int] = {}
    for ln in body:
        first.setdefault(ln.speaker, ln.no)
    guests = sorted(
        (lbl for lbl, r in roles.items() if r == SpeakerRole.GUEST),
        key=lambda lbl: first.get(lbl, 10**9),
    )
    entries = [
        SpeakerMapEntry(label=lbl, name=moderator_name, role=SpeakerRole.MODERATOR, confidence=0.5)
        for lbl, r in roles.items()
        if r == SpeakerRole.MODERATOR
    ]
    assigned = _evidence_assignment(body, guests, _moderator_label(roles), roster)
    for lbl in guests:
        name = assigned.get(lbl)
        entries.append(
            SpeakerMapEntry(
                label=lbl,
                name=name or lbl,
                role=SpeakerRole.GUEST,
                confidence=0.3 if name else 0.0,
            )
        )
    return entries


def map_speakers(
    transcript: str,
    *,
    guests: list[str] | None = None,
    moderator: str | None = None,
    llm: Callable[[str], SpeakerAssignment] | None = None,
    briefing_text: str = "",
) -> SpeakerMap:
    lines = parse_lines(transcript)
    roster = list(guests or [])
    roles = heuristic_roles(lines, n_main=len(roster) + 1 if roster else None)
    start = detect_debate_start(lines, roles)
    start_s = ts_seconds(start) if start else 0
    body = [ln for ln in lines if ln.seconds >= start_s]
    moderator_name = moderator or DEFAULT_MODERATOR
    notes: list[str] = []
    entries: list[SpeakerMapEntry] = []
    ok = False
    source = "cli" if roster else "llm"

    if llm is not None:
        prompt = build_speaker_prompt(lines, body, roles, roster, moderator_name, briefing_text)
        try:
            assign: SpeakerAssignment | None = llm(prompt)
        except Exception as exc:  # noqa: BLE001 - fall back to heuristics, never abort the run
            logger.warning("Speaker mapping LLM failed: %s", exc)
            notes.append(f"Speaker map LLM failed: {exc}"[:300])
            assign = None
        if assign is not None:
            if not roster:
                roster = _roster_from_intro(assign.guests_from_intro, lines)
            entries, v_notes, ok = validate_assignment(assign, lines, roles, roster, moderator_name)
            notes.extend(v_notes)
    if not entries:
        entries = _heuristic_entries(body, roles, roster, moderator_name)
        source = "heuristic"
        ok = False
    if not roster:
        notes.append("Speaker map: no guest names known (no --guests, none found in the intro)")
        ok = False

    mapped = {e.label for e in entries}
    for label, role in roles.items():
        if label not in mapped and role in (SpeakerRole.CLIP, SpeakerRole.UNKNOWN):
            entries.append(
                SpeakerMapEntry(
                    label=label,
                    name=CLIP_NAME,
                    role=role,
                    confidence=1.0 if role == SpeakerRole.CLIP else 0.5,
                )
            )
    named_guest = any(e.role == SpeakerRole.GUEST and e.name in roster for e in entries)
    status = "ok" if ok else ("partial" if named_guest else "failed")
    return SpeakerMap(
        status=status, source=source, debate_start=start, entries=entries, notes=notes
    )


def default_speaker_llm(settings) -> Callable[[str], SpeakerAssignment]:
    from src.llm import generate_json

    return lambda prompt: generate_json(
        prompt, SpeakerAssignment, settings, label="speaker_map"
    )
