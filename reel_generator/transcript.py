"""Parse `Speaker [MM:SS]: text` transcripts. Local copy of the line format.

Kept inside this package so reel generation does not import the analyzer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

LINE_RE = re.compile(
    r"^(?P<speaker>[^\[\]]{1,80}?)\s*\[(?P<ts>(?:\d{1,3}:)?\d{1,2}:\d{2})\]\s*:\s*(?P<text>.*)$"
)

# Slovak debate speech. Used when two lines share a timestamp and the gap
# would otherwise dump a whole answer into a fraction of a second.
WORDS_PER_SECOND = 2.45


def parse_timestamp(ts: str) -> float:
    total = 0
    for part in ts.strip().split(":"):
        total = total * 60 + int(part)
    return float(total)


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class TranscriptLine:
    speaker: str
    start: float
    end: float
    text: str
    timestamp: str


def load_transcript(path: Path) -> list[TranscriptLine]:
    raw: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        raw.append((match.group("speaker").strip(), match.group("ts"), match.group("text").strip()))
    drafted = [
        TranscriptLine(speaker, parse_timestamp(ts), parse_timestamp(ts), text, ts)
        for speaker, ts, text in raw
    ]
    return _repair_ends(drafted)


def _repair_ends(lines: list[TranscriptLine]) -> list[TranscriptLine]:
    """Give each line a duration that matches how long those words take to say.

    A short overlap often shares the long turn's timestamp (`Erik [27:11]` plus
    `Moderátor [27:11]: V rovnakom`). Using the next stamp as the end crushed
    the long answer into 0.2s, so captions flashed and then froze.
    """
    counts = [len(line.text.split()) for line in lines]
    repaired: list[TranscriptLine] = []
    for index, line in enumerate(lines):
        expected = max(0.4, counts[index] / WORDS_PER_SECOND)
        next_distinct = next(
            (
                lines[later].start
                for later in range(index + 1, len(lines))
                if lines[later].start > line.start + 0.45
            ),
            None,
        )
        gap = (next_distinct - line.start) if next_distinct is not None else expected
        siblings = [
            later
            for later, other in enumerate(lines)
            if abs(other.start - line.start) < 0.45
        ]
        longest = max(siblings, key=lambda later: counts[later])
        if len(siblings) > 1 and index != longest and counts[index] <= 8:
            end = line.start
        else:
            end = line.start + min(gap, expected * 1.2)
        repaired.append(
            TranscriptLine(line.speaker, line.start, end, line.text, line.timestamp)
        )
    return repaired


def speech_segments(
    lines: list[TranscriptLine], start: float, end: float
) -> list[tuple[float, float, str]]:
    """Merged speaker turns inside the clip. Empty-duration overlap lines are skipped."""
    segments: list[tuple[float, float, str]] = []
    for line in lines:
        if line.end <= line.start + 0.2 or line.end <= start or line.start >= end:
            continue
        seg_start = max(line.start, start)
        seg_end = min(line.end, end)
        if seg_end - seg_start < 0.25:
            continue
        if (
            segments
            and speaker_match(segments[-1][2], line.speaker)
            and seg_start - segments[-1][1] < 0.6
        ):
            segments[-1] = (segments[-1][0], seg_end, line.speaker)
        else:
            segments.append((seg_start, seg_end, line.speaker))
    return segments


def speaker_match(left: str, right: str) -> bool:
    return _fold(left) == _fold(right)


def _fold(name: str) -> str:
    import unicodedata

    stripped = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    return " ".join(stripped.casefold().split())
