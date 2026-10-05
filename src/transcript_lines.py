"""Parse 'Label [MM:SS]: text' transcript lines.

Leaf module: imported by scoring, speakers, correction, questions, and
validation, so it must not import anything from `src`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LINE_RE = re.compile(
    r"^(?P<speaker>[^\[\]]{1,60}?)\s*\[(?P<ts>(?:\d{1,3}:)?\d{1,2}:\d{2})\]\s*:\s*(?P<text>.*)$"
)
_MARKER_RE = re.compile(r"\((?:hovorenie\s+)?cez\s+seba[^)]*\)", re.IGNORECASE)


def ts_seconds(ts: str) -> int:
    total = 0
    for part in ts.split(":"):
        total = total * 60 + int(part)
    return total


def strip_markers(text: str) -> str:
    """Drop cross-talk annotations so they are not counted as spoken words."""
    return " ".join(_MARKER_RE.sub(" ", text or "").split())


@dataclass
class Line:
    no: int
    speaker: str
    ts: str
    text: str
    raw_no: int | None = None
    edit_ids: list[int] = field(default_factory=list)

    @property
    def crosstalk(self) -> bool:
        return bool(_MARKER_RE.search(self.text))

    @property
    def seconds(self) -> int:
        return ts_seconds(self.ts)

    @property
    def words(self) -> int:
        return len(strip_markers(self.text).split())


def parse_lines(transcript: str) -> list[Line]:
    out: list[Line] = []
    for raw in (transcript or "").splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        n = len(out)
        out.append(
            Line(
                no=n,
                speaker=m.group("speaker").strip(),
                ts=m.group("ts"),
                text=m.group("text").strip(),
                raw_no=n,
            )
        )
    return out


def format_lines(lines: list[Line]) -> str:
    return "".join(f"{ln.speaker} [{ln.ts}]: {ln.text}\n" for ln in lines)
