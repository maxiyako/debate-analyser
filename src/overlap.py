"""Overlap regions, voice enrollment and separated-stream assignment."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SR = 16000
Turn = tuple[float, float, str]


@dataclass(frozen=True)
class Region:
    start: float
    end: float
    speakers: frozenset[str]
    group: int


def _overlap_spans(turns: list[Turn]) -> list[tuple[float, float, frozenset[str]]]:
    events: list[tuple[float, int, str]] = []
    for start, end, speaker in turns:
        if end > start:
            events.append((start, 1, speaker))
            events.append((end, -1, speaker))
    events.sort(key=lambda ev: (ev[0], ev[1]))
    active: dict[str, int] = {}
    spans: list[tuple[float, float, frozenset[str]]] = []
    prev_t: float | None = None
    for t, kind, speaker in events:
        if prev_t is not None and t > prev_t:
            live = frozenset(k for k, v in active.items() if v > 0)
            if len(live) >= 2:
                spans.append((prev_t, t, live))
        active[speaker] = active.get(speaker, 0) + kind
        prev_t = t
    return spans


def find_overlap_regions(
    turns: list[Turn],
    *,
    min_seconds: float,
    merge_gap: float = 0.3,
    max_chunk_seconds: float = 8.0,
) -> list[Region]:
    merged: list[tuple[float, float, frozenset[str]]] = []
    for start, end, speakers in _overlap_spans(turns):
        if merged and merged[-1][2] == speakers and start - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], end, speakers)
        else:
            merged.append((start, end, speakers))

    regions: list[Region] = []
    kept = [m for m in merged if m[1] - m[0] >= min_seconds]
    for group, (start, end, speakers) in enumerate(kept):
        n = max(1, math.ceil((end - start) / max_chunk_seconds - 1e-9))
        step = (end - start) / n
        for i in range(n):
            chunk_end = end if i == n - 1 else start + (i + 1) * step
            regions.append(Region(start + i * step, chunk_end, speakers, group))
    return regions


def _subtract(start: float, end: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for c0, c1 in cuts:
        nxt: list[tuple[float, float]] = []
        for p0, p1 in pieces:
            if c1 <= p0 or c0 >= p1:
                nxt.append((p0, p1))
                continue
            if c0 > p0:
                nxt.append((p0, c0))
            if c1 < p1:
                nxt.append((c1, p1))
        pieces = nxt
    return pieces


def trim_spans(spans: list[Turn], regions: list[Region]) -> list[Turn]:
    cuts = [(r.start, r.end) for r in regions]
    out: list[Turn] = []
    for start, end, speaker in spans:
        out.extend((p0, p1, speaker) for p0, p1 in _subtract(start, end, cuts))
    return out


def select_enrollment(
    turns: list[Turn], *, max_seconds: float, min_seconds: float
) -> dict[str, list[tuple[float, float]]]:
    cuts = [(s, e) for s, e, _ in _overlap_spans(turns)]
    pieces: dict[str, list[tuple[float, float]]] = {}
    for start, end, speaker in turns:
        pieces.setdefault(speaker, []).extend(_subtract(start, end, cuts))

    out: dict[str, list[tuple[float, float]]] = {}
    for speaker, spans in pieces.items():
        chosen: list[tuple[float, float]] = []
        total = 0.0
        for p0, p1 in sorted(spans, key=lambda p: (-(p[1] - p[0]), p[0])):
            if total >= max_seconds:
                break
            take = min(p1 - p0, max_seconds - total)
            chosen.append((p0, p1 if take >= p1 - p0 else p0 + take))
            total += take
        if total >= min_seconds:
            out[speaker] = chosen
    return out


def lead_speaker(region: Region, turns: list[Turn]) -> str:
    active = [
        (start, speaker)
        for start, end, speaker in turns
        if speaker in region.speakers and start <= region.start + 1e-6 and end > region.start
    ]
    if active:
        return min(active)[1]
    return sorted(region.speakers)[0]
