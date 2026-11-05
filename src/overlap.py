"""Overlap regions, voice enrollment and separated-stream assignment."""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from config import Settings

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


class Embedder(Protocol):
    def embed(self, audio: np.ndarray) -> np.ndarray: ...


class Separator(Protocol):
    def separate(self, audio: np.ndarray) -> list[np.ndarray]: ...


@dataclass
class OverlapTools:
    separator: Separator | None
    embedder: Embedder


@dataclass
class RegionResult:
    region: Region
    lead: str
    streams: dict[str, np.ndarray] = field(default_factory=dict)
    reason: str | None = None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def assign_streams(sims: np.ndarray, speakers: list[str], min_sim: float) -> dict[str, int]:
    straight = sims[0, 0] + sims[1, 1]
    crossed = sims[0, 1] + sims[1, 0]
    pairs = [(0, 0), (1, 1)] if straight >= crossed else [(0, 1), (1, 0)]
    return {speakers[col]: row for row, col in pairs if sims[row, col] >= min_sim}


def _slice(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    return audio[max(0, int(start * SR)) : min(len(audio), int(end * SR))]


def enroll(
    audio: np.ndarray, spans: dict[str, list[tuple[float, float]]], embedder: Embedder
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for speaker, pieces in spans.items():
        vecs = [embedder.embed(_slice(audio, s, e)) for s, e in pieces]
        out[speaker] = np.mean(np.stack(vecs), axis=0)
    return out


def _peak_normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if peak == 0.0 else (x * (0.9 / peak)).astype(np.float32)


def _separate_one(
    audio: np.ndarray,
    region: Region,
    enrollments: dict[str, np.ndarray],
    settings: Settings,
    tools: OverlapTools,
) -> tuple[dict[str, np.ndarray], str | None]:
    if len(region.speakers) != 2:
        return {}, "too_many_speakers"
    speakers = sorted(region.speakers)
    if any(s not in enrollments for s in speakers):
        return {}, "no_enrollment"
    if tools.separator is None:
        return {}, "separator_unavailable"

    pad = settings.overlap_pad_seconds
    s0 = max(0, int((region.start - pad) * SR))
    e0 = min(len(audio), int((region.end + pad) * SR))
    try:
        raw = tools.separator.separate(audio[s0:e0])
    except Exception:
        logger.exception("Separator failed on region %.2f-%.2f", region.start, region.end)
        return {}, "separator_error"

    offset = int(region.start * SR) - s0
    length = int(region.end * SR) - int(region.start * SR)
    streams = [_peak_normalize(np.asarray(x[offset : offset + length], dtype=np.float32)) for x in raw[:2]]
    vecs = [tools.embedder.embed(x) for x in streams]
    sims = np.array([[cosine(v, enrollments[s]) for s in speakers] for v in vecs])
    assigned = assign_streams(sims, speakers, settings.overlap_assign_min_sim)
    if not assigned:
        return {}, "low_similarity"
    return {speaker: streams[idx] for speaker, idx in assigned.items()}, None


def process_regions(
    audio: np.ndarray,
    regions: list[Region],
    turns: list[Turn],
    settings: Settings,
    tools: OverlapTools,
) -> list[RegionResult]:
    if not regions:
        return []
    spans = select_enrollment(
        turns, max_seconds=settings.enroll_max_seconds, min_seconds=settings.enroll_min_seconds
    )
    enrollments = enroll(audio, spans, tools.embedder)
    results: list[RegionResult] = []
    for region in regions:
        streams, reason = _separate_one(audio, region, enrollments, settings, tools)
        results.append(RegionResult(region, lead_speaker(region, turns), streams, reason))
    return results


def log_summary(results: list[RegionResult]) -> str:
    separated = sum(1 for r in results if r.reason is None)
    reasons = Counter(r.reason for r in results if r.reason is not None)
    line = f"Overlap regions: {len(results)} found, {separated} separated"
    if reasons:
        line += ", fallback " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
    logger.info(line)
    return line
