"""Overlap gold set: candidate regions, token scoring and the enable gate."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from src.overlap import SR, Region, Turn, lead_speaker
from src.transcript_lines import Line, parse_lines, strip_markers
from src.transcription import Diarization, Segment

CONTEXT_SECONDS = 2.0
LINE_TOLERANCE = 1.0
LAST_LINE_SECONDS = 30.0
_TOKEN = re.compile(r"\w+", re.UNICODE)

Span = tuple[float, float, str, str]


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(strip_markers(text))]


def matched(gold: list[str], hyp: list[str], min_block: int = 1) -> int:
    blocks = SequenceMatcher(None, gold, hyp, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= min_block)


def score_region(gold: dict[str, str], dominant: str, hyp: dict[str, str]) -> dict:
    interrupter = next(s for s in gold if s != dominant)
    out: dict = {}
    for role, speaker in (("dominant", dominant), ("interrupter", interrupter)):
        g = tokens(gold[speaker])
        out[role] = {"gold": len(g), "found": matched(g, tokens(hyp.get(speaker, "")))}
    out["misattributed"] = sum(
        matched(tokens(gold[g_spk]), tokens(text), min_block=2)
        for g_spk in gold
        for h_spk, text in hyp.items()
        if h_spk != g_spk
    )
    return out


def summarize(scores: list[dict]) -> dict:
    def recall(role: str) -> float:
        g = sum(s[role]["gold"] for s in scores)
        return round(sum(s[role]["found"] for s in scores) / g, 4) if g else 0.0

    return {
        "dominant_recall": recall("dominant"),
        "interrupter_recall": recall("interrupter"),
        "misattributed": sum(s["misattributed"] for s in scores),
        "regions": len(scores),
    }


def gate(base: dict, new: dict) -> dict:
    gain = round(new["interrupter_recall"] - base["interrupter_recall"], 4)
    drop = round(base["dominant_recall"] - new["dominant_recall"], 4)
    delta = new["misattributed"] - base["misattributed"]
    return {
        "interrupter_gain": gain,
        "dominant_drop": drop,
        "misattributed_delta": delta,
        "passed": gain >= 0.25 and drop <= 0.05 and delta <= 0,
    }


def line_spans(lines: list[Line]) -> list[Span]:
    out: list[Span] = []
    for i, ln in enumerate(lines):
        start = float(ln.seconds)
        end = float(lines[i + 1].seconds) if i + 1 < len(lines) else start + LAST_LINE_SECONDS
        out.append((start, end, ln.speaker, ln.text))
    return out


def map_labels(spans: list[Span], turns: list[Turn]) -> dict[str, str]:
    totals: dict[str, dict[str, float]] = {}
    for s, e, label, _ in spans:
        for t0, t1, raw in turns:
            ov = min(e, t1) - max(s, t0)
            if ov > 0:
                totals.setdefault(label, {}).setdefault(raw, 0.0)
                totals[label][raw] += ov
    return {label: max(by_raw, key=by_raw.get) for label, by_raw in totals.items()}


def _join(parts: dict[str, list[str]]) -> dict[str, str]:
    return {speaker: " ".join(texts) for speaker, texts in sorted(parts.items())}


def hyp_from_lines(spans: list[Span], mapping: dict[str, str], start: float, end: float) -> dict[str, str]:
    parts: dict[str, list[str]] = {}
    for s, e, label, text in spans:
        if e > start - LINE_TOLERANCE and s < end + LINE_TOLERANCE and label in mapping:
            parts.setdefault(mapping[label], []).append(text)
    return _join(parts)


def hyp_from_segments(segments: list[Segment], start: float, end: float) -> dict[str, str]:
    parts: dict[str, list[str]] = {}
    for seg in segments:
        if seg.end >= start and seg.start <= end:
            parts.setdefault(seg.speaker, []).append(seg.text)
    return _join(parts)


def build_candidates(
    episode: str,
    diar: Diarization,
    audio: np.ndarray,
    transcript: str,
    regions: list[Region],
    clip_dir: Path,
    top: int,
) -> list[dict]:
    import soundfile as sf

    spans = line_spans(parse_lines(transcript))
    pairs = [r for r in regions if len(r.speakers) == 2]
    chosen = sorted(pairs, key=lambda r: -(r.end - r.start))[:top]
    clip_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for r in sorted(chosen, key=lambda r: r.start):
        c0 = max(0, int((r.start - CONTEXT_SECONDS) * SR))
        c1 = min(len(audio), int((r.end + CONTEXT_SECONDS) * SR))
        clip = clip_dir / f"{episode}_{r.start:.2f}.wav"
        sf.write(str(clip), audio[c0:c1], SR)
        speakers = sorted(r.speakers)
        out.append(
            {
                "episode": episode,
                "start": round(r.start, 2),
                "end": round(r.end, 2),
                "speakers": speakers,
                "dominant": lead_speaker(r, diar.turns),
                "clip": str(clip),
                "transcript_lines": [
                    f"{label} [{int(s) // 60:02d}:{int(s) % 60:02d}]: {text}"
                    for s, e, label, text in spans
                    if e > r.start - LINE_TOLERANCE and s < r.end + LINE_TOLERANCE
                ],
                "gold": {s: "" for s in speakers},
            }
        )
    return out


def merge_gold(existing: list[dict], new: list[dict]) -> list[dict]:
    seen = {(e["episode"], e["start"]) for e in existing}
    return existing + [n for n in new if (n["episode"], n["start"]) not in seen]
