"""Speaker diarization + Slovak Whisper ASR, merged into tagged transcript lines."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from config import Settings, get_settings
from src.gcs import upload_file
from src.overlap import (
    SR,
    OverlapTools,
    Region,
    Turn,
    find_overlap_regions,
    load_overlap_tools,
    log_summary,
    process_regions,
    trim_spans,
)

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"
    crosstalk: bool = False  # another speaker talks at the same time
    region: int | None = None
    rank: int = 0


def _fmt_ts(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _speaker_label(raw: str, mapping: dict[str, str]) -> str:
    if raw not in mapping:
        mapping[raw] = f"Speaker {chr(ord('A') + len(mapping))}"
    return mapping[raw]


@dataclass
class Diarization:
    model: str
    turns: list[Turn]
    exclusive_turns: list[Turn]


def _annotation_turns(annotation) -> list[Turn]:
    return [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def diarization_from_output(output, model: str) -> Diarization:
    regular = getattr(output, "speaker_diarization", None)
    if regular is None:
        turns = _annotation_turns(output)
        return Diarization(model, turns, list(turns))
    exclusive = getattr(output, "exclusive_speaker_diarization", None) or regular
    return Diarization(model, _annotation_turns(regular), _annotation_turns(exclusive))


def diarize(audio_path: Path, settings: Settings) -> Diarization:
    """Overlap-aware and exclusive speaker turns from pyannote."""
    model_id = settings.diarization_model_id
    if not settings.huggingface_token:
        msg = (
            "HUGGINGFACE_TOKEN missing — diarization cannot run, so the whole "
            "transcript would collapse to a single speaker. Accept terms at "
            f"https://huggingface.co/{model_id} and set the token."
        )
        if settings.require_diarization:
            raise RuntimeError(msg)
        logger.warning("%s Falling back to a single speaker (Speaker A).", msg)
        return Diarization(model_id, [], [])

    from pyannote.audio import Pipeline

    logger.info("Loading diarization model %s", model_id)
    try:
        pipeline = Pipeline.from_pretrained(model_id, token=settings.huggingface_token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=settings.huggingface_token)

    import torch

    pipeline.to(torch.device(_resolve_device(settings)))

    logger.info("Running diarization on %s", audio_path)
    diar = diarization_from_output(pipeline(str(audio_path)), model_id)
    logger.info(
        "Found %d speaker turns (%d exclusive)", len(diar.turns), len(diar.exclusive_turns)
    )
    return diar


def diarization_cache_path(transcript_dir: Path, episode_id: str) -> Path:
    return transcript_dir / f"{episode_id}.diarization.v2.json"


def save_diarization(path: Path, diar: Diarization) -> None:
    payload = {"model": diar.model, "turns": diar.turns, "exclusive_turns": diar.exclusive_turns}
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_diarization(path: Path) -> Diarization | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))

    def turns(key: str) -> list[Turn]:
        return [(float(a), float(b), str(s)) for a, b, s in raw[key]]

    return Diarization(raw["model"], turns("turns"), turns("exclusive_turns"))


def _resolve_device(settings: Settings) -> str:
    import torch

    device = settings.whisper_device
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return device


def _window_starts(total_s: float, window_s: float, step_s: float) -> list[float]:
    starts = [0.0]
    while starts[-1] + window_s < total_s:
        starts.append(starts[-1] + step_s)
    return starts


def _merge_turns(
    turns: list[tuple[float, float, str]], max_gap: float = 0.6
) -> list[tuple[float, float, str]]:
    """Merge adjacent same-speaker diarization turns into utterances."""
    merged: list[tuple[float, float, str]] = []
    for start, end, spk in sorted(turns, key=lambda t: t[0]):
        if merged and merged[-1][2] == spk and start - merged[-1][1] <= max_gap:
            p_start, p_end, p_spk = merged[-1]
            merged[-1] = (p_start, max(p_end, end), p_spk)
        else:
            merged.append((start, end, spk))
    return merged


def _chunk_spans(start: float, end: float, max_len: float = 28.0) -> list[tuple[float, float]]:
    """Split a span longer than Whisper's 30s receptive field into sub-spans."""
    spans: list[tuple[float, float]] = []
    s = start
    while s < end - 1e-3:
        spans.append((s, min(end, s + max_len)))
        s += max_len
    return spans or [(start, end)]


MIN_SLICE_SECONDS = 0.3


def load_audio(audio_path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise RuntimeError(f"Expected 16 kHz WAV, got {sr} Hz — re-extract audio")
    return audio


class ASR(Protocol):
    def transcribe(self, chunks: list[np.ndarray]) -> list[str]: ...


class WhisperASR:
    """Batched Slovak Whisper over in-memory audio chunks.

    This model does not reliably emit Whisper timestamp tokens, and word-level
    timestamps via cross-attention DTW are ~7x slower on MPS, so each chunk is
    one diarization slice or separated stream and its speaker is known upfront.
    """

    def __init__(self, settings: Settings, batch_size: int | None = None) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self._torch = torch
        self.device = _resolve_device(settings)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_id = settings.whisper_model_id
        logger.info("Loading Whisper model %s on %s (dtype=%s)", model_id, self.device, self.dtype)
        load_kwargs: dict = {}
        if settings.huggingface_token and not Path(model_id).exists():
            load_kwargs["token"] = settings.huggingface_token
        self.processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=self.dtype, low_cpu_mem_usage=True, **load_kwargs
        )
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size or settings.asr_batch_size

    def transcribe(self, chunks: list[np.ndarray]) -> list[str]:
        torch = self._torch
        texts: list[str] = []
        bs = self.batch_size
        for b in range(0, len(chunks), bs):
            batch = chunks[b : b + bs]
            inputs = self.processor(
                batch, sampling_rate=SR, return_tensors="pt", return_attention_mask=True
            )
            features = inputs.input_features.to(device=self.device, dtype=self.dtype)
            mask = inputs.attention_mask.to(device=self.device)
            with torch.inference_mode():
                predicted = self.model.generate(
                    features,
                    attention_mask=mask,
                    language="slovak",
                    task="transcribe",
                    no_repeat_ngram_size=3,
                )
            texts.extend(t.strip() for t in self.processor.batch_decode(predicted, skip_special_tokens=True))
            done = min(b + bs, len(chunks))
            if done % (bs * 5) == 0 or done == len(chunks):
                logger.info("ASR progress %d/%d (%.0f%%)", done, len(chunks), 100.0 * done / len(chunks))
        return texts

    def close(self) -> None:
        del self.model
        if self.device == "mps":
            self._torch.mps.empty_cache()


def clean_spans(exclusive_turns: list[Turn], total_s: float, regions: list[Region]) -> list[Turn]:
    spans: list[Turn] = []
    for start, end, speaker in trim_spans(_merge_turns(exclusive_turns), regions):
        for s, e in _chunk_spans(start, min(end, total_s)):
            if e - s >= MIN_SLICE_SECONDS:
                spans.append((s, e, speaker))
    return spans


def _clip(spans: list[Turn], window: tuple[float, float] | None) -> list[Turn]:
    if window is None:
        return spans
    w0, w1 = window
    out: list[Turn] = []
    for s, e, speaker in spans:
        s, e = max(s, w0), min(e, w1)
        if e - s >= MIN_SLICE_SECONDS:
            out.append((s, e, speaker))
    return out


def build_segments(
    audio: np.ndarray,
    diar: Diarization,
    settings: Settings,
    asr: ASR,
    *,
    tools: OverlapTools | None = None,
    window: tuple[float, float] | None = None,
) -> list[Segment]:
    """Speaker-tagged ASR segments: clean slices, fallback slices, separated streams."""
    total_s = len(audio) / SR
    pending: list[tuple[np.ndarray, Segment]] = []

    def slice_of(s: float, e: float) -> np.ndarray:
        return audio[int(s * SR) : int(e * SR)]

    if not diar.exclusive_turns:
        logger.warning("No diarization turns — falling back to coarse 30s windows")
        spans = [(s, min(total_s, s + 30.0), "SPEAKER_00") for s in _window_starts(total_s, 30.0, 30.0)]
        for s, e, speaker in _clip(spans, window):
            pending.append((slice_of(s, e), Segment(s, e, "", speaker)))
    else:
        regions: list[Region] = []
        results = []
        if tools is not None:
            regions = find_overlap_regions(diar.turns, min_seconds=settings.overlap_min_seconds)
            if window is not None:
                regions = [r for r in regions if r.start >= window[0] and r.end <= window[1]]
            results = process_regions(audio, regions, diar.turns, settings, tools)
            log_summary(results)

        for s, e, speaker in _clip(clean_spans(diar.exclusive_turns, total_s, regions), window):
            pending.append((slice_of(s, e), Segment(s, e, "", speaker)))

        for res in results:
            r = res.region
            if res.reason is None:
                for speaker, stream in res.streams.items():
                    rank = 0 if speaker == res.lead else 1
                    pending.append(
                        (stream, Segment(r.start, r.end, "", speaker, True, r.group, rank))
                    )
                continue
            for s, e, speaker in diar.exclusive_turns:
                s, e = max(s, r.start), min(e, r.end)
                if e - s >= MIN_SLICE_SECONDS:
                    rank = 0 if speaker == res.lead else 1
                    pending.append((slice_of(s, e), Segment(s, e, "", speaker, True, r.group, rank)))

    texts = asr.transcribe([chunk for chunk, _ in pending])
    segments: list[Segment] = []
    for (_, seg), text in zip(pending, texts):
        if text:
            seg.text = text
            segments.append(seg)
    segments.sort(key=lambda seg: (seg.start, seg.rank))
    logger.info("ASR produced %d segments", len(segments))
    return segments


def merge_segments(segments: list[Segment], max_gap_seconds: float = 2.0) -> list[Segment]:
    """Relabel to Speaker A.. and merge consecutive same-speaker segments."""
    label_map: dict[str, str] = {}
    merged: list[Segment] = []
    for seg in segments:
        label = _speaker_label(seg.speaker, label_map)
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.speaker == label
            and prev.crosstalk == seg.crosstalk
            and prev.region == seg.region
            and seg.start - prev.end <= max_gap_seconds
        ):
            prev.text = f"{prev.text} {seg.text}".strip()
            prev.end = max(prev.end, seg.end)
        else:
            merged.append(
                Segment(seg.start, seg.end, seg.text, label, seg.crosstalk, seg.region, seg.rank)
            )
    return merged


def format_transcript(segments: list[Segment]) -> str:
    lines: list[str] = []
    for seg in segments:
        prefix = "(cez seba) " if seg.crosstalk else ""
        lines.append(f"{seg.speaker} [{_fmt_ts(seg.start)}]: {prefix}{seg.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def load_or_diarize(audio_path: Path, episode_id: str, settings: Settings) -> Diarization:
    cache = diarization_cache_path(settings.transcript_dir, episode_id)
    diar = load_diarization(cache)
    if diar is not None and diar.model == settings.diarization_model_id:
        logger.info("Loaded cached diarization (%d turns) from %s", len(diar.turns), cache)
        return diar
    diar = diarize(audio_path, settings)
    if diar.turns:
        save_diarization(cache, diar)
        logger.info("Cached diarization -> %s", cache)
    return diar


def run_transcription(audio_path: Path, episode_id: str, settings: Settings | None = None) -> Path:
    """Diarize + ASR + merge; write transcript file and return its path."""
    settings = settings or get_settings()
    settings.ensure_dirs()

    diar = load_or_diarize(audio_path, episode_id, settings)
    tools = None
    if settings.overlap_separation_enabled and diar.turns:
        tools = load_overlap_tools(settings)

    audio = load_audio(audio_path)
    asr = WhisperASR(settings)
    try:
        raw = build_segments(audio, diar, settings, asr, tools=tools)
    finally:
        asr.close()
    merged = merge_segments(raw)
    text = format_transcript(merged)

    out_path = settings.transcript_dir / f"{episode_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote transcript (%d lines) -> %s", len(merged), out_path)
    upload_file(out_path, object_name=f"transcripts/{out_path.name}", settings=settings)
    return out_path
