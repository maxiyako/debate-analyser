"""Speaker diarization + Slovak Whisper ASR, merged into tagged transcript lines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from config import Settings, get_settings
from src.gcs import upload_file

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"
    crosstalk: bool = False  # another speaker substantially overlaps this segment


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


def diarize(audio_path: Path, settings: Settings) -> list[tuple[float, float, str]]:
    """Return list of (start, end, speaker_id) from Pyannote."""
    if not settings.huggingface_token:
        msg = (
            "HUGGINGFACE_TOKEN missing — diarization cannot run, so the whole "
            "transcript would collapse to a single speaker. Accept terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 and set the token."
        )
        if settings.require_diarization:
            raise RuntimeError(msg)
        logger.warning("%s Falling back to a single speaker (Speaker A).", msg)
        return []

    from pyannote.audio import Pipeline

    logger.info("Loading diarization model %s", settings.diarization_model_id)
    try:
        pipeline = Pipeline.from_pretrained(
            settings.diarization_model_id,
            token=settings.huggingface_token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            settings.diarization_model_id,
            use_auth_token=settings.huggingface_token,
        )

    import torch

    device = _resolve_device(settings)
    pipeline.to(torch.device(device))

    logger.info("Running diarization on %s", audio_path)
    output = pipeline(str(audio_path))

    # pyannote.audio >= 4 returns DiarizeOutput; older versions return Annotation.
    annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(
        output, "speaker_diarization", None
    )
    if annotation is None:
        annotation = output  # Annotation (legacy)

    turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    logger.info("Found %d speaker turns", len(turns))
    return turns


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


def transcribe(
    audio_path: Path,
    settings: Settings,
    *,
    diar_turns: list[tuple[float, float, str]] | None = None,
    batch_size: int | None = None,
) -> list[tuple[float, float, str]]:
    """Diarization-guided ASR: transcribe each speaker turn's audio slice.

    This model does not reliably emit Whisper timestamp tokens, and word-level
    timestamps via cross-attention DTW (return_token_timestamps) are ~7x slower
    on MPS. Instead we transcribe the audio slice of each (merged) diarization
    turn directly, so speaker attribution is exact by construction. Slices are
    batched through a single plain generate() call for throughput. When no
    diarization is available we fall back to coarse fixed 30s windows.
    """
    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = _resolve_device(settings)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = settings.whisper_model_id

    logger.info("Loading Whisper model %s on %s (dtype=%s)", model_id, device, dtype)
    load_kwargs: dict = {}
    if settings.huggingface_token and not Path(model_id).exists():
        load_kwargs["token"] = settings.huggingface_token

    processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )
    model.to(device)
    model.eval()

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f"Expected 16 kHz WAV, got {sr} Hz — re-extract audio")
    total_s = len(audio) / sr

    if diar_turns:
        spans: list[tuple[float, float]] = []
        for start, end, _spk in _merge_turns(diar_turns):
            spans.extend(_chunk_spans(start, min(end, total_s)))
    else:
        logger.warning("No diarization turns — falling back to coarse 30s windows")
        spans = [(s, min(total_s, s + 30.0)) for s in _window_starts(total_s, 30.0, 30.0)]

    spans = [(s, min(e, total_s)) for s, e in spans if (e - s) >= 0.3]
    bs = batch_size or settings.asr_batch_size
    logger.info(
        "Transcribing %.1f min audio in %d turn slices (batch %d)",
        total_s / 60.0,
        len(spans),
        bs,
    )

    segments: list[tuple[float, float, str]] = []
    for b in range(0, len(spans), bs):
        batch = spans[b : b + bs]
        chunks = [audio[int(s * sr) : int(e * sr)] for s, e in batch]
        inputs = processor(
            chunks, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)
        attention_mask = inputs.attention_mask.to(device=device)
        with torch.inference_mode():
            predicted = model.generate(
                input_features,
                attention_mask=attention_mask,
                language="slovak",
                task="transcribe",
                no_repeat_ngram_size=3,
            )
        texts = processor.batch_decode(predicted, skip_special_tokens=True)
        for (s, e), text in zip(batch, texts):
            text = text.strip()
            if text:
                segments.append((s, e, text))

        done = min(b + bs, len(spans))
        if done % (bs * 5) == 0 or done == len(spans):
            logger.info(
                "ASR progress %d/%d (%.0f%%), %d segments so far",
                done,
                len(spans),
                100.0 * done / len(spans),
                len(segments),
            )

    del model
    if device == "mps":
        torch.mps.empty_cache()

    segments.sort(key=lambda s: s[0])
    logger.info("ASR produced %d segments", len(segments))
    return segments


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _assign_speaker(
    start: float,
    end: float,
    diar_turns: list[tuple[float, float, str]],
    crosstalk_min: float,
) -> tuple[str, bool]:
    """Dominant diarization speaker for a segment + whether a 2nd speaker overlaps it."""
    per_speaker: dict[str, float] = {}
    for d_start, d_end, speaker in diar_turns:
        ov = _overlap(start, end, d_start, d_end)
        if ov > 0.0:
            per_speaker[speaker] = per_speaker.get(speaker, 0.0) + ov
    if not per_speaker:
        return "SPEAKER_00", False
    ordered = sorted(per_speaker.items(), key=lambda kv: -kv[1])
    best_speaker = ordered[0][0]
    second_ov = ordered[1][1] if len(ordered) > 1 else 0.0
    duration = max(1e-6, end - start)
    return best_speaker, (second_ov / duration) >= crosstalk_min


def merge_diarization_and_asr(
    asr_segments: list[tuple[float, float, str]],
    diar_turns: list[tuple[float, float, str]],
    *,
    crosstalk_min: float = 0.35,
    max_gap_seconds: float = 2.0,
) -> list[Segment]:
    """Assign a speaker per ASR segment, then merge consecutive same-speaker turns."""
    label_map: dict[str, str] = {}
    merged: list[Segment] = []

    for start, end, text in asr_segments:
        raw_speaker, crosstalk = _assign_speaker(start, end, diar_turns, crosstalk_min)
        label = _speaker_label(raw_speaker, label_map)
        if (
            merged
            and merged[-1].speaker == label
            and start - merged[-1].end <= max_gap_seconds
        ):
            prev = merged[-1]
            prev.text = f"{prev.text} {text}".strip()
            prev.end = end
            prev.crosstalk = prev.crosstalk or crosstalk
        else:
            merged.append(
                Segment(
                    start=start,
                    end=end,
                    text=text,
                    speaker=label,
                    crosstalk=crosstalk,
                )
            )
    return merged


def format_transcript(segments: list[Segment]) -> str:
    lines: list[str] = []
    for seg in segments:
        prefix = "(cez seba) " if seg.crosstalk else ""
        lines.append(f"{seg.speaker} [{_fmt_ts(seg.start)}]: {prefix}{seg.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def run_transcription(audio_path: Path, episode_id: str, settings: Settings | None = None) -> Path:
    """Diarize + ASR + merge; write transcript file and return its path."""
    import json

    settings = settings or get_settings()
    settings.ensure_dirs()

    diar_cache = settings.transcript_dir / f"{episode_id}.diarization.json"
    if diar_cache.exists():
        raw = json.loads(diar_cache.read_text(encoding="utf-8"))
        diar_turns = [(float(a), float(b), str(s)) for a, b, s in raw]
        logger.info("Loaded cached diarization (%d turns) from %s", len(diar_turns), diar_cache)
    else:
        diar_turns = diarize(audio_path, settings)
        diar_cache.write_text(json.dumps(diar_turns), encoding="utf-8")
        logger.info("Cached diarization -> %s", diar_cache)

    asr_segments = transcribe(audio_path, settings, diar_turns=diar_turns)
    merged = merge_diarization_and_asr(
        asr_segments, diar_turns, crosstalk_min=settings.crosstalk_min_overlap
    )
    text = format_transcript(merged)

    out_path = settings.transcript_dir / f"{episode_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote transcript (%d lines) -> %s", len(merged), out_path)
    upload_file(out_path, object_name=f"transcripts/{out_path.name}", settings=settings)
    return out_path