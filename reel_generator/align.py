"""Word-level timings for dynamic subtitles.

Order for `--align auto`: WhisperX, then stable-ts, then an even split of the
transcript line across its timestamp span. The fallback needs no model download
and is what `--dry-run` uses when the raw video is absent.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from reel_generator.transcript import TranscriptLine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    speaker: str


def align_words(
    lines: list[TranscriptLine],
    window_start: float,
    window_end: float,
    *,
    method: str = "auto",
    media: Path | None = None,
    work_dir: Path | None = None,
    language: str = "sk",
    device: str = "cpu",
) -> list[Word]:
    selected = [
        line
        for line in lines_in_window(lines, window_start, window_end)
        if line.end - line.start >= 0.2
    ]
    if method == "proportional" or media is None or not media.is_file():
        if method not in {"auto", "proportional"}:
            raise FileNotFoundError(f"alignment media missing: {media}")
        return align_proportional(selected, window_start, window_end)
    wav = _extract_wav(media, window_start, window_end, work_dir or media.parent)
    if method in {"auto", "whisperx"}:
        try:
            return _align_whisperx(wav, selected, window_start, language, device)
        except Exception as exc:
            if method == "whisperx":
                raise
            logger.warning("WhisperX alignment unavailable (%s). Trying stable-ts.", exc)
    if method in {"auto", "stable-ts"}:
        try:
            return _align_stable(wav, selected, window_start, language)
        except Exception as exc:
            if method == "stable-ts":
                raise
            logger.warning("stable-ts alignment unavailable (%s). Using transcript timings.", exc)
    return align_proportional(selected, window_start, window_end)


def lines_in_window(
    lines: list[TranscriptLine], start: float, end: float
) -> list[TranscriptLine]:
    return [line for line in lines if line.end > start and line.start < end]


def align_proportional(
    lines: list[TranscriptLine], window_start: float, window_end: float
) -> list[Word]:
    words: list[Word] = []
    for line in lines:
        if line.end - line.start < 0.2:
            continue
        tokens = [token for token in line.text.split() if token.strip()]
        if not tokens:
            continue
        duration = line.end - line.start
        step = duration / len(tokens)
        for index, token in enumerate(tokens):
            start = line.start + index * step
            end = start + step
            if end <= window_start or start >= window_end:
                continue
            words.append(Word(token, start, end, line.speaker))
    return words


def _extract_wav(media: Path, start: float, end: float, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / "align.wav"
    duration = max(0.2, end - start)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(media),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav),
        ],
        check=True,
    )
    return wav


def _align_whisperx(
    wav: Path,
    lines: list[TranscriptLine],
    window_start: float,
    language: str,
    device: str,
) -> list[Word]:
    import whisperx

    audio = whisperx.load_audio(str(wav))
    model, metadata = whisperx.load_align_model(language_code=language, device=device)
    segments = [
        {
            "text": line.text,
            "start": max(0.0, line.start - window_start),
            "end": max(0.05, line.end - window_start),
        }
        for line in lines
    ]
    aligned = whisperx.align(
        segments, model, metadata, audio, device, return_char_alignments=False
    )
    words: list[Word] = []
    for segment in aligned.get("segments") or []:
        for token in segment.get("words") or []:
            if "start" not in token or "end" not in token:
                continue
            text = str(token.get("word") or "").strip()
            if not text:
                continue
            start = float(token["start"]) + window_start
            end = float(token["end"]) + window_start
            words.append(Word(text, start, end, _speaker_at(lines, start)))
    if not words:
        raise RuntimeError("WhisperX returned no words")
    return words


def _align_stable(
    wav: Path,
    lines: list[TranscriptLine],
    window_start: float,
    language: str,
) -> list[Word]:
    import stable_whisper

    model = stable_whisper.load_model("base")
    text = " ".join(line.text for line in lines)
    result = model.align(str(wav), text, language=language)
    words: list[Word] = []
    for segment in result.segments:
        for token in segment.words:
            text_token = str(token.word).strip()
            if not text_token:
                continue
            start = float(token.start) + window_start
            end = float(token.end) + window_start
            words.append(Word(text_token, start, end, _speaker_at(lines, start)))
    if not words:
        raise RuntimeError("stable-ts returned no words")
    return words


def _speaker_at(lines: list[TranscriptLine], moment: float) -> str:
    speaker = lines[0].speaker if lines else ""
    for line in lines:
        if line.start <= moment + 0.05:
            speaker = line.speaker
        else:
            break
    return speaker
