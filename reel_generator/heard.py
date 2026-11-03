"""Retiming captions from a transcript of the clip that will actually play.

Proportional timings follow the debate transcript, which starts each turn on a
coarse stamp. A reel that begins inside the first word then shows that word
while the audio is already on the next one. Hearing the clip and dropping any
word that started before the cut removes the shift.

Fast path is mlx-whisper. When that package is absent, the same word clock
comes from the local transformers Whisper (the analyzer model on disk). A word
timestamp often includes the pause before the voice, and silence gets
hallucinated tokens, so each word is trimmed to the samples that are actually loud.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from reel_generator.align import Word, _extract_wav
from reel_generator.timeline import ReelWindow, find_quote_span

logger = logging.getLogger(__name__)

HEARD_MODEL = "mlx-community/whisper-large-v3-turbo"
_TOKEN = re.compile(r"\w+", re.UNICODE)
_SPEECH_FLOOR = 0.02
_HF_LANGUAGE = {"sk": "slovak", "cs": "czech", "en": "english"}
_PIPE = None


def transcribe_span(
    media: Path,
    start: float,
    end: float,
    work_dir: Path,
    language: str = "sk",
) -> list[Word]:
    wav = _extract_wav(media, start, end, work_dir)
    try:
        payload = _mlx_transcribe(wav, language)
    except ImportError:
        logger.warning(
            "mlx_whisper is not installed. Hearing the clip with the local Whisper model."
        )
        payload = _transformers_transcribe(wav, language)
    return trim_to_speech(words_from_transcript(payload, start), wav, start)


def _mlx_transcribe(wav: Path, language: str) -> dict:
    import mlx_whisper

    return mlx_whisper.transcribe(
        str(wav),
        path_or_hf_repo=HEARD_MODEL,
        word_timestamps=True,
        language=language,
        verbose=False,
        condition_on_previous_text=False,
        temperature=0.0,
    )


def _local_whisper_id() -> str:
    env = os.environ.get("WHISPER_MODEL_ID", "").strip()
    if env:
        return env
    candidate = Path(__file__).resolve().parents[1] / "models" / "whisper-large-v3-sk"
    if candidate.is_dir():
        return str(candidate)
    return "kinit/whisper-large-v3-sk"


def _transformers_transcribe(wav: Path, language: str) -> dict:
    """Word timestamps from the Slovak Whisper already on disk.

    CPU on purpose. The same pass on MPS stalled once the buffer got past
    about 15 seconds, and a finished CPU pass is what the captions need.
    """
    global _PIPE
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if _PIPE is None:
        model_id = _local_whisper_id()
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=torch.float32, low_cpu_mem_usage=True
        )
        model.to("cpu")
        model.eval()
        _PIPE = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=torch.float32,
            device="cpu",
        )
        logger.warning(
            "Hearing clips with %s on CPU. Word timestamps take a few minutes per reel.",
            model_id,
        )
    hf_language = _HF_LANGUAGE.get(language, language)
    return _PIPE(
        str(wav),
        return_timestamps="word",
        generate_kwargs={"language": hf_language, "task": "transcribe"},
    )


def words_from_transcript(payload: dict, offset: float) -> list[Word]:
    """Absolute word times. Accepts mlx segments or a transformers chunk list."""
    words: list[Word] = []
    chunks = payload.get("chunks")
    if chunks:
        for chunk in chunks:
            stamp = chunk.get("timestamp") or (None, None)
            text = str(chunk.get("text") or "").strip()
            if not text or stamp[0] is None:
                continue
            start = float(stamp[0])
            end = float(stamp[1]) if stamp[1] is not None else start + 0.3
            if end <= start:
                end = start + 0.08
            words.append(Word(text, start + offset, end + offset, ""))
        return words
    for segment in payload.get("segments") or []:
        tokens = segment.get("words") or []
        if not tokens and segment.get("text"):
            tokens = [
                {
                    "word": segment["text"],
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                }
            ]
        for token in tokens:
            if "start" not in token or "end" not in token:
                continue
            text = str(token.get("word") or "").strip()
            if not text:
                continue
            words.append(
                Word(
                    text,
                    float(token["start"]) + offset,
                    float(token["end"]) + offset,
                    "",
                )
            )
    return words


def trim_to_speech(
    words: list[Word],
    wav: Path,
    origin: float,
    floor: float = _SPEECH_FLOOR,
) -> list[Word]:
    """Drop tokens that sit in silence, and start each word when the voice does.

    Whisper's first timestamp often covers the pause before the word, so the
    highlight lights up while nobody is talking. `origin` is the absolute time
    of sample 0 in `wav`.
    """
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(wav), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    hop = max(1, int(0.02 * sr))
    kept: list[Word] = []
    for word in words:
        a = int(round((word.start - origin) * sr))
        b = int(round((word.end - origin) * sr))
        a = max(0, min(len(audio), a))
        b = max(a, min(len(audio), b))
        span = audio[a:b]
        if span.size == 0:
            continue
        loud_from = None
        loud_to = None
        for index in range(0, len(span), hop):
            piece = span[index : index + hop]
            rms = float(np.sqrt(np.mean(np.square(piece))))
            if rms >= floor:
                if loud_from is None:
                    loud_from = index
                loud_to = index + len(piece)
        if loud_from is None or loud_to is None:
            continue
        start = word.start + loud_from / sr
        end = word.start + loud_to / sr
        if end - start < 0.04:
            end = start + 0.04
        kept.append(Word(word.text, start, end, word.speaker))
    return kept


def snap_window(
    reel: ReelWindow,
    quote: str,
    heard: list[Word],
    *,
    lead: float,
    tail: float,
    lock_end: bool,
    max_span: float = 24.0,
) -> ReelWindow:
    """Move the clip onto the words that were actually spoken.

    `lock_end` keeps a dodge running through the next turn. Only the start
    slides onto the heard question.
    """
    span = find_quote_span(heard, quote)
    if span is None:
        return reel
    spoken_at = heard[span[0]].start
    quote_words = max(8, len(_TOKEN.findall(quote)))
    last = min(len(heard) - 1, span[0] + quote_words - 1)
    spoken_end = heard[last].end
    start = max(0.0, spoken_at - lead)
    if lock_end:
        end = max(reel.end, spoken_end + tail)
        stamp = max(reel.stamp_at, spoken_at)
        if stamp < start:
            stamp = spoken_at
        flag_end = reel.flag_end if reel.flag_end > stamp else end
        return ReelWindow(start, end, stamp, flag_end)
    end = spoken_end + tail
    if end < spoken_at + 3.0:
        end = spoken_at + 3.0 + tail
    if end - start > max_span:
        end = start + max_span
        spoken_end = min(spoken_end, end)
    return ReelWindow(start, end, spoken_at, spoken_end)


def captions_from_heard(
    planned: list[Word],
    heard: list[Word],
    clip_start: float,
    clip_end: float,
) -> list[Word]:
    """Clock from the clip transcript. Spelling from the debate transcript when it matches.

    Words the clip never plays, including a first word cut off at the start, are omitted.
    """
    pool = [word for word in planned if clip_start - 2.0 <= word.start <= clip_end + 2.0]
    cursor = 0
    captions: list[Word] = []
    for spoken in heard:
        if spoken.start < clip_start - 0.05 or spoken.start >= clip_end:
            continue
        match = None
        for look in range(cursor, min(cursor + 6, len(pool))):
            if _same(pool[look].text, spoken.text):
                match = pool[look]
                cursor = look + 1
                break
        text = match.text if match is not None else spoken.text.strip()
        speaker = match.speaker if match is not None else _speaker_near(pool, spoken.start)
        captions.append(Word(text, spoken.start, spoken.end, speaker))
    return captions


def _same(left: str, right: str) -> bool:
    a = _norm(left)
    b = _norm(right)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and long.startswith(short)


def _norm(text: str) -> str:
    return "".join(_TOKEN.findall(text.casefold()))


def _speaker_near(words: list[Word], moment: float) -> str:
    speaker = ""
    for word in words:
        if word.start <= moment + 0.2:
            speaker = word.speaker
        else:
            break
    return speaker
