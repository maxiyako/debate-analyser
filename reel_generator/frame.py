"""One Gemini call to pad a highlight: room before the sentence, room after it.

Question dodges also run until the next speaker takes a turn, so the clip
shows the question was not answered. Offline fallback is the ReelStyle pads.
The reel module does not import the analyzer.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from reel_generator.highlight import Highlight
from reel_generator.paths import REPO_ROOT
from reel_generator.transcript import TranscriptLine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameChoice:
    lead_seconds: float
    tail_seconds: float
    source: str  # "llm" | "fallback"


def frame_clip(
    highlight: Highlight,
    lines: list[TranscriptLine],
    *,
    fallback_lead: float,
    fallback_tail: float,
    next_turn: str = "",
) -> FrameChoice:
    fallback = FrameChoice(fallback_lead, fallback_tail, "fallback")
    excerpt = _excerpt(highlight, lines)
    if not excerpt:
        return fallback
    try:
        choice = _ask(highlight, excerpt, next_turn)
    except Exception as exc:
        logger.warning("Frame call failed (%s). Using lead %.1fs tail %.1fs.", exc, fallback_lead, fallback_tail)
        return fallback
    if highlight.kind == "question_dodging":
        lead = _clamp(choice.lead_seconds, 0.8, 1.8, fallback_lead)
    else:
        lead = _clamp(choice.lead_seconds, 2.8, 4.2, fallback_lead)
    return FrameChoice(
        lead,
        _clamp(choice.tail_seconds, 0.8, 2.2, fallback_tail),
        "llm",
    )


def _excerpt(highlight: Highlight, lines: list[TranscriptLine]) -> str:
    chunks: list[str] = []
    for line in lines:
        if line.end <= line.start:
            continue
        if line.start < highlight.timestamp_s - 20 or line.start > highlight.timestamp_s + 160:
            continue
        chunks.append(f"{line.speaker}: {line.text}")
    return "\n".join(chunks)[:2200]


def _ask(highlight: Highlight, excerpt: str, next_turn: str) -> FrameChoice:
    project, location, model = _vertex_env()
    if not project:
        raise RuntimeError("GCP_PROJECT_ID is not set")
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=project,
        location=location or "global",
        http_options=types.HttpOptions(timeout=60_000),
    )
    if highlight.kind == "question_dodging":
        task = (
            "A moderator asks a question and the guest does not answer. "
            "The clip already runs until the next speaker takes a turn. "
            "lead_seconds is 0.9 to 1.5, pad BEFORE the first word of the question. "
            "tail_seconds is 1.0 to 2.0, pad AFTER that next speaker has started. "
            f"Next turn: {next_turn or 'unknown'}\n"
        )
    else:
        task = (
            f"The clip shows one error, stamped {highlight.stamp_text}. "
            "lead_seconds is 3.0 to 4.0 of the words BEFORE the flagged sentence, "
            "so a viewer knows the topic. Still a short reel. Do not start mid-word. "
            "tail_seconds is 1.0 to 1.8 AFTER the last flagged word. "
            "The stamp blinks only while the flagged words are spoken, not during the lead or the tail.\n"
        )
    prompt = (
        "You edit a vertical political clip. Choose only the pads.\n"
        f"{task}"
        f"Quote: {highlight.quote}\n"
        f"Why it is flagged: {highlight.detail}\n"
        f"Transcript:\n{excerpt}\n"
        'Reply JSON only: {"lead_seconds": number, "tail_seconds": number}'
    )
    response = client.models.generate_content(
        model=model or "gemini-2.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "lead_seconds": {"type": "number"},
                    "tail_seconds": {"type": "number"},
                },
                "required": ["lead_seconds", "tail_seconds"],
            },
        ),
    )
    payload = getattr(response, "parsed", None)
    if not isinstance(payload, dict):
        payload = json.loads(response.text or "{}")
    return FrameChoice(float(payload["lead_seconds"]), float(payload["tail_seconds"]), "llm")


def _vertex_env() -> tuple[str, str, str]:
    _load_dotenv(REPO_ROOT / ".env")
    return (
        os.environ.get("GCP_PROJECT_ID", ""),
        os.environ.get("GCP_LOCATION", "global"),
        os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"),
    )


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _clamp(value: float, low: float, high: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(high, max(low, number))
