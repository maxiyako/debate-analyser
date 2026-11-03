"""Turn a highlight plus aligned words into a reel window and a stamp time."""

from __future__ import annotations

import re
from dataclasses import dataclass

from reel_generator.align import Word
from reel_generator.config import ReelStyle
from reel_generator.highlight import Highlight
from reel_generator.transcript import TranscriptLine, speaker_match

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Question-shaped highlights: the stamp hits when the guest starts the dodge,
# and the clip keeps the tail of the question as the hook.
_QUESTION_KINDS = {"question_dodging", "question_audit"}


@dataclass(frozen=True)
class ReelWindow:
    start: float
    end: float
    stamp_at: float
    # Last moment the stamp should blink. 0 means the clip end.
    flag_end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.1, self.end - self.start)

    @property
    def stamp_offset(self) -> float:
        return max(0.0, self.stamp_at - self.start)


def search_bounds(highlight: Highlight, pad_after: float = 180.0) -> tuple[float, float]:
    return max(0.0, highlight.timestamp_s - 12.0), highlight.timestamp_s + pad_after


def plan_window(
    highlight: Highlight,
    lines: list[TranscriptLine],
    words: list[Word],
    style: ReelStyle,
    *,
    answer_seconds: float | None = None,
    lead_seconds: float | None = None,
    tail_seconds: float | None = None,
) -> ReelWindow:
    span = find_quote_span(words, highlight.quote)
    if highlight.kind in _QUESTION_KINDS and highlight.outcome in {"", "dodged", "partial", "interrupted"}:
        return _question_window(
            highlight,
            lines,
            words,
            span,
            answer_seconds if answer_seconds is not None else style.dodge_answer_s,
            lead_seconds if lead_seconds is not None else style.question_lead_s,
            tail_seconds if tail_seconds is not None else style.dodge_tail_s,
            style.dodge_max_s,
        )
    return _claim_window(
        highlight,
        words,
        span,
        style,
        lead_seconds if lead_seconds is not None else style.hook_lead_s,
        tail_seconds if tail_seconds is not None else style.claim_tail_s,
    )


def find_quote_span(words: list[Word], quote: str) -> tuple[int, int] | None:
    needles = _tokens(quote)[:12]
    if len(needles) < 3:
        return None
    hay = [_tokens(word.text) for word in words]
    flat: list[tuple[int, str]] = []
    for index, parts in enumerate(hay):
        for part in parts:
            flat.append((index, part))
    if len(flat) < 3:
        return None
    width = len(needles)
    best_score = 0.0
    best_at: tuple[int, int] | None = None
    for offset in range(0, len(flat) - 2):
        window = flat[offset : offset + width]
        if len(window) < 3:
            break
        score = _overlap([token for _, token in window], needles[: len(window)])
        if score > best_score:
            best_score = score
            best_at = (window[0][0], window[-1][0] + 1)
    if best_at is None or best_score < 0.55:
        return None
    return best_at


def slice_words(words: list[Word], start: float, end: float) -> list[Word]:
    sliced: list[Word] = []
    for word in words:
        # A word already underway at the cut plays as the first caption while
        # the audio is on the next word. Drop it. The clip lead keeps the real first word.
        if word.start < start - 0.08 or word.start >= end or word.end <= start:
            continue
        sliced.append(
            Word(
                text=word.text,
                start=max(0.0, word.start - start),
                end=max(0.04, min(word.end, end) - start),
                speaker=word.speaker,
            )
        )
    return sliced


def next_other_turn(
    lines: list[TranscriptLine], after: float, speaker: str
) -> TranscriptLine | None:
    """First real turn by someone else. A shared timestamp with no duration is not a turn."""
    for line in lines:
        if line.start < after + 0.8:
            continue
        if line.end - line.start < 1.0:
            continue
        if speaker_match(line.speaker, speaker):
            continue
        return line
    return None


def _question_window(
    highlight: Highlight,
    lines: list[TranscriptLine],
    words: list[Word],
    span: tuple[int, int] | None,
    answer_seconds: float,
    lead_seconds: float,
    tail_seconds: float,
    max_answer: float,
) -> ReelWindow:
    answer_at = _answer_start(highlight, lines, words, span)
    if span is not None:
        question_at = words[span[0]].start
    else:
        question_at = highlight.timestamp_s
    clip_start = max(0.0, question_at - lead_seconds)
    if clip_start > answer_at:
        clip_start = max(0.0, answer_at - lead_seconds)
    turn = next_other_turn(lines, answer_at, highlight.speaker)
    if turn is not None:
        flag_end = turn.start
    else:
        flag_end = answer_at + answer_seconds
    clip_end = min(flag_end + tail_seconds, answer_at + max_answer)
    if clip_end < answer_at + 8.0:
        clip_end = answer_at + min(answer_seconds, max_answer)
        flag_end = min(flag_end, clip_end)
    return ReelWindow(clip_start, clip_end, answer_at, flag_end)


def _claim_window(
    highlight: Highlight,
    words: list[Word],
    span: tuple[int, int] | None,
    style: ReelStyle,
    lead_seconds: float,
    tail_seconds: float,
) -> ReelWindow:
    if span is not None:
        stamp_at, quote_end = _flag_bounds(words, span, highlight.quote)
    else:
        stamp_at = highlight.timestamp_s
        quote_end = stamp_at + style.body_seconds
    clip_start = max(0.0, stamp_at - lead_seconds)
    clip_end = quote_end + tail_seconds
    if clip_end < stamp_at + 3.0:
        clip_end = stamp_at + 3.0 + tail_seconds
    if clip_end - clip_start > style.claim_max_s:
        clip_end = clip_start + style.claim_max_s
        quote_end = min(quote_end, clip_end)
    return ReelWindow(clip_start, clip_end, stamp_at, quote_end)


def _flag_bounds(words: list[Word], span: tuple[int, int], quote: str) -> tuple[float, float]:
    """First and last word of the flagged quote. The lead before this is not part of the flag."""
    start_i = span[0]
    speaker = words[start_i].speaker
    quote_words = max(4, len(_TOKEN_RE.findall(quote)))
    end_i = min(len(words) - 1, start_i + quote_words - 1)
    while end_i > start_i and not speaker_match(words[end_i].speaker, speaker):
        end_i -= 1
    while end_i + 1 < len(words) and speaker_match(words[end_i + 1].speaker, speaker):
        if words[end_i].text.endswith((".", "?", "!", "…")) and end_i >= span[1] - 1:
            break
        if words[end_i].end - words[start_i].start > 14.0:
            break
        end_i += 1
    return words[start_i].start, words[end_i].end


def _answer_start(
    highlight: Highlight,
    lines: list[TranscriptLine],
    words: list[Word],
    span: tuple[int, int] | None,
) -> float:
    after = words[span[1] - 1].end if span is not None else highlight.timestamp_s
    for word in words:
        if word.start + 0.05 >= after and speaker_match(word.speaker, highlight.speaker):
            return word.start
    for line in lines:
        if line.start + 0.05 >= highlight.timestamp_s and speaker_match(line.speaker, highlight.speaker):
            return line.start
    return highlight.timestamp_s


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _overlap(window: list[str], needles: list[str]) -> float:
    if not needles:
        return 0.0
    hits = sum(1 for left, right in zip(window, needles) if left == right)
    return hits / len(needles)
