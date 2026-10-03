"""Question audit: which moderator questions demanded a concrete answer, and
whether the addressed guest gave one.

Candidates and answer spans are found in code; an LLM classifies them; code
validates every dodge against the answer text. Doubt resolves to `answered`.
"""

from __future__ import annotations

import logging
import re

from src.report_models import QuestionItem
from src.transcript_lines import Line, strip_markers

logger = logging.getLogger(__name__)

MIN_QUESTION_WORDS = 4
MIN_ANSWER_WORDS = 15
_QUESTION = re.compile(r"[^.!?]*\?")
_VOCATIVE = re.compile(r"\b(?:pán|pani|pane)\s+([^\W\d_]+)", re.IGNORECASE)


def _questions(text: str) -> list[str]:
    return [
        q.strip()
        for q in _QUESTION.findall(strip_markers(text))
        if len(q.split()) >= MIN_QUESTION_WORDS
    ]


def _addressee(text: str, guests: list[str]) -> str | None:
    """The guest addressed by name ('pán Viskupič'); the last such vocative wins."""
    for m in reversed(list(_VOCATIVE.finditer(text))):
        stem = m.group(1).casefold()[:5]
        if len(stem) < 4:
            continue
        hits = [g for g in guests if any(t.casefold().startswith(stem) for t in g.split())]
        if len(hits) == 1:
            return hits[0]
    return None


def extract_question_candidates(
    lines: list[Line], moderator: str, guests: list[str], start_s: int = 0
) -> list[QuestionItem]:
    """One candidate per moderator turn (consecutive moderator lines) that
    contains a real question; the last question of the turn is the one asked."""
    items: list[QuestionItem] = []
    i = 0
    while i < len(lines):
        if lines[i].speaker != moderator:
            i += 1
            continue
        end = i
        while end < len(lines) and lines[end].speaker == moderator:
            end += 1
        turn = lines[i:end]
        i = end
        asked = [ln for ln in turn if _questions(ln.text)]
        if not asked or asked[0].seconds < start_s:
            continue
        question = _questions(asked[-1].text)[-1]
        addressee = next(
            (a for ln in reversed(turn) if (a := _addressee(ln.text, guests))), None
        )
        if addressee is None:
            for nl in lines[end:]:
                if nl.speaker == moderator and _questions(nl.text):
                    break
                if nl.speaker in guests:
                    addressee = nl.speaker
                    break
        if addressee is None:
            continue
        answer: list[str] = []
        for nl in lines[end:]:
            if nl.speaker == moderator and _questions(nl.text):
                break
            if nl.speaker == addressee:
                answer.append(strip_markers(nl.text))
        text = " ".join(a for a in answer if a)
        items.append(
            QuestionItem(
                id=len(items) + 1,
                timestamp=asked[0].ts,
                addressee=addressee,
                question=question[:600],
                answer_words=len(text.split()),
                answer_text=text,
            )
        )
    return items
