"""Question audit: which moderator questions demanded a concrete answer, and
whether the addressed guest gave one.

Candidates and answer spans are found in code; an LLM classifies them; code
validates every dodge against the answer text. Doubt resolves to `answered`.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from pydantic import BaseModel, Field

from src.report_models import QuestionItem, QuestionKind, QuestionOutcome, SpeakerMap
from src.transcript_lines import Line, strip_markers, ts_seconds

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


_MIN_EVIDENCE_CHARS = 15
_TS_PREFIX = re.compile(r"^\s*\[[\d:]+\]\s*")
_BALANCE_MIN_GAP = 4

_PROMPT = (
    "Classify each moderator question from a Slovak TV debate and judge the "
    "addressed guest's answer.\n"
    "kind:\n"
    "- challenging: a direct question on substance that requires a concrete "
    "answer: a position (yes/no), a number, responsibility, a deadline, or a "
    "confrontation with a contradiction or a fact.\n"
    "- open: a soft open invitation ('Ako to vidíte?').\n"
    "- procedural: running the debate ('Dokončíte?', 'Môžeme ísť ďalej?').\n"
    "- rhetorical: no answer expected.\n"
    "outcome (only for challenging):\n"
    "- answered: the answer addresses the core of the question, even if it is "
    "unpleasant, hedged, or combative.\n"
    "- partial: answers part of it but omits the core (the number, the yes/no, "
    "the responsibility).\n"
    "- dodged: switches topic, counter-attacks, or refuses without a "
    "substantive answer.\n"
    "- interrupted: the guest had no real chance to answer.\n"
    "A tough, uncomfortable, or critical answer is answered, not dodged. Be "
    "symmetric across coalition and opposition.\n"
    "For partial and dodged copy 1-2 verbatim excerpts from ANSWER that show "
    "the evasion into `evidence`. `reason`: one sentence in Slovak.\n"
)


class ClassifiedQuestion(BaseModel):
    id: int
    kind: QuestionKind
    outcome: QuestionOutcome | None = None
    evidence: list[str] = Field(default_factory=list)
    reason: str = ""


class QuestionClassification(BaseModel):
    items: list[ClassifiedQuestion] = Field(default_factory=list)


def build_question_prompt(items: list[QuestionItem]) -> str:
    blocks = [
        f"#{it.id} [{it.timestamp}] to {it.addressee}\n"
        f"QUESTION: {it.question}\n"
        f"ANSWER ({it.answer_words} words): {it.answer_text[:1500]}"
        for it in items
    ]
    return _PROMPT + "\n\n" + "\n\n".join(blocks)


def classify_questions(
    items: list[QuestionItem], *, llm: Callable[[str], QuestionClassification]
) -> list[str]:
    """Merge the LLM classification into `items`; every dodge must quote the answer."""
    from src.validation import normalize, quote_grounded

    notes: list[str] = []
    if not items:
        return notes
    by_id = {c.id: c for c in llm(build_question_prompt(items)).items}
    for it in items:
        c = by_id.get(it.id)
        if c is None:
            it.kind, it.outcome, it.reason = QuestionKind.OPEN, None, "neklasifikované"
            notes.append(f"Question audit: #{it.id} [{it.timestamp}] not classified; not counted")
            continue
        it.kind, it.reason = c.kind, c.reason
        if it.kind != QuestionKind.CHALLENGING:
            it.outcome = None
            continue
        if it.answer_words < MIN_ANSWER_WORDS:
            it.outcome = QuestionOutcome.INTERRUPTED
            continue
        outcome = c.outcome or QuestionOutcome.ANSWERED
        if outcome in (QuestionOutcome.PARTIAL, QuestionOutcome.DODGED):
            quotes = [_TS_PREFIX.sub("", ev) for ev in c.evidence]
            grounded = [
                q for q in quotes
                if len(normalize(q)) >= _MIN_EVIDENCE_CHARS and quote_grounded(q, it.answer_text)
            ]
            if not grounded:
                notes.append(
                    f"Question audit: #{it.id} [{it.timestamp}] {outcome.value} without a "
                    "grounded answer quote -> answered"
                )
                outcome = QuestionOutcome.ANSWERED
            else:
                it.evidence = [
                    f"[{it.timestamp}] otázka: {it.question[:200]}",
                    *(f"odpoveď: {q[:200]}" for q in grounded[:2]),
                ]
        it.outcome = outcome
    return notes


def question_counts(items: list[QuestionItem]) -> dict[str, int]:
    challenging = [i for i in items if i.kind == QuestionKind.CHALLENGING]
    return {
        "received": len(items),
        "challenging": len(challenging),
        "interrupted": sum(1 for i in challenging if i.outcome == QuestionOutcome.INTERRUPTED),
        "dodged": sum(1 for i in challenging if i.outcome == QuestionOutcome.DODGED),
        "partial": sum(1 for i in challenging if i.outcome == QuestionOutcome.PARTIAL),
    }


def render_dodges(items: list[QuestionItem]) -> list[str]:
    return [
        f"[{i.timestamp}] {i.question[:160]} → {i.outcome.value}: {i.reason}"
        for i in items
        if i.outcome in (QuestionOutcome.PARTIAL, QuestionOutcome.DODGED)
    ]


def question_balance_finding(items: list[QuestionItem], guests: list[str]) -> str | None:
    counts = {
        g: sum(1 for i in items if i.addressee == g and i.kind == QuestionKind.CHALLENGING)
        for g in guests
    }
    if len(counts) < 2:
        return None
    hi, lo = max(counts.values()), min(counts.values())
    if hi - lo >= _BALANCE_MIN_GAP and hi >= 2 * max(lo, 1):
        return "Nepomer podstatných otázok moderátora: " + ", ".join(
            f"{g}: {n}" for g, n in counts.items()
        )
    return None


def run_question_audit(
    lines: list[Line], smap: SpeakerMap, *, llm: Callable[[str], QuestionClassification]
) -> tuple[list[QuestionItem], list[str]]:
    moderator, guests = smap.moderator(), smap.guests()
    if not moderator or not guests:
        return [], ["Question audit skipped: no moderator or guests in the speaker map"]
    start_s = ts_seconds(smap.debate_start) if smap.debate_start else 0
    items = extract_question_candidates(lines, moderator, guests, start_s)
    try:
        notes = classify_questions(items, llm=llm)
    except Exception as exc:  # noqa: BLE001 - responsiveness becomes "no data"
        logger.warning("Question audit failed: %s", exc)
        return [], [f"Question audit failed: {exc}"[:300]]
    return items, notes


def default_question_llm(settings) -> Callable[[str], QuestionClassification]:
    from src.llm import generate_json

    return lambda prompt: generate_json(
        prompt, QuestionClassification, settings, label="question_audit"
    )
