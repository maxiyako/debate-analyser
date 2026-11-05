"""Accusation fidelity gate: does a verdict refute what the speaker actually said?

Checkers see a paraphrased claim and search results, never the debate around
the quote, and each claim is checked apart from the others. So a verdict can
refute a stronger or different statement than the one made, or rest on not
finding a record. This gate re-reads every False/Misleading verdict with the
transcript around the quote and the debate's other verdicts, and may only
downgrade to Unverified — never upgrade, never change a True.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field

from src.agents import Verdict, VerifiedFact
from src.transcript_lines import Line, format_lines

_ACCUSATIONS = (Verdict.FALSE, Verdict.MISLEADING)

logger = logging.getLogger(__name__)


class Reading(str, Enum):
    SAME = "same"
    STRONGER = "stronger_reading"
    DIFFERENT = "different_referent"
    ABSENCE = "absence_only"
    NON_SEQUITUR = "invalid_inference"
    CONTEXT = "contradicted_by_context"


class FidelityOutput(BaseModel):
    asserted: str = Field(
        description="What the speaker asserted, read fairly in its debate context, in Slovak."
    )
    refuted: str = Field(
        description="What the rationale's evidence actually shows to be false, in Slovak."
    )
    reading: Reading
    issue: str = Field(
        default="",
        description="One short Slovak sentence naming the mismatch. Empty when reading is same.",
    )


_INSTRUCTIONS = """\
You audit one accusation in a fact-check of a Slovak political debate. You do
NOT re-check the facts and you do NOT use your own knowledge of them. Take the
rationale's findings as given and decide one thing: do they refute what the
speaker actually asserted?

First write `asserted`: the speaker's proposition read fairly, as a listener
in the studio would understand it from the transcript around the quote —
including what the speaker said earlier in the debate on the same topic. Keep
the speaker's own qualifiers and do not add new ones ("automaticky", "iba",
"všetci", "definitívne"). Then write `refuted`: what the rationale's evidence
shows to be false. Then pick `reading`:

- same: the evidence contradicts the asserted proposition itself.
- stronger_reading: it only contradicts a stronger, more literal or more
  absolute version than the one asserted.
- different_referent: it is about a different thing, event, period, actor or
  stage (e.g. production start instead of allocation, final vote instead of
  first reading, a ministry instead of the whole government).
- absence_only: it rests on not finding a record, figure, quote or vote.
- invalid_inference: the cited facts are true but do not contradict the
  assertion (e.g. "rose last month" against "historically lowest").
- contradicted_by_context: the transcript or another verdict of this debate
  shows the asserted part did happen or is true.

A READING RECORDED BEFORE CHECKING, when given, was written from the
transcript before any search and may be wrong: still write your own
`asserted`. When what the evidence refutes matches one of its "not
asserted" items, the reading is stronger_reading or different_referent.

Pick same whenever the evidence does hit the asserted proposition, even if the
speaker's wording is loose. A rounding or a slightly wrong figure the
rationale documents is same, not stronger_reading.
"""


def transcript_window(
    lines: list[Line], fact: VerifiedFact, before: int = 4, after: int = 4
) -> str:
    """Transcript lines around the fact's quoted line; empty when it cannot be located."""
    if not fact.timestamp:
        return ""
    for i, ln in enumerate(lines):
        if ln.speaker == fact.speaker and ln.ts == fact.timestamp:
            return format_lines(lines[max(0, i - before) : i + after + 1])
    return ""


def build_fidelity_prompt(
    fact: VerifiedFact, window: str, facts: list[VerifiedFact]
) -> str:
    siblings = "\n".join(
        f"- [{f.timestamp or '?'}] {f.speaker}: {f.claim} => {f.verdict.value}. "
        f"{f.rationale[:300]}"
        for f in facts
        if f is not fact
    )
    recorded = ""
    if fact.interpretation:
        recorded = (
            "READING RECORDED BEFORE CHECKING (may be wrong):\n"
            f"  interpretation: {fact.interpretation}\n"
            f"  not asserted: {'; '.join(fact.not_asserted) or '(none)'}\n"
        )
    return (
        f"{_INSTRUCTIONS}\n"
        f"TRANSCRIPT AROUND THE QUOTE:\n{window or '(unavailable)'}\n\n"
        f"SPEAKER: {fact.speaker}\n"
        f"QUOTE: {fact.quote}\n"
        f"CLAIM AS EXTRACTED: {fact.claim}\n"
        f"{recorded}"
        f"VERDICT: {fact.verdict.value}"
        f"{' / ' + fact.severity.value if fact.severity else ''}\n"
        f"RATIONALE: {fact.rationale[:1500]}\n\n"
        f"OTHER VERDICTS IN THIS DEBATE:\n{siblings or '(none)'}\n"
    )


def audit_accusations(
    facts: list[VerifiedFact],
    lines: list[Line],
    *,
    llm: Callable[[str], FidelityOutput],
    siblings: list[VerifiedFact] | None = None,
) -> list[str]:
    """Downgrade, in place, every accusation whose rationale misses the assertion."""
    notes: list[str] = []
    context = facts if siblings is None else siblings
    for fact in facts:
        if fact.verdict not in _ACCUSATIONS:
            continue
        prompt = build_fidelity_prompt(fact, transcript_window(lines, fact), context)
        try:
            result = llm(prompt)
        except Exception as exc:  # noqa: BLE001 - one failed audit must not sink the report
            logger.warning("Fidelity audit failed for %r: %s", fact.claim[:80], exc)
            notes.append(f'Fidelity audit failed, verdict kept: "{fact.claim[:120]}" ({exc})')
            continue
        if result.reading == Reading.SAME:
            continue
        notes.append(
            f"Fidelity downgraded {fact.verdict.value}->Unverified "
            f'({result.reading.value}): "{fact.claim[:120]}" — {result.issue}'
        )
        fact.verdict = Verdict.UNVERIFIED
        fact.severity = None
        fact.rationale = (
            f"{fact.rationale} [Overenie vyvracia iné tvrdenie, než rečník povedal: "
            f"{result.issue or result.refuted}]"
        ).strip()
    return notes


def default_fidelity_llm(settings) -> Callable[[str], FidelityOutput]:
    from src.llm import generate_json

    model = settings.judge_model or settings.gemini_model
    return lambda prompt: generate_json(
        prompt, FidelityOutput, settings, label="fidelity", model=model
    )
