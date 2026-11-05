"""Claim interpretation (Phase A2): what exactly did the speaker assert?

Extraction rewords claims and checkers never see the transcript, so a verdict
can refute a stronger or different statement than the one made. One batched
pass reads each selected claim against the transcript around its quote and
records the asserted proposition, drift from the quote, resolved referents,
readings the speaker did not make, and what evidence would refute it. Code
guards decide which readings reach the checkers.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel, Field

from src.agents import ExtractedClaim, Referent, ReferentOrigin
from src.briefing import DebateBriefing
from src.transcript_lines import Line, format_lines
from src.validation import DEFAULT_MIN_RATIO, best_window

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger(__name__)

_PROBE_WORDS = 12
_DIGITS = re.compile(r"\d+(?:[.,]\d+)?")


class DriftKind(str, Enum):
    ADDED_QUALIFIER = "added_qualifier"
    DROPPED_QUALIFIER = "dropped_qualifier"
    WRONG_REFERENT = "wrong_referent"
    CHANGED_METRIC = "changed_metric"
    CHANGED_PERIOD = "changed_period"


class Drift(BaseModel):
    kind: DriftKind
    detail: str = Field(default="", description="One Slovak sentence naming the drift.")


class ClaimReading(BaseModel):
    claim_id: int = Field(description="Must match the claim id.")
    interpretation: str = Field(
        default="",
        description="The exact proposition the speaker asserted, in Slovak, with the "
        "speaker's qualifiers kept and referents resolved; nothing added.",
    )
    drift: list[Drift] = Field(
        default_factory=list,
        description="Every difference between the extracted claim and the quote/transcript.",
    )
    claim: str = Field(
        default="",
        description="Faithful Slovak claim text; required when drift is not empty.",
    )
    referents: list[Referent] = Field(default_factory=list)
    not_asserted: list[str] = Field(
        default_factory=list,
        description="Stronger or different readings a checker might test that the speaker did NOT make.",
    )
    refutation_would_be: str = Field(
        default="",
        description="What evidence would refute the interpretation: metric, period, actor, stage.",
    )
    related_claim_ids: list[int] = Field(
        default_factory=list,
        description="Ids of other listed claims about the same fact or dispute.",
    )


class ClaimReadings(BaseModel):
    readings: list[ClaimReading] = Field(default_factory=list)


def locate_window(lines: list[Line], claim: ExtractedClaim, size: int) -> str:
    """Transcript lines around the line where the quote starts; empty when not found."""
    probe = " ".join((claim.quote or "").split()[:_PROBE_WORDS])
    if not probe:
        return ""
    own = [i for i, ln in enumerate(lines) if ln.speaker == claim.speaker]
    for pool in (own, range(len(lines))):
        best_i, best = -1, 0.0
        for i in pool:
            score, _ = best_window(probe, lines[i].text)
            if score > best:
                best, best_i = score, i
        if best_i >= 0 and best >= DEFAULT_MIN_RATIO:
            return format_lines(lines[max(0, best_i - size) : best_i + size + 1])
    return ""


def referent_context(briefing: DebateBriefing | None) -> str:
    """Who holds which office, role references and terms: enough to resolve referents, no events."""
    if briefing is None:
        return ""
    out = ["ÚČASTNÍCI A FUNKCIE k dátumu debaty (iba na určenie, koho alebo čo rečník myslí; NIE dôkaz):"]
    for p in briefing.participants:
        bits = ", ".join(b for b in (p.party, p.office) if b)
        out.append(f"- {p.name}" + (f" ({bits})" if bits else ""))
    for e in briefing.entity_index:
        out.append(f'- "{e.reference}" = {e.person}')
    if briefing.glossary:
        out.append("POJMY:")
        out.extend(f"- {g.term}: {g.definition}" for g in briefing.glossary)
    return "\n".join(out)


def _norm_number(tok: str) -> str:
    tok = tok.replace(",", ".")
    return tok if "." in tok else str(int(tok))


def digit_tokens(text: str) -> set[str]:
    return {_norm_number(m.group()) for m in _DIGITS.finditer(text or "")}


def unsupported_numbers(
    reading: ClaimReading,
    claim: ExtractedClaim,
    window: str,
    ref_context: str,
    debate_date: date | None,
) -> list[str]:
    """Digit tokens in the reading that neither the transcript, the extracted claim nor a briefing referent backs."""
    allowed = (
        digit_tokens(claim.quote)
        | digit_tokens(claim.claim)
        | digit_tokens(window)
        | digit_tokens(claim.time_window)
        | digit_tokens(debate_date.isoformat() if debate_date else "")
    )
    briefing_digits = digit_tokens(ref_context)
    for r in reading.referents:
        toks = digit_tokens(r.resolved)
        if r.origin == ReferentOrigin.BRIEFING and toks <= briefing_digits:
            allowed |= toks
    texts = [
        reading.interpretation,
        reading.claim,
        reading.refutation_would_be,
        *reading.not_asserted,
        *(r.resolved for r in reading.referents),
    ]
    bad: set[str] = set()
    for text in texts:
        bad |= digit_tokens(text) - allowed
    return sorted(bad)


def apply_readings(
    claims: list[ExtractedClaim],
    readings: list[ClaimReading],
    windows: dict[int, str],
    ref_context: str,
    debate_date: date | None = None,
) -> list[str]:
    """Copy accepted readings onto the claims, in place; returns notes."""
    notes: list[str] = []
    by_id = {c.id: c for c in claims}
    handled: set[int] = set()
    for r in readings:
        c = by_id.get(r.claim_id)
        if c is None:
            notes.append(f"Interpretation: ignored reading for unknown claim {r.claim_id}.")
            continue
        if r.claim_id in handled:
            notes.append(f"Interpretation: ignored duplicate reading for claim {r.claim_id}.")
            continue
        handled.add(r.claim_id)
        bad = unsupported_numbers(r, c, windows.get(c.id, ""), ref_context, debate_date)
        if bad:
            notes.append(
                f"Interpretation rejected for claim {c.id}: numbers not in quote, "
                f"transcript or briefing referents ({', '.join(bad)})."
            )
            continue
        if not r.interpretation.strip():
            notes.append(f"Interpretation: empty reading for claim {c.id}; checked as extracted.")
            continue
        c.interpretation = r.interpretation.strip()
        c.not_asserted = [s.strip() for s in r.not_asserted if s.strip()]
        c.refutation_would_be = r.refutation_would_be.strip()
        c.referents = list(r.referents)
        c.related_claim_ids = sorted({i for i in r.related_claim_ids if i in by_id and i != c.id})
        if not r.drift:
            continue
        kinds = "; ".join(f"{d.kind.value}: {d.detail}" for d in r.drift)
        if r.claim.strip():
            c.claim_extracted = c.claim
            c.claim = r.claim.strip()
            notes.append(
                f'Interpretation rewrote claim {c.id} ({kinds}): '
                f'"{c.claim_extracted[:100]}" -> "{c.claim[:100]}"'
            )
        else:
            notes.append(
                f"Interpretation found drift in claim {c.id} but gave no claim text; "
                f"claim kept ({kinds})."
            )
    for c in claims:
        if c.id not in handled:
            notes.append(f"Interpretation: no reading for claim {c.id}; checked as extracted.")
    return notes


def clear_interpretation(claims: list[ExtractedClaim], *, restore_claim: bool = False) -> None:
    """Blank the code-owned reading fields; optionally undo a claim rewrite."""
    for c in claims:
        if restore_claim and c.claim_extracted:
            c.claim = c.claim_extracted
        c.claim_extracted = ""
        c.interpretation = ""
        c.not_asserted = []
        c.refutation_would_be = ""
        c.referents = []
        c.related_claim_ids = []


_INSTRUCTIONS = """\
You record what speakers in a Slovak political debate asserted, before anyone
fact-checks them. You do NOT check facts, you do NOT say whether a claim is
true, and you do NOT use your own knowledge of the facts.

For every CLAIM below, read the QUOTE as a listener in the studio would
understand it from the TRANSCRIPT AROUND THE QUOTE and from what the speaker
said in the other listed claims. Return one reading per claim:

- interpretation: the exact proposition the speaker asserted, in Slovak. Keep
  the speaker's own qualifiers and scope ("pre živnostníkov", "za posledné 3
  roky", "my", "približne"). Add none ("automaticky", "iba", "všetci",
  "definitívne", "úplne"). Resolve who or what "my", "oni", "to", "vtedy",
  role names and similar words refer to.
- drift: compare the EXTRACTED CLAIM with the quote and the transcript. List
  every added qualifier, dropped qualifier, wrong referent, changed metric
  (e.g. production start instead of allocation, ministry instead of
  government) and changed period. Empty when the claim is faithful.
- claim: when drift is not empty, a faithful one-sentence Slovak claim text,
  the proposition in the speaker's own voice. No attribution frame ("Podľa
  vyjadrenia ...", "X tvrdí, že ...") and no quotation marks; the speaker's
  hedges and uncertainty ("myslím", "pokiaľ si pamätám") belong in
  interpretation, not in claim.
- referents: each resolution as {text, resolved, origin}. origin is
  "transcript" when the transcript settles it, "briefing" when only the
  PARTICIPANTS AND OFFICES list does. Use that list for nothing else.
- not_asserted: stronger or different readings a checker might wrongly test,
  which the speaker did NOT make (e.g. full abolition when the speaker said an
  exemption; the packages' own total when the speaker spoke of debt growth).
- refutation_would_be: what evidence would show the interpretation false:
  the metric, period, actor and stage it must be about. Describe the kind of
  evidence; do not state any fact or figure you know.
- related_claim_ids: ids of other listed claims about the same fact, vote,
  figure or dispute.

Write no number, date or figure that is not in the quote, the extracted
claim, the transcript around it, the claim's time window, or the
PARTICIPANTS AND OFFICES list.
"""


def _claim_block(c: ExtractedClaim, window: str) -> str:
    time = ""
    if c.time_reference or c.time_window:
        time = f"TIME: {c.time_reference} -> {c.time_window}\n"
    return (
        f"CLAIM {c.id} — {c.speaker}\n"
        f"EXTRACTED CLAIM: {c.claim}\n"
        f"QUOTE: {c.quote}\n"
        f"{time}"
        f"TRANSCRIPT AROUND THE QUOTE:\n{window or '(unavailable)'}\n"
    )


def build_interpretation_prompt(
    chunk: list[ExtractedClaim],
    windows: dict[int, str],
    siblings: list[ExtractedClaim],
    ref_context: str,
    debate_date: date | None,
) -> str:
    sibs = "\n".join(f"- #{c.id} {c.speaker}: {c.claim}" for c in siblings)
    ref = f"DEBATE DATE: {debate_date.isoformat()}\n\n" if debate_date else ""
    return (
        f"{_INSTRUCTIONS}\n"
        f"{ref}"
        f"PARTICIPANTS AND OFFICES:\n{ref_context or '(none)'}\n\n"
        f"ALL CLAIMS OF THIS DEBATE:\n{sibs}\n\n"
        + "\n".join(_claim_block(c, windows.get(c.id, "")) for c in chunk)
    )


def interpret_claims(
    claims: list[ExtractedClaim],
    lines: list[Line],
    *,
    llm: Callable[[str], ClaimReadings],
    briefing: DebateBriefing | None = None,
    debate_date: date | None = None,
    window: int = 6,
    chunk: int = 20,
) -> list[str]:
    """Read every claim in chunks and apply the accepted readings in place."""
    ref_context = referent_context(briefing)
    windows = {c.id: locate_window(lines, c, window) for c in claims}
    readings: list[ClaimReading] = []
    notes: list[str] = []
    for start in range(0, len(claims), max(1, chunk)):
        part = claims[start : start + max(1, chunk)]
        prompt = build_interpretation_prompt(part, windows, claims, ref_context, debate_date)
        try:
            readings.extend(llm(prompt).readings)
        except Exception as exc:  # noqa: BLE001 - one failed chunk must not sink the rest
            logger.warning("Interpretation failed for claims %d-%d: %s", part[0].id, part[-1].id, exc)
            notes.append(
                f"Interpretation call failed for claims {part[0].id}-{part[-1].id} ({exc}); "
                "they are checked as extracted."
            )
    notes.extend(apply_readings(claims, readings, windows, ref_context, debate_date))
    return notes


def run_interpretation(
    claims: list[ExtractedClaim],
    lines: list[Line],
    settings: "Settings",
    *,
    briefing: DebateBriefing | None = None,
    debate_date: date | None = None,
    llm: Callable[[str], ClaimReadings] | None = None,
) -> list[str]:
    """Phase A2 entry point: flag-guarded, and never fails the run."""
    if not settings.interpretation_enabled or not claims:
        return []
    try:
        return interpret_claims(
            claims,
            lines,
            llm=llm or default_interpretation_llm(settings),
            briefing=briefing,
            debate_date=debate_date,
            window=settings.interpretation_window,
            chunk=settings.interpretation_chunk,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Interpretation failed; checking claims as extracted")
        clear_interpretation(claims, restore_claim=True)
        return [f"Interpretation failed ({exc}); claims checked as extracted."]


def default_interpretation_llm(settings: "Settings") -> Callable[[str], ClaimReadings]:
    from src.llm import generate_json

    model = settings.judge_model or settings.gemini_model
    return lambda prompt: generate_json(
        prompt, ClaimReadings, settings, label="interpretation", model=model
    )
