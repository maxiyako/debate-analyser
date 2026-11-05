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

from pydantic import BaseModel, Field

from src.agents import ExtractedClaim, Referent, ReferentOrigin
from src.briefing import DebateBriefing
from src.transcript_lines import Line, format_lines
from src.validation import DEFAULT_MIN_RATIO, best_window

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
    """Digit tokens in the reading that neither the transcript nor a briefing referent backs."""
    allowed = (
        digit_tokens(claim.quote)
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
