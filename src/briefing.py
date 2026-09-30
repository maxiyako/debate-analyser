"""Phase 0: political desk briefing researched before analysis.

The briefing gives the pipeline what a desk journalist knows before watching a
debate: who the speakers are, what role references mean on the debate date,
what the live disputes are and what is at stake, and what happened recently.

It is BACKGROUND ONLY. It shapes claim selection, claim context and query
construction; it must never be handed to a verdict-producing step as evidence.

Cycle rule: `src.agents` imports this module, and `src.validation` imports
`src.agents`. Do NOT import `src.validation` (or `src.agents`) at module level
here; import inside functions.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Side(str, Enum):
    COALITION = "coalition"
    OPPOSITION = "opposition"
    OTHER = "other"


class Participant(BaseModel):
    name: str
    party: str = ""
    office: str = Field(default="", description="Office held ON THE DEBATE DATE.")
    side: Side = Side.OTHER
    prior_roles: list[str] = Field(default_factory=list)
    dossiers: list[str] = Field(default_factory=list, description="Policy areas the person owns.")
    sources: list[str] = Field(default_factory=list)


class EntityEntry(BaseModel):
    reference: str = Field(description='Role reference as spoken, e.g. "prezident Poľska".')
    person: str
    valid_on: str = Field(default="", description="ISO date the mapping is valid for.")
    sources: list[str] = Field(default_factory=list)


class Dispute(BaseModel):
    title: str
    factual_question: str = Field(description="The checkable question underneath the dispute.")
    stakes: str = Field(
        default="",
        description="What a claim about this would mean for a voter's judgement (descriptive, not evaluative).",
    )
    sources: list[str] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    date: str = Field(description="ISO date YYYY-MM-DD.")
    event: str
    sources: list[str] = Field(default_factory=list)


class GlossaryEntry(BaseModel):
    term: str
    definition: str
    status: str = Field(default="", description="Status of the matter on the debate date.")
    sources: list[str] = Field(default_factory=list)


class DebateBriefing(BaseModel):
    debate_date: str = ""
    topic: str = ""
    participants: list[Participant] = Field(default_factory=list)
    entity_index: list[EntityEntry] = Field(default_factory=list)
    live_disputes: list[Dispute] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)


def filter_briefing(
    briefing: DebateBriefing, allowed_urls: set[str], debate_date
) -> tuple[DebateBriefing, list[str]]:
    """Enforce provenance and anachronism in code (the model's output is untrusted).

    - URLs not returned by a search this run, or on the disinfo/party blocklist,
      are stripped.
    - participants / entity_index / timeline / glossary items left with no
      source are dropped. Disputes are transcript-derived and exempt.
    - timeline events dated after the debate (or undated) are dropped.
    """
    from src.validation import _normalize_url, is_blocked_source

    allowed = {_normalize_url(u) for u in allowed_urls}
    notes: list[str] = []

    def clean(urls: list[str]) -> list[str]:
        out: list[str] = []
        for u in urls:
            if is_blocked_source(u) or _normalize_url(u) not in allowed or u in out:
                continue
            out.append(u)
        return out

    def keep(items, label_of, kind: str):
        kept = []
        for it in items:
            it = it.model_copy(update={"sources": clean(it.sources)})
            if not it.sources:
                notes.append(f'Briefing: dropped unsourced {kind} "{label_of(it)}"')
                continue
            kept.append(it)
        return kept

    participants = keep(briefing.participants, lambda p: p.name, "participant")
    entities = keep(briefing.entity_index, lambda e: e.reference, "entity mapping")
    glossary = keep(briefing.glossary, lambda g: g.term, "glossary term")

    from src.tools.page import parse_iso_date

    timeline = []
    for ev in keep(briefing.timeline, lambda t: t.event[:60], "timeline event"):
        d = parse_iso_date(ev.date)
        if d is None:
            notes.append(f'Briefing: dropped undated timeline event "{ev.event[:60]}"')
            continue
        if d > debate_date:
            notes.append(
                f'Briefing: dropped timeline event after debate date ({ev.date}): "{ev.event[:60]}"'
            )
            continue
        timeline.append(ev)

    disputes = [
        d.model_copy(update={"sources": clean(d.sources)}) for d in briefing.live_disputes
    ]
    return (
        briefing.model_copy(
            update={
                "participants": participants,
                "entity_index": entities,
                "glossary": glossary,
                "timeline": sorted(timeline, key=lambda t: t.date),
                "live_disputes": disputes,
            }
        ),
        notes,
    )


_SIDE_SK = {Side.COALITION: "koalícia", Side.OPPOSITION: "opozícia", Side.OTHER: "iné"}


def render_briefing(briefing: DebateBriefing, max_chars: int = 7000) -> str:
    """Compact Slovak rendering for the extraction prompt."""
    lines = [
        "POLITICKÝ KONTEXT (podklad na výber a pochopenie tvrdení — NIE dôkaz; "
        "nepoužívaj ho ako zdroj verdiktu):",
        f"Dátum debaty: {briefing.debate_date}. Téma: {briefing.topic}",
        "",
        "ÚČASTNÍCI:",
    ]
    for p in briefing.participants:
        bits = ", ".join(b for b in (p.party, p.office, _SIDE_SK[p.side]) if b)
        dossiers = f" — agendy: {', '.join(p.dossiers)}" if p.dossiers else ""
        lines.append(f"- {p.name} ({bits}){dossiers}")
    lines += ["", f"FUNKCIE k dátumu {briefing.debate_date}:"]
    for e in briefing.entity_index:
        lines.append(f'- "{e.reference}" = {e.person}')
    lines += ["", "SPORNÉ TÉMY (z nich odvodzuj dôležitosť tvrdení):"]
    for i, d in enumerate(briefing.live_disputes, 1):
        stakes = f"; v stávke: {d.stakes}" if d.stakes else ""
        lines.append(f"{i}. {d.title} — otázka: {d.factual_question}{stakes}")
    lines += ["", "UDALOSTI PRED DEBATOU:"]
    for t in briefing.timeline:
        lines.append(f"- {t.date}: {t.event}")
    lines += ["", "POJMY:"]
    for g in briefing.glossary:
        status = f" (stav: {g.status})" if g.status else ""
        lines.append(f"- {g.term}: {g.definition}{status}")

    out: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) + 1 > max_chars:
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)
