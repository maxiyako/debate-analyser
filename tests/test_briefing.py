"""Offline tests for the political briefing (src/briefing.py)."""

from __future__ import annotations

from datetime import date

from src.briefing import (
    DebateBriefing,
    Dispute,
    EntityEntry,
    GlossaryEntry,
    Participant,
    Side,
    TimelineEvent,
    filter_briefing,
    render_briefing,
)

DEBATE = date(2026, 4, 12)
GOOD = "https://www.sme.sk/c/1/clanok.html"
OTHER = "https://dennikn.sk/2/clanok/"
FAKE = "https://www.sme.sk/c/999/vymysleny.html"
PARTY = "https://www.smer.sk/clanok"


def make_briefing() -> DebateBriefing:
    return DebateBriefing(
        debate_date="2026-04-12",
        topic="Maďarské voľby",
        participants=[
            Participant(name="Ján Novák", party="Smer", office="poslanec", side=Side.COALITION, sources=[GOOD]),
            Participant(name="Fiktívny Hosť", party="X", sources=[FAKE]),  # only fabricated source
        ],
        entity_index=[
            EntityEntry(reference="predseda vlády", person="Robert Fico", valid_on="2026-04-12", sources=[GOOD, PARTY]),
        ],
        live_disputes=[
            Dispute(title="Eurofondy", factual_question="Boli fondy pozastavené?", stakes="Dopad na rozpočet"),
        ],
        timeline=[
            TimelineEvent(date="2026-03-01", event="Komisia zmrazila fondy", sources=[OTHER]),
            TimelineEvent(date="2026-05-01", event="Udalosť po debate", sources=[OTHER]),
            TimelineEvent(date="2026-02-01", event="Bez zdroja", sources=[]),
        ],
        glossary=[GlossaryEntry(term="konsolidácia", definition="Znižovanie deficitu", sources=[GOOD])],
    )


def test_filter_drops_anachronistic_unsourced_and_fabricated() -> None:
    out, notes = filter_briefing(make_briefing(), {GOOD, OTHER}, DEBATE)
    assert [p.name for p in out.participants] == ["Ján Novák"]
    assert [t.event for t in out.timeline] == ["Komisia zmrazila fondy"]
    assert out.entity_index[0].sources == [GOOD]  # party URL stripped
    assert len(out.glossary) == 1
    assert len(out.live_disputes) == 1  # disputes are exempt from the source rule
    joined = " | ".join(notes)
    assert "after debate date" in joined
    assert "Fiktívny Hosť" in joined


def test_filter_keeps_undated_timeline_out() -> None:
    b = make_briefing()
    b.timeline = [TimelineEvent(date="", event="Neurčený dátum", sources=[GOOD])]
    out, _ = filter_briefing(b, {GOOD}, DEBATE)
    assert out.timeline == []  # a dateless event cannot be shown to precede the debate


def test_render_contains_sections_and_is_background_labelled() -> None:
    out, _ = filter_briefing(make_briefing(), {GOOD, OTHER}, DEBATE)
    text = render_briefing(out)
    assert "podklad" in text.lower()
    assert "Ján Novák" in text and "koalícia" in text
    assert '"predseda vlády" = Robert Fico' in text
    assert "Eurofondy" in text and "Boli fondy pozastavené?" in text
    assert "2026-03-01" in text
    assert "konsolidácia" in text


def test_render_respects_max_chars_on_line_boundary() -> None:
    b = make_briefing()
    b.timeline = [
        TimelineEvent(date="2026-03-01", event="E" * 100 + str(i), sources=[OTHER]) for i in range(100)
    ]
    text = render_briefing(b, max_chars=1000)
    assert len(text) <= 1000
    assert not text.endswith("E" * 3 + "\n\n")
