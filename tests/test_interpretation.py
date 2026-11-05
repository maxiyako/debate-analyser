"""Offline tests for the claim interpretation pass (src/interpretation.py).

Checkers refuted claims the speaker never made: full abolition of a tax
instead of the exemption for živnostníci, a ministry instead of the
government. The pass records the asserted proposition before any search;
code decides which readings reach the checkers.
"""

from __future__ import annotations

from datetime import date

from src.agents import ExtractedClaim, Referent, ReferentOrigin
from src.briefing import (
    DebateBriefing,
    Dispute,
    EntityEntry,
    GlossaryEntry,
    Participant,
    TimelineEvent,
)
from src.interpretation import (
    ClaimReading,
    Drift,
    DriftKind,
    apply_readings,
    clear_interpretation,
    digit_tokens,
    locate_window,
    referent_context,
    unsupported_numbers,
)
from src.transcript_lines import parse_lines

TRANSCRIPT = """\
Moderátor [08:40]: Hovorme o daniach.
Tomáš Taraba [08:50]: Opozícia nič nepresadila.
Moderátor [08:58]: Pán Šimečka?
Michal Šimečka [09:04]: Keď sa hlasovalo o transakčnej dani a jej zrušení pre živnostníkov, rovnako hlasovala opozícia za.
Michal Šimečka [09:12]: A prešlo to preto, že aj SNS hlasovala za.
Moderátor [09:20]: Ďakujem.
Tomáš Taraba [09:25]: To nie je pravda.
"""
LINES = parse_lines(TRANSCRIPT)
DEBATE = date(2026, 9, 27)


def claim(**kw) -> ExtractedClaim:
    base = dict(
        id=1,
        claim="Zrušenie transakčnej dane prešlo v parlamente hlasmi opozície a SNS.",
        speaker="Michal Šimečka",
        quote="o transakčnej dani a jej zrušení pre živnostníkov, rovnako hlasovala opozícia za",
    )
    base.update(kw)
    return ExtractedClaim(**base)


def reading(**kw) -> ClaimReading:
    base = dict(
        claim_id=1,
        interpretation="Výnimka z transakčnej dane pre živnostníkov prešla s hlasmi opozície a SNS.",
        drift=[Drift(kind=DriftKind.DROPPED_QUALIFIER, detail="vypadlo 'pre živnostníkov'")],
        claim="Zrušenie transakčnej dane pre živnostníkov prešlo hlasmi opozície a SNS.",
        not_asserted=["úplné zrušenie transakčnej dane"],
        refutation_would_be="záznam hlasovania o výnimke pre živnostníkov bez hlasov opozície alebo SNS",
    )
    base.update(kw)
    return ClaimReading(**base)


def briefing() -> DebateBriefing:
    return DebateBriefing(
        debate_date="2026-09-27",
        participants=[Participant(name="Michal Šimečka", party="PS", office="poslanec NR SR")],
        entity_index=[EntityEntry(reference="vláda", person="vláda Roberta Fica (od 25. 10. 2023)")],
        glossary=[GlossaryEntry(term="transakčná daň", definition="daň z platobných transakcií", status="výnimka schválená 2025-09-10")],
        live_disputes=[Dispute(title="Dane", factual_question="Kto zrušil daň?")],
        timeline=[TimelineEvent(date="2025-09-10", event="NR SR schválila výnimku")],
    )


# --- window -----------------------------------------------------------------


def test_window_surrounds_the_quoted_line() -> None:
    w = locate_window(LINES, claim(), size=1)
    assert "Michal Šimečka [09:04]" in w
    assert "Moderátor [08:58]" in w and "Michal Šimečka [09:12]" in w
    assert "[08:50]" not in w and "[09:20]" not in w


def test_window_for_quote_spanning_two_lines_covers_both() -> None:
    c = claim(quote="rovnako hlasovala opozícia za. A prešlo to preto, že aj SNS hlasovala za")
    w = locate_window(LINES, c, size=1)
    assert "Michal Šimečka [09:04]" in w and "Michal Šimečka [09:12]" in w


def test_window_is_empty_without_a_match() -> None:
    assert locate_window(LINES, claim(quote="qqq zzz xxx www vvv kkk"), size=2) == ""


def test_window_is_clipped_at_transcript_edges() -> None:
    c = claim(speaker="Moderátor", quote="Hovorme o daniach.")
    w = locate_window(LINES, c, size=3)
    assert w.startswith("Moderátor [08:40]")
    assert "[09:12]" not in w


# --- briefing context -----------------------------------------------------------


def test_referent_context_has_only_offices_entities_and_glossary_terms() -> None:
    ctx = referent_context(briefing())
    assert "Michal Šimečka" in ctx and "poslanec NR SR" in ctx
    assert '"vláda" = vláda Roberta Fica (od 25. 10. 2023)' in ctx
    assert "transakčná daň: daň z platobných transakcií" in ctx
    assert "2025-09-10" not in ctx  # glossary status and timeline are facts, not referents
    assert "Kto zrušil daň?" not in ctx
    assert referent_context(None) == ""


# --- number guard ---------------------------------------------------------------


def test_digit_tokens_normalize_decimal_comma_and_leading_zeros() -> None:
    assert digit_tokens("11,5 mld. v 09/2026") == {"11.5", "9", "2026"}
    assert digit_tokens("bez čísel") == set()


def test_numbers_from_quote_window_and_time_window_are_allowed() -> None:
    c = claim(quote="za tie 3 balíky, keď sa vybralo cez 11,5 mld.", time_window="2023-10")
    r = reading(interpretation="Za 3 balíky sa vybralo 11.5 mld. od 2023 a dlh rástol od 10/2023.", not_asserted=[])
    assert unsupported_numbers(r, c, "Moderátor [09:20]: Ďakujem.", "", DEBATE) == []


def test_number_from_nowhere_is_reported() -> None:
    r = reading(refutation_would_be="súčet balíkov 7,4 mld.")
    assert unsupported_numbers(r, claim(), "", "", DEBATE) == ["7.4"]


def test_briefing_referent_digits_need_the_briefing() -> None:
    ctx = referent_context(briefing())
    ok = reading(
        interpretation="Dlh vzrástol od nástupu vlády v roku 2023.",
        referents=[Referent(text="keď preberali", resolved="nástup vlády Roberta Fica (25. 10. 2023)", origin=ReferentOrigin.BRIEFING)],
        not_asserted=[],
    )
    assert unsupported_numbers(ok, claim(), "", ctx, DEBATE) == []
    bad = ok.model_copy(update={"referents": [Referent(text="keď preberali", resolved="október 2020", origin=ReferentOrigin.BRIEFING)]})
    assert unsupported_numbers(bad, claim(), "", ctx, DEBATE) == ["2020", "2023"]


def test_debate_date_digits_are_allowed() -> None:
    r = reading(interpretation="K 27. 9. 2026 platí výnimka.", not_asserted=[])
    assert unsupported_numbers(r, claim(), "", "", DEBATE) == []


# --- apply_readings ---------------------------------------------------------------


def test_drift_rewrites_claim_and_keeps_original() -> None:
    c = claim()
    notes = apply_readings([c], [reading()], {1: ""}, "", DEBATE)
    assert c.claim == "Zrušenie transakčnej dane pre živnostníkov prešlo hlasmi opozície a SNS."
    assert c.claim_extracted == "Zrušenie transakčnej dane prešlo v parlamente hlasmi opozície a SNS."
    assert c.interpretation.startswith("Výnimka z transakčnej dane")
    assert c.not_asserted == ["úplné zrušenie transakčnej dane"]
    assert c.refutation_would_be.startswith("záznam hlasovania")
    assert any("rewrote claim 1" in n and "dropped_qualifier" in n for n in notes)


def test_no_drift_keeps_claim_text() -> None:
    c = claim()
    apply_readings([c], [reading(drift=[], claim="iný text")], {1: ""}, "", DEBATE)
    assert c.claim.startswith("Zrušenie transakčnej dane prešlo v parlamente")
    assert c.claim_extracted == ""
    assert c.interpretation


def test_drift_without_claim_text_keeps_claim_and_notes_it() -> None:
    c = claim()
    notes = apply_readings([c], [reading(claim="")], {1: ""}, "", DEBATE)
    assert c.claim_extracted == "" and c.claim.startswith("Zrušenie transakčnej dane prešlo v")
    assert c.interpretation
    assert any("gave no claim text" in n for n in notes)


def test_rejected_reading_leaves_claim_untouched() -> None:
    c = claim()
    notes = apply_readings([c], [reading(refutation_would_be="súčet 7,4 mld.")], {1: ""}, "", DEBATE)
    assert c.interpretation == "" and c.claim_extracted == ""
    assert any("rejected for claim 1" in n and "7.4" in n for n in notes)


def test_unknown_missing_and_related_ids() -> None:
    a, b = claim(id=1), claim(id=2, claim="Iné tvrdenie.")
    notes = apply_readings(
        [a, b],
        [reading(related_claim_ids=[1, 2, 99]), reading(claim_id=7)],
        {1: "", 2: ""},
        "",
        DEBATE,
    )
    assert a.related_claim_ids == [2]
    assert b.interpretation == ""
    assert any("unknown claim 7" in n for n in notes)
    assert any("no reading for claim 2" in n for n in notes)


def test_protected_fields_never_change() -> None:
    c = claim(time_reference="v septembri", time_window="2025-09", consequence=5)
    before = (c.quote, c.speaker, c.category, c.checkability, c.consequence, c.time_reference, c.time_window)
    apply_readings([c], [reading()], {1: ""}, "", DEBATE)
    after = (c.quote, c.speaker, c.category, c.checkability, c.consequence, c.time_reference, c.time_window)
    assert before == after


def test_clear_interpretation_blanks_and_optionally_restores() -> None:
    c = claim()
    apply_readings([c], [reading()], {1: ""}, "", DEBATE)
    clear_interpretation([c], restore_claim=True)
    assert c.claim == "Zrušenie transakčnej dane prešlo v parlamente hlasmi opozície a SNS."
    assert (c.claim_extracted, c.interpretation, c.refutation_would_be) == ("", "", "")
    assert c.not_asserted == [] and c.referents == [] and c.related_claim_ids == []

    d = claim(claim_extracted="extractor filled this", interpretation="x")
    clear_interpretation([d])
    assert d.claim.startswith("Zrušenie") and d.claim_extracted == "" and d.interpretation == ""
