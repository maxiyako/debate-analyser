"""Unit tests for the fact-checker hallucination guard (src/validation.py).

These reproduce a real hallucination observed in data/reports/604147.json:
the fact-checker turned a debate remark about Poland revoking an award from
a *Ukrainian military unit* ("velebia fašistov") into a fabricated claim that
"the President of Poland took away a state award from Zelenskyy", then
confidently verified that fabrication against an unrelated 2022 story. Since
each VerifiedFact must carry a verbatim `quote` copied from the transcript,
validate_report() can mechanically catch and drop this kind of claim.
"""

from __future__ import annotations

from typing import Callable

import requests

from src.agents import AnalysisReport, BehavioralAnalysis, Civility, SpeakerTactics, Verdict, VerifiedFact
from src.validation import (
    extract_quoted_spans,
    quote_grounded,
    url_reachable,
    validate_evidence_list,
    validate_facts,
    validate_fact_sources,
    validate_report,
)

# Real excerpts copied verbatim from data/transcripts/604147.txt.
REAL_QUOTE_DEBT = (
    "Táto vláda zadlžuje Slovensko oveľa rýchlejšie v absolútnych aj "
    "relatívnych číslach viac ako tie predchádzajúce vlády."
)
REAL_QUOTE_CIRKUSANT = "Čo si to vy dovoľujete, vy ste cirkusant."
REAL_QUOTE_POLAND_AWARD = (
    "poľský prezident povie, že jednoducho odníma mu kvôli tomu, kvôli tomu, "
    "že velebia fašistov, štátne vyznamenie"
)

# Fabricated claim reproducing the actual hallucination shipped in 604147.json.
HALLUCINATED_QUOTE_ZELENSKYY = (
    "Poľský prezident odobral štátne vyznamenanie priamo Volodymyrovi "
    "Zelenskému za to, že velebí fašizmus."
)
HALLUCINATED_QUOTE_UNRELATED = (
    "Predseda vlády Slovenskej republiky osobne odovzdal Nobelovu cenu za "
    "mier Elonovi Muskovi v roku 2025."
)


# Real source URLs cited in data/reports/604147.json. Verified by hand (curl) —
# the "fabricated" ones are real-looking article slugs on real domains that
# actually 404, i.e. exactly the kind of hallucination this module must catch.
REAL_SOURCE_URL = "https://www.nrsr.sk/web/Default.aspx?sid=schodze/hlasovanie/hlasovanie&ID=53394"
FABRICATED_SOURCE_URL = (
    "https://dennikn.sk/2030140/danko-sa-chvali-ako-sns-zvysovala-platy-"
    "ucitelov-o-40-percent-zabudol-vsak-povedat-ze-to-bolo-za-styri-roky/"
)
NONEXISTENT_DOMAIN_URL = "https://this-domain-definitely-does-not-exist-12345.sk/article"


def make_fact(
    claim: str,
    quote: str,
    verdict: Verdict = Verdict.TRUE,
    sources: list[str] | None = None,
) -> VerifiedFact:
    return VerifiedFact(
        claim=claim,
        speaker="Andrej Danko",
        quote=quote,
        verdict=verdict,
        sources=sources or [],
    )


def fake_checker(status_map: dict[str, bool | None]) -> Callable[[str], bool | None]:
    """Deterministic stand-in for url_reachable() so tests don't hit the network."""

    def _checker(url: str) -> bool | None:
        return status_map.get(url)

    return _checker


class TestQuoteGrounded:
    def test_exact_verbatim_quote_is_grounded(self, transcript: str) -> None:
        assert quote_grounded(REAL_QUOTE_DEBT, transcript)

    def test_quote_with_minor_whitespace_case_noise_is_grounded(self, transcript: str) -> None:
        noisy = REAL_QUOTE_CIRKUSANT.upper().replace(" ", "  ")
        assert quote_grounded(noisy, transcript)

    def test_fabricated_quote_reproducing_real_hallucination_is_not_grounded(
        self, transcript: str
    ) -> None:
        assert not quote_grounded(HALLUCINATED_QUOTE_ZELENSKYY, transcript)

    def test_completely_unrelated_fabricated_quote_is_not_grounded(
        self, transcript: str
    ) -> None:
        assert not quote_grounded(HALLUCINATED_QUOTE_UNRELATED, transcript)

    def test_very_short_quote_is_not_blocked(self, transcript: str) -> None:
        # Too short to verify meaningfully -> should not falsely fail.
        assert quote_grounded("dane", transcript)

    def test_ellipsis_joined_real_excerpts_are_grounded(self, transcript: str) -> None:
        # LLM evidence commonly stitches two real spans with "..."; both parts
        # exist verbatim, so the whole must count as grounded.
        stitched = f"{REAL_QUOTE_DEBT} ... {REAL_QUOTE_CIRKUSANT}"
        assert quote_grounded(stitched, transcript)

    def test_ellipsis_with_one_fabricated_part_is_not_grounded(
        self, transcript: str
    ) -> None:
        stitched = f"{REAL_QUOTE_DEBT} ... {HALLUCINATED_QUOTE_UNRELATED}"
        assert not quote_grounded(stitched, transcript)


class TestExtractQuotedSpans:
    def test_extracts_slovak_quote_from_evidence_string(self) -> None:
        entry = "Personal Insult: 'Čo si to vy dovoľujete, vy ste cirkusant.'"
        spans = extract_quoted_spans(entry)
        assert any("cirkusant" in span for span in spans)

    def test_no_quotes_returns_empty(self) -> None:
        assert extract_quoted_spans("A free-text note with no quotes at all") == []


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class TestUrlReachable:
    """Pure unit tests — network calls are stubbed so these never touch the wire."""

    def test_returns_true_for_200(self, monkeypatch) -> None:
        monkeypatch.setattr(requests, "head", lambda *a, **k: _FakeResponse(200))
        assert url_reachable(REAL_SOURCE_URL) is True

    def test_returns_false_for_404(self, monkeypatch) -> None:
        monkeypatch.setattr(requests, "head", lambda *a, **k: _FakeResponse(404))
        assert url_reachable(FABRICATED_SOURCE_URL) is False

    def test_returns_false_on_connection_error(self, monkeypatch) -> None:
        def _raise(*_a, **_k):
            raise requests.exceptions.ConnectionError("no DNS")

        monkeypatch.setattr(requests, "head", _raise)
        assert url_reachable(NONEXISTENT_DOMAIN_URL) is False

    def test_returns_none_on_timeout(self, monkeypatch) -> None:
        def _raise(*_a, **_k):
            raise requests.exceptions.Timeout("slow")

        monkeypatch.setattr(requests, "head", _raise)
        assert url_reachable(REAL_SOURCE_URL) is None

    def test_returns_none_for_bot_blocked_403(self, monkeypatch) -> None:
        # Legitimate sites often block scrapers; must not be treated as fabricated.
        monkeypatch.setattr(requests, "head", lambda *a, **k: _FakeResponse(403))
        assert url_reachable(REAL_SOURCE_URL) is None

    def test_falls_back_to_get_when_head_not_allowed(self, monkeypatch) -> None:
        monkeypatch.setattr(requests, "head", lambda *a, **k: _FakeResponse(405))
        monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(200))
        assert url_reachable(REAL_SOURCE_URL) is True


class TestValidateFactSources:
    def test_keeps_reachable_source(self) -> None:
        fact = make_fact("A", "quote", sources=[REAL_SOURCE_URL])
        checker = fake_checker({REAL_SOURCE_URL: True})
        facts, notes = validate_fact_sources([fact], url_checker=checker)
        assert facts[0].sources == [REAL_SOURCE_URL]
        assert notes == []

    def test_drops_fabricated_404_source(self) -> None:
        fact = make_fact("A", "quote", sources=[FABRICATED_SOURCE_URL])
        checker = fake_checker({FABRICATED_SOURCE_URL: False})
        facts, notes = validate_fact_sources([fact], url_checker=checker)
        assert facts[0].sources == []
        assert len(notes) == 2  # dropped-source note + "all sources dead" note
        assert any("dead/fabricated source URL" in n for n in notes)

    def test_keeps_unverifiable_source_instead_of_dropping(self) -> None:
        # None (timeout/bot-block) must not be punished like a confirmed 404.
        fact = make_fact("A", "quote", sources=[REAL_SOURCE_URL])
        checker = fake_checker({REAL_SOURCE_URL: None})
        facts, notes = validate_fact_sources([fact], url_checker=checker)
        assert facts[0].sources == [REAL_SOURCE_URL]
        assert notes == []

    def test_mixed_sources_keeps_only_reachable_ones(self) -> None:
        fact = make_fact("A", "quote", sources=[REAL_SOURCE_URL, FABRICATED_SOURCE_URL])
        checker = fake_checker({REAL_SOURCE_URL: True, FABRICATED_SOURCE_URL: False})
        facts, notes = validate_fact_sources([fact], url_checker=checker)
        assert facts[0].sources == [REAL_SOURCE_URL]
        assert len(notes) == 1

    def test_drops_disinfo_gray_zone_source(self) -> None:
        blocked = "https://ereport.sk/exekucia-pre-matku-politika"
        fact = make_fact("A", "quote", sources=[REAL_SOURCE_URL, blocked])
        # Reachable (200) yet must still be dropped for being gray-zone.
        checker = fake_checker({REAL_SOURCE_URL: True, blocked: True})
        facts, notes = validate_fact_sources([fact], url_checker=checker)
        assert facts[0].sources == [REAL_SOURCE_URL]
        assert any("disinfo/partisan-primary-source" in n for n in notes)

    def test_drops_political_party_own_site_as_source(self) -> None:
        # A party's own site reporting on a rival is not an independent source,
        # regardless of which side of the aisle it's on.
        blocked = "https://progresivne.sk/ficova-vlada-beznych-ludi-dohnala-na-hranicu"
        fact = make_fact("A", "quote", sources=[REAL_SOURCE_URL, blocked])
        checker = fake_checker({REAL_SOURCE_URL: True, blocked: True})
        facts, notes = validate_fact_sources([fact], url_checker=checker)
        assert facts[0].sources == [REAL_SOURCE_URL]
        assert any("disinfo/partisan-primary-source" in n for n in notes)


class TestValidateFacts:
    def test_grounded_fact_is_kept(self, transcript: str) -> None:
        fact = make_fact("The debt is rising faster than under previous governments.", REAL_QUOTE_DEBT)
        kept, notes = validate_facts([fact], transcript)
        assert kept == [fact]
        assert notes == []

    def test_hallucinated_fact_is_dropped(self, transcript: str) -> None:
        fact = make_fact(
            "The President of Poland took away a state award from Zelenskyy.",
            HALLUCINATED_QUOTE_ZELENSKYY,
            verdict=Verdict.FALSE,
        )
        kept, notes = validate_facts([fact], transcript)
        assert kept == []
        assert len(notes) == 1
        assert "hallucinated" in notes[0] or "ungrounded" in notes[0]

    def test_fact_without_quote_is_kept_but_flagged(self, transcript: str) -> None:
        fact = make_fact("Some claim with no supporting quote.", "")
        kept, notes = validate_facts([fact], transcript)
        assert kept == [fact]
        assert len(notes) == 1
        assert "no supporting quote" in notes[0]

    def test_mixed_batch_keeps_only_grounded_facts(self, transcript: str) -> None:
        good = make_fact("Grounded claim.", REAL_QUOTE_DEBT)
        bad = make_fact("Hallucinated claim.", HALLUCINATED_QUOTE_UNRELATED)
        kept, notes = validate_facts([good, bad], transcript)
        assert kept == [good]
        assert len(notes) == 1


class TestValidateEvidenceList:
    def test_grounded_evidence_quote_is_kept(self, transcript: str) -> None:
        entries = [f"Personal Insult: '{REAL_QUOTE_CIRKUSANT}'"]
        kept, notes = validate_evidence_list(entries, transcript)
        assert kept == entries
        assert notes == []

    def test_fabricated_evidence_quote_is_dropped(self, transcript: str) -> None:
        entries = [f"Fabricated attack: '{HALLUCINATED_QUOTE_UNRELATED}'"]
        kept, notes = validate_evidence_list(entries, transcript)
        assert kept == []
        assert len(notes) == 1

    def test_free_text_without_quotes_is_kept(self, transcript: str) -> None:
        entries = ["General observation with no quoted evidence."]
        kept, notes = validate_evidence_list(entries, transcript)
        assert kept == entries
        assert notes == []


class TestValidateReport:
    def test_end_to_end_removes_hallucinated_fact_and_evidence(self, transcript: str) -> None:
        report = AnalysisReport(
            summary="Test report",
            behavioral_analysis=BehavioralAnalysis(
                speakers=[
                    SpeakerTactics(
                        speaker="Andrej Danko",
                        civility=Civility(
                            incivility=[f"Insult: '{REAL_QUOTE_CIRKUSANT}'"],
                            fair_conduct=[f"Fabricated: '{HALLUCINATED_QUOTE_UNRELATED}'"],
                        ),
                    )
                ]
            ),
            facts=[
                make_fact("Grounded claim.", REAL_QUOTE_DEBT, sources=[REAL_SOURCE_URL]),
                make_fact(
                    "The President of Poland took away a state award from Zelenskyy.",
                    HALLUCINATED_QUOTE_ZELENSKYY,
                    verdict=Verdict.FALSE,
                    sources=[FABRICATED_SOURCE_URL],
                ),
            ],
        )
        checker = fake_checker({REAL_SOURCE_URL: True, FABRICATED_SOURCE_URL: False})

        result = validate_report(report, transcript, url_checker=checker)

        assert len(result.facts) == 1
        assert result.facts[0].quote == REAL_QUOTE_DEBT
        assert result.facts[0].sources == [REAL_SOURCE_URL]
        assert result.behavioral_analysis.speakers[0].civility.incivility == [
            f"Insult: '{REAL_QUOTE_CIRKUSANT}'"
        ]
        assert result.behavioral_analysis.speakers[0].civility.fair_conduct == []
        assert any("hallucinated" in note or "ungrounded" in note for note in result.critic_notes)

    def test_end_to_end_drops_fabricated_source_from_otherwise_valid_fact(
        self, transcript: str
    ) -> None:
        report = AnalysisReport(
            facts=[
                make_fact(
                    "Grounded claim with one fake and one real citation.",
                    REAL_QUOTE_DEBT,
                    sources=[REAL_SOURCE_URL, FABRICATED_SOURCE_URL],
                )
            ],
        )
        checker = fake_checker({REAL_SOURCE_URL: True, FABRICATED_SOURCE_URL: False})

        result = validate_report(report, transcript, url_checker=checker)

        assert len(result.facts) == 1
        assert result.facts[0].sources == [REAL_SOURCE_URL]
        assert any("dead/fabricated source URL" in note for note in result.critic_notes)
