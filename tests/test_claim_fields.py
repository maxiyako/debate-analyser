"""New claim/fact fields carry the interpretation from extraction to the report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents import (
    AnalysisReport,
    ClaimCategory,
    ExtractedClaim,
    ExtractedClaims,
    Referent,
    ReferentOrigin,
    VerifiedFact,
    Verdict,
)
from src.reconcile import reconcile_checks

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"


def test_new_fields_default_empty() -> None:
    c = ExtractedClaim(id=1, claim="c", speaker="X", quote="q")
    assert (c.claim_extracted, c.interpretation, c.refutation_would_be) == ("", "", "")
    assert c.not_asserted == [] and c.referents == [] and c.related_claim_ids == []
    f = VerifiedFact(claim="c", speaker="X", verdict=Verdict.TRUE)
    assert f.interpretation == "" and f.not_asserted == [] and f.claim_extracted == ""
    assert f.category is None


def test_build_fact_copies_reading_and_category() -> None:
    claim = ExtractedClaim(
        id=1,
        claim="Zrušenie transakčnej dane pre živnostníkov prešlo hlasmi opozície a SNS.",
        claim_extracted="Zrušenie transakčnej dane prešlo hlasmi opozície a SNS.",
        speaker="Michal Šimečka",
        quote="o transakčnej dani a jej zrušení pre živnostníkov",
        category=ClaimCategory.HISTORY_POLITICS,
        interpretation="Výnimka z transakčnej dane pre živnostníkov prešla s hlasmi opozície a SNS.",
        not_asserted=["úplné zrušenie transakčnej dane"],
        referents=[Referent(text="to", resolved="výnimka pre živnostníkov", origin=ReferentOrigin.TRANSCRIPT)],
    )
    facts, _ = reconcile_checks(ExtractedClaims(claims=[claim]), None, None, set())
    f = facts[0]
    assert f.claim == claim.claim
    assert f.claim_extracted == claim.claim_extracted
    assert f.interpretation == claim.interpretation
    assert f.not_asserted == ["úplné zrušenie transakčnej dane"]
    assert f.category == ClaimCategory.HISTORY_POLITICS


def test_stored_reports_still_load() -> None:
    from src.eval import report_files

    if not REPORTS.exists():
        pytest.skip("no local reports")
    paths = report_files(REPORTS)
    if not paths:
        pytest.skip("no local reports")
    for p in paths:
        AnalysisReport.model_validate(json.loads(p.read_text(encoding="utf-8")))
