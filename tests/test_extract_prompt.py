"""The extraction prompt switches from rhetoric-based salience to briefing-based consequence."""

from __future__ import annotations

from src.agents import AnalysisReport, _importance_block


def test_legacy_block_when_no_briefing() -> None:
    block = _importance_block("")
    assert "salience" in block
    assert "consequence" not in block.lower()


def test_consequence_block_embeds_briefing_and_rules() -> None:
    block = _importance_block("POLITICKÝ KONTEXT: Eurofondy")
    assert "POLITICKÝ KONTEXT: Eurofondy" in block
    assert "consequence" in block
    assert "checkability" in block
    assert "definitional" in block
    assert "entity" in block.lower()  # role references resolved via the briefing


def test_report_briefing_defaults_to_none() -> None:
    assert AnalysisReport().briefing is None


def test_publisher_prompt_excludes_code_made_sections() -> None:
    import inspect

    from src.agents import generate_facebook_post

    src = inspect.getsource(generate_facebook_post)
    assert 'exclude={"briefing", "speaker_map", "transcript_quality"}' in src
    assert "scoring_status" in src  # no winner claims when scoring is degraded


def test_retry_empty_extraction_recovers() -> None:
    from src.agents import ExtractedClaim, ExtractedClaims, retry_empty_extraction

    good = ExtractedClaims(claims=[ExtractedClaim(id=1, claim="c", speaker="s", quote="q")])
    calls = []

    def rerun():
        calls.append(1)
        return ExtractedClaims(claims=[]) if len(calls) == 1 else good

    result, notes = retry_empty_extraction(ExtractedClaims(claims=[]), rerun, attempts=2)
    assert result is good and len(calls) == 2 and "recovered 1 claims" in notes[-1]


def test_retry_skipped_when_claims_present_and_survives_errors() -> None:
    from src.agents import ExtractedClaim, ExtractedClaims, retry_empty_extraction

    ok = ExtractedClaims(claims=[ExtractedClaim(id=1, claim="c", speaker="s", quote="q")])
    result, notes = retry_empty_extraction(ok, lambda: 1 / 0)
    assert result is ok and notes == []

    def boom():
        raise RuntimeError("x")

    result, notes = retry_empty_extraction(None, boom, attempts=2)
    assert result is None and len(notes) == 2


def test_kickoff_with_retry_rebuilds_and_gives_up() -> None:
    import pytest

    from src.agents import kickoff_with_retry

    class Crew:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def kickoff(self):
            if self.fail:
                raise ValueError("empty")
            return "ok"

    built = []

    def make():
        built.append(1)
        return Crew(fail=len(built) < 3), {"t": 1}

    assert kickoff_with_retry(make, "x") == ("ok", {"t": 1})
    assert len(built) == 3

    with pytest.raises(ValueError):
        kickoff_with_retry(lambda: (Crew(True), None), "x", attempts=2)
