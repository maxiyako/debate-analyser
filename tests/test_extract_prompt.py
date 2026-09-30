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
