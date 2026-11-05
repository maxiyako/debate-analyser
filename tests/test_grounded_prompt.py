"""vertex_grounded_search returns evidence, never a verdict.

The tool's inner model sees only the query, not the claim. In stored runs it
answered with its own guess of what the speaker meant ("Ak rečník tvrdil, že
… výrok je nepravdivý") and the checkers copied that verdict.
"""

from __future__ import annotations

from src.agents import _grounded_prompt


def test_prompt_keeps_debate_date_and_query() -> None:
    p = _grounded_prompt("Volkswagen Bratislava nový model", "2026-11-01")
    assert "2026-11-01" in p
    assert p.rstrip().endswith("Volkswagen Bratislava nový model")


def test_prompt_forbids_judging_a_claim() -> None:
    p = _grounded_prompt("q", "2026-11-01").lower()
    assert "nehodnoť" in p
    assert "verdikt" in p
    assert "rečník" in p
