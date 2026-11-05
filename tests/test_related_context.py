"""Checkers and the manager see the recorded reading and related claims."""

from __future__ import annotations

from src.agents import (
    _CHECK_RULES,
    ExtractedClaim,
    Severity,
    Verdict,
    VerifiedFact,
    _check_task_description,
    _related_block,
)


def c(cid: int, related: list[int] | None = None, interpretation: str = "") -> ExtractedClaim:
    return ExtractedClaim(
        id=cid,
        claim=f"Tvrdenie {cid}.",
        speaker="Michal Šimečka",
        quote="q",
        related_claim_ids=related or [],
        interpretation=interpretation,
    )


def test_check_rules_bind_the_reading() -> None:
    for word in ("interpretation", "not_asserted", "refutation_would_be", "RELATED CLAIMS"):
        assert word in _CHECK_RULES
    assert "judge the quote" in _CHECK_RULES


def test_related_block_lists_only_outside_related_claims() -> None:
    pool = [c(1, [2, 3]), c(2, interpretation="Výnimka prešla."), c(3), c(4)]
    block = _related_block([pool[0], pool[2]], pool)
    assert block.startswith("RELATED CLAIMS (read-only")
    assert "#2 Michal Šimečka: Tvrdenie 2." in block
    assert "výklad: Výnimka prešla." in block
    assert "#3" not in block  # claim 3 is checked in this task, not context
    assert "#4" not in block
    assert _related_block([pool[3]], pool) == ""


def test_related_block_with_verdicts() -> None:
    pool = [c(1, [2]), c(2)]
    fact = VerifiedFact(
        claim="Tvrdenie 2.",
        speaker="Michal Šimečka",
        verdict=Verdict.FALSE,
        severity=Severity.MATERIAL,
        rationale="x" * 400,
    )
    block = _related_block([pool[0]], pool, {2: fact})
    assert block.startswith("RELATED CLAIMS AND CURRENT VERDICTS")
    assert "aktuálny verdikt: False / material." in block
    assert "x" * 300 in block and "x" * 301 not in block


def test_task_description_appends_related_block() -> None:
    d = _check_task_description("HEADER", [c(1)], None, related="RELATED CLAIMS (read-only): -")
    assert d.rstrip().endswith("RELATED CLAIMS (read-only): -")
    assert "RELATED CLAIMS (read-only)" not in _check_task_description("HEADER", [c(1)], None)


def test_manager_rules_require_consistency_with_related_verdicts() -> None:
    from src.agents import _MANAGER_RULES

    assert "CONSISTENCY" in _MANAGER_RULES
    assert "RELATED CLAIMS AND CURRENT VERDICTS" in _MANAGER_RULES


def test_manager_crew_accepts_related_block() -> None:
    import inspect

    from src.agents import build_manager_crew

    assert inspect.signature(build_manager_crew).parameters["related"].default == ""


def test_check_claims_signature_and_empty_input() -> None:
    import inspect

    from config import Settings
    from src.agents import check_claims

    params = inspect.signature(check_claims).parameters
    assert list(params)[:3] == ["claims", "settings", "debate_date"]
    assert params["pool"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["pool_facts"].kind is inspect.Parameter.KEYWORD_ONLY
    assert check_claims([], Settings(_env_file=None)) == ([], set(), [])
