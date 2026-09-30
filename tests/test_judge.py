"""Offline tests for the evidence judge (src/judge.py)."""

from __future__ import annotations

from src.agents import Severity, Verdict, VerifiedFact
from src.judge import (
    FactJudgement,
    JudgeAxes,
    apply_judge_downgrades,
    forced_downgrade,
    quality_score,
    summarize,
)


def axes(s=2, m=2, d=2, c=2, i=2) -> JudgeAxes:
    return JudgeAxes(support=s, metric_fidelity=m, date_fit=d, coverage=c, independence=i)


def make_fact(verdict: Verdict = Verdict.FALSE) -> VerifiedFact:
    return VerifiedFact(
        claim="Deficit je 6 % HDP v roku 2025.",
        speaker="X",
        quote="q",
        verdict=verdict,
        severity=Severity.MATERIAL if verdict in (Verdict.FALSE, Verdict.MISLEADING) else None,
        sources=["https://a.sk/x"],
    )


def test_quality_score_bounds_and_weights() -> None:
    assert quality_score(axes()) == 100.0
    assert quality_score(axes(0, 0, 0, 0, 0)) == 0.0
    assert quality_score(axes(2, 0, 0, 0, 0)) == 35.0
    assert quality_score(axes(0, 2, 0, 0, 0)) == 25.0


def test_forced_downgrade_rules() -> None:
    assert forced_downgrade(Verdict.FALSE, axes(s=0), False) == Verdict.UNVERIFIED
    assert forced_downgrade(Verdict.FALSE, axes(m=0), False) == Verdict.UNVERIFIED
    assert forced_downgrade(Verdict.MISLEADING, axes(d=0), False) == Verdict.UNVERIFIED
    assert forced_downgrade(Verdict.FALSE, axes(), True) == Verdict.UNVERIFIED
    assert forced_downgrade(Verdict.FALSE, axes(), False) is None
    # True is only downgraded when its evidence does not support it at all.
    assert forced_downgrade(Verdict.TRUE, axes(s=0), False) == Verdict.UNVERIFIED
    assert forced_downgrade(Verdict.TRUE, axes(s=1, m=1, d=1, c=1, i=1), False) is None
    # Never touches non-judged verdicts.
    assert forced_downgrade(Verdict.UNVERIFIED, axes(0, 0, 0, 0, 0), True) is None
    assert forced_downgrade(Verdict.CONTESTED, axes(0, 0, 0, 0, 0), True) is None


def _j(idx: int, verdict: Verdict, quality: float | None, down: Verdict | None = None, ok: bool = True):
    return FactJudgement(
        fact_index=idx,
        claim="c",
        verdict=verdict.value,
        assessable=ok,
        quality=quality,
        downgrade_to=down,
    )


def test_summarize_means_counts_and_by_verdict() -> None:
    js = [
        _j(0, Verdict.TRUE, 80.0),
        _j(1, Verdict.FALSE, 40.0, Verdict.UNVERIFIED),
        _j(2, Verdict.FALSE, 60.0),
        _j(3, Verdict.UNVERIFIED, None, ok=False),
    ]
    s = summarize(js)
    assert s.judged == 3
    assert s.skipped == 1
    assert s.mean_quality == 60.0
    assert s.downgrades == 1
    assert s.mean_quality_by_verdict == {"True": 80.0, "False": 50.0}


def test_summarize_empty() -> None:
    s = summarize([])
    assert s.judged == 0 and s.mean_quality is None


def test_apply_downgrades_mutates_and_clears_severity() -> None:
    facts = [make_fact(Verdict.FALSE), make_fact(Verdict.TRUE)]
    notes = apply_judge_downgrades(
        facts, [_j(0, Verdict.FALSE, 20.0, Verdict.UNVERIFIED), _j(1, Verdict.TRUE, 90.0)]
    )
    assert facts[0].verdict == Verdict.UNVERIFIED
    assert facts[0].severity is None
    assert facts[1].verdict == Verdict.TRUE
    assert len(notes) == 1 and "False->Unverified" in notes[0]
