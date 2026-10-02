"""Offline tests for the evidence judge (src/judge.py)."""

from __future__ import annotations

from datetime import date

from src.agents import Severity, Verdict, VerifiedFact
from src.judge import (
    FactJudgement,
    JudgeAxes,
    JudgeOutput,
    apply_judge_downgrades,
    build_judge_prompt,
    forced_downgrade,
    judge_fact,
    judge_facts,
    quality_score,
    summarize,
)
from src.tools.page import PageResult


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


DEBATE = date(2026, 4, 12)


def page_ok(url="https://a.sk/x", published="2026-03-01", text="Deficit verejných financií za rok 2025 dosiahol 5,3 % HDP."):
    return PageResult("ok", url, final_url=url, title="t", published=published, text=text)


class RecordingLLM:
    def __init__(self, out: JudgeOutput):
        self.out = out
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> JudgeOutput:
        self.prompts.append(prompt)
        return self.out


def good_out(**kw) -> JudgeOutput:
    return JudgeOutput(axes=axes(**kw), issues=[], recommend_downgrade=False)


def test_prompt_contains_excerpt_dates_and_flags_anachronism() -> None:
    fact = make_fact()
    late = page_ok("https://b.sk/y", published="2026-05-01")
    prompt = build_judge_prompt(fact, DEBATE, [page_ok(), late])
    assert "5,3 % HDP" in prompt
    assert "2026-04-12" in prompt
    assert prompt.count("PUBLISHED AFTER THE DEBATE DATE") == 1
    assert fact.claim in prompt


def test_prompt_marks_unread_sources() -> None:
    prompt = build_judge_prompt(
        make_fact(), DEBATE, [PageResult("inconclusive", "https://c.sk/z", detail="HTTP 403")]
    )
    assert "NOT READ" in prompt and "HTTP 403" in prompt


def test_judge_fact_scores_from_axes_not_from_model() -> None:
    llm = RecordingLLM(good_out(s=2, m=1, d=2, c=2, i=1))
    j = judge_fact(make_fact(), 0, DEBATE, fetch=lambda u: page_ok(u), llm=llm)
    assert j.assessable and j.sources_read == 1
    assert j.quality == quality_score(axes(2, 1, 2, 2, 1))
    assert j.downgrade_to is None
    assert len(llm.prompts) == 1


def test_judge_fact_downgrades_on_zero_support() -> None:
    llm = RecordingLLM(JudgeOutput(axes=axes(s=0), issues=["zdroj tvrdenie nepodporuje"]))
    j = judge_fact(make_fact(Verdict.FALSE), 3, DEBATE, fetch=lambda u: page_ok(u), llm=llm)
    assert j.fact_index == 3
    assert j.downgrade_to == Verdict.UNVERIFIED
    assert j.issues == ["zdroj tvrdenie nepodporuje"]


def test_judge_fact_unassessable_when_nothing_readable_skips_llm() -> None:
    llm = RecordingLLM(good_out())
    j = judge_fact(
        make_fact(),
        0,
        DEBATE,
        fetch=lambda u: PageResult("inconclusive", u, detail="HTTP 403"),
        llm=llm,
    )
    assert not j.assessable
    assert llm.prompts == []


def test_judge_fact_skips_unverified_and_unsourced() -> None:
    llm = RecordingLLM(good_out())
    unv = make_fact(Verdict.UNVERIFIED)
    assert not judge_fact(unv, 0, DEBATE, fetch=lambda u: page_ok(u), llm=llm).assessable
    nosrc = make_fact(Verdict.TRUE)
    nosrc.sources = []
    assert not judge_fact(nosrc, 0, DEBATE, fetch=lambda u: page_ok(u), llm=llm).assessable
    assert llm.prompts == []


def test_judge_facts_keeps_order_and_summarizes() -> None:
    llm = RecordingLLM(good_out())
    facts = [make_fact(Verdict.TRUE), make_fact(Verdict.UNVERIFIED), make_fact(Verdict.FALSE)]
    summary = judge_facts(facts, DEBATE, fetch=lambda u: page_ok(u), llm=llm, max_workers=2)
    assert [j.fact_index for j in summary.judgements] == [0, 1, 2]
    assert summary.judged == 2 and summary.skipped == 1
    assert summary.mean_quality == 100.0


def test_judge_facts_isolates_a_failing_fact() -> None:
    def flaky_fetch(url: str) -> PageResult:
        if url.endswith("/boom"):
            raise RuntimeError("parser crashed")
        return page_ok(url)

    bad = make_fact(Verdict.FALSE)
    bad.sources = ["https://a.sk/boom"]
    facts = [make_fact(Verdict.TRUE), bad, make_fact(Verdict.FALSE)]
    summary = judge_facts(
        facts, DEBATE, fetch=flaky_fetch, llm=RecordingLLM(good_out()), max_workers=2
    )
    assert [j.fact_index for j in summary.judgements] == [0, 1, 2]
    assert summary.judgements[1].assessable is False
    assert "parser crashed" in summary.judgements[1].reason
    assert summary.judged == 2
