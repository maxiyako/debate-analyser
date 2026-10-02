"""Evidence judge: does the evidence cited for a verdict actually justify it?

The judge reads the cited pages, rates five axes (0-2 each), and may force a
DOWNGRADE to Unverified — never an upgrade. Quality score and downgrade
decisions are computed in code from the axes; the model's own opinion of
"overall quality" is never trusted. Aggregated, the per-fact quality becomes a
run-level number stored with the report (the evaluation harness the spec asks
for).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Callable

from pydantic import BaseModel, Field

from src.agents import Verdict, VerifiedFact
from src.tools.page import (
    PageResult,
    claim_keywords,
    fetch_page,
    parse_iso_date,
    window_around_keywords,
)

_JUDGED = (Verdict.TRUE, Verdict.FALSE, Verdict.MISLEADING)
_ACCUSATIONS = (Verdict.FALSE, Verdict.MISLEADING)
_WEIGHTS = {
    "support": 0.35,
    "metric_fidelity": 0.25,
    "date_fit": 0.15,
    "coverage": 0.15,
    "independence": 0.10,
}

logger = logging.getLogger(__name__)


class JudgeAxes(BaseModel):
    support: int = Field(
        ge=0, le=2,
        description="2 = cited evidence entails the verdict; 1 = partially; 0 = does not support or contradicts.",
    )
    metric_fidelity: int = Field(
        ge=0, le=2,
        description="2 = evidence is about exactly the claimed metric, period, territory; 1 = close; 0 = different quantity.",
    )
    date_fit: int = Field(
        ge=0, le=2,
        description="2 = dated at/before the debate and fresh enough for the claim's time frame; 1 = dated but somewhat stale or undated; 0 = published after the debate or about the wrong period.",
    )
    coverage: int = Field(
        ge=0, le=2,
        description="2 = addresses the claim as a whole; 1 = addresses part of it; 0 = addresses none of it.",
    )
    independence: int = Field(
        ge=0, le=2,
        description="2 = two or more genuinely independent sources agree; 1 = one source, or several copying one origin; 0 = none usable.",
    )


class JudgeOutput(BaseModel):
    axes: JudgeAxes
    issues: list[str] = Field(
        default_factory=list,
        description="Short concrete problems found, in Slovak. Empty if none.",
    )
    recommend_downgrade: bool = Field(
        default=False,
        description="True only if the verdict should be withdrawn to Unverified because the evidence cannot carry it.",
    )


class FactJudgement(BaseModel):
    fact_index: int
    claim: str
    verdict: str
    assessable: bool = True
    reason: str = ""
    sources_read: int = 0
    quality: float | None = None
    axes: JudgeAxes | None = None
    issues: list[str] = Field(default_factory=list)
    downgrade_to: Verdict | None = None


class JudgeSummary(BaseModel):
    judgements: list[FactJudgement] = Field(default_factory=list)
    judged: int = 0
    skipped: int = 0
    mean_quality: float | None = None
    downgrades: int = 0
    mean_quality_by_verdict: dict[str, float] = Field(default_factory=dict)


def quality_score(axes: JudgeAxes) -> float:
    return round(
        100.0 * sum(w * getattr(axes, k) / 2.0 for k, w in _WEIGHTS.items()), 1
    )


def forced_downgrade(
    verdict: Verdict, axes: JudgeAxes, recommend: bool
) -> Verdict | None:
    """Downgrade target (always Unverified) or None. Never upgrades."""
    if verdict in _ACCUSATIONS:
        if recommend or axes.support == 0 or axes.metric_fidelity == 0 or axes.date_fit == 0:
            return Verdict.UNVERIFIED
        return None
    if verdict == Verdict.TRUE:
        return Verdict.UNVERIFIED if axes.support == 0 else None
    return None


def summarize(judgements: list[FactJudgement]) -> JudgeSummary:
    scored = [j for j in judgements if j.assessable and j.quality is not None]
    by_verdict: dict[str, list[float]] = {}
    for j in scored:
        by_verdict.setdefault(j.verdict, []).append(j.quality)  # type: ignore[arg-type]
    return JudgeSummary(
        judgements=judgements,
        judged=len(scored),
        skipped=len(judgements) - len(scored),
        mean_quality=(
            round(sum(j.quality for j in scored) / len(scored), 1) if scored else None  # type: ignore[misc]
        ),
        downgrades=sum(1 for j in judgements if j.downgrade_to is not None),
        mean_quality_by_verdict={
            v: round(sum(q) / len(q), 1) for v, q in by_verdict.items()
        },
    )


def apply_judge_downgrades(
    facts: list[VerifiedFact], judgements: list[FactJudgement]
) -> list[str]:
    """Apply forced downgrades in place. Returns human-readable notes."""
    notes: list[str] = []
    for j in judgements:
        if j.downgrade_to is None or not (0 <= j.fact_index < len(facts)):
            continue
        fact = facts[j.fact_index]
        notes.append(
            f"Judge downgraded {fact.verdict.value}->{j.downgrade_to.value} "
            f'(quality {j.quality}): "{fact.claim[:120]}" — '
            + "; ".join(j.issues[:3])
        )
        fact.verdict = j.downgrade_to
        fact.severity = None
    return notes


_JUDGE_INSTRUCTIONS = """\
You are a strict evidence auditor for a fact-checking desk. You do NOT decide
whether the claim is true from your own knowledge. You judge ONLY whether the
evidence below justifies the verdict that was given.

Rate five axes, each 0, 1 or 2:
- support: does the excerpted evidence entail the verdict? (for True: confirm
  the claim; for False: explicitly contradict it; for Misleading: show it
  distorts reality). 0 = it does not, or it points the other way.
- metric_fidelity: is the evidence about exactly the same metric, period and
  territory as the claim? (headline CPI is not food inflation; nominal is not
  real; year-on-year is not cumulative; a different year is not this year.)
- date_fit: is every relied-on source dated at or before the debate date and
  fresh enough for the claim's time frame? A source marked "PUBLISHED AFTER
  THE DEBATE DATE" cannot support anything. Undated sources score at most 1.
- coverage: does the evidence address the claim as a WHOLE, or only a part?
- independence: do two or more genuinely independent sources agree (not one
  wire story republished)? One source = 1.

Set recommend_downgrade=true only if the verdict should be withdrawn to
Unverified because this evidence cannot carry it. Sources marked NOT READ
tell you nothing either way. List concrete problems in `issues`, in Slovak,
one short sentence each. Be strict but fair: do not invent problems.
"""


def build_judge_prompt(
    fact: VerifiedFact, debate_date: date | None, pages: list[PageResult]
) -> str:
    ref = debate_date.isoformat() if debate_date else "unknown"
    kws = claim_keywords(fact.claim)
    blocks: list[str] = []
    for i, p in enumerate(pages, 1):
        if p.status == "ok":
            late = ""
            pub = parse_iso_date(p.published)
            if debate_date and pub and pub > debate_date:
                late = "  !! PUBLISHED AFTER THE DEBATE DATE\n"
            blocks.append(
                f"[{i}] {p.url}\n  title: {p.title or '(none)'}\n"
                f"  published: {p.published or 'unknown'}\n{late}"
                f"  excerpt: {window_around_keywords(p.text, kws)}"
            )
        else:
            blocks.append(f"[{i}] {p.url}\n  NOT READ ({p.status}: {p.detail})")
    return (
        f"{_JUDGE_INSTRUCTIONS}\n"
        f"DEBATE DATE (reference 'now'): {ref}\n\n"
        f"CLAIM: {fact.claim}\n"
        f"SPEAKER QUOTE: {fact.quote}\n"
        f"VERDICT GIVEN: {fact.verdict.value}"
        f"{' / ' + fact.severity.value if fact.severity else ''}\n"
        f"RATIONALE GIVEN: {fact.rationale[:1200]}\n\n"
        "CITED EVIDENCE:\n" + "\n\n".join(blocks)
    )


def judge_fact(
    fact: VerifiedFact,
    fact_index: int,
    debate_date: date | None,
    *,
    fetch: Callable[[str], PageResult] = fetch_page,
    llm: Callable[[str], JudgeOutput],
    max_sources: int = 4,
) -> FactJudgement:
    base = dict(fact_index=fact_index, claim=fact.claim, verdict=fact.verdict.value)
    if fact.verdict not in _JUDGED:
        return FactJudgement(**base, assessable=False, reason="verdict not judged")
    if not fact.sources:
        return FactJudgement(**base, assessable=False, reason="no sources cited")

    pages = [fetch(u) for u in fact.sources[:max_sources]]
    readable = [p for p in pages if p.status == "ok"]
    if not readable:
        return FactJudgement(**base, assessable=False, reason="no cited source could be read")

    out = llm(build_judge_prompt(fact, debate_date, pages))
    return FactJudgement(
        **base,
        assessable=True,
        sources_read=len(readable),
        quality=quality_score(out.axes),
        axes=out.axes,
        issues=list(out.issues),
        downgrade_to=forced_downgrade(fact.verdict, out.axes, out.recommend_downgrade),
    )


def judge_facts(
    facts: list[VerifiedFact],
    debate_date: date | None,
    *,
    fetch: Callable[[str], PageResult] = fetch_page,
    llm: Callable[[str], JudgeOutput],
    max_workers: int = 4,
) -> JudgeSummary:
    def _one(item: tuple[int, VerifiedFact]) -> FactJudgement:
        i, f = item
        try:
            return judge_fact(f, i, debate_date, fetch=fetch, llm=llm)
        except Exception as exc:  # noqa: BLE001 - one bad fact must not sink the pass
            logger.warning("Judge failed for fact %d: %s", i, exc)
            return FactJudgement(
                fact_index=i,
                claim=f.claim,
                verdict=f.verdict.value,
                assessable=False,
                reason=f"judge error: {exc}"[:300],
            )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        judgements = list(pool.map(_one, enumerate(facts)))  # map preserves order
    return summarize(judgements)


def default_judge_llm(settings) -> Callable[[str], JudgeOutput]:
    from src.llm import generate_json

    model = settings.judge_model or settings.gemini_model
    return lambda prompt: generate_json(
        prompt, JudgeOutput, settings, label="judge", model=model
    )
