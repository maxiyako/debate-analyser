"""Evidence judge: does the evidence cited for a verdict actually justify it?

The judge reads the cited pages, rates five axes (0-2 each), and may force a
DOWNGRADE to Unverified — never an upgrade. Quality score and downgrade
decisions are computed in code from the axes; the model's own opinion of
"overall quality" is never trusted. Aggregated, the per-fact quality becomes a
run-level number stored with the report (the evaluation harness the spec asks
for).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agents import Verdict, VerifiedFact

_JUDGED = (Verdict.TRUE, Verdict.FALSE, Verdict.MISLEADING)
_ACCUSATIONS = (Verdict.FALSE, Verdict.MISLEADING)
_WEIGHTS = {
    "support": 0.35,
    "metric_fidelity": 0.25,
    "date_fit": 0.15,
    "coverage": 0.15,
    "independence": 0.10,
}


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
