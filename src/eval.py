"""Offline evaluation harness. Currently: re-judge stored reports.

    python -m src.eval rejudge --out data/judge_baseline.json

Establishes the evidence-quality baseline for the CURRENT pipeline before any
change lands, and scores every later run against it.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Callable

import click
from pydantic import BaseModel

from src.agents import ClaimCategory, ExtractedClaim, Verdict, VerifiedFact
from src.fidelity import FidelityOutput, audit_accusations
from src.judge import JudgeOutput, judge_facts
from src.tools.page import PageResult, fetch_page, parse_iso_date

_STEM = re.compile(r"^(\d+)(?:_v\d+)?$")


def base_id(stem: str) -> str | None:
    m = _STEM.match(stem)
    return m.group(1) if m else None


def load_dates(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def report_files(reports_dir: Path) -> list[Path]:
    return sorted(p for p in reports_dir.glob("*.json") if base_id(p.stem))


def rejudge_reports(
    reports_dir: Path,
    dates: dict[str, str],
    *,
    fetch: Callable[[str], PageResult] = fetch_page,
    llm: Callable[[str], JudgeOutput],
    only: set[str] | None = None,
) -> dict:
    reports: dict[str, dict] = {}
    skipped_no_date: list[str] = []
    total_judged = 0
    weighted = 0.0
    total_downgrades = 0

    for path in report_files(reports_dir):
        stem = path.stem
        if only and stem not in only and base_id(stem) not in only:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_date = payload.get("debate_date") or dates.get(base_id(stem) or "", "")
        debate: date | None = parse_iso_date(raw_date)
        if debate is None:
            skipped_no_date.append(stem)
            continue
        facts = [VerifiedFact.model_validate(f) for f in payload.get("facts", [])]
        summary = judge_facts(facts, debate, fetch=fetch, llm=llm)
        reports[stem] = {"date": debate.isoformat(), "summary": summary.model_dump(mode="json")}
        total_judged += summary.judged
        weighted += (summary.mean_quality or 0.0) * summary.judged
        total_downgrades += summary.downgrades

    return {
        "reports": reports,
        "skipped_no_date": skipped_no_date,
        "overall": {
            "judged": total_judged,
            "mean_quality": round(weighted / total_judged, 1) if total_judged else None,
            "downgrades": total_downgrades,
        },
    }


def fidelity_eval(
    gold: list[dict],
    reports_dir: Path,
    transcripts_dir: Path,
    *,
    llm: Callable[[str], FidelityOutput],
) -> dict:
    """Run the fidelity gate over stored reports and score it against hand labels.

    Labels: downgrade (the verdict refutes something not said), keep (sound
    accusation), doubtful. A case whose stored claim no longer starts with its
    `claim_prefix` is reported as stale instead of scored.
    """
    from src.transcript_lines import parse_lines

    by_report: dict[str, list[dict]] = {}
    for case in gold:
        by_report.setdefault(case["report"], []).append(case)

    cases: list[dict] = []
    stale: list[str] = []
    notes: list[str] = []
    for report, report_cases in by_report.items():
        payload = json.loads((reports_dir / f"{report}.json").read_text(encoding="utf-8"))
        facts = [VerifiedFact.model_validate(f) for f in payload.get("facts", [])]
        before = [f.verdict for f in facts]
        tpath = transcripts_dir / f"{report}.corrected.txt"
        lines = parse_lines(tpath.read_text(encoding="utf-8")) if tpath.exists() else []
        notes.extend(audit_accusations(facts, lines, llm=llm))
        for case in report_cases:
            i = case["index"]
            if not (0 <= i < len(facts)) or not facts[i].claim.startswith(case["claim_prefix"]):
                stale.append(f"{report}#{i}")
                continue
            cases.append(
                {
                    **case,
                    "before": before[i].value,
                    "after": facts[i].verdict.value,
                    "downgraded": facts[i].verdict != before[i],
                }
            )

    summary = {
        label: {
            "n": sum(1 for c in cases if c["label"] == label),
            "downgraded": sum(1 for c in cases if c["label"] == label and c["downgraded"]),
        }
        for label in ("downgrade", "keep", "doubtful")
    }
    return {"summary": summary, "cases": cases, "stale": stale, "notes": notes}


_ACCUSATIONS = (Verdict.FALSE, Verdict.MISLEADING)


class CategoryAssignment(BaseModel):
    claim_id: int
    category: ClaimCategory


class CategoryList(BaseModel):
    items: list[CategoryAssignment] = []


Recheck = Callable[
    [list[ExtractedClaim], list[ExtractedClaim], dict[int, VerifiedFact], date | None],
    list[VerifiedFact],
]


def rebuild_claims(facts: list[VerifiedFact]) -> list[ExtractedClaim]:
    """Stored facts as claims (id = index + 1); time fields are not stored and stay empty."""
    return [
        ExtractedClaim(
            id=i + 1,
            claim=f.claim,
            speaker=f.speaker,
            quote=f.quote,
            salience=f.salience,
            consequence=f.salience,
            category=f.category or ClaimCategory.CURRENT_EVENTS,
        )
        for i, f in enumerate(facts)
    ]


def assign_categories(
    claims: list[ExtractedClaim],
    facts: list[VerifiedFact],
    llm: Callable[[str], CategoryList],
) -> None:
    """Classify claims whose stored fact has no category, for specialist routing."""
    missing = [c for c, f in zip(claims, facts) if f.category is None]
    if not missing:
        return
    prompt = (
        "Assign each claim one category for specialist fact-checker routing:\n"
        "- economy_finance: budget, deficit, debt, consolidation, taxes, salaries of "
        "officials, public spending.\n"
        "- statistics: official statistical indicators (inflation/CPI, wages, "
        "unemployment, GDP, demographics).\n"
        "- history_politics: parliamentary votes, laws, past governments, past "
        "statements or positions.\n"
        "- current_events: recent events, quote attributions, media affairs.\n\n"
        + "\n".join(f"{c.id}. {c.claim}" for c in missing)
    )
    by_id = {c.id: c for c in missing}
    for item in llm(prompt).items:
        if item.claim_id in by_id:
            by_id[item.claim_id].category = item.category


def replay_report(
    report: str,
    cases: list[dict],
    reports_dir: Path,
    transcripts_dir: Path,
    dates: dict[str, str],
    *,
    interpret_llm,
    category_llm,
    recheck: Recheck | None,
    gate_llm,
    no_check: bool = False,
    window: int = 6,
    chunk: int = 20,
    run: int = 1,
) -> dict:
    """Re-interpret one stored report and re-check its gold cases."""
    from src.briefing import DebateBriefing
    from src.interpretation import interpret_claims
    from src.transcript_lines import parse_lines

    payload = json.loads((reports_dir / f"{report}.json").read_text(encoding="utf-8"))
    facts = [VerifiedFact.model_validate(f) for f in payload.get("facts", [])]
    debate = parse_iso_date(payload.get("debate_date") or dates.get(base_id(report) or "", ""))
    briefing = (
        DebateBriefing.model_validate(payload["briefing"]) if payload.get("briefing") else None
    )
    tpath = transcripts_dir / f"{report}.corrected.txt"
    lines = parse_lines(tpath.read_text(encoding="utf-8")) if tpath.exists() else []

    valid: list[dict] = []
    stale: list[str] = []
    for case in cases:
        i = case["index"]
        if not (0 <= i < len(facts)) or not facts[i].claim.startswith(case["claim_prefix"]):
            stale.append(f"{report}#{i}")
        else:
            valid.append(case)

    claims = rebuild_claims(facts)
    notes = interpret_claims(
        claims, lines, llm=interpret_llm, briefing=briefing, debate_date=debate,
        window=window, chunk=chunk,
    )
    after: dict[int, VerifiedFact] = {}
    if valid and not no_check:
        assert recheck is not None
        assign_categories(claims, facts, category_llm)
        targets = [claims[c["index"]] for c in valid]
        pool_facts = {i + 1: f for i, f in enumerate(facts)}
        new_facts = recheck(targets, claims, pool_facts, debate)
        for case, nf in zip(valid, new_facts):
            nf.timestamp = facts[case["index"]].timestamp
            after[case["index"]] = nf
        siblings = [after.get(i, f) for i, f in enumerate(facts)]
        notes.extend(audit_accusations(list(after.values()), lines, llm=gate_llm, siblings=siblings))

    rows = []
    for case in valid:
        i = case["index"]
        c, nf = claims[i], after.get(i)
        rows.append(
            {
                **case,
                "run": run,
                "claim_after": c.claim,
                "interpretation": c.interpretation,
                "not_asserted": c.not_asserted,
                "refutation_would_be": c.refutation_would_be,
                "before": facts[i].verdict.value,
                "after": nf.verdict.value if nf else None,
                "rationale": nf.rationale if nf else "",
            }
        )
    return {"cases": rows, "stale": stale, "notes": notes}


def interpret_replay(
    gold: list[dict],
    reports_dir: Path,
    transcripts_dir: Path,
    dates: dict[str, str],
    *,
    interpret_llm,
    category_llm,
    recheck: Recheck | None,
    gate_llm,
    labels: set[str],
    only: set[str] | None = None,
    runs: int = 1,
    no_check: bool = False,
    window: int = 6,
    chunk: int = 20,
) -> dict:
    """Re-interpret and re-check gold cases; count which stay accusations per label."""
    picked = [
        c
        for c in gold
        if c["label"] in labels and (not only or f"{c['report']}#{c['index']}" in only)
    ]
    by_report: dict[str, list[dict]] = {}
    for case in picked:
        by_report.setdefault(case["report"], []).append(case)

    cases: list[dict] = []
    stale: list[str] = []
    notes: list[str] = []
    for run in range(1, runs + 1):
        for report, report_cases in by_report.items():
            r = replay_report(
                report, report_cases, reports_dir, transcripts_dir, dates,
                interpret_llm=interpret_llm, category_llm=category_llm, recheck=recheck,
                gate_llm=gate_llm, no_check=no_check, window=window, chunk=chunk, run=run,
            )
            cases.extend(r["cases"])
            notes.extend(f"[run {run}] {report}: {n}" for n in r["notes"])
            if run == 1:
                stale.extend(r["stale"])

    accusations = {v.value for v in _ACCUSATIONS}
    summary = {
        label: {
            "n": sum(1 for c in cases if c["label"] == label),
            "accusation": sum(
                1 for c in cases if c["label"] == label and c["after"] in accusations
            ),
        }
        for label in sorted(labels)
    }
    return {"summary": summary, "cases": cases, "stale": stale, "notes": notes}


@click.group()
def cli() -> None:
    """Evaluation tools."""


@cli.command()
@click.option("--reports", "reports_dir", type=click.Path(path_type=Path, exists=True), default=Path("data/reports"))
@click.option("--dates", "dates_path", type=click.Path(path_type=Path), default=Path("data/debate_dates.json"))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path("data/judge_baseline.json"))
@click.option("--only", multiple=True, help="Episode id(s) to judge; default all.")
def rejudge(reports_dir: Path, dates_path: Path, out_path: Path, only: tuple[str, ...]) -> None:
    """Re-judge stored reports and print the run-level evidence quality."""
    from config import get_settings
    from src.costs import TRACKER
    from src.judge import default_judge_llm

    settings = get_settings()
    TRACKER.reset()
    result = rejudge_reports(
        reports_dir,
        load_dates(dates_path),
        llm=default_judge_llm(settings),
        only=set(only) or None,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    click.echo(f"{'report':<14}{'date':<12}{'judged':>7}{'skipped':>8}{'quality':>9}{'down':>6}")
    for stem, r in result["reports"].items():
        s = r["summary"]
        q = "-" if s["mean_quality"] is None else f"{s['mean_quality']:.1f}"
        click.echo(f"{stem:<14}{r['date']:<12}{s['judged']:>7}{s['skipped']:>8}{q:>9}{s['downgrades']:>6}")
    o = result["overall"]
    click.echo(f"OVERALL judged={o['judged']} mean_quality={o['mean_quality']} downgrades={o['downgrades']}")
    if result["skipped_no_date"]:
        click.echo("Skipped (no debate date): " + ", ".join(result["skipped_no_date"]))
    click.echo(TRACKER.summary(settings.judge_model or settings.gemini_model))
    click.echo(f"Written: {out_path}")


@cli.command()
@click.option("--gold", "gold_path", type=click.Path(path_type=Path, exists=True), default=Path("data/eval/fidelity_gold.json"))
@click.option("--reports", "reports_dir", type=click.Path(path_type=Path, exists=True), default=Path("data/reports"))
@click.option("--transcripts", "transcripts_dir", type=click.Path(path_type=Path, exists=True), default=Path("data/transcripts"))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path("data/eval/fidelity_result.json"))
def fidelity(gold_path: Path, reports_dir: Path, transcripts_dir: Path, out_path: Path) -> None:
    """Score the fidelity gate against the hand-labelled accusations."""
    from config import get_settings
    from src.costs import TRACKER
    from src.fidelity import default_fidelity_llm

    settings = get_settings()
    TRACKER.reset()
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    result = fidelity_eval(gold, reports_dir, transcripts_dir, llm=default_fidelity_llm(settings))
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for label, s in result["summary"].items():
        click.echo(f"{label:<10} n={s['n']:>3} downgraded={s['downgraded']:>3}")
    if result["stale"]:
        click.echo("Stale gold cases (report changed): " + ", ".join(result["stale"]))
    click.echo(TRACKER.summary(settings.judge_model or settings.gemini_model))
    click.echo(f"Written: {out_path}")


@cli.command("interpret-replay")
@click.option("--gold", "gold_path", type=click.Path(path_type=Path, exists=True), default=Path("data/eval/fidelity_gold.json"))
@click.option("--reports", "reports_dir", type=click.Path(path_type=Path, exists=True), default=Path("data/reports"))
@click.option("--transcripts", "transcripts_dir", type=click.Path(path_type=Path, exists=True), default=Path("data/transcripts"))
@click.option("--dates", "dates_path", type=click.Path(path_type=Path), default=Path("data/debate_dates.json"))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path("data/eval/interpret_replay.json"))
@click.option("--labels", default="downgrade,keep", help="Comma-separated gold labels to replay.")
@click.option("--only", multiple=True, help="Gold case(s) as REPORT#INDEX; default all with --labels.")
@click.option("--runs", default=1, type=int, help="Repeat the replay to expose variance.")
@click.option("--no-check", is_flag=True, help="Only re-interpret; print readings, no re-check.")
def interpret_replay_cmd(
    gold_path: Path,
    reports_dir: Path,
    transcripts_dir: Path,
    dates_path: Path,
    out_path: Path,
    labels: str,
    only: tuple[str, ...],
    runs: int,
    no_check: bool,
) -> None:
    """Re-interpret and re-check gold accusations without a full pipeline run."""
    from config import get_settings
    from src.agents import check_claims
    from src.costs import TRACKER
    from src.fidelity import default_fidelity_llm
    from src.interpretation import default_interpretation_llm
    from src.llm import generate_json

    settings = get_settings()
    TRACKER.reset()
    model = settings.judge_model or settings.gemini_model

    def category_llm(prompt: str) -> CategoryList:
        return generate_json(prompt, CategoryList, settings, label="replay_category", model=model)

    def recheck(targets, pool, pool_facts, debate):
        facts, _urls, _notes = check_claims(targets, settings, debate, pool=pool, pool_facts=pool_facts)
        return facts

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    result = interpret_replay(
        gold,
        reports_dir,
        transcripts_dir,
        load_dates(dates_path),
        interpret_llm=default_interpretation_llm(settings),
        category_llm=category_llm,
        recheck=recheck,
        gate_llm=default_fidelity_llm(settings),
        labels={s.strip() for s in labels.split(",") if s.strip()},
        only=set(only) or None,
        runs=runs,
        no_check=no_check,
        window=settings.interpretation_window,
        chunk=settings.interpretation_chunk,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for c in result["cases"]:
        click.echo(
            f"[run {c['run']}] {c['report']}#{c['index']} {c['label']:<9} "
            f"{c['before']} -> {c['after'] or '-'}\n"
            f"    claim: {c['claim_after']}\n"
            f"    reading: {c['interpretation'] or '(none)'}\n"
            f"    not asserted: {'; '.join(c['not_asserted']) or '(none)'}"
        )
    for label, s in result["summary"].items():
        click.echo(f"{label:<10} n={s['n']:>3} still_accusation={s['accusation']:>3}")
    if result["stale"]:
        click.echo("Stale gold cases (report changed): " + ", ".join(result["stale"]))
    click.echo(TRACKER.summary(model))
    click.echo(f"Written: {out_path}")


@cli.command("overlap-candidates")
@click.argument("episode")
@click.option("--top", default=10, type=int, help="Longest two-speaker regions to export.")
@click.option("--gold", "gold_path", type=click.Path(path_type=Path), default=Path("data/eval/overlap_gold.json"))
@click.option("--clips", "clips_dir", type=click.Path(path_type=Path), default=Path("data/eval/overlap_clips"))
def overlap_candidates(episode: str, top: int, gold_path: Path, clips_dir: Path) -> None:
    """Export overlap regions of an episode as clips plus gold entries to fill in."""
    from config import get_settings
    from src.overlap import find_overlap_regions
    from src.overlap_eval import build_candidates, merge_gold
    from src.transcription import load_audio, load_or_diarize

    settings = get_settings()
    audio_path = settings.audio_dir / f"{episode}.wav"
    diar = load_or_diarize(audio_path, episode, settings)
    regions = find_overlap_regions(diar.turns, min_seconds=settings.overlap_min_seconds)
    transcript = (settings.transcript_dir / f"{episode}.txt").read_text(encoding="utf-8")
    new = build_candidates(episode, diar, load_audio(audio_path), transcript, regions, clips_dir / episode, top)

    existing = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.exists() else []
    merged = merge_gold(existing, new)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(f"{len(regions)} overlap regions, exported {len(new)}; gold now has {len(merged)} entries")
    click.echo(f"Fill in 'gold' per speaker in {gold_path}, listening to each 'clip'.")


@cli.command()
@click.option("--gold", "gold_path", type=click.Path(path_type=Path, exists=True), default=Path("data/eval/overlap_gold.json"))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path("data/eval/overlap_result.json"))
def overlap(gold_path: Path, out_path: Path) -> None:
    """Score existing, exclusive and separated transcripts on the overlap gold set."""
    from config import get_settings
    from src.overlap import load_overlap_tools
    from src.overlap_eval import run_overlap_eval
    from src.transcription import WhisperASR, load_audio, load_or_diarize

    settings = get_settings()
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    episodes = sorted({g["episode"] for g in gold})
    audio, diar, transcripts = {}, {}, {}
    for ep in episodes:
        audio_path = settings.audio_dir / f"{ep}.wav"
        audio[ep] = load_audio(audio_path)
        diar[ep] = load_or_diarize(audio_path, ep, settings)
        transcripts[ep] = (settings.transcript_dir / f"{ep}.txt").read_text(encoding="utf-8")

    tools = load_overlap_tools(settings)
    asr = WhisperASR(settings)
    try:
        result = run_overlap_eval(
            gold, transcripts=transcripts, audio=audio, diar=diar, settings=settings, asr=asr, tools=tools
        )
    finally:
        asr.close()
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    click.echo(f"{'row':<11}{'regions':>8}{'dominant':>10}{'interrupter':>13}{'misattr':>9}")
    for name, row in result["rows"].items():
        click.echo(
            f"{name:<11}{row['regions']:>8}{row['dominant_recall']:>10.2f}"
            f"{row['interrupter_recall']:>13.2f}{row['misattributed']:>9}"
        )
    g = result["gate"]
    verdict = "PASS" if g["passed"] else "FAIL"
    click.echo(
        f"GATE {verdict}: interrupter +{g['interrupter_gain']:.2f} (need >= 0.25), "
        f"dominant -{g['dominant_drop']:.2f} (max 0.05), misattributed {g['misattributed_delta']:+d} (max 0)"
    )
    click.echo(f"Written: {out_path}")


if __name__ == "__main__":
    cli()
