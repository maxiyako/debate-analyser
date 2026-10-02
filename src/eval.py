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

from src.agents import VerifiedFact
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


if __name__ == "__main__":
    cli()
