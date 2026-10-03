#!/usr/bin/env python3
"""CLI orchestrator: STVR URL -> transcript + multi-agent JSON report."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from config import get_settings
from src.gcs import upload_file
from src.ingestion import ingest
from src.transcription import run_transcription

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _checked_name(name: str, option: str) -> str:
    """A speaker name the transcript can carry, or a usage error before any work."""
    from src.speakers import is_valid_speaker_name

    if not is_valid_speaker_name(name):
        raise click.UsageError(
            f"Invalid {option} name {name!r}: a speaker name must be 1-60 "
            "characters and must not contain '[' or ']'."
        )
    return name.strip()


def _split_guests(value: str | None) -> list[str] | None:
    """Parse the ';'-separated --guests value; no usable name means no roster."""
    names = [_checked_name(g, "--guests") for g in (value or "").split(";") if g.strip()]
    return names or None


def _verdict_line(verdict) -> str:
    """The overall-result line of the scoreboard print."""
    if verdict.scoring_status != "ok":
        return (
            f"  Hodnotenie je degradované (scoring_status={verdict.scoring_status}): "
            "víťaz sa neurčuje"
        )
    if verdict.winner:
        gap = f" (náskok {verdict.margin:.1f})" if verdict.margin is not None else ""
        return f"  Celkový víťaz: {verdict.winner}{gap}"
    if verdict.margin is not None:
        return f"  Výsledok nerozhodný (náskok lídra {verdict.margin:.1f})"
    return "  Celkový víťaz: —"


@click.command()
@click.option(
    "--url",
    required=True,
    help="STVR archive episode URL, e.g. https://www.stvr.sk/televizia/archiv/14036/<id>",
)
@click.option(
    "--skip-agents",
    is_flag=True,
    default=False,
    help="Stop after transcription (useful before GCP/Vertex is configured).",
)
@click.option(
    "--transcript",
    "transcript_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Reuse an existing transcript file and skip download/ASR.",
)
@click.option(
    "--episode-id",
    default=None,
    help="Episode id when using --transcript without downloading.",
)
@click.option(
    "--debate-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help=(
        "Broadcast date of the debate (YYYY-MM-DD). Used as the reference 'now' "
        "for fact-checking so post-debate events are treated as anachronistic."
    ),
)
@click.option(
    "--guests",
    default=None,
    help=(
        'Invited guests, ";"-separated, e.g. "Erik Tomáš;Marián Viskupič". '
        "Names the diarization labels."
    ),
)
@click.option("--moderator", default=None, help="Moderator name (default: Moderátor).")
def main(
    url: str,
    skip_agents: bool,
    transcript_path: Path | None,
    episode_id: str | None,
    debate_date,
    guests: str | None,
    moderator: str | None,
) -> None:
    # Names first: a name the transcript grammar cannot carry must not cost a
    # download, an ASR run, and a full agent pipeline before it is noticed.
    guest_names = _split_guests(guests)
    moderator_name = _checked_name(moderator, "--moderator") if moderator else None

    settings = get_settings()  # ensures dirs + applies credentials

    if transcript_path is not None:
        if not transcript_path.exists():
            raise click.ClickException(f"Transcript not found: {transcript_path}")
        ep_id = episode_id or transcript_path.stem
        transcript_file = transcript_path
        logger.info("Using existing transcript %s", transcript_file)
    else:
        logger.info("=== Ingestion ===")
        ep_id, _video, audio = ingest(url, settings=settings)
        logger.info("=== Transcription ===")
        transcript_file = run_transcription(audio, ep_id, settings=settings)

    transcript_text = transcript_file.read_text(encoding="utf-8")
    click.echo(f"Transcript: {transcript_file} ({len(transcript_text)} chars)")

    if skip_agents:
        click.echo("Skipping agents (--skip-agents).")
        return

    if not settings.vertex_ready():
        click.echo(
            "GCP/Vertex not configured (set GCP_PROJECT_ID). "
            "Transcription done; re-run without --skip-agents after auth.",
            err=True,
        )
        sys.exit(0)

    logger.info("=== Multi-agent analysis (Vertex Gemini) ===")
    try:
        from src.agents import run_analysis
        from src.costs import TRACKER
    except ImportError as exc:
        raise click.ClickException(f"Agent dependencies missing: {exc}") from exc

    TRACKER.reset()
    debate_day = debate_date.date() if debate_date is not None else None
    try:
        report, corrected = run_analysis(
            transcript_text,
            settings=settings,
            debate_date=debate_day,
            guests=guest_names,
            moderator=moderator_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent pipeline failed")
        raise click.ClickException(
            f"Agent analysis failed: {exc}. "
            "Ensure ADC is set (gcloud auth application-default login) "
            f"for project {settings.gcp_project_id}."
        ) from exc

    corrected_path = settings.transcript_dir / f"{ep_id}.corrected.txt"
    corrected_path.write_text(corrected.text.strip() + "\n", encoding="utf-8")
    upload_file(
        corrected_path,
        object_name=f"transcripts/{corrected_path.name}",
        settings=settings,
    )
    click.echo(f"Corrected transcript: {corrected_path}")
    if corrected.notes:
        for note in corrected.notes[:8]:
            click.echo(f"  · {note}")

    corrections_path = settings.transcript_dir / f"{ep_id}.corrections.json"
    corrections_path.write_text(
        json.dumps(
            [r.model_dump(mode="json") for r in corrected.log],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    upload_file(
        corrections_path,
        object_name=f"transcripts/{corrections_path.name}",
        settings=settings,
    )
    click.echo(f"Transcript edit log: {corrections_path}")
    if report.transcript_quality is not None:
        report.transcript_quality.log_path = f"transcripts/{corrections_path.name}"

    logger.info("=== Fair-play scoring + Facebook post ===")
    from src.agents import generate_facebook_post
    from src.scoring import score_report

    judge_summary = None
    if settings.judge_enabled:
        from src.judge import apply_judge_downgrades, default_judge_llm, judge_facts

        logger.info("=== Evidence judge ===")
        try:
            judge_summary = judge_facts(
                report.facts, debate_day, llm=default_judge_llm(settings)
            )
            judge_notes = apply_judge_downgrades(report.facts, judge_summary.judgements)
            report.critic_notes = [*report.critic_notes, *judge_notes]
            click.echo(
                f"Judge: mean evidence quality {judge_summary.mean_quality} over "
                f"{judge_summary.judged} facts, {judge_summary.downgrades} downgrades"
            )
        except Exception:  # noqa: BLE001 - never lose a finished report to the judge
            logger.exception("Evidence judge failed; writing report without it")
            judge_summary = None

    verdict = score_report(report, transcript=corrected.text)
    report_path = settings.report_dir / f"{ep_id}.json"
    payload = report.model_dump(mode="json")
    payload["verdict"] = verdict.model_dump(mode="json")
    payload["debate_date"] = debate_day.isoformat() if debate_day else None
    if judge_summary is not None:
        payload["judge"] = judge_summary.model_dump(mode="json")
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    upload_file(report_path, object_name=f"reports/{report_path.name}", settings=settings)
    click.echo(f"Report written: {report_path}")
    click.echo("Scoreboard:")
    click.echo(f"  Stav skórovania: {verdict.scoring_status}")
    for row in verdict.scoreboard:
        parts = ", ".join(
            f"{d.label}={d.score:.0f}" if d.score is not None else f"{d.label}=–"
            for d in row.disciplines
        )
        click.echo(
            f"  {row.speaker}: {row.score:.1f} (slová {row.words}, podiel slov "
            f"hostí {row.word_share_percent:.1f}%, prehovory {row.turns}, podstatné "
            f"otázky {row.challenging_questions}, vyhnutia {row.questions_dodged}, "
            f"tvrdenia {row.checked_claims}/{row.claims_selected}/"
            f"{row.claims_extracted}) [{parts}]"
        )
    for disc, name in verdict.discipline_winners.items():
        click.echo(f"  Víťaz disciplíny {disc}: {name}")
    click.echo(_verdict_line(verdict))
    try:
        post = generate_facebook_post(report, verdict, settings=settings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Facebook post generation failed")
        raise click.ClickException(f"Facebook post generation failed: {exc}") from exc

    post_txt = settings.posts_dir / f"{ep_id}.txt"
    post_json = settings.posts_dir / f"{ep_id}.json"
    post_txt.write_text(post.body, encoding="utf-8")
    post_json.write_text(
        json.dumps(
            {
                "verdict": verdict.model_dump(mode="json"),
                "post": post.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    upload_file(post_txt, object_name=f"posts/{post_txt.name}", settings=settings)
    upload_file(post_json, object_name=f"posts/{post_json.name}", settings=settings)
    click.echo(f"Post written: {post_txt}")
    click.echo(f"Post JSON: {post_json}")
    click.echo(TRACKER.summary(settings.gemini_model))


if __name__ == "__main__":
    main()
