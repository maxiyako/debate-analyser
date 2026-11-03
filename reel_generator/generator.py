"""Orchestrate one reel: highlight -> clip -> karaoke ASS -> match HUD -> MP4.

A counted foul (lie, misleading claim, attack, dodge) gets the scoreboard
instead of the blinking stamp. Other highlights still use the stamp.

    python -m reel_generator.generator \\
        --id 620752 \\
        --highlight 'behavioral_analysis.speakers[speaker="Erik Tomáš"].question_dodging[1]'

    python -m reel_generator.generator \\
        --id 620752 --speaker "Erik Tomáš" --type question_dodging --index 1 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from reel_generator.align import align_words
from reel_generator.config import ReelStyle
from reel_generator.crop import plan_speaker_crops, probe_media
from reel_generator.frame import frame_clip
from reel_generator.heard import captions_from_heard, snap_window, transcribe_span
from reel_generator.highlight import (
    CIVILITY_LISTS,
    BEHAVIOUR_LISTS,
    build_highlight_path,
    highlight_from_report,
    load_report,
)
from reel_generator.paths import (
    DEFAULT_DATA_DIR,
    PACKAGE_ROOT,
    expected_video,
    find_report,
    find_transcript,
    find_video,
)
from reel_generator.render import (
    build_ffmpeg_command,
    build_with_ffmpeg_python,
    ffmpeg_has_filter,
    format_command,
    run_ffmpeg,
)
from reel_generator.scoreboard import (
    board_for_highlight,
    burn_hud,
    collect_score_events,
    render_hud,
    scoreboard_applies,
)
from reel_generator.stamps import render_stamp
from reel_generator.subtitles import generate_ass, render_caption_video
from reel_generator.timeline import next_other_turn, plan_window, search_bounds, slice_words
from reel_generator.transcript import load_transcript, speech_segments

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReelResult:
    highlight_path: str
    speaker: str
    stamp_text: str
    clip_start: float
    clip_end: float
    stamp_at: float
    ass_path: Path
    stamp_path: Path | None
    video_path: Path
    output_path: Path
    command: list[str]
    rendered: bool


def generate(
    discussion_id: str,
    highlight: str,
    *,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    video: Path | None = None,
    style: ReelStyle | None = None,
    align: str = "auto",
    language: str = "sk",
    device: str = "cpu",
    dry_run: bool = False,
    engine: str = "argv",
    frame_with_llm: bool = True,
    hear: bool | None = None,
    answer_speed: float | None = None,
) -> ReelResult:
    style = style or ReelStyle()
    data_dir = data_dir or DEFAULT_DATA_DIR
    report = load_report(find_report(data_dir, discussion_id))
    item = highlight_from_report(report, highlight, discussion_id)
    if answer_speed is not None and item.outcome != "dodged":
        raise ValueError("answer_speed is only for a dodged question")
    lines = load_transcript(find_transcript(data_dir, discussion_id))
    window_start, window_end = search_bounds(item)
    media = video or find_video(data_dir, discussion_id)
    out_dir = output_dir or (PACKAGE_ROOT / "output" / discussion_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    words = align_words(
        lines,
        window_start,
        window_end,
        method="proportional",
        media=None,
    )
    question = item.kind in {"question_dodging", "question_audit"} and item.outcome in {
        "",
        "dodged",
        "partial",
        "interrupted",
    }
    if item.kind == "question_dodging":
        fallback_lead = style.question_lead_s
    else:
        fallback_lead = style.hook_lead_s
    fallback_tail = style.dodge_tail_s if question else style.claim_tail_s
    turn = next_other_turn(lines, item.timestamp_s, item.speaker) if question else None
    next_turn = ""
    if turn is not None:
        next_turn = f"{turn.speaker} [{turn.timestamp}]: {turn.text[:180]}"
    if frame_with_llm:
        choice = frame_clip(
            item,
            lines,
            fallback_lead=fallback_lead,
            fallback_tail=fallback_tail,
            next_turn=next_turn,
        )
        logger.warning(
            "Frame (%s): %.1fs before, %.1fs after.",
            choice.source,
            choice.lead_seconds,
            choice.tail_seconds,
        )
        lead_seconds = choice.lead_seconds
        tail_seconds = choice.tail_seconds
    else:
        lead_seconds = fallback_lead
        tail_seconds = fallback_tail
    reel = plan_window(
        item,
        lines,
        words,
        style,
        lead_seconds=lead_seconds,
        tail_seconds=tail_seconds,
    )
    if hear is None:
        hear = not dry_run
    heard_captions: list = []
    if hear and media is not None and media.is_file():
        try:
            heard = transcribe_span(
                media,
                max(0.0, reel.start - 4.0),
                reel.end + 3.0,
                out_dir,
                language,
            )
        except Exception as exc:
            logger.warning("Clip transcript failed (%s). Captions stay on the debate transcript.", exc)
            heard = []
        if heard:
            before = reel.start
            reel = snap_window(
                reel,
                item.quote,
                heard,
                lead=lead_seconds,
                tail=tail_seconds,
                lock_end=question,
                max_span=style.claim_max_s,
            )
            logger.warning(
                "Heard snap: clip start %.2fs -> %.2fs (stamp %.2fs).",
                before,
                reel.start,
                reel.stamp_at,
            )
            heard_captions = captions_from_heard(words, heard, reel.start, reel.end)
            if heard_captions:
                words = heard_captions
    if align != "proportional" and not heard_captions and media is not None and media.is_file():
        words = align_words(
            lines,
            max(0.0, reel.start - 0.4),
            reel.end + 0.4,
            method=align,
            media=media,
            work_dir=out_dir,
            language=language,
            device=device,
        )
    captions = slice_words(words, reel.start, reel.end)
    uses_board = scoreboard_applies(item)
    stamp_path = None
    if not uses_board:
        stamp_path = render_stamp(item, out_dir / f"{item.slug}.png", style)
    stamp_start = max(0.0, reel.stamp_at - reel.start)
    flag_end = reel.flag_end if reel.flag_end > reel.stamp_at else reel.end
    stamp_end = min(reel.duration, max(stamp_start + 0.4, flag_end - reel.start))
    ass_path = generate_ass(
        captions,
        out_dir / f"{item.slug}.ass",
        style,
        flag_start=stamp_start,
        flag_end=stamp_end,
    )

    video_ready = media is not None and media.is_file()
    video_path = media if video_ready else expected_video(data_dir, discussion_id)
    segments = speech_segments(lines, reel.start, reel.end)
    crop = plan_speaker_crops(
        media if video_ready else None,
        segments,
        reel.start,
        reel.duration,
        style,
        out_dir,
    )
    has_audio = True
    if video_ready:
        has_audio = probe_media(media).has_audio
    blink = True
    common = dict(
        clip_start=reel.start,
        duration=reel.duration,
        crop=crop,
        style=style,
        stamp_path=stamp_path,
        stamp_start=stamp_start,
        stamp_end=stamp_end,
        has_audio=has_audio,
        blink=blink,
    )
    picture = out_dir / f"{item.slug}.mp4"
    destination = out_dir / f"{item.slug}.hud.mp4" if uses_board else picture
    caption_video = None
    use_ass = ffmpeg_has_filter("ass") or ffmpeg_has_filter("subtitles")
    if engine == "argv" and not use_ass and not dry_run and video_ready:
        logger.warning(
            "ffmpeg has no libass (no ass/subtitles filter). "
            "Burning the same karaoke cues as a transparent overlay. "
            "The .ass file is still written for an ffmpeg built with libass."
        )
        caption_video = render_caption_video(
            captions,
            out_dir / f"{item.slug}.captions.mov",
            style,
            reel.duration,
            flag_start=stamp_start,
            flag_end=stamp_end,
        )
    if engine == "ffmpeg-python":
        command = build_with_ffmpeg_python(video_path, ass_path, picture, **common)
    elif engine == "argv":
        command = build_ffmpeg_command(
            video_path, ass_path, picture, caption_video=caption_video, **common
        )
    else:
        raise ValueError(f"unknown engine {engine!r}")

    rendered = False
    if not dry_run:
        if media is None or not media.is_file():
            raise FileNotFoundError(
                f"raw video missing for {discussion_id}. Expected {video_path}. "
                "ASS, stamp, and the ffmpeg command were still written."
            )
        run_ffmpeg(command)
        if uses_board:
            events = collect_score_events(report, lines, discussion_id)
            board = board_for_highlight(events, item, report)
            with tempfile.TemporaryDirectory(prefix="hud-") as raw:
                frames = render_hud(board, Path(raw), style)
                burn_hud(
                    picture,
                    destination,
                    frames,
                    stamp_start=stamp_start,
                    stamp_end=stamp_end,
                    style=style,
                    answer_speed=answer_speed,
                )
        rendered = True

    return ReelResult(
        highlight_path=highlight,
        speaker=item.speaker,
        stamp_text=item.stamp_text,
        clip_start=reel.start,
        clip_end=reel.end,
        stamp_at=reel.stamp_at,
        ass_path=ass_path,
        stamp_path=stamp_path,
        video_path=video_path,
        output_path=destination,
        command=command,
        rendered=rendered,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a 9:16 political reel from one report highlight.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--id", required=True, help="Discussion id, same stem as the report file.")
    parser.add_argument(
        "--highlight",
        help='JSON path, e.g. behavioral_analysis.speakers[speaker="Erik Tomáš"].question_dodging[1]',
    )
    parser.add_argument("--speaker", help="Used with --type and --index when --highlight is omitted.")
    parser.add_argument(
        "--type",
        choices=[*BEHAVIOUR_LISTS, *CIVILITY_LISTS, "facts", "question_audit"],
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--verdict", help='Narrow facts, e.g. False.')
    parser.add_argument("--outcome", help='Narrow question_audit, e.g. dodged.')
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--align", choices=["auto", "whisperx", "stable-ts", "proportional"], default="auto")
    parser.add_argument("--language", default="sk")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--engine", choices=["argv", "ffmpeg-python"], default="argv")
    parser.add_argument("--layout", choices=["auto", "cover", "blur"], default="auto")
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--font", default=None, help="ASS font name, e.g. 'Montserrat Black'.")
    parser.add_argument("--highlight-color", default=None, help="ASS colour &H00BBGGRR, default yellow.")
    parser.add_argument("--dry-run", action="store_true", help="Write the ASS and print ffmpeg. Do not encode.")
    parser.add_argument(
        "--answer-speed",
        type=float,
        default=None,
        help="Dodge reels only. Play the answer at this speed after the flag (2.5 is the match pace).",
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip the one framing call. Use the default answer length.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    path = args.highlight
    if not path:
        if not args.speaker or not args.type:
            parser.error("pass --highlight, or pass --speaker and --type")
        path = build_highlight_path(
            args.speaker, args.type, args.index, verdict=args.verdict, outcome=args.outcome
        )
    style = ReelStyle(layout=args.layout)
    if args.max_duration is not None:
        style = replace(style, max_duration=args.max_duration)
    if args.font:
        style = replace(style, font_name=args.font)
    if args.highlight_color:
        style = replace(style, highlight_ass=args.highlight_color)

    try:
        result = generate(
            args.id,
            path,
            data_dir=args.data_dir,
            output_dir=args.out_dir,
            video=args.video,
            style=style,
            align=args.align,
            language=args.language,
            device=args.device,
            dry_run=args.dry_run,
            engine=args.engine,
            frame_with_llm=not args.no_llm,
            answer_speed=args.answer_speed,
        )
    except (FileNotFoundError, KeyError, IndexError, ValueError, TypeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"highlight: {result.highlight_path}")
    print(f"speaker:   {result.speaker}")
    print(f"stamp:     {result.stamp_text} at {result.stamp_at:.2f}s")
    print(f"clip:      {result.clip_start:.2f}s -> {result.clip_end:.2f}s")
    print(f"ass:       {result.ass_path}")
    print(f"png:       {result.stamp_path or '(scoreboard, no stamp)'}")
    print(f"video:     {result.video_path}")
    print(f"output:    {result.output_path}")
    print(format_command(result.command))
    if not ffmpeg_has_filter("ass") and not ffmpeg_has_filter("subtitles"):
        print(
            "note: this ffmpeg has no libass, so encode burns karaoke from the "
            ".ass timings as a transparent overlay. Install ffmpeg with libass "
            "to use the ass= filter directly."
        )
    if result.rendered:
        print(f"wrote {result.output_path}")
    elif args.dry_run:
        print("dry-run: mp4 not encoded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
