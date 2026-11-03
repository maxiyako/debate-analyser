"""Match-style HUD. One discipline, two speakers, score ticks when the stamp hits.

The number on screen is how many events of this category that speaker already
has at this moment in the debate. The caught action adds one, in time with
the blinking stamp.
"""

from __future__ import annotations

import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from reel_generator.config import ReelStyle, find_font_file
from reel_generator.highlight import (
    Highlight,
    build_highlight_path,
    highlight_from_report,
)
from reel_generator.transcript import TranscriptLine, speaker_match

# key, on-screen label, chip colour
_CATEGORIES: dict[tuple[str, str], tuple[str, str, str]] = {
    ("fact", "False"): ("loz", "LOŽ", "#FF0000"),
    ("fact", "Misleading"): ("misleading", "ZAVÁDZAJÚCE", "#FF6A00"),
    ("incivility", ""): ("incivility", "ÚTOK", "#FF0000"),
    ("question_dodging", "dodged"): ("dodge", "NEODPOVEDANIE", "#FF0000"),
    ("question_audit", "dodged"): ("dodge", "NEODPOVEDANIE", "#FF0000"),
}

_STRIP_H = 96
_BAR_TOP = 8
_BAR_BOTTOM = 84
_FLASH_YELLOW = (255, 230, 0, 255)
_END_CARD_S = 3.0
_BANNER_BLINK_S = 1.0
_END_CARD_SIZE = 96
_END_CARD_LINES = (
    ("Chcete vedieť ako dopadol tento zápas?", (255, 255, 255, 255)),
    ("Sledujte stránku Politický VAR", _FLASH_YELLOW),
)
_FOOTER = "Finálne skóre zápasu je na stránke Politický VAR..."


@dataclass(frozen=True)
class ScoreEvent:
    speaker: str
    category: str
    timestamp_s: float
    order: int
    source_path: str


@dataclass(frozen=True)
class ScoreBoard:
    left_name: str
    right_name: str
    left_speaker: str
    right_speaker: str
    label: str
    accent: str
    left_before: int
    left_after: int
    right_before: int
    right_after: int
    scoring_side: str  # "left" or "right"


@dataclass(frozen=True)
class HudFrames:
    before: Path
    after: Path
    flash: Path


def scoreboard_applies(highlight: Highlight) -> bool:
    """True when the match panel, not the blinking stamp, marks this highlight."""
    try:
        category_of(highlight)
    except ValueError:
        return False
    return True


def category_of(highlight: Highlight) -> tuple[str, str, str]:
    key = (highlight.kind, highlight.outcome if highlight.kind == "fact" or highlight.outcome == "dodged" else "")
    if highlight.kind == "incivility":
        key = ("incivility", "")
    found = _CATEGORIES.get(key)
    if found is None:
        raise ValueError(
            f"no scoreboard category for {highlight.kind}/{highlight.outcome or '—'}"
        )
    return found


def collect_score_events(
    report: dict,
    lines: list[TranscriptLine],
    discussion_id: str,
) -> list[ScoreEvent]:
    """Every countable foul, in report order, with a debate timestamp."""
    events: list[ScoreEvent] = []
    order = 0
    facts = report.get("facts") or []
    seen: dict[tuple[str, str], int] = {}
    for fact in facts:
        speaker = str(fact.get("speaker") or "")
        verdict = str(fact.get("verdict") or "")
        if verdict not in {"False", "Misleading"}:
            continue
        index = seen.get((speaker, verdict), 0)
        seen[(speaker, verdict)] = index + 1
        path = build_highlight_path(speaker, "facts", index, verdict=verdict)
        item = highlight_from_report(report, path, discussion_id, lines)
        cat, _label, _accent = category_of(item)
        events.append(
            ScoreEvent(item.speaker, cat, item.timestamp_s, order, item.source_path)
        )
        order += 1

    for block in (report.get("behavioral_analysis") or {}).get("speakers") or []:
        speaker = str(block.get("speaker") or "")
        civility = block.get("civility") or {}
        for index, _text in enumerate(civility.get("incivility") or []):
            path = build_highlight_path(speaker, "incivility", index)
            item = highlight_from_report(report, path, discussion_id, lines)
            events.append(
                ScoreEvent(item.speaker, "incivility", item.timestamp_s, order, item.source_path)
            )
            order += 1
        for index, _text in enumerate(block.get("question_dodging") or []):
            path = build_highlight_path(speaker, "question_dodging", index)
            item = highlight_from_report(report, path, discussion_id, lines)
            if item.outcome != "dodged":
                continue
            events.append(
                ScoreEvent(item.speaker, "dodge", item.timestamp_s, order, item.source_path)
            )
            order += 1
    return events


def board_for_highlight(
    events: list[ScoreEvent],
    highlight: Highlight,
    report: dict,
) -> ScoreBoard:
    category, label, accent = category_of(highlight)
    current = next(
        (event for event in events if event.source_path == highlight.source_path),
        None,
    )
    if current is None:
        raise KeyError(f"highlight {highlight.source_path} is not in the score events")
    left_speaker, right_speaker = _debaters(report)
    moment = (current.timestamp_s, current.order)

    def tally(speaker: str) -> int:
        return sum(
            1
            for event in events
            if event.category == category
            and speaker_match(event.speaker, speaker)
            and (event.timestamp_s, event.order) < moment
        )

    left_before = tally(left_speaker)
    right_before = tally(right_speaker)
    scoring_side = "left" if speaker_match(highlight.speaker, left_speaker) else "right"
    return ScoreBoard(
        left_name=_short_name(left_speaker),
        right_name=_short_name(right_speaker),
        left_speaker=left_speaker,
        right_speaker=right_speaker,
        label=label,
        accent=accent,
        left_before=left_before,
        left_after=left_before + (1 if scoring_side == "left" else 0),
        right_before=right_before,
        right_after=right_before + (1 if scoring_side == "right" else 0),
        scoring_side=scoring_side,
    )


def render_hud(board: ScoreBoard, destination_dir: Path, style: ReelStyle | None = None) -> HudFrames:
    style = style or ReelStyle()
    destination_dir.mkdir(parents=True, exist_ok=True)
    font_file = find_font_file()
    before = destination_dir / "hud_before.png"
    after = destination_dir / "hud_after.png"
    flash = destination_dir / "hud_flash.png"
    _paint(before, board, board.left_before, board.right_before, style.width, font_file, flash=False)
    _paint(after, board, board.left_after, board.right_after, style.width, font_file, flash=False)
    _paint(flash, board, board.left_after, board.right_after, style.width, font_file, flash=True)
    return HudFrames(before, after, flash)


def _retimed_video(stamp_start: float, answer_speed: float | None) -> str:
    """Question stays at 1x. From the stamp on, the dodge answer can run faster."""
    if not answer_speed:
        return "[0:v]setpts=PTS[src];"
    start = f"{stamp_start:.3f}"
    return (
        f"[0:v]split[vhead][vtail];"
        f"[vhead]trim=end={start},setpts=PTS-STARTPTS,fps=30[head];"
        f"[vtail]trim=start={start},setpts=(PTS-STARTPTS)/{answer_speed:.3f},fps=30[tail];"
        f"[head][tail]concat=n=2:v=1:a=0[src];"
    )


def _retimed_audio(stamp_start: float, answer_speed: float | None, _duration_s: float) -> str:
    if not answer_speed:
        return f"[0:a]apad=pad_dur={_END_CARD_S:.3f}[aout]"
    start = f"{stamp_start:.3f}"
    return (
        f"[0:a]asplit[ahead][atail];"
        f"[ahead]atrim=end={start},asetpts=PTS-STARTPTS[ah];"
        f"[atail]atrim=start={start},asetpts=PTS-STARTPTS,atempo={answer_speed:.3f}[at];"
        f"[ah][at]concat=n=2:v=0:a=1,apad=pad_dur={_END_CARD_S:.3f}[aout]"
    )


def burn_hud(
    video: Path,
    destination: Path,
    frames: HudFrames,
    *,
    stamp_start: float,
    stamp_end: float,
    style: ReelStyle | None = None,
    answer_speed: float | None = None,
) -> None:
    """Overlay the HUD on a finished reel. Score swaps when the flagged line starts.

    The bar blinks yellow for one second from that moment, then holds the new
    score. The last 3 seconds are a blurred end card. `answer_speed` plays a
    dodge answer (from the flag on) faster.
    """
    style = style or ReelStyle()
    duration = _duration(video)
    if answer_speed is not None and answer_speed > 1:
        content = stamp_start + max(0.0, duration - stamp_start) / answer_speed
    else:
        content = duration
        answer_speed = None
    total = content + _END_CARD_S
    # Banner only. One second, twice the old stamp rate, then locked.
    period = (style.blink_on_s + style.blink_off_s) / 2
    on = style.blink_on_s / 2
    start = f"{stamp_start:.3f}"
    blink_end = stamp_start + _BANNER_BLINK_S
    end = f"{blink_end:.3f}"
    cut = f"{content:.3f}"
    phase = f"lt(mod(t-{start}\\,{period:.3f})\\,{on:.3f})"
    shown = f"gte(t\\,{start})*(gte(t\\,{end})+lt(t\\,{end})*{phase})"
    src = _retimed_video(stamp_start, answer_speed)
    graph = (
        f"{src}"
        f"[src][1:v]overlay=x=0:y=0:enable='lt(t\\,{start})'[pre];"
        f"[pre][2:v]overlay=x=0:y=0:enable='{shown}'[scored];"
        f"[scored][3:v]overlay=x=0:y=0:enable='gte(t\\,{start})*lt(t\\,{end})*{phase}'[played];"
        f"[played]tpad=stop_mode=clone:stop_duration={_END_CARD_S:.3f}[padded];"
        f"[padded]split[clean][toblur];"
        f"[toblur]gblur=sigma={style.blur_sigma:.0f}[blurred];"
        f"[clean][blurred]overlay=enable='gte(t\\,{cut})'[soft];"
        f"[soft][4:v]overlay=enable='gte(t\\,{cut})'[card];"
        f"[card][5:v]overlay=x=0:y=0[vout]"
    )
    end_dir = Path(tempfile.mkdtemp(prefix="endcard-"))
    end_card = end_dir / "endcard.png"
    footer = end_dir / "footer.png"
    render_end_card(end_card, style)
    render_footer(footer, style)
    if _has_audio(video):
        graph += ";" + _retimed_audio(stamp_start, answer_speed, duration)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-loop",
        "1",
        "-t",
        f"{total:.3f}",
        "-i",
        str(frames.before),
        "-loop",
        "1",
        "-t",
        f"{total:.3f}",
        "-i",
        str(frames.after),
        "-loop",
        "1",
        "-t",
        f"{total:.3f}",
        "-i",
        str(frames.flash),
        "-loop",
        "1",
        "-t",
        f"{total:.3f}",
        "-i",
        str(end_card),
        "-loop",
        "1",
        "-t",
        f"{total:.3f}",
        "-i",
        str(footer),
        "-filter_complex",
        graph,
        "-map",
        "[vout]",
    ]
    if _has_audio(video):
        command += ["-map", "[aout]", "-c:a", "aac", "-b:a", style.audio_bitrate]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        style.preset,
        "-crf",
        str(style.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    finally:
        end_card.unlink(missing_ok=True)
        footer.unlink(missing_ok=True)
        end_dir.rmdir()
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()[-20:]
        raise RuntimeError("scoreboard ffmpeg failed:\n" + "\n".join(tail))


def detect_stamp_window(video: Path, style: ReelStyle | None = None) -> tuple[float, float]:
    """First and last moment the coloured stamp is on screen.

    Samples the upper band, where the stamp is burned and the lower-thirds
    graphics are not.
    """
    style = style or ReelStyle()
    with tempfile.TemporaryDirectory(prefix="stampscan-") as raw:
        work = Path(raw)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(video),
                "-vf",
                "fps=10,scale=270:480",
                str(work / "f%03d.png"),
            ],
            check=True,
        )
        hits: list[float] = []
        frames = sorted(work.glob("f*.png"))
        for index, path in enumerate(frames):
            if _stamp_pixels(path) > 400:
                hits.append(index / 10.0)
    if not hits:
        raise RuntimeError(f"no stamp found in {video.name}")
    return hits[0], hits[-1] + style.blink_on_s * 0.5


def _stamp_pixels(path: Path) -> int:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    crop = image.crop((width // 5, 0, 4 * width // 5, int(height * 0.32)))
    count = 0
    for red, green, blue in crop.getdata():
        if red > 190 and blue < 100 and red > green + 50:
            count += 1
    return count


def _paint(
    destination: Path,
    board: ScoreBoard,
    left_score: int,
    right_score: int,
    width: int,
    font_file: Path | None,
    *,
    flash: bool,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGBA", (width, _STRIP_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    name_font = _font(ImageFont, font_file, 34)
    score_font = _font(ImageFont, font_file, 52)
    label_font = _font(ImageFont, font_file, 30)
    if flash:
        bar_fill = _FLASH_YELLOW
        name_fill = (10, 10, 10, 255)
        stroke = _FLASH_YELLOW
        chip_text = (255, 255, 255, 255)
    else:
        bar_fill = (0, 0, 0, 214)
        name_fill = (255, 255, 255, 255)
        stroke = (0, 0, 0, 255)
        chip_text = (255, 255, 255, 255)
    draw.rounded_rectangle(
        (16, _BAR_TOP, width - 16, _BAR_BOTTOM),
        radius=22,
        fill=bar_fill,
    )
    _chip(draw, board.label, label_font, width, _rgba(board.accent, 255), chip_text)
    for side, score, anchor, align in (
        ("left", left_score, 40, "left"),
        ("right", right_score, width - 40, "right"),
    ):
        _side(
            draw,
            f"{board.left_name if side == 'left' else board.right_name}:",
            str(score),
            name_font,
            score_font,
            x=anchor,
            align=align,
            name_fill=name_fill,
            score_fill=_score_fill(board, side, score, flash=flash),
            stroke_fill=stroke,
        )
    canvas.save(destination)


def _score_is_hot(board: ScoreBoard, side: str, score: int) -> bool:
    before = board.left_before if side == "left" else board.right_before
    return side == board.scoring_side and score != before


def _score_fill(
    board: ScoreBoard, side: str, score: int, *, flash: bool
) -> tuple[int, int, int, int]:
    if _score_is_hot(board, side, score):
        if flash:
            return (10, 10, 10, 255)
        return (255, 230, 0, 255)
    if flash:
        return (10, 10, 10, 255)
    return (255, 255, 255, 255)


def _chip(
    draw,
    label: str,
    font,
    width: int,
    fill: tuple[int, int, int, int],
    text_fill: tuple[int, int, int, int],
) -> None:
    box = draw.textbbox((0, 0), label, font=font)
    text_w = box[2] - box[0]
    text_h = box[3] - box[1]
    cx = width / 2
    cy = (_BAR_TOP + _BAR_BOTTOM) / 2
    pad_x, pad_y = 16, 10
    rect = (
        cx - text_w / 2 - pad_x,
        cy - text_h / 2 - pad_y,
        cx + text_w / 2 + pad_x,
        cy + text_h / 2 + pad_y,
    )
    draw.rounded_rectangle(rect, radius=14, fill=fill)
    draw.text(
        (cx - text_w / 2 - box[0], cy - text_h / 2 - box[1]),
        label,
        font=font,
        fill=text_fill,
    )


def _side(
    draw,
    name: str,
    score: str,
    name_font,
    score_font,
    *,
    x: float,
    align: str,
    name_fill: tuple[int, int, int, int],
    score_fill: tuple[int, int, int, int],
    stroke_fill: tuple[int, int, int, int],
) -> None:
    name_box = draw.textbbox((0, 0), name, font=name_font)
    score_box = draw.textbbox((0, 0), score, font=score_font, stroke_width=5)
    name_w = name_box[2] - name_box[0]
    score_w = score_box[2] - score_box[0]
    gap = 12
    total = name_w + gap + score_w
    origin = x if align == "left" else x - total
    cy = (_BAR_TOP + _BAR_BOTTOM) / 2
    name_h = name_box[3] - name_box[1]
    score_h = score_box[3] - score_box[1]
    name_y = cy - name_h / 2 - name_box[1]
    score_y = cy - score_h / 2 - score_box[1]
    draw.text(
        (origin, name_y),
        name,
        font=name_font,
        fill=name_fill,
    )
    draw.text(
        (origin + name_w + gap, score_y),
        score,
        font=score_font,
        fill=score_fill,
        stroke_width=5,
        stroke_fill=stroke_fill,
    )


def _debaters(report: dict) -> tuple[str, str]:
    rows = ((report.get("verdict") or {}).get("scoreboard")) or []
    names = [str(row.get("speaker") or "") for row in rows if row.get("speaker")]
    if len(names) >= 2:
        return names[0], names[1]
    blocks = (report.get("behavioral_analysis") or {}).get("speakers") or []
    names = [str(block.get("speaker") or "") for block in blocks if block.get("speaker")]
    if len(names) >= 2:
        return names[0], names[1]
    raise ValueError("report has fewer than two speakers")


def _short_name(full: str) -> str:
    last = full.split()[-1]
    folded = unicodedata.normalize("NFKD", last)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return folded[:1].upper() + folded[1:]


def _font(image_font, font_file: Path | None, size: int):
    if font_file is not None:
        return image_font.truetype(str(font_file), size=size)
    return image_font.load_default()


def _rgba(hex_colour: str, alpha: int) -> tuple[int, int, int, int]:
    hex_colour = hex_colour.lstrip("#")
    return (int(hex_colour[0:2], 16), int(hex_colour[2:4], 16), int(hex_colour[4:6], 16), alpha)


def render_end_card(destination: Path, style: ReelStyle | None = None) -> Path:
    """Centered end-card line in the same face as the scoreboard and stamps."""
    from PIL import Image, ImageDraw, ImageFont

    style = style or ReelStyle()
    destination.parent.mkdir(parents=True, exist_ok=True)
    font_file = find_font_file()
    canvas = Image.new("RGBA", (style.width, style.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    top_limit = 360
    bottom_limit = style.height - 180
    size = _END_CARD_SIZE
    font = _font(ImageFont, font_file, size)
    stroke = max(6, size // 10)
    max_width = style.width - 100
    rows = _end_lines(draw, font, max_width, stroke)
    gap = 22
    total_h = _block_height(draw, [text for text, _fill in rows], font, stroke, gap)
    while size > 72 and (
        total_h > bottom_limit - top_limit
        or _widest(draw, [text for text, _fill in rows], font, stroke) > max_width
    ):
        size -= 4
        font = _font(ImageFont, font_file, size)
        stroke = max(8, size // 8)
        rows = _end_lines(draw, font, max_width, stroke)
        total_h = _block_height(draw, [text for text, _fill in rows], font, stroke, gap)
    boxes = [draw.textbbox((0, 0), text, font=font, stroke_width=stroke) for text, _fill in rows]
    heights = [box[3] - box[1] for box in boxes]
    y = top_limit + max(0, (bottom_limit - top_limit - total_h) / 2)
    for (text, fill), box, height in zip(rows, boxes, heights):
        width = box[2] - box[0]
        x = (style.width - width) / 2 - box[0]
        draw.text(
            (x, y - box[1]),
            text,
            font=font,
            fill=fill,
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, 255),
        )
        y += height + gap
    canvas.save(destination)
    return destination


def render_footer(destination: Path, style: ReelStyle | None = None) -> Path:
    """Small disclaimer along the bottom edge, on for the whole reel."""
    from PIL import Image, ImageDraw, ImageFont

    style = style or ReelStyle()
    destination.parent.mkdir(parents=True, exist_ok=True)
    font_file = find_font_file()
    canvas = Image.new("RGBA", (style.width, style.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _font(ImageFont, font_file, 36)
    stroke = 4
    box = draw.textbbox((0, 0), _FOOTER, font=font, stroke_width=stroke)
    text_w = box[2] - box[0]
    text_h = box[3] - box[1]
    cx = style.width / 2
    pad_x, pad_y = 28, 8
    bottom = style.height - 16
    rect = (
        cx - text_w / 2 - pad_x,
        bottom - text_h - pad_y * 2,
        cx + text_w / 2 + pad_x,
        bottom,
    )
    draw.rounded_rectangle(rect, radius=14, fill=(72, 72, 72, 168))
    draw.text(
        (cx - text_w / 2 - box[0], rect[1] + pad_y - box[1]),
        _FOOTER,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 255),
    )
    canvas.save(destination)
    return destination


def _end_lines(draw, font, max_width: int, stroke: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    rows: list[tuple[str, tuple[int, int, int, int]]] = []
    for sentence, fill in _END_CARD_LINES:
        for line in _wrap_line(draw, sentence, font, max_width, stroke):
            rows.append((line, fill))
    return rows


def _block_height(draw, lines: list[str], font, stroke: int, gap: int) -> int:
    heights = [
        draw.textbbox((0, 0), line, font=font, stroke_width=stroke)[3]
        - draw.textbbox((0, 0), line, font=font, stroke_width=stroke)[1]
        for line in lines
    ]
    return sum(heights) + gap * (len(lines) - 1)


def _widest(draw, lines: list[str], font, stroke: int) -> int:
    return max(_line_width(draw, line, font, stroke) for line in lines)


def _wrap_line(draw, text: str, font, max_width: int, stroke: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if _line_width(draw, trial, font, stroke) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _line_width(draw, line: str, font, stroke: int = 8) -> int:
    box = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
    return box[2] - box[0]


def _has_audio(video: Path) -> bool:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool((completed.stdout or "").strip())


def _duration(video: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(completed.stdout.strip())
