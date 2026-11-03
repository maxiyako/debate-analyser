"""Hormozi-style karaoke captions as Advanced SubStation Alpha.

Only the word being spoken is yellow. Earlier words in the same 3-word chunk
stay white. Standard ASS `\\k` would leave already-sung words filled, so each
chunk is one Dialogue event per word, with that word recoloured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reel_generator.align import Word
from reel_generator.config import ReelStyle


@dataclass(frozen=True)
class CaptionCue:
    start: float
    end: float
    words: list[Word]
    active: int


def generate_ass(
    words: list[Word],
    destination: Path,
    style: ReelStyle | None = None,
) -> Path:
    """Write an `.ass` file. `words` are already clip-relative seconds."""
    style = style or ReelStyle()
    destination.parent.mkdir(parents=True, exist_ok=True)
    events = [_dialogue(cue, style) for cue in caption_cues(words, style)]
    destination.write_text(_header(style) + "".join(events), encoding="utf-8")
    return destination


def caption_cues(words: list[Word], style: ReelStyle) -> list[CaptionCue]:
    cues: list[CaptionCue] = []
    for chunk in _chunks(words, style):
        for index, _word in enumerate(chunk):
            start = chunk[index].start
            if index + 1 < len(chunk):
                end = chunk[index + 1].start
            else:
                end = max(chunk[index].end, start + 0.12)
            if end - start < 0.04:
                end = start + 0.04
            cues.append(CaptionCue(start, end, chunk, index))
    return cues


def render_caption_video(
    words: list[Word],
    destination: Path,
    style: ReelStyle,
    duration: float,
) -> Path:
    """Burn the same karaoke cues into a transparent movie.

    Used when the local ffmpeg was built without libass, so the `ass` filter
    does not exist. The `.ass` file is still written for a libass ffmpeg.
    """
    import subprocess

    from PIL import Image, ImageDraw, ImageFont

    from reel_generator.config import find_font_file

    destination.parent.mkdir(parents=True, exist_ok=True)
    cues = caption_cues(words, style)
    font_file = find_font_file()
    font = _pil_font(ImageFont, font_file, style.font_size)
    active_font = _pil_font(
        ImageFont, font_file, int(style.font_size * style.active_scale / 100)
    )
    blank = Image.new("RGBA", (style.width, style.height), (0, 0, 0, 0))
    blank_bytes = blank.tobytes()
    drawn: dict[int, bytes] = {}
    log_path = destination.with_suffix(".ffmpeg.log")
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{style.width}x{style.height}",
        "-r",
        str(style.fps),
        "-i",
        "-",
        "-c:v",
        "qtrle",
        "-pix_fmt",
        "argb",
        str(destination),
    ]
    frames = max(1, int(round(duration * style.fps)))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        assert proc.stdin is not None
        try:
            for index in range(frames):
                moment = index / style.fps
                cue_index = _cue_index(cues, moment)
                if cue_index is None:
                    raw = blank_bytes
                elif cue_index in drawn:
                    raw = drawn[cue_index]
                else:
                    image = _draw_cue(cues[cue_index], style, font, active_font, ImageDraw)
                    raw = image.tobytes()
                    drawn[cue_index] = raw
                proc.stdin.write(raw)
        finally:
            proc.stdin.close()
            code = proc.wait()
    if code != 0:
        detail = log_path.read_text(encoding="utf-8").strip()
        raise RuntimeError(f"caption overlay encode failed: {detail}")
    return destination


def _header(style: ReelStyle) -> str:
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {style.width}\n"
        f"PlayResY: {style.height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Reel,"
        f"{style.font_name},{style.font_size},"
        f"{style.primary_ass},{style.highlight_ass},{style.outline_ass},{style.shadow_ass},"
        f"-1,0,0,0,100,100,0,0,1,{style.outline},{style.shadow},5,40,40,0,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _dialogue(cue: CaptionCue, style: ReelStyle) -> str:
    text = _chunk_text(cue.words, cue.active, style)
    return (
        f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},Reel,,0,0,0,,{text}\n"
    )


def _chunks(words: list[Word], style: ReelStyle) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    chars = 0
    for word in words:
        token = word.text.strip()
        if not token:
            continue
        gap = word.start - current[-1].end if current else 0.0
        overflow = len(current) >= style.max_words or chars + len(token) > style.max_chars
        if current and (overflow or gap > style.max_gap_s):
            chunks.append(current)
            current = []
            chars = 0
        current.append(word)
        chars += len(token) + 1
    if current:
        chunks.append(current)
    return chunks


def _chunk_text(chunk: list[Word], active: int, style: ReelStyle) -> str:
    x = style.width // 2
    parts = [f"{{\\an5\\pos({x},{style.subtitle_y})}}"]
    for index, word in enumerate(chunk):
        token = _escape(word.text.strip())
        if index == active:
            parts.append(
                f"{{\\fscx{style.active_scale}\\fscy{style.active_scale}"
                f"\\c{style.highlight_ass}}}{token}"
            )
        else:
            parts.append(f"{{\\fscx100\\fscy100\\c{style.primary_ass}}}{token}")
        if index + 1 < len(chunk):
            parts.append(" ")
    return "".join(parts)


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    secs, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _cue_index(cues: list[CaptionCue], moment: float) -> int | None:
    for index, cue in enumerate(cues):
        if cue.start <= moment < cue.end:
            return index
    return None


def _draw_cue(cue: CaptionCue, style: ReelStyle, font, active_font, image_draw) -> object:
    from PIL import Image

    image = Image.new("RGBA", (style.width, style.height), (0, 0, 0, 0))
    draw = image_draw.Draw(image)
    gap = 16
    pieces: list[tuple[str, object, int, int]] = []
    for index, word in enumerate(cue.words):
        face = active_font if index == cue.active else font
        box = draw.textbbox((0, 0), word.text, font=face, stroke_width=style.outline)
        pieces.append((word.text, face, box[2] - box[0], box[3] - box[1]))
    total = sum(width for _, _, width, _ in pieces) + gap * max(0, len(pieces) - 1)
    x = (style.width - total) / 2
    for index, (token, face, width, height) in enumerate(pieces):
        y = style.subtitle_y - height / 2
        fill = (255, 255, 0, 255) if index == cue.active else (255, 255, 255, 255)
        draw.text((x + 4, y + 4), token, font=face, fill=(0, 0, 0, 170))
        draw.text(
            (x, y),
            token,
            font=face,
            fill=fill,
            stroke_width=style.outline,
            stroke_fill=(0, 0, 0, 255),
        )
        x += width + gap
    return image


def _pil_font(image_font, font_file: Path | None, size: int):
    if font_file is not None:
        return image_font.truetype(str(font_file), size=size)
    return image_font.load_default()


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")
