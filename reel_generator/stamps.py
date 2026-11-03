"""Transparent PNG stamps. Colours come from ReelStyle.stamps."""

from __future__ import annotations

import subprocess
from pathlib import Path

from reel_generator.config import ReelStyle, StampStyle, find_font_file
from reel_generator.highlight import Highlight


def render_stamp(highlight: Highlight, destination: Path, style: ReelStyle | None = None) -> Path:
    style = style or ReelStyle()
    look = style.stamp_style(highlight.stamp_key)
    text = highlight.stamp_text.strip()
    layout = "banner" if len(text) > 12 else look.layout
    destination.parent.mkdir(parents=True, exist_ok=True)
    font = find_font_file()
    try:
        _render_pillow(destination, text, look, layout, font)
    except Exception:
        _render_ffmpeg(destination, text, look, layout, font)
    return destination


def _render_pillow(
    destination: Path,
    text: str,
    look: StampStyle,
    layout: str,
    font_file: Path | None,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if layout == "stamp":
        canvas = Image.new("RGBA", (760, 760), (0, 0, 0, 0))
        plate = Image.new("RGBA", (760, 760), (0, 0, 0, 0))
        draw = ImageDraw.Draw(plate)
        draw.rounded_rectangle((80, 250, 680, 510), radius=36, fill=_rgba(look.background, 235))
        draw.rounded_rectangle((80, 250, 680, 510), radius=36, outline=(255, 255, 255, 255), width=10)
        font = _load_font(ImageFont, font_file, 92)
        _draw_centered(draw, text, font, (380, 380), _rgba(look.foreground, 255))
        canvas = plate.rotate(-12, expand=True, resample=Image.Resampling.BICUBIC)
        canvas = _trim_alpha(canvas)
    else:
        lines = _banner_lines(text)
        height = 300 if len(lines) > 1 else 240
        canvas = Image.new("RGBA", (1000, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((16, 28, 984, height - 28), radius=28, fill=_rgba(look.background, 240))
        draw.rounded_rectangle(
            (16, 28, 984, height - 28), radius=28, outline=(0, 0, 0, 255), width=8
        )
        font = _load_font(ImageFont, font_file, 54 if len(lines) == 1 else 46)
        for index, line in enumerate(lines):
            y = height / 2 + (index - (len(lines) - 1) / 2) * 62
            _draw_centered(draw, line, font, (500, y), _rgba(look.foreground, 255))
        canvas = _trim_alpha(canvas)
    canvas.save(destination)


def _render_ffmpeg(
    destination: Path,
    text: str,
    look: StampStyle,
    layout: str,
    font_file: Path | None,
) -> None:
    if font_file is None:
        raise RuntimeError("no font file for stamp text, and Pillow is unavailable")
    width, height = (1000, 260) if layout == "banner" else (720, 420)
    colour = look.background.lstrip("#")
    font_colour = "black" if look.foreground.upper() in {"#111111", "#000000"} else "white"
    safe = text.replace(":", "\\:").replace("'", "")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x{colour}@{0.92}:s={width}x{height},format=rgba",
            "-vf",
            f"drawtext=fontfile='{font_file}':text='{safe}':fontsize=64:"
            f"fontcolor={font_colour}:x=(w-text_w)/2:y=(h-text_h)/2",
            "-frames:v",
            "1",
            str(destination),
        ],
        check=True,
    )


def _trim_alpha(image):
    bbox = image.getbbox()
    if bbox is None:
        return image
    pad = 6
    return image.crop(
        (
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(image.width, bbox[2] + pad),
            min(image.height, bbox[3] + pad),
        )
    )


def _banner_lines(text: str) -> list[str]:
    if ": " in text and len(text) > 18:
        head, tail = text.split(": ", 1)
        return [head + ":", tail]
    return [text]


def _load_font(image_font, font_file: Path | None, size: int):
    if font_file is not None:
        return image_font.truetype(str(font_file), size=size)
    return image_font.load_default()


def _draw_centered(draw, text: str, font, center: tuple[float, float], fill: tuple[int, int, int, int]) -> None:
    box = draw.textbbox((0, 0), text, font=font, stroke_width=6)
    width = box[2] - box[0]
    height = box[3] - box[1]
    x = center[0] - width / 2
    y = center[1] - height / 2
    stroke = (0, 0, 0, 255) if fill[0] > 200 else (255, 255, 255, 255)
    draw.text((x, y), text, font=font, fill=fill, stroke_width=6, stroke_fill=stroke)


def _rgba(hex_colour: str, alpha: int) -> tuple[int, int, int, int]:
    hex_colour = hex_colour.lstrip("#")
    return (int(hex_colour[0:2], 16), int(hex_colour[2:4], 16), int(hex_colour[4:6], 16), alpha)
