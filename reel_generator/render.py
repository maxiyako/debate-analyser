"""One-pass FFmpeg graph: crop (or blur bars) + stamp pop + ASS karaoke.

`build_ffmpeg_command` is the command the pipeline runs.
`build_with_ffmpeg_python` is the same graph via the ffmpeg-python API.
"""

from __future__ import annotations

import shlex
import subprocess
from functools import lru_cache
from pathlib import Path

from reel_generator.config import ReelStyle
from reel_generator.crop import (
    CropPlan,
    cover_crop_filter,
    cover_width,
    crop_x_expr,
    wide_enable_expr,
)


def build_ffmpeg_command(
    video: Path,
    ass_path: Path,
    destination: Path,
    *,
    clip_start: float,
    duration: float,
    crop: CropPlan,
    style: ReelStyle,
    stamp_path: Path | None = None,
    stamp_start: float = 0.0,
    stamp_end: float = 0.0,
    has_audio: bool = True,
    caption_video: Path | None = None,
    blink: bool = False,
) -> list[str]:
    """Return the argv that burns crop, subtitles, and the PNG overlay together."""
    caption_input = None
    if caption_video is not None:
        caption_input = 2 if stamp_path is not None else 1
    graph = filtergraph(
        ass_path,
        crop,
        style,
        stamp=stamp_path is not None,
        stamp_start=stamp_start,
        stamp_end=stamp_end,
        caption_input=caption_input,
        blink=blink,
    )
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{clip_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video),
    ]
    if stamp_path is not None:
        command += [
            "-loop",
            "1",
            "-framerate",
            str(style.fps),
            "-t",
            f"{duration:.3f}",
            "-i",
            str(stamp_path),
        ]
    if caption_video is not None:
        command += ["-i", str(caption_video)]
    command += [
        "-filter_complex",
        graph,
        "-map",
        "[vout]",
    ]
    if has_audio:
        command += ["-map", "0:a", "-c:a", "aac", "-b:a", style.audio_bitrate, "-ac", "2"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        style.preset,
        "-crf",
        str(style.crf),
        "-r",
        str(style.fps),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    return command


def filtergraph(
    ass_path: Path,
    crop: CropPlan,
    style: ReelStyle,
    *,
    stamp: bool,
    stamp_start: float,
    stamp_end: float,
    caption_input: int | None = None,
    blink: bool = False,
) -> str:
    """Filtergraph string. Commas inside option values are escaped for FFmpeg.

    `caption_input` is set when ffmpeg has no libass filter. Captions are then
    a transparent movie on that input instead of `ass=filename=...`.
    """
    if crop.mode == "cover":
        base = (
            f"[0:v]{cover_crop_filter(crop)},"
            f"scale={style.width}:{style.height}:flags=lanczos,setsar=1[base]"
        )
    elif crop.mode == "fit":
        # Crop width cannot change per frame. Close-ups stay 9:16. A two-face
        # shot swaps in the full frame, fitted to the width, with blur bars.
        wide = wide_enable_expr(crop)
        x_expr = crop_x_expr(crop)
        base = (
            "[0:v]split=3[bg_src][cover_src][wide_src];"
            f"[bg_src]scale={style.width}:{style.height}:force_original_aspect_ratio=increase,"
            f"crop={style.width}:{style.height},gblur=sigma={style.blur_sigma:.0f}[bg];"
            f"[cover_src]crop={cover_width(crop)}:{crop.h}:x='{x_expr}':y={crop.y},"
            f"scale={style.width}:{style.height}:flags=lanczos[cover];"
            f"[wide_src]scale={style.width}:-2:flags=lanczos[wide];"
            f"[bg][cover]overlay=x=(W-w)/2:y=(H-h)/2:enable='lt({wide}+0\\,0.5)'[mid];"
            f"[mid][wide]overlay=x=(W-w)/2:y=(H-h)/2:enable='gte({wide}\\,0.5)'[base]"
        )
    else:
        base = (
            "[0:v]split[bg_src][fg_src];"
            f"[bg_src]scale={style.width}:{style.height}:force_original_aspect_ratio=increase,"
            f"crop={style.width}:{style.height},gblur=sigma={style.blur_sigma:.0f}[bg];"
            f"[fg_src]scale={style.width}:-2:flags=lanczos[fg];"
            "[bg][fg]overlay=x=(W-w)/2:y=(H-h)/2[base]"
        )
    parts = [base]
    current = "[base]"
    if stamp:
        pop = _pop_scale(stamp_start)
        fades = f"fade=t=in:st={stamp_start:.3f}:d=0.12:alpha=1,"
        if not blink:
            fade_out = max(stamp_start, stamp_end - 0.15)
            fades += f"fade=t=out:st={fade_out:.3f}:d=0.15:alpha=1,"
        parts.append(
            "[1:v]format=rgba,"
            f"{fades}"
            f"{pop},format=rgba[stamp]"
        )
        parts.append(
            f"{current}[stamp]overlay=x=(W-w)/2:y={style.stamp_y}:"
            f"enable='{_stamp_enable(stamp_start, stamp_end, blink, style)}'[withstamp]"
        )
        current = "[withstamp]"
    if caption_input is None:
        parts.append(_ass_filter(current, ass_path, style))
    else:
        parts.append(
            f"{current}[{caption_input}:v]overlay=x=0:y=0:format=auto[vout]"
        )
    return ";".join(parts)


def build_with_ffmpeg_python(
    video: Path,
    ass_path: Path,
    destination: Path,
    *,
    clip_start: float,
    duration: float,
    crop: CropPlan,
    style: ReelStyle,
    stamp_path: Path | None = None,
    stamp_start: float = 0.0,
    stamp_end: float = 0.0,
    has_audio: bool = True,
    blink: bool = False,
) -> list[str]:
    """Same render expressed with ffmpeg-python. Swap this in for the argv builder."""
    import ffmpeg

    inp = ffmpeg.input(str(video), ss=clip_start, t=duration)
    if crop.mode == "cover":
        video_stream = (
            inp.video.filter("crop", crop.w, crop.h, crop.x, crop.y)
            .filter("scale", style.width, style.height, flags="lanczos")
            .filter("setsar", "1")
        )
    else:
        split = inp.video.filter_multi_output("split")
        background = (
            split[0]
            .filter(
                "scale",
                style.width,
                style.height,
                force_original_aspect_ratio="increase",
            )
            .filter("crop", style.width, style.height)
            .filter("gblur", sigma=style.blur_sigma)
        )
        foreground = split[1].filter("scale", style.width, -2, flags="lanczos")
        video_stream = ffmpeg.overlay(background, foreground, x="(W-w)/2", y="(H-h)/2")
    if stamp_path is not None:
        stamp = ffmpeg.input(str(stamp_path), loop=1, framerate=style.fps, t=duration)
        stamp = stamp.filter("format", "rgba").filter(
            "fade", t="in", st=stamp_start, d=0.12, alpha=1
        )
        video_stream = ffmpeg.overlay(
            video_stream,
            stamp,
            x="(W-w)/2",
            y=style.stamp_y,
            enable=_stamp_enable(stamp_start, stamp_end, blink, style).replace("\\,", ","),
        )
    fonts = {}
    if style.fonts_dir is not None:
        fonts["fontsdir"] = str(style.fonts_dir)
    video_stream = video_stream.filter("ass", str(ass_path), **fonts)
    streams = [video_stream]
    if has_audio:
        streams.append(inp.audio)
    node = ffmpeg.output(
        *streams,
        str(destination),
        vcodec="libx264",
        acodec="aac",
        pix_fmt="yuv420p",
        preset=style.preset,
        crf=style.crf,
        r=style.fps,
        movflags="+faststart",
        **{"b:a": style.audio_bitrate},
    )
    return ffmpeg.compile(node, overwrite_output=True)


@lru_cache(maxsize=None)
def ffmpeg_has_filter(name: str) -> bool:
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in (completed.stdout or "").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == name:
            return True
    return False


def run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()[-20:]
        raise RuntimeError("ffmpeg failed:\n" + "\n".join(tail))


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def _ass_filter(source: str, ass_path: Path, style: ReelStyle) -> str:
    options = f"filename='{_escape_path(ass_path)}'"
    if style.fonts_dir is not None:
        options += f":fontsdir='{_escape_path(style.fonts_dir)}'"
    return f"{source}ass={options}[vout]"


def _stamp_enable(start: float, end: float, blink: bool, style: ReelStyle) -> str:
    if not blink:
        return f"between(t\\,{start:.3f}\\,{end:.3f})"
    period = style.blink_on_s + style.blink_off_s
    return (
        f"gte(t\\,{start:.3f})*lte(t\\,{end:.3f})*"
        f"lt(mod(t-{start:.3f}\\,{period:.3f})\\,{style.blink_on_s:.3f})"
    )


def _pop_scale(stamp_start: float, grow: float = 0.16) -> str:
    begin = f"{stamp_start:.3f}"
    done = f"{stamp_start + grow:.3f}"
    width = (
        f"if(lt(t\\,{begin})\\,iw\\,"
        f"if(lt(t\\,{done})\\,iw*(0.55+0.45*(t-{begin})/{grow:.3f})\\,iw))"
    )
    height = width.replace("iw", "ih")
    return f"scale=w='{width}':h='{height}':eval=frame"


def _escape_path(path: Path) -> str:
    text = str(path)
    for char in ("\\", ":", "'", ",", "[", "]"):
        text = text.replace(char, "\\" + char)
    return text
