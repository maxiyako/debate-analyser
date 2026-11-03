"""9:16 crop locked on the person who is talking.

Samples a handful of frames in the reel window, detects faces, and picks the
face whose mouth region moves the most. A static studio seat stays in one
crop for the whole clip. If that crop would cut the face off, the plan switches
to a full-frame centre band with blurred bars above and below.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from reel_generator.config import ReelStyle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Face:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class CropSpan:
    """One camera shot. `start`/`end` are seconds from the start of the reel.

    `w` is the crop width. A split screen is wider than the 9:16 cover so both
    faces stay in frame. 0 means "use the plan width".
    """

    start: float
    end: float
    x: int
    w: int = 0


@dataclass(frozen=True)
class CropPlan:
    mode: str  # cover | blur
    src_w: int
    src_h: int
    x: int
    y: int
    w: int
    h: int
    face_found: bool
    spans: tuple[CropSpan, ...] = ()


@dataclass(frozen=True)
class MediaInfo:
    width: int
    height: int
    has_audio: bool


def probe_media(path: Path) -> MediaInfo:
    import json

    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ]
    )
    streams = json.loads(raw).get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}")
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    return MediaInfo(int(video["width"]), int(video["height"]), has_audio)


def plan_crop(
    video: Path | None,
    start: float,
    end: float,
    style: ReelStyle,
    work_dir: Path | None = None,
) -> CropPlan:
    if video is None or not video.is_file():
        return crop_plan_for_face(1920, 1080, None, style)
    info = probe_media(video)
    if style.layout == "blur":
        return crop_plan_for_face(info.width, info.height, None, style, force="blur")
    face = None
    if style.layout != "cover":
        face = detect_talking_face(video, start, end, work_dir)
    return crop_plan_for_face(info.width, info.height, face, style, force=style.layout)


def crop_plan_for_face(
    src_w: int,
    src_h: int,
    face: Face | None,
    style: ReelStyle,
    force: str = "auto",
) -> CropPlan:
    if force == "blur":
        return _blur(src_w, src_h, False)
    crop_h = _even(src_h)
    crop_w = _even(round(crop_h * 9 / 16))
    if crop_w > src_w or crop_w < 2:
        return _blur(src_w, src_h, face is not None)
    y = _even(max(0, (src_h - crop_h) // 2))
    if face is None:
        x = _even(max(0, (src_w - crop_w) // 2))
        return CropPlan("cover", src_w, src_h, x, y, crop_w, crop_h, False)
    x = int(round(face.cx - crop_w / 2))
    x = _even(max(0, min(x, src_w - crop_w)))
    if crop_h < src_h:
        y = int(round(face.cy - crop_h / 2))
        y = _even(max(0, min(y, src_h - crop_h)))
    return CropPlan("cover", src_w, src_h, x, y, crop_w, crop_h, True)


def plan_speaker_crops(
    video: Path | None,
    segments: list[tuple[float, float, str]],
    clip_start: float,
    duration: float,
    style: ReelStyle,
    work_dir: Path | None = None,
) -> CropPlan:
    """9:16 on one face, held until that face nears the edge. Two faces: the whole frame."""
    if video is None or not video.is_file():
        return crop_plan_for_face(1920, 1080, None, style)
    info = probe_media(video)
    if style.layout == "blur":
        return crop_plan_for_face(info.width, info.height, None, style, force="blur")
    base = crop_plan_for_face(info.width, info.height, None, style, force="cover")
    if base.mode != "cover":
        return base
    track = _face_track(video, clip_start, duration, info.width, info.height)
    spans = spans_along_track(
        track,
        [(clip_start, clip_start + duration)],
        info.width,
        info.height,
        clip_start,
        duration,
    )
    spans = _merge_spans(spans, duration)
    primary = spans[0] if spans else CropSpan(0.0, duration, base.x, base.w)
    wide = any(span.w > base.w + 8 for span in spans)
    return CropPlan(
        "fit" if wide else "cover",
        info.width,
        info.height,
        primary.x,
        base.y,
        primary.w or base.w,
        base.h,
        any(span.x != base.x for span in spans) or bool(spans),
        tuple(spans),
    )


def crop_plan_for_faces(
    src_w: int, src_h: int, faces: list[Face], style: ReelStyle
) -> CropPlan:
    """Cover on one face. Wider than 9:16 when two faces would not both fit."""
    base = crop_plan_for_face(src_w, src_h, None, style, force="cover")
    x, width = window_for_faces(src_w, src_h, faces)
    span = CropSpan(0.0, 1.0, x, width)
    wide = width > base.w + 8
    return CropPlan(
        "fit" if wide else "cover",
        src_w,
        src_h,
        x,
        0,
        width,
        base.h,
        bool(faces),
        (span,),
    )


def distinct_faces(faces: list[Face]) -> list[Face]:
    """Drop nested boxes on the same head, and boxes much smaller than the real face."""
    ordered = sorted(faces, key=lambda face: face.w * face.h, reverse=True)
    kept: list[Face] = []
    for face in ordered:
        if kept and face.h < 0.45 * kept[0].h:
            continue
        if any(abs(face.cx - other.cx) < max(50, 0.45 * (face.w + other.w)) for other in kept):
            continue
        kept.append(face)
    return kept


def people_in_frame(faces: list[Face]) -> list[Face]:
    """Real people. A small box on the same head does not count as a second face."""
    people = distinct_faces(faces)
    if len(people) < 2:
        return people
    biggest = max(people, key=lambda face: face.h)
    if biggest.h >= 90:
        return [face for face in people if face.h >= 0.55 * biggest.h]
    far = [face for face in people if abs(face.cx - biggest.cx) > 120 and face.h >= 30]
    if far:
        return [biggest, *far]
    return [biggest]


def window_for_faces(src_w: int, src_h: int, faces: list[Face]) -> tuple[int, int]:
    """One face: 9:16 on that face. More than one: the whole frame."""
    cover_w = _cover_width(src_w, src_h)
    people = people_in_frame(faces)
    if len(people) >= 2:
        return 0, _even(src_w)
    if not people:
        return _even(max(0, (src_w - cover_w) // 2)), cover_w
    return _cover_on(people[0], src_w, cover_w)


def _cover_width(src_w: int, src_h: int) -> int:
    cover_w = _even(round(_even(src_h) * 9 / 16))
    return max(2, min(cover_w, _even(src_w)))


def _cover_on(face: Face, src_w: int, cover_w: int) -> tuple[int, int]:
    x = int(round(face.cx - cover_w / 2))
    return _even(max(0, min(x, src_w - cover_w))), cover_w


def _face_near_crop_edge(face: Face, crop_x: int, cover_w: int) -> bool:
    """True when the face has drifted into the outer band of the current crop."""
    margin = max(28, int(cover_w * 0.16))
    return face.cx < crop_x + margin or face.cx > crop_x + cover_w - margin


def spans_along_track(
    track: list[tuple[float, list[Face]]],
    shots: list[tuple[float, float]],
    src_w: int,
    src_h: int,
    clip_start: float,
    duration: float,
) -> list[CropSpan]:
    """One face keeps its crop until that face reaches the edge. Two faces use the whole frame.

    A missed sample holds the current crop. Two samples must agree before the mode flips,
    so one false box does not punch the picture in and out.
    """
    full_w = _even(src_w)
    cover_w = _cover_width(src_w, src_h)
    samples: list[tuple[float, list[Face]]] = []
    for shot_start, shot_end in shots:
        samples.extend(item for item in track if shot_start - 0.05 <= item[0] < shot_end)
    samples.sort(key=lambda item: item[0])
    end = shots[-1][1] if shots else clip_start + duration
    if not samples:
        return [CropSpan(0.0, max(0.1, end - clip_start), 0, full_w)]

    # Two samples agree before a flip, and a wide shot stays up until one face
    # has been alone for a couple of seconds. A one-frame extra box must not punch in.
    wide_enter = 3
    crop_return = 4
    spans: list[CropSpan] = []
    cursor = samples[0][0]
    crop_x = 0
    crop_w = full_w
    mode = ""
    many_run = 0
    one_run = 0
    placed = False

    def emit(until: float) -> None:
        nonlocal cursor
        if until - cursor < 0.2:
            return
        spans.append(CropSpan(cursor - clip_start, until - clip_start, crop_x, crop_w))
        cursor = until

    for moment, faces in samples:
        people = people_in_frame(faces)
        if len(people) >= 2:
            many_run += 1
            one_run = 0
        elif len(people) == 1:
            one_run += 1
            many_run = 0
        else:
            continue
        if not placed:
            if len(people) >= 2 and many_run < wide_enter:
                crop_x, crop_w = _cover_on(max(people, key=lambda face: face.h), src_w, cover_w)
                mode = "one"
                placed = True
                continue
            if len(people) >= 2:
                crop_x, crop_w = 0, full_w
                mode = "many"
            else:
                crop_x, crop_w = _cover_on(people[0], src_w, cover_w)
                mode = "one"
            placed = True
            continue
        if len(people) >= 2 and many_run >= wide_enter and mode != "many":
            emit(moment)
            crop_x, crop_w = 0, full_w
            mode = "many"
        elif len(people) == 1 and one_run >= crop_return and mode == "many":
            emit(moment)
            crop_x, crop_w = _cover_on(people[0], src_w, cover_w)
            mode = "one"
        elif len(people) == 1 and mode == "one" and _face_near_crop_edge(people[0], crop_x, cover_w):
            emit(moment)
            crop_x, crop_w = _cover_on(people[0], src_w, cover_w)
    emit(end)
    if not spans:
        spans.append(CropSpan(0.0, max(0.1, end - clip_start), crop_x, crop_w))
    spans[0] = CropSpan(0.0, spans[0].end, spans[0].x, spans[0].w)
    spans[-1] = CropSpan(spans[-1].start, max(duration, spans[-1].end), spans[-1].x, spans[-1].w)
    return spans


def _face_track(
    video: Path, clip_start: float, duration: float, src_w: int, src_h: int
) -> list[tuple[float, list[Face]]]:
    """Every half second, frontal plus profile, so a pull-back to two people is seen."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("opencv-python is not installed. Crop stays centred.")
        return []
    fps = 2
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{clip_start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video),
            "-vf",
            f"fps={fps},scale={src_w}:{src_h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )
    frame_bytes = src_w * src_h
    count = len(raw) // frame_bytes
    if count == 0:
        return []
    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    track: list[tuple[float, list[Face]]] = []
    for index in range(count):
        gray = np.frombuffer(raw[index * frame_bytes : (index + 1) * frame_bytes], dtype=np.uint8)
        gray = gray.reshape(src_h, src_w)
        track.append((clip_start + index / fps, _faces_in_gray(gray, frontal, profile)))
    return track


def _faces_in_gray(gray, frontal, profile) -> list[Face]:
    import cv2

    width = gray.shape[1]
    boxes: list[tuple[int, int, int, int]] = []
    for x, y, w, h in frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(28, 28)):
        boxes.append((int(x), int(y), int(w), int(h)))
    for flipped in (False, True):
        view = cv2.flip(gray, 1) if flipped else gray
        for x, y, w, h in profile.detectMultiScale(view, scaleFactor=1.08, minNeighbors=3, minSize=(28, 28)):
            x = int(x)
            w = int(w)
            if flipped:
                x = width - x - w
            boxes.append((x, int(y), w, int(h)))
    return distinct_faces([Face(*box) for box in boxes])


def _portrait_width(plan: CropPlan) -> int:
    return max(2, _even(round(plan.h * 9 / 16)))


def wide_enable_expr(plan: CropPlan) -> str:
    """1 while a shot is too wide for 9:16, else 0. Crop width itself stays fixed."""
    cover_w = _portrait_width(plan)
    parts = [
        f"between(t\\,{span.start:.3f}\\,{span.end:.3f})"
        for span in plan.spans
        if (span.w or plan.w) > cover_w + 8
    ]
    return "+".join(parts) if parts else "0"


def crop_x_expr(plan: CropPlan) -> str:
    return _span_expr(plan.spans, "x", plan.x)


def cover_width(plan: CropPlan) -> int:
    return _portrait_width(plan)


def cover_crop_filter(plan: CropPlan) -> str:
    """`crop=` clause. `x` and `w` switch when the camera cuts."""
    spans = plan.spans
    if plan.mode == "fit":
        width = _span_expr(spans, "w", plan.w)
        x = _span_expr(spans, "x", plan.x)
        return f"crop=w='{width}':h={plan.h}:x='{x}':y={plan.y}"
    if len(spans) <= 1:
        x = spans[0].x if spans else plan.x
        return f"crop={plan.w}:{plan.h}:{x}:{plan.y}"
    return f"crop={plan.w}:{plan.h}:x='{_span_expr(spans, 'x', plan.x)}':y={plan.y}"


def _span_expr(spans: tuple[CropSpan, ...] | list[CropSpan], field: str, fallback: int) -> str:
    if not spans:
        return str(fallback)
    def value(span: CropSpan) -> int:
        raw = getattr(span, field)
        return fallback if field == "w" and not raw else raw

    expr = str(value(spans[-1]))
    for span in reversed(list(spans)[:-1]):
        expr = f"if(lt(t\\,{span.end:.3f})\\,{value(span)}\\,{expr})"
    return expr


def faces_in_shot(
    video: Path, start: float, end: float, work_dir: Path | None
) -> list[Face]:
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python is not installed. Crop stays centred.")
        return []
    folder = work_dir or video.parent
    folder.mkdir(parents=True, exist_ok=True)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    samples: list[list[Face]] = []
    for index, moment in enumerate(_sample_times(start, end, 3)):
        image_path = folder / f"shot_{int(start)}_{index}.jpg"
        _grab_frame(video, moment, image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
        samples.append(distinct_faces([Face(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]))
    groups = [sample for sample in samples if len(sample) >= 2]
    if groups:
        return max(groups, key=len)
    singles = [sample[0] for sample in samples if sample]
    if not singles:
        return []
    singles.sort(key=lambda face: face.cx)
    return [singles[len(singles) // 2]]


def largest_face(
    video: Path, start: float, end: float, work_dir: Path | None
) -> Face | None:
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python is not installed. Crop stays centred.")
        return None
    folder = work_dir or video.parent
    folder.mkdir(parents=True, exist_ok=True)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    found: list[Face] = []
    for index, moment in enumerate(_sample_times(start, end, 3)):
        image_path = folder / f"shot_{int(start)}_{index}.jpg"
        _grab_frame(video, moment, image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        faces = [Face(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]
        if faces:
            found.append(max(faces, key=lambda face: face.w * face.h))
    if not found:
        return None
    found.sort(key=lambda face: face.cx)
    return found[len(found) // 2]


def scene_cut_times(ffmpeg_log: str) -> list[float]:
    """First-frame timestamps of camera cuts, from ffmpeg `showinfo` on `scene`."""
    times: list[float] = []
    for match in re.finditer(r"\bpts_time:([0-9.]+)", ffmpeg_log):
        moment = float(match.group(1))
        if times and moment - times[-1] < 0.12:
            continue
        times.append(moment)
    return times


def _shot_spans(video: Path, clip_start: float, duration: float) -> list[tuple[float, float]]:
    """Camera cuts at the real frame, not on a half-second grid.

    Source keyframes here are a fixed 2s GOP and the cuts are P-frames, so
    keyframe times do not mark a camera change. `scene` scores every frame.
    """
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-ss",
            f"{clip_start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video),
            "-vf",
            "select='gte(scene,0.35)',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    cuts = [0.0]
    for moment in scene_cut_times(completed.stderr):
        if 0.08 < moment < duration - 0.08:
            cuts.append(moment)
    spans: list[tuple[float, float]] = []
    for index, cut in enumerate(cuts):
        rel_end = cuts[index + 1] if index + 1 < len(cuts) else duration
        spans.append((clip_start + cut, clip_start + rel_end))
    return spans


def _merge_spans(spans: list[CropSpan], duration: float) -> list[CropSpan]:
    if not spans:
        return [CropSpan(0.0, duration, 0)]
    merged: list[CropSpan] = [spans[0]]
    last = len(spans) - 1
    for index, span in enumerate(spans[1:], start=1):
        previous = merged[-1]
        same_width = abs((span.w or 0) - (previous.w or 0)) <= 16
        blip = index < last and span.end - span.start < 0.8
        if (abs(span.x - previous.x) <= 16 and same_width) or blip:
            merged[-1] = CropSpan(previous.start, span.end, previous.x, previous.w)
        else:
            merged.append(span)
    merged[0] = CropSpan(0.0, merged[0].end, merged[0].x, merged[0].w)
    merged[-1] = CropSpan(merged[-1].start, max(duration, merged[-1].end), merged[-1].x, merged[-1].w)
    return merged


def _grab_frame(video: Path, moment: float, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{moment:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(destination),
        ],
        check=True,
    )


def detect_talking_face(
    video: Path, start: float, end: float, work_dir: Path | None
) -> Face | None:
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python is not installed. Crop stays centred.")
        return None
    folder = work_dir or video.parent
    folder.mkdir(parents=True, exist_ok=True)
    times = _sample_times(start, end, 5)
    frames: list[tuple[object, list[Face]]] = []
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    for index, moment in enumerate(times):
        image_path = folder / f"face_{index}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                f"{moment:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(image_path),
            ],
            check=True,
        )
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        faces = [Face(int(x), int(y), int(w), int(h)) for (x, y, w, h) in found]
        frames.append((gray, faces))
    return _pick_talker(frames)


def _pick_talker(frames: list[tuple[object, list[Face]]]) -> Face | None:
    tracks: list[list[tuple[int, Face]]] = []
    for frame_index, (_gray, faces) in enumerate(frames):
        for face in faces:
            placed = False
            for track in tracks:
                anchor = track[-1][1]
                if abs(face.cx - anchor.cx) < max(80, anchor.w):
                    track.append((frame_index, face))
                    placed = True
                    break
            if not placed:
                tracks.append([(frame_index, face)])
    if not tracks:
        return None
    best_score = -1.0
    best: list[tuple[int, Face]] | None = None
    for track in tracks:
        score = 0.0
        for (index_a, face_a), (index_b, face_b) in zip(track, track[1:]):
            if index_b != index_a + 1:
                continue
            gray_a = frames[index_a][0]
            gray_b = frames[index_b][0]
            score += _mouth_motion(gray_a, gray_b, face_a, face_b)
        score *= len(track) / max(1, len(frames))
        if score > best_score:
            best_score = score
            best = track
    if best is None:
        return None
    faces = [face for _, face in best]
    mid = faces[len(faces) // 2]
    return mid


def _mouth_motion(gray_a, gray_b, face_a: Face, face_b: Face) -> float:
    import numpy as np

    box = _mouth_box(face_a)
    other = _mouth_box(face_b)
    height, width = gray_a.shape[:2]
    ax0, ay0, ax1, ay1 = _clip_box(box, width, height)
    bx0, by0, bx1, by1 = _clip_box(other, width, height)
    crop_w = min(ax1 - ax0, bx1 - bx0)
    crop_h = min(ay1 - ay0, by1 - by0)
    if crop_w < 8 or crop_h < 8:
        return 0.0
    left = gray_a[ay0 : ay0 + crop_h, ax0 : ax0 + crop_w]
    right = gray_b[by0 : by0 + crop_h, bx0 : bx0 + crop_w]
    return float(np.mean(np.abs(left.astype("float32") - right.astype("float32"))))


def _mouth_box(face: Face) -> tuple[int, int, int, int]:
    top = face.y + int(face.h * 0.55)
    return face.x, top, face.right, face.bottom


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        max(0, min(width - 1, x0)),
        max(0, min(height - 1, y0)),
        max(1, min(width, x1)),
        max(1, min(height, y1)),
    )


def _sample_times(start: float, end: float, count: int) -> list[float]:
    duration = max(0.2, end - start)
    if duration < 0.8:
        return [start + duration / 2]
    return [start + duration * (index + 0.5) / count for index in range(count)]


def _blur(src_w: int, src_h: int, face_found: bool) -> CropPlan:
    return CropPlan("blur", src_w, src_h, 0, 0, src_w, src_h, face_found)


def _even(value: int) -> int:
    value = int(value)
    return value - (value % 2)
