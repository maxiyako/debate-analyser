"""Swap fonts, colours, paths, and pacing here. Nothing else hardcodes them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StampStyle:
    background: str
    foreground: str
    layout: str  # "stamp" (short punch) or "banner" (typed label)


@dataclass(frozen=True)
class ReelStyle:
    width: int = 1080
    height: int = 1920
    fps: int = 30

    # Subtitles. ASS colour is &HAABBGGRR. #FFFF00 -> &H0000FFFF.
    font_name: str = "Arial Black"
    font_size: int = 68
    max_words: int = 3
    max_chars: int = 26
    max_gap_s: float = 0.45
    primary_ass: str = "&H00FFFFFF"  # white, words not currently spoken
    highlight_ass: str = "&H0000FFFF"  # #FFFF00, the spoken word
    # Flagged line only. Light red for the rest of that line, red for the word being said.
    flag_primary_ass: str = "&H008080FF"  # #FF8080
    flag_highlight_ass: str = "&H000000FF"  # #FF0000
    outline_ass: str = "&H00000000"
    shadow_ass: str = "&H78000000"
    outline: int = 8
    shadow: int = 3
    active_scale: int = 108
    subtitle_y: int = 1460
    fonts_dir: Path | None = None

    # Crop / blur bars.
    blur_sigma: float = 28.0
    layout: str = "auto"  # auto | cover | blur

    # Pacing.
    max_duration: float = 18.0
    body_seconds: float = 12.0
    lead_in: float = 0.45
    question_preroll: float = 4.0
    # Pad before the first word of the question, and how long the dodge answer stays up.
    question_lead_s: float = 1.05
    dodge_answer_s: float = 32.0
    dodge_tail_s: float = 1.6
    dodge_max_s: float = 150.0
    # Non-dodge clips: a few seconds of setup so the viewer knows the topic. Still a reel.
    hook_lead_s: float = 3.6
    claim_tail_s: float = 1.4
    claim_max_s: float = 26.0
    stamp_hold_s: float = 3.5
    blink_on_s: float = 0.50
    blink_off_s: float = 0.22
    stamp_y: int = 70

    crf: int = 18
    preset: str = "medium"
    audio_bitrate: str = "192k"

    stamps: dict[str, StampStyle] = field(
        default_factory=lambda: {
            "loz": StampStyle("#FF0000", "#FFFFFF", "stamp"),
            "misleading": StampStyle("#FF6A00", "#FFFFFF", "banner"),
            "unverified": StampStyle("#5A5A5A", "#FFFFFF", "banner"),
            "true": StampStyle("#1F7A3A", "#FFFFFF", "stamp"),
            "fallacy": StampStyle("#FFE600", "#111111", "banner"),
            "manipulation": StampStyle("#FFD000", "#111111", "banner"),
            "dodge": StampStyle("#FF0000", "#FFFFFF", "stamp"),
            "partial": StampStyle("#FF6A00", "#FFFFFF", "banner"),
            "incivility": StampStyle("#FF0000", "#FFFFFF", "stamp"),
            "fair": StampStyle("#1F7A3A", "#FFFFFF", "stamp"),
        }
    )

    def stamp_style(self, key: str) -> StampStyle:
        return self.stamps.get(key, StampStyle("#FF0000", "#FFFFFF", "banner"))


# Pillow / drawtext look here, in order. First file that exists wins.
FONT_FILES: tuple[str, ...] = (
    "/Library/Fonts/Montserrat-Black.ttf",
    "/System/Library/Fonts/Supplemental/Montserrat-Black.ttf",
    str(Path.home() / "Library/Fonts/Montserrat-Black.ttf"),
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/Library/Fonts/Arial Black.ttf",
    "/usr/share/fonts/truetype/montserrat/Montserrat-Black.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def find_font_file() -> Path | None:
    for raw in FONT_FILES:
        path = Path(raw)
        if path.is_file():
            return path
    return None
