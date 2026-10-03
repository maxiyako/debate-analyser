"""Offline tests for transcript line parsing (src/transcript_lines.py)."""

from __future__ import annotations

from src.transcript_lines import format_lines, parse_lines, strip_markers, ts_seconds


def test_parse_lines_skips_noise_and_reads_fields() -> None:
    text = (
        "Moderátor [01:02]: Dobrý deň?\n"
        "noise without a timestamp\n"
        "Erik Tomáš [1:02:03]: (cez seba) Áno, to je pravda.\n"
    )
    lines = parse_lines(text)
    assert [ln.speaker for ln in lines] == ["Moderátor", "Erik Tomáš"]
    assert lines[1].crosstalk is True
    assert lines[1].words == 4
    assert lines[1].seconds == 3723
    assert [ln.raw_no for ln in lines] == [0, 1]
    assert format_lines(lines) == (
        "Moderátor [01:02]: Dobrý deň?\n"
        "Erik Tomáš [1:02:03]: (cez seba) Áno, to je pravda.\n"
    )


def test_strip_markers_variants() -> None:
    assert strip_markers("a (hovorenie cez seba) b") == "a b"
    assert strip_markers("(cez seba) slovo") == "slovo"


def test_ts_seconds() -> None:
    assert ts_seconds("01:58") == 118
    assert ts_seconds("1:00:00") == 3600
