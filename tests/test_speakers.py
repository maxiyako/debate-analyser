"""Offline tests for speaker mapping (src/speakers.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.report_models import SpeakerMap, SpeakerMapEntry, SpeakerRole
from src.speakers import (
    apply_speaker_map,
    canonical_speaker,
    detect_debate_start,
    heuristic_roles,
)
from src.transcript_lines import parse_lines

REAL = Path(__file__).resolve().parents[1] / "data" / "transcripts" / "620752.txt"
GUESTS = ["Erik Tomáš", "Marián Viskupič"]


def _ts(n: int) -> str:
    return f"{n // 60:02d}:{n % 60:02d}"


def synthetic() -> str:
    """Recap clip X, moderator M with short questions, guests G1 (minister) and G2 (SaS)."""
    rows = ["X [00:01]: krátky zostrih z ulice"]
    t = 10

    def add(spk: str, text: str) -> None:
        nonlocal t
        rows.append(f"{spk} [{_ts(t)}]: {text}")
        t += 7

    add("M", "Vitajte, v štúdiu sú Erik Tomáš a Marián Viskupič.")
    for i in range(6):
        add("M", f"Pán minister, otázka {i}?")
        add("G1", "Ja ako minister práce poviem, že " + "dôchodky rastú " * 15)
        add("M", f"Pán poslanec, otázka {i}?")
        add("G2", "My v SaS tvrdíme, že " + "dane rastú " * 12)
    return "\n".join(rows) + "\n"


def test_heuristic_roles_and_debate_start() -> None:
    lines = parse_lines(synthetic())
    roles = heuristic_roles(lines)
    assert roles == {
        "X": SpeakerRole.CLIP,
        "M": SpeakerRole.MODERATOR,
        "G1": SpeakerRole.GUEST,
        "G2": SpeakerRole.GUEST,
    }
    assert detect_debate_start(lines, roles) == "00:10"


def test_roster_size_widens_main_labels() -> None:
    lines = parse_lines(synthetic())
    roles = heuristic_roles(lines, n_main=4)
    assert roles["X"] == SpeakerRole.CLIP  # still under the clip share


def test_apply_speaker_map_changes_only_labels() -> None:
    raw = synthetic()
    smap = SpeakerMap(
        status="ok",
        entries=[
            SpeakerMapEntry(label="M", name="Moderátor", role=SpeakerRole.MODERATOR),
            SpeakerMapEntry(label="G1", name="Erik Tomáš", role=SpeakerRole.GUEST),
        ],
    )
    named = apply_speaker_map(raw, smap)
    raw_lines, named_lines = parse_lines(raw), parse_lines(named)
    assert [ln.text for ln in raw_lines] == [ln.text for ln in named_lines]
    assert [ln.ts for ln in raw_lines] == [ln.ts for ln in named_lines]
    assert {ln.speaker for ln in named_lines} == {"X", "Moderátor", "Erik Tomáš", "G2"}


@pytest.mark.parametrize(
    "name,expected",
    [
        ("erik tomáš", "Erik Tomáš"),
        ("Tomáš", "Erik Tomáš"),
        ("Erika Tomáša", "Erik Tomáš"),
        ("Viskupič", "Marián Viskupič"),
        ("Novák", None),
        ("Speaker H", None),
        ("", None),
    ],
)
def test_canonical_speaker(name: str, expected: str | None) -> None:
    assert canonical_speaker(name, GUESTS) == expected


@pytest.mark.skipif(not REAL.exists(), reason="local transcript not available")
def test_heuristics_on_620752() -> None:
    lines = parse_lines(REAL.read_text(encoding="utf-8"))
    roles = heuristic_roles(lines)
    assert roles["Speaker A"] == SpeakerRole.MODERATOR
    assert roles["Speaker H"] == SpeakerRole.GUEST
    assert roles["Speaker D"] == SpeakerRole.GUEST
    assert roles["Speaker B"] == SpeakerRole.CLIP
    assert detect_debate_start(lines, roles) == "01:58"
