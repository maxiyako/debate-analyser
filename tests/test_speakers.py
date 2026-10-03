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
        # ASR and agents drop diacritics; the roster name is still returned.
        ("Erik Tomas", "Erik Tomáš"),
        ("erik tomas", "Erik Tomáš"),
        ("Tomasa", "Erik Tomáš"),
        ("Viskupica", "Marián Viskupič"),
    ],
)
def test_canonical_speaker(name: str, expected: str | None) -> None:
    assert canonical_speaker(name, GUESTS) == expected


def test_canonical_speaker_ambiguous_surname_stays_unresolved() -> None:
    roster = ["Erik Tomáš", "Peter Tomas"]
    assert canonical_speaker("Tomáš", roster) is None
    assert canonical_speaker("Tomasa", roster) is None
    assert canonical_speaker("Peter Tomáš", roster) == "Peter Tomas"


@pytest.mark.skipif(not REAL.exists(), reason="local transcript not available")
def test_heuristics_on_620752() -> None:
    lines = parse_lines(REAL.read_text(encoding="utf-8"))
    roles = heuristic_roles(lines)
    assert roles["Speaker A"] == SpeakerRole.MODERATOR
    assert roles["Speaker H"] == SpeakerRole.GUEST
    assert roles["Speaker D"] == SpeakerRole.GUEST
    assert roles["Speaker B"] == SpeakerRole.CLIP
    assert detect_debate_start(lines, roles) == "01:58"


from src.speakers import (  # noqa: E402
    LabelAssignment,
    SpeakerAssignment,
    build_speaker_prompt,
    map_speakers,
)


def _good(prompt: str) -> SpeakerAssignment:
    return SpeakerAssignment(
        assignments=[
            LabelAssignment(
                label="M", name="Moderátor", confidence=0.95,
                evidence=["[00:10] Vitajte, v štúdiu sú Erik Tomáš a Marián Viskupič."],
            ),
            LabelAssignment(
                label="G1", name="Erik Tomáš", confidence=0.9,
                evidence=["[00:24] Ja ako minister práce poviem"],
            ),
            LabelAssignment(
                label="G2", name="Marián Viskupič", confidence=0.9,
                evidence=["[00:38] My v SaS tvrdíme, že dane rastú"],
            ),
        ]
    )


def test_map_speakers_ok() -> None:
    smap = map_speakers(synthetic(), guests=GUESTS, llm=_good)
    assert smap.status == "ok"
    assert smap.source == "cli"
    assert smap.debate_start == "00:10"
    assert smap.name_for("G1") == "Erik Tomáš"
    assert smap.name_for("G2") == "Marián Viskupič"
    assert smap.name_for("M") == "Moderátor"
    assert smap.name_for("X") == "Záznam"


def test_name_outside_roster_is_partial() -> None:
    def fake(prompt: str) -> SpeakerAssignment:
        a = _good(prompt)
        a.assignments[2].name = "Igor Matovič"
        return a

    smap = map_speakers(synthetic(), guests=GUESTS, llm=fake)
    assert smap.status == "partial"
    assert smap.name_for("G2") is None
    assert any("not in the roster" in n for n in smap.notes)


def test_ungrounded_evidence_zeroes_confidence() -> None:
    def fake(prompt: str) -> SpeakerAssignment:
        a = _good(prompt)
        a.assignments[1].evidence = ["[00:24] Som minister financií a dane znížim"]
        return a

    smap = map_speakers(synthetic(), guests=GUESTS, llm=fake)
    g1 = next(e for e in smap.entries if e.label == "G1")
    assert g1.confidence == 0.0
    assert smap.status == "partial"


def _boom(prompt: str) -> SpeakerAssignment:
    raise RuntimeError("vertex down")


def addressed() -> str:
    """Like synthetic(), but the moderator addresses each guest by surname."""
    rows = []
    for row in synthetic().splitlines():
        if row.startswith("M [") and "Pán minister" in row:
            row = row.replace("Pán minister", "Pán Tomáš")
        elif row.startswith("M [") and "Pán poslanec" in row:
            row = row.replace("Pán poslanec", "Pán Viskupič")
        rows.append(row)
    return "\n".join(rows) + "\n"


def test_llm_failure_leaves_guests_unnamed_without_evidence() -> None:
    # The moderator never says a surname, so which label is which person is a
    # coin flip: naming them would ship a swapped scoreboard as fact.
    smap = map_speakers(synthetic(), guests=GUESTS, llm=_boom)
    assert smap.source == "heuristic"
    assert smap.status == "failed"
    assert smap.name_for("M") == "Moderátor"
    assert smap.name_for("G1") == "G1"
    assert smap.name_for("G2") == "G2"
    assert all(e.confidence == 0.0 for e in smap.entries if e.label in ("G1", "G2"))
    assert all(
        e.role == SpeakerRole.GUEST for e in smap.entries if e.label in ("G1", "G2")
    )


def test_heuristic_names_guests_the_moderator_addresses_by_surname() -> None:
    smap = map_speakers(addressed(), guests=GUESTS, llm=_boom)
    assert smap.source == "heuristic"
    assert smap.status == "partial"
    assert smap.name_for("G1") == "Erik Tomáš"
    assert smap.name_for("G2") == "Marián Viskupič"
    assert smap.name_for("M") == "Moderátor"


@pytest.mark.skipif(not REAL.exists(), reason="local transcript not available")
def test_heuristic_fallback_names_620752_labels_by_evidence() -> None:
    smap = map_speakers(REAL.read_text(encoding="utf-8"), guests=GUESTS, llm=_boom)
    assert smap.source == "heuristic"
    assert smap.name_for("Speaker H") == "Erik Tomáš"
    assert smap.name_for("Speaker D") == "Marián Viskupič"


def test_roster_from_intro_when_no_cli_names() -> None:
    def fake(prompt: str) -> SpeakerAssignment:
        a = _good(prompt)
        a.guests_from_intro = ["Erik Tomáš", "Marián Viskupič", "Robert Fico"]
        return a

    smap = map_speakers(synthetic(), llm=fake)
    assert smap.status == "ok"
    assert smap.source == "llm"
    assert smap.guests() == ["Erik Tomáš", "Marián Viskupič"]


def test_near_miss_fabricated_evidence_is_rejected() -> None:
    def fake(prompt: str) -> SpeakerAssignment:
        a = _good(prompt)
        a.assignments[1].evidence = ["[00:24] Ja ako minister financií"]
        return a

    smap = map_speakers(synthetic(), guests=GUESTS, llm=fake)
    g1 = next(e for e in smap.entries if e.label == "G1")
    assert g1.confidence == 0.0
    assert any("no grounded evidence for G1" in n for n in smap.notes)


def test_intro_greeting_evidence_is_grounded() -> None:
    rows = [
        "X [00:01]: krátky zostrih z ulice",
        "M [00:03]: Dobrý večer, mojimi hosťami sú Erik Tomáš a Marián Viskupič.",
        "G1 [00:05]: Ďakujem za pozvanie, dobrý večer všetkým divákom.",
        "Y [00:07]: ďalší krátky zostrih",
    ]
    raw = "\n".join(rows) + "\n" + synthetic().split("\n", 1)[1]

    def fake(prompt: str) -> SpeakerAssignment:
        a = _good(prompt)
        a.assignments[1].evidence = ["[00:05] Ďakujem za pozvanie, dobrý večer všetkým divákom."]
        return a

    smap = map_speakers(raw, guests=GUESTS, llm=fake)
    assert smap.debate_start == "00:10"
    g1 = next(e for e in smap.entries if e.label == "G1")
    assert g1.confidence == 0.9
    assert not any("no grounded evidence" in n for n in smap.notes)


def test_bug_in_validation_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.speakers as speakers

    def broken(*args, **kwargs):
        raise ValueError("bug in our code")

    monkeypatch.setattr(speakers, "validate_assignment", broken)
    with pytest.raises(ValueError, match="bug in our code"):
        map_speakers(synthetic(), guests=GUESTS, llm=_good)


def test_duplicate_label_keeps_first_assignment() -> None:
    def fake(prompt: str) -> SpeakerAssignment:
        a = _good(prompt)
        a.assignments.append(
            LabelAssignment(
                label="G1", name="Marián Viskupič", confidence=0.9,
                evidence=["[00:24] Ja ako minister práce poviem"],
            )
        )
        return a

    smap = map_speakers(synthetic(), guests=GUESTS, llm=fake)
    assert smap.name_for("G1") == "Erik Tomáš"
    assert "Speaker map: duplicate assignment for G1" in smap.notes
    assert smap.status == "partial"


def test_moderator_label_keeps_the_known_moderator_name() -> None:
    """The heuristic owns the moderator label; an LLM-invented person name is replaced."""

    def fake(prompt: str) -> SpeakerAssignment:
        a = _good(prompt)
        a.assignments[0].name = "Jana Nováková"
        return a

    smap = map_speakers(synthetic(), guests=GUESTS, llm=fake)
    m = next(e for e in smap.entries if e.label == "M")
    assert m.name == "Moderátor"
    assert m.role == SpeakerRole.MODERATOR
    assert m.confidence == 0.95
    assert smap.status == "ok"
    assert (
        "Speaker map: moderator label M named 'Jana Nováková' by LLM; using 'Moderátor'"
        in smap.notes
    )


def test_cli_moderator_name_overrides_the_llm() -> None:
    smap = map_speakers(synthetic(), guests=GUESTS, moderator="Jana Nováková", llm=_good)
    assert smap.name_for("M") == "Jana Nováková"
    assert smap.moderator() == "Jana Nováková"
    assert smap.status == "ok"


def test_roster_guest_on_the_moderator_label_contradicts_the_heuristic() -> None:
    def fake(prompt: str) -> SpeakerAssignment:
        a = _good(prompt)
        a.assignments[0].name = "Erik Tomáš"
        return a

    smap = map_speakers(synthetic(), guests=GUESTS, llm=fake)
    m = next(e for e in smap.entries if e.label == "M")
    assert m.role == SpeakerRole.GUEST
    assert smap.status == "partial"
    assert any("contradicts the turn-taking heuristic" in n for n in smap.notes)


def test_prompt_lists_only_main_labels() -> None:
    lines = parse_lines(synthetic())
    roles = heuristic_roles(lines)
    prompt = build_speaker_prompt(lines, lines, roles, GUESTS, "Moderátor")
    assert "LABEL G1" in prompt and "LABEL M" in prompt
    assert "LABEL X" not in prompt
