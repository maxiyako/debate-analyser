"""Unit tests for the deterministic ASR/diarization alignment helpers.

The Whisper forward pass itself needs the model + audio and is out of scope
here; these cover the windowing, speaker assignment, grouping and cross-talk
logic that turn raw ASR segments + diarization turns into a tagged transcript.
"""

from __future__ import annotations

from src.transcription import (
    _assign_speaker,
    _window_starts,
    format_transcript,
    merge_diarization_and_asr,
)


def test_window_starts_overlap() -> None:
    # 70s audio, 30s windows, 5s overlap -> step 25s.
    assert _window_starts(70.0, 30.0, 25.0) == [0.0, 25.0, 50.0]


def test_window_starts_single_short_clip() -> None:
    assert _window_starts(10.0, 30.0, 25.0) == [0.0]


def test_assign_speaker_dominant_no_crosstalk() -> None:
    turns = [(0.0, 10.0, "S0"), (10.0, 20.0, "S1")]
    spk, crosstalk = _assign_speaker(1.0, 4.0, turns, crosstalk_min=0.35)
    assert spk == "S0"
    assert crosstalk is False


def test_assign_speaker_detects_crosstalk() -> None:
    # Segment 3-6s: S0 overlaps 3s (dominant), S1 overlaps 2s -> 2/3 >= 0.35.
    turns = [(0.0, 10.0, "S0"), (4.0, 7.0, "S1")]
    spk, crosstalk = _assign_speaker(3.0, 6.0, turns, crosstalk_min=0.35)
    assert spk == "S0"
    assert crosstalk is True


def test_assign_speaker_no_diarization() -> None:
    assert _assign_speaker(0.0, 5.0, [], crosstalk_min=0.35) == ("SPEAKER_00", False)


def test_merge_groups_consecutive_same_speaker_and_tags_crosstalk() -> None:
    asr = [(0.0, 3.0, "a1"), (3.0, 6.0, "a2"), (11.0, 14.0, "b1")]
    turns = [(0.0, 10.0, "S0"), (4.0, 7.0, "S1"), (10.0, 20.0, "S2")]
    merged = merge_diarization_and_asr(asr, turns, crosstalk_min=0.35)
    assert len(merged) == 2
    assert merged[0].speaker == "Speaker A"
    assert merged[0].text == "a1 a2"
    assert merged[0].crosstalk is True
    assert merged[1].speaker == "Speaker B"
    assert merged[1].text == "b1"


def test_merge_splits_same_speaker_when_gap_too_large() -> None:
    asr = [(0.0, 3.0, "a1"), (30.0, 33.0, "a2")]
    turns = [(0.0, 60.0, "S0")]
    merged = merge_diarization_and_asr(asr, turns, crosstalk_min=0.35, max_gap_seconds=2.0)
    assert len(merged) == 2
    assert all(seg.speaker == "Speaker A" for seg in merged)


def test_format_transcript_marks_crosstalk() -> None:
    asr = [(3.0, 6.0, "prekrikujeme sa")]
    turns = [(0.0, 10.0, "S0"), (3.0, 6.0, "S1")]
    text = format_transcript(merge_diarization_and_asr(asr, turns, crosstalk_min=0.3))
    assert text.startswith("Speaker A [00:03]: (cez seba) prekrikujeme sa")
