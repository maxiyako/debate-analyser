"""Unit tests for diarization parsing/caching and the ASR segment building.

The Whisper forward pass itself needs the model + audio and is out of scope
here; ASR is faked. These cover windowing, slice trimming around overlap
regions, separated-stream segments, fallbacks and the merge into tagged lines.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from config import Settings
from src.overlap import OverlapTools, Region
from src.transcription import (
    Diarization,
    Segment,
    _window_starts,
    build_segments,
    clean_spans,
    diarization_cache_path,
    diarization_from_output,
    format_transcript,
    load_diarization,
    merge_segments,
    save_diarization,
)

SR = 16000


def _settings(**kw) -> Settings:
    base = dict(enroll_min_seconds=1.0, overlap_min_seconds=0.5, overlap_pad_seconds=0.25)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _rows(segments):
    return [(s.speaker, s.start, s.end, s.crosstalk) for s in segments]


def test_window_starts_overlap() -> None:
    # 70s audio, 30s windows, 5s overlap -> step 25s.
    assert _window_starts(70.0, 30.0, 25.0) == [0.0, 25.0, 50.0]


def test_window_starts_single_short_clip() -> None:
    assert _window_starts(10.0, 30.0, 25.0) == [0.0]


# --- diarization parsing and cache -------------------------------------------


class FakeAnnotation:
    def __init__(self, rows):
        self.rows = rows

    def itertracks(self, yield_label=False):
        for start, end, speaker in self.rows:
            yield SimpleNamespace(start=start, end=end), None, speaker


def test_diarization_from_output_keeps_both_views() -> None:
    output = SimpleNamespace(
        speaker_diarization=FakeAnnotation([(0.0, 10.0, "S0"), (4.0, 6.0, "S1")]),
        exclusive_speaker_diarization=FakeAnnotation([(0.0, 10.0, "S0")]),
    )
    diar = diarization_from_output(output, "m")
    assert diar == Diarization("m", [(0.0, 10.0, "S0"), (4.0, 6.0, "S1")], [(0.0, 10.0, "S0")])


def test_diarization_from_legacy_annotation() -> None:
    diar = diarization_from_output(FakeAnnotation([(0.0, 5.0, "S0")]), "m")
    assert diar.turns == diar.exclusive_turns == [(0.0, 5.0, "S0")]


def test_diarization_cache_round_trip(tmp_path) -> None:
    path = diarization_cache_path(tmp_path, "ep1")
    assert path.name == "ep1.diarization.v2.json"
    diar = Diarization("m", [(0.0, 1.5, "S0")], [(0.0, 1.5, "S0")])
    save_diarization(path, diar)
    assert load_diarization(path) == diar


def test_old_cache_is_never_read(tmp_path) -> None:
    (tmp_path / "ep1.diarization.json").write_text('[[0.0, 1.0, "S0"]]', encoding="utf-8")
    assert load_diarization(diarization_cache_path(tmp_path, "ep1")) is None


# --- merging ------------------------------------------------------------------


def test_merge_groups_consecutive_same_speaker() -> None:
    raw = [Segment(0.0, 3.0, "a1", "S0"), Segment(3.0, 6.0, "a2", "S0"), Segment(11.0, 14.0, "b1", "S2")]
    merged = merge_segments(raw)
    assert [(m.speaker, m.text) for m in merged] == [("Speaker A", "a1 a2"), ("Speaker B", "b1")]


def test_merge_splits_same_speaker_when_gap_too_large() -> None:
    raw = [Segment(0.0, 3.0, "a1", "S0"), Segment(30.0, 33.0, "a2", "S0")]
    assert len(merge_segments(raw, max_gap_seconds=2.0)) == 2


def test_crosstalk_never_merges_into_clean_neighbour() -> None:
    raw = [
        Segment(0.0, 4.0, "a", "S0"),
        Segment(4.0, 6.0, "x", "S0", crosstalk=True, region=0),
        Segment(6.0, 8.0, "b", "S0"),
    ]
    assert [m.text for m in merge_segments(raw)] == ["a", "x", "b"]


def test_crosstalk_merges_only_within_one_region() -> None:
    raw = [
        Segment(4.0, 5.0, "x", "S0", crosstalk=True, region=0),
        Segment(5.0, 6.0, "y", "S0", crosstalk=True, region=0),
        Segment(6.0, 7.0, "z", "S0", crosstalk=True, region=1),
    ]
    assert [m.text for m in merge_segments(raw)] == ["x y", "z"]


def test_format_transcript_marks_crosstalk() -> None:
    raw = [Segment(3.0, 6.0, "prekrikujeme sa", "S0", crosstalk=True, region=0)]
    assert format_transcript(merge_segments(raw)) == "Speaker A [00:03]: (cez seba) prekrikujeme sa\n"


# --- spans and segment building ------------------------------------------------


def test_clean_spans_merge_chunk_and_trim() -> None:
    turns = [(0.0, 10.0, "S0"), (10.3, 40.0, "S0"), (40.0, 40.2, "S1")]
    regions = [Region(4.0, 6.0, frozenset({"S0", "S1"}), 0)]
    assert clean_spans(turns, 45.0, regions) == [
        (0.0, 4.0, "S0"),
        (6.0, 34.0, "S0"),
        (34.0, 40.0, "S0"),
    ]


def test_flag_off_matches_exclusive_turn_slicing(fake_asr) -> None:
    audio = np.zeros(20 * SR, dtype=np.float32)
    diar = Diarization("m", [(0.0, 10.0, "S0"), (8.0, 20.0, "S1")], [(0.0, 10.0, "S0"), (10.0, 20.0, "S1")])
    raw = build_segments(audio, diar, _settings(), fake_asr)
    assert _rows(raw) == [("S0", 0.0, 10.0, False), ("S1", 10.0, 20.0, False)]
    assert format_transcript(merge_segments(raw)) == "Speaker A [00:00]: c0\nSpeaker B [00:10]: c1\n"


def test_no_diarization_falls_back_to_30s_windows(fake_asr) -> None:
    audio = np.zeros(70 * SR, dtype=np.float32)
    raw = build_segments(audio, Diarization("m", [], []), _settings(), fake_asr)
    assert _rows(raw) == [
        ("SPEAKER_00", 0.0, 30.0, False),
        ("SPEAKER_00", 30.0, 60.0, False),
        ("SPEAKER_00", 60.0, 70.0, False),
    ]
    assert format_transcript(merge_segments(raw)) == "Speaker A [00:00]: c0 c1 c2\n"


OVERLAP_DIAR = dict(
    turns=[(0.0, 10.0, "A"), (4.0, 6.0, "B"), (12.0, 16.0, "B")],
    exclusive_turns=[(0.0, 10.0, "A"), (12.0, 16.0, "B")],
)


def test_separated_region_gives_each_speaker_a_line(voice_audio, fake_separator, fake_embedder, fake_asr) -> None:
    tools = OverlapTools(separator=fake_separator, embedder=fake_embedder)
    raw = build_segments(voice_audio, Diarization("m", **OVERLAP_DIAR), _settings(), fake_asr, tools=tools)
    assert _rows(raw) == [
        ("A", 0.0, 4.0, False),
        ("A", 4.0, 6.0, True),
        ("B", 4.0, 6.0, True),
        ("A", 6.0, 10.0, False),
        ("B", 12.0, 16.0, False),
    ]
    lines = format_transcript(merge_segments(raw)).splitlines()
    assert [ln.split(":")[0] for ln in lines] == [
        "Speaker A [00",
        "Speaker A [00",
        "Speaker B [00",
        "Speaker A [00",
        "Speaker B [00",
    ]
    assert "(cez seba)" in lines[1] and "(cez seba)" in lines[2]
    assert "(cez seba)" not in lines[0] and "(cez seba)" not in lines[3]


def test_fallback_region_uses_exclusive_slice_marked(voice_audio, fake_embedder, fake_asr) -> None:
    tools = OverlapTools(separator=None, embedder=fake_embedder)
    raw = build_segments(voice_audio, Diarization("m", **OVERLAP_DIAR), _settings(), fake_asr, tools=tools)
    assert _rows(raw) == [
        ("A", 0.0, 4.0, False),
        ("A", 4.0, 6.0, True),
        ("A", 6.0, 10.0, False),
        ("B", 12.0, 16.0, False),
    ]


def test_window_limits_segments(voice_audio, fake_separator, fake_embedder, fake_asr) -> None:
    tools = OverlapTools(separator=fake_separator, embedder=fake_embedder)
    raw = build_segments(
        voice_audio, Diarization("m", **OVERLAP_DIAR), _settings(), fake_asr, tools=tools, window=(3.0, 7.0)
    )
    assert _rows(raw) == [
        ("A", 3.0, 4.0, False),
        ("A", 4.0, 6.0, True),
        ("B", 4.0, 6.0, True),
        ("A", 6.0, 7.0, False),
    ]


# --- run_transcription wiring --------------------------------------------------

import src.transcription as tr


def _wire(monkeypatch, tmp_path, fake_asr, *, diar, flag=False, tools=None):
    calls = {"diarize": 0, "tools": 0}

    def fake_diarize(audio_path, settings):
        calls["diarize"] += 1
        return diar

    def fake_tools(settings):
        calls["tools"] += 1
        return tools

    monkeypatch.setattr(tr, "diarize", fake_diarize)
    monkeypatch.setattr(tr, "load_audio", lambda p: np.zeros(20 * SR, dtype=np.float32))
    monkeypatch.setattr(
        tr, "WhisperASR", lambda settings: SimpleNamespace(transcribe=fake_asr.transcribe, close=lambda: None)
    )
    monkeypatch.setattr(tr, "load_overlap_tools", fake_tools)
    monkeypatch.setattr(tr, "upload_file", lambda *a, **k: None)
    settings = _settings(
        transcript_dir=tmp_path, overlap_separation_enabled=flag, diarization_model_id="m"
    )
    return settings, calls


def test_run_transcription_writes_v2_cache_and_transcript(monkeypatch, tmp_path, fake_asr) -> None:
    diar = tr.Diarization("m", [(0.0, 10.0, "S0")], [(0.0, 10.0, "S0")])
    settings, calls = _wire(monkeypatch, tmp_path, fake_asr, diar=diar)
    (tmp_path / "ep.diarization.json").write_text("[]", encoding="utf-8")
    out = tr.run_transcription(tmp_path / "ep.wav", "ep", settings=settings)
    assert out.read_text(encoding="utf-8") == "Speaker A [00:00]: c0\n"
    assert tr.load_diarization(tmp_path / "ep.diarization.v2.json") == diar
    assert (tmp_path / "ep.diarization.json").read_text(encoding="utf-8") == "[]"
    assert calls == {"diarize": 1, "tools": 0}


def test_run_transcription_reuses_matching_cache(monkeypatch, tmp_path, fake_asr) -> None:
    diar = tr.Diarization("m", [(0.0, 10.0, "S0")], [(0.0, 10.0, "S0")])
    settings, calls = _wire(monkeypatch, tmp_path, fake_asr, diar=diar)
    tr.save_diarization(tmp_path / "ep.diarization.v2.json", diar)
    tr.run_transcription(tmp_path / "ep.wav", "ep", settings=settings)
    assert calls["diarize"] == 0


def test_run_transcription_rediarizes_on_model_change(monkeypatch, tmp_path, fake_asr) -> None:
    diar = tr.Diarization("m", [(0.0, 10.0, "S0")], [(0.0, 10.0, "S0")])
    settings, calls = _wire(monkeypatch, tmp_path, fake_asr, diar=diar)
    tr.save_diarization(tmp_path / "ep.diarization.v2.json", tr.Diarization("old", [], []))
    tr.run_transcription(tmp_path / "ep.wav", "ep", settings=settings)
    assert calls["diarize"] == 1


def test_run_transcription_does_not_cache_empty_diarization(monkeypatch, tmp_path, fake_asr) -> None:
    settings, _ = _wire(monkeypatch, tmp_path, fake_asr, diar=tr.Diarization("m", [], []))
    tr.run_transcription(tmp_path / "ep.wav", "ep", settings=settings)
    assert not (tmp_path / "ep.diarization.v2.json").exists()


def test_run_transcription_loads_tools_only_with_flag(monkeypatch, tmp_path, fake_asr, fake_embedder) -> None:
    diar = tr.Diarization("m", [(0.0, 10.0, "S0")], [(0.0, 10.0, "S0")])
    tools = OverlapTools(separator=None, embedder=fake_embedder)
    settings, calls = _wire(monkeypatch, tmp_path, fake_asr, diar=diar, flag=True, tools=tools)
    tr.run_transcription(tmp_path / "ep.wav", "ep", settings=settings)
    assert calls["tools"] == 1
