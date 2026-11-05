"""Unit tests for overlap regions, enrollment and stream assignment."""

from __future__ import annotations

from src.overlap import Region, find_overlap_regions, lead_speaker, select_enrollment, trim_spans

AB = frozenset({"A", "B"})


def test_touching_turns_are_not_overlap() -> None:
    assert find_overlap_regions([(0.0, 5.0, "A"), (5.0, 9.0, "B")], min_seconds=0.5) == []


def test_two_speaker_overlap() -> None:
    regions = find_overlap_regions([(0.0, 10.0, "A"), (4.0, 6.0, "B")], min_seconds=0.5)
    assert regions == [Region(4.0, 6.0, AB, 0)]


def test_short_overlap_dropped() -> None:
    assert find_overlap_regions([(0.0, 10.0, "A"), (4.0, 4.3, "B")], min_seconds=0.5) == []


def test_close_spans_with_same_speakers_merge() -> None:
    turns = [(0.0, 20.0, "A"), (4.0, 5.0, "B"), (5.2, 6.0, "B")]
    regions = find_overlap_regions(turns, min_seconds=0.5, merge_gap=0.3)
    assert regions == [Region(4.0, 6.0, AB, 0)]


def test_three_speaker_span_stays_separate() -> None:
    turns = [(0.0, 10.0, "A"), (2.0, 8.0, "B"), (3.0, 5.0, "C")]
    regions = find_overlap_regions(turns, min_seconds=0.5)
    assert [(r.start, r.end, r.speakers) for r in regions] == [
        (2.0, 3.0, AB),
        (3.0, 5.0, frozenset({"A", "B", "C"})),
        (5.0, 8.0, AB),
    ]
    assert [r.group for r in regions] == [0, 1, 2]


def test_long_region_split_into_equal_chunks() -> None:
    regions = find_overlap_regions(
        [(0.0, 30.0, "A"), (2.0, 20.0, "B")], min_seconds=0.5, max_chunk_seconds=8.0
    )
    assert [(r.start, r.end) for r in regions] == [(2.0, 8.0), (8.0, 14.0), (14.0, 20.0)]
    assert {r.group for r in regions} == {0}


def test_trim_spans_cuts_region_time() -> None:
    regions = [Region(4.0, 6.0, AB, 0)]
    spans = [(0.0, 10.0, "A"), (10.0, 12.0, "B")]
    assert trim_spans(spans, regions) == [(0.0, 4.0, "A"), (6.0, 10.0, "A"), (10.0, 12.0, "B")]


def test_select_enrollment_longest_clean_first_with_cap() -> None:
    turns = [(0.0, 10.0, "A"), (4.0, 6.0, "B"), (12.0, 14.0, "B"), (20.0, 21.0, "C")]
    spans = select_enrollment(turns, max_seconds=5.0, min_seconds=1.5)
    assert spans["A"] == [(0.0, 4.0), (6.0, 7.0)]
    assert spans["B"] == [(12.0, 14.0)]
    assert "C" not in spans


def test_select_enrollment_excludes_short_overlaps_too() -> None:
    turns = [(0.0, 10.0, "A"), (4.0, 4.2, "B"), (12.0, 16.0, "B")]
    spans = select_enrollment(turns, max_seconds=30.0, min_seconds=1.0)
    assert spans["A"] == [(4.2, 10.0), (0.0, 4.0)]


def test_lead_speaker_is_the_one_already_talking() -> None:
    region = Region(4.0, 6.0, AB, 0)
    assert lead_speaker(region, [(0.0, 10.0, "A"), (4.0, 6.0, "B")]) == "A"
    assert lead_speaker(region, [(1.0, 6.0, "B"), (3.0, 9.0, "A")]) == "B"


def test_lead_speaker_tie_is_alphabetical() -> None:
    region = Region(4.0, 6.0, AB, 0)
    assert lead_speaker(region, [(4.0, 6.0, "B"), (4.0, 9.0, "A")]) == "A"


import logging

import numpy as np
import pytest

from config import Settings
from src.overlap import (
    OverlapTools,
    assign_streams,
    cosine,
    log_summary,
    process_regions,
)

SR = 16000
TURNS = [(0.0, 10.0, "A"), (4.0, 6.0, "B"), (12.0, 16.0, "B")]


def _settings(**kw) -> Settings:
    base = dict(enroll_min_seconds=1.0, enroll_max_seconds=30.0, overlap_pad_seconds=0.25)
    base.update(kw)
    return Settings(_env_file=None, **base)


def test_cosine_handles_zero_vector() -> None:
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 0.0])) == 0.0
    assert cosine(np.array([2.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_assign_streams_picks_crossed_pairing() -> None:
    sims = np.array([[0.1, 0.9], [0.8, 0.2]])
    assert assign_streams(sims, ["A", "B"], min_sim=0.3) == {"A": 1, "B": 0}


def test_assign_streams_drops_weak_stream() -> None:
    sims = np.array([[0.9, 0.1], [0.2, 0.25]])
    assert assign_streams(sims, ["A", "B"], min_sim=0.3) == {"A": 0}


def test_assign_streams_all_weak() -> None:
    assert assign_streams(np.zeros((2, 2)), ["A", "B"], min_sim=0.3) == {}


def test_process_region_separates_and_assigns(voice_audio, fake_separator, fake_embedder) -> None:
    region = Region(4.0, 6.0, AB, 0)
    tools = OverlapTools(separator=fake_separator, embedder=fake_embedder)
    [result] = process_regions(voice_audio, [region], TURNS, _settings(), tools)
    assert result.reason is None
    assert result.lead == "A"
    assert set(result.streams) == {"A", "B"}
    assert len(result.streams["A"]) == 2 * SR
    assert np.allclose(result.streams["A"], 0.9)
    assert np.allclose(result.streams["B"], -0.9)
    assert fake_separator.calls == [int(6.25 * SR) - int(3.75 * SR)]


def test_process_region_too_many_speakers(voice_audio, fake_separator, fake_embedder) -> None:
    region = Region(4.0, 6.0, frozenset({"A", "B", "C"}), 0)
    tools = OverlapTools(separator=fake_separator, embedder=fake_embedder)
    [result] = process_regions(voice_audio, [region], TURNS, _settings(), tools)
    assert result.reason == "too_many_speakers"
    assert result.streams == {}
    assert fake_separator.calls == []


def test_process_region_no_enrollment(voice_audio, fake_separator, fake_embedder) -> None:
    turns = [(0.0, 10.0, "A"), (4.0, 6.0, "B")]
    tools = OverlapTools(separator=fake_separator, embedder=fake_embedder)
    [result] = process_regions(voice_audio, [Region(4.0, 6.0, AB, 0)], turns, _settings(), tools)
    assert result.reason == "no_enrollment"


def test_process_region_separator_unavailable(voice_audio, fake_embedder) -> None:
    tools = OverlapTools(separator=None, embedder=fake_embedder)
    [result] = process_regions(voice_audio, [Region(4.0, 6.0, AB, 0)], TURNS, _settings(), tools)
    assert result.reason == "separator_unavailable"


def test_process_region_separator_error(voice_audio, fake_embedder) -> None:
    class Broken:
        def separate(self, audio):
            raise RuntimeError("boom")

    tools = OverlapTools(separator=Broken(), embedder=fake_embedder)
    [result] = process_regions(voice_audio, [Region(4.0, 6.0, AB, 0)], TURNS, _settings(), tools)
    assert result.reason == "separator_error"


def test_process_region_low_similarity(voice_audio, fake_embedder) -> None:
    class Silent:
        def separate(self, audio):
            return [np.zeros(len(audio), dtype=np.float32), np.zeros(len(audio), dtype=np.float32)]

    tools = OverlapTools(separator=Silent(), embedder=fake_embedder)
    [result] = process_regions(voice_audio, [Region(4.0, 6.0, AB, 0)], TURNS, _settings(), tools)
    assert result.reason == "low_similarity"
    assert result.streams == {}


def test_log_summary_counts_reasons(caplog) -> None:
    from src.overlap import RegionResult

    r = Region(4.0, 6.0, AB, 0)
    results = [
        RegionResult(r, "A", {"A": np.zeros(1), "B": np.zeros(1)}, None),
        RegionResult(r, "A", {}, "no_enrollment"),
        RegionResult(r, "A", {}, "too_many_speakers"),
    ]
    with caplog.at_level(logging.INFO, logger="src.overlap"):
        line = log_summary(results)
    assert line == "Overlap regions: 3 found, 1 separated, fallback no_enrollment=1, too_many_speakers=1"
    assert line in caplog.text
