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
