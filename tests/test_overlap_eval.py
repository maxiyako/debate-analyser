"""Unit tests for the overlap gold set: candidates, scoring and the gate."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from src.overlap import Region
from src.overlap_eval import (
    build_candidates,
    gate,
    hyp_from_lines,
    hyp_from_segments,
    line_spans,
    map_labels,
    matched,
    merge_gold,
    score_region,
    summarize,
    tokens,
)
from src.transcript_lines import parse_lines
from src.transcription import Diarization, Segment

SR = 16000


def test_tokens_lowercase_and_drop_markers() -> None:
    assert tokens("(cez seba) To NIE je pravda, pán Kollár!") == ["to", "nie", "je", "pravda", "pán", "kollár"]


def test_matched_counts_aligned_tokens() -> None:
    assert matched(["a", "b", "c", "d"], ["x", "a", "b", "y", "d"]) == 3
    assert matched(["a", "b", "c", "d"], ["x", "a", "b", "y", "d"], min_block=2) == 2


def test_score_region_recall_and_misattribution() -> None:
    gold = {"S0": "to nie je pravda", "S1": "klamete ľuďom"}
    hyp = {"S0": "to nie je pravda klamete ľuďom", "S1": ""}
    score = score_region(gold, "S0", hyp)
    assert score == {
        "dominant": {"gold": 4, "found": 4},
        "interrupter": {"gold": 2, "found": 0},
        "misattributed": 2,
    }


def test_summarize_micro_averages() -> None:
    scores = [
        {"dominant": {"gold": 4, "found": 4}, "interrupter": {"gold": 2, "found": 0}, "misattributed": 2},
        {"dominant": {"gold": 6, "found": 3}, "interrupter": {"gold": 2, "found": 2}, "misattributed": 0},
    ]
    assert summarize(scores) == {
        "dominant_recall": 0.7,
        "interrupter_recall": 0.5,
        "misattributed": 2,
        "regions": 2,
    }


def test_gate_pass_and_fail() -> None:
    base = {"dominant_recall": 0.9, "interrupter_recall": 0.1, "misattributed": 10, "regions": 20}
    good = {"dominant_recall": 0.87, "interrupter_recall": 0.5, "misattributed": 4, "regions": 20}
    bad = {"dominant_recall": 0.8, "interrupter_recall": 0.5, "misattributed": 4, "regions": 20}
    assert gate(base, good)["passed"] is True
    assert gate(base, bad)["passed"] is False
    assert gate(base, good)["interrupter_gain"] == 0.4


TRANSCRIPT = "Speaker A [00:00]: dlhá odpoveď\nSpeaker B [00:12]: krátka\nSpeaker A [00:20]: koniec\n"


def test_line_spans_and_label_mapping() -> None:
    spans = line_spans(parse_lines(TRANSCRIPT))
    assert spans == [
        (0.0, 12.0, "Speaker A", "dlhá odpoveď"),
        (12.0, 20.0, "Speaker B", "krátka"),
        (20.0, 50.0, "Speaker A", "koniec"),
    ]
    turns = [(0.0, 11.0, "SPEAKER_01"), (11.0, 20.0, "SPEAKER_00"), (20.0, 40.0, "SPEAKER_01")]
    assert map_labels(spans, turns) == {"Speaker A": "SPEAKER_01", "Speaker B": "SPEAKER_00"}


def test_hyp_from_lines_uses_one_second_tolerance() -> None:
    spans = line_spans(parse_lines(TRANSCRIPT))
    mapping = {"Speaker A": "SPEAKER_01", "Speaker B": "SPEAKER_00"}
    assert hyp_from_lines(spans, mapping, 4.0, 6.0) == {"SPEAKER_01": "dlhá odpoveď"}
    assert hyp_from_lines(spans, mapping, 12.5, 12.9) == {"SPEAKER_00": "krátka", "SPEAKER_01": "dlhá odpoveď"}


def test_hyp_from_segments_joins_touching_segments() -> None:
    segs = [
        Segment(0.0, 4.0, "a", "S0"),
        Segment(4.0, 6.0, "(cez seba) b", "S0", crosstalk=True, region=0),
        Segment(4.0, 6.0, "x", "S1", crosstalk=True, region=0),
        Segment(7.0, 9.0, "c", "S0"),
    ]
    assert hyp_from_segments(segs, 4.0, 6.0) == {"S0": "a (cez seba) b", "S1": "x"}


def test_build_candidates_writes_clips_and_entries(tmp_path) -> None:
    audio = np.zeros(20 * SR, dtype=np.float32)
    diar = Diarization(
        "m",
        [(0.0, 10.0, "S0"), (4.0, 6.0, "S1"), (12.0, 16.0, "S1"), (13.0, 13.8, "S0")],
        [(0.0, 10.0, "S0"), (12.0, 16.0, "S1")],
    )
    regions = [
        Region(4.0, 6.0, frozenset({"S0", "S1"}), 0),
        Region(13.0, 13.8, frozenset({"S0", "S1"}), 1),
    ]
    out = build_candidates("ep", diar, audio, TRANSCRIPT, regions, tmp_path, top=1)
    assert len(out) == 1
    entry = out[0]
    assert (entry["start"], entry["end"]) == (4.0, 6.0)
    assert entry["speakers"] == ["S0", "S1"]
    assert entry["dominant"] == "S0"
    assert entry["gold"] == {"S0": "", "S1": ""}
    assert entry["transcript_lines"] == ["Speaker A [00:00]: dlhá odpoveď"]
    clip, sr = sf.read(entry["clip"])
    assert sr == SR and len(clip) == 6 * SR


def test_merge_gold_skips_duplicates_and_keeps_filled_text() -> None:
    old = [{"episode": "ep", "start": 4.0, "gold": {"S0": "vyplnené"}}]
    new = [{"episode": "ep", "start": 4.0, "gold": {"S0": ""}}, {"episode": "ep", "start": 9.0, "gold": {}}]
    merged = merge_gold(old, new)
    assert [m["start"] for m in merged] == [4.0, 9.0]
    assert merged[0]["gold"] == {"S0": "vyplnené"}
