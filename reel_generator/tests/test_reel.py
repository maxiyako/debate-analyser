from __future__ import annotations

import re
from pathlib import Path

import pytest

from reel_generator.align import align_proportional
from reel_generator.config import ReelStyle
from reel_generator.crop import Face, crop_plan_for_face
from reel_generator.generator import generate
from reel_generator.highlight import highlight_from_report, load_report
from reel_generator.paths import DEFAULT_DATA_DIR
from reel_generator.render import build_ffmpeg_command, filtergraph
from reel_generator.subtitles import generate_ass
from reel_generator.timeline import plan_window
from reel_generator.transcript import load_transcript
from reel_generator.align import Word

REPORT = DEFAULT_DATA_DIR / "reports" / "620752.json"
TRANSCRIPT = DEFAULT_DATA_DIR / "transcripts" / "620752.corrected.txt"
DODGE = 'behavioral_analysis.speakers[speaker="Erik Tomáš"].question_dodging[1]'
FALLACY = 'behavioral_analysis.speakers[speaker="Erik Tomáš"].logical_fallacies[0]'
LIE = 'facts[speaker="Erik Tomáš"][verdict="False"][0]'


def test_question_dodge_path_matches_report_line():
    report = load_report(REPORT)
    item = highlight_from_report(report, DODGE, "620752")
    assert item.speaker == "Erik Tomáš"
    assert item.kind == "question_dodging"
    assert item.index == 1
    assert item.timestamp_label == "26:47"
    assert item.timestamp_s == 26 * 60 + 47
    assert "baby boom" in item.quote
    assert item.stamp_text == "UHNUTIE"
    assert item.stamp_key == "dodge"
    assert item.outcome == "dodged"


def test_fallacy_uses_parenthetical_and_evidence_timestamp():
    item = highlight_from_report(load_report(REPORT), FALLACY, "620752")
    assert item.stamp_text == "FAUL: TY TIEŽ"
    assert item.stamp_key == "fallacy"
    assert item.timestamp_label == "10:18"
    assert "padnuté" in item.quote or "vlády" in item.quote


def test_behaviour_without_bracket_time_uses_the_transcript_line():
    report = load_report(DEFAULT_DATA_DIR / "reports" / "621954.json")
    item = highlight_from_report(
        report,
        'behavioral_analysis.speakers[speaker="Juraj Krúpa"].manipulation[1]',
        "621954",
    )
    assert item.kind == "manipulation"
    assert item.slug.endswith("_juraj_krupa")
    assert "spackan" in item.quote.casefold()
    assert 22 * 60 + 31 <= item.timestamp_s <= 22 * 60 + 31 + 40
    dukla = highlight_from_report(
        report,
        'behavioral_analysis.speakers[speaker="Robert Kaliňák"].logical_fallacies[0]',
        "621954",
    )
    assert "dukl" in dukla.quote.casefold()
    assert 1 * 3600 + 3 * 60 + 56 <= dukla.timestamp_s <= 1 * 3600 + 4 * 60 + 20


def test_false_fact_stamps_loz():
    item = highlight_from_report(load_report(REPORT), LIE, "620752")
    assert item.stamp_text == "LOŽ"
    assert item.stamp_key == "loz"
    assert item.outcome == "False"
    assert item.speaker == "Erik Tomáš"


def test_dodge_window_starts_at_answer_not_whole_question():
    item = highlight_from_report(load_report(REPORT), DODGE, "620752")
    lines = load_transcript(TRANSCRIPT)
    words = align_proportional(lines, item.timestamp_s - 1, item.timestamp_s + 90)
    window = plan_window(item, lines, words, ReelStyle())
    assert item.timestamp_s < window.stamp_at < item.timestamp_s + 90
    assert window.start < window.stamp_at < window.end
    assert window.end - window.stamp_at >= 22
    question = next(word for word in words if word.text.lower().startswith("očakávate"))
    assert window.start <= question.start - 0.6
    erik = [word for word in words if word.speaker.startswith("Erik") and word.start >= window.stamp_at]
    assert len(erik) > 30
    assert erik[24].start - erik[0].start > 5


def test_ass_highlights_only_current_word_and_caps_chunk_size(tmp_path: Path):
    words = [
        Word("Očakávate", 0.0, 0.4, "Moderátor"),
        Word("teraz", 0.4, 0.7, "Moderátor"),
        Word("že", 0.7, 0.9, "Moderátor"),
        Word("nastane", 0.9, 1.3, "Moderátor"),
        Word("nejaký", 1.3, 1.7, "Moderátor"),
    ]
    path = generate_ass(words, tmp_path / "clip.ass", ReelStyle())
    text = path.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in text
    assert "PlayResY: 1920" in text
    assert "Arial Black" in text
    assert "&H0000FFFF" in text
    assert "&H00FFFFFF" in text
    assert "\\pos(540,1460)" in text
    dialogues = [line for line in text.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == 5
    for line in dialogues:
        payload = line.split(",,", 1)[-1]
        visible = re.sub(r"\{[^}]*\}", "", payload)
        assert len(visible.split()) <= 3
        assert visible.count("Očakávate") + visible.count("nastane") <= 3
    # First event paints only the first word yellow.
    assert dialogues[0].index("\\c&H0000FFFF") < dialogues[0].index("Očakávate")
    assert dialogues[0].index("Očakávate") < dialogues[0].index("\\c&H00FFFFFF")


def test_ffmpeg_graph_crops_overlays_and_burns_ass(tmp_path: Path):
    from reel_generator.crop import CropPlan

    crop = CropPlan("cover", 1920, 1080, 400, 0, 608, 1080, True)
    style = ReelStyle()
    ass = tmp_path / "clip.ass"
    stamp = tmp_path / "loz.png"
    out = tmp_path / "reel.mp4"
    graph = filtergraph(ass, crop, style, stamp=True, stamp_start=2.5, stamp_end=6.0)
    assert "crop=608:1080:400:0" in graph
    assert "scale=1080:1920" in graph
    assert "overlay=x=(W-w)/2:y=70" in graph
    assert "between(t\\,2.500\\,6.000)" in graph
    assert "ass=filename='" in graph
    assert "eval=frame" in graph
    command = build_ffmpeg_command(
        Path("data/raw/620752.mp4"),
        ass,
        out,
        clip_start=1607.0,
        duration=14.0,
        crop=crop,
        style=style,
        stamp_path=stamp,
        stamp_start=2.5,
        stamp_end=6.0,
    )
    assert command[0] == "ffmpeg"
    assert "-filter_complex" in command
    assert "[vout]" in command
    joined = " ".join(command)
    assert "libx264" in joined
    assert "yuv420p" in joined


def test_scene_cuts_use_the_frame_timestamp_not_a_half_second_grid():
    from reel_generator.crop import scene_cut_times

    log = (
        "pts_time:3.88 duration_time:0.04\n"
        "pts_time:3.92 duration_time:0.04\n"
        "pts_time:10.4 duration_time:0.04\n"
    )
    assert scene_cut_times(log) == [3.88, 10.4]


def test_crop_x_switches_when_the_speaker_changes():
    from reel_generator.crop import CropPlan, CropSpan, cover_crop_filter

    plan = CropPlan(
        "cover",
        640,
        360,
        8,
        0,
        202,
        360,
        True,
        (CropSpan(0.0, 8.0, 8), CropSpan(8.0, 40.0, 248)),
    )
    clause = cover_crop_filter(plan)
    assert "if(lt(t\\,8.000)\\,8\\,248)" in clause


def test_face_at_edge_stays_a_cover_crop():
    face = Face(x=1700, y=80, w=150, h=220)
    plan = crop_plan_for_face(1920, 1080, face, ReelStyle())
    assert plan.mode == "cover"
    assert plan.x + plan.w <= 1920


def test_face_in_frame_shifts_cover_crop():
    face = Face(x=200, y=300, w=180, h=180)
    plan = crop_plan_for_face(1920, 1080, face, ReelStyle())
    assert plan.mode == "cover"
    assert plan.w / plan.h == 608 / 1080 or abs(plan.w / plan.h - 9 / 16) < 0.02
    assert plan.x <= face.x
    assert plan.x + plan.w >= face.x + face.w
    assert plan.x < (1920 - plan.w) / 2


def test_two_faces_on_a_split_screen_widen_the_crop():
    from reel_generator.crop import crop_plan_for_faces

    left = Face(114, 65, 181, 181)
    right = Face(390, 80, 142, 142)
    plan = crop_plan_for_faces(640, 360, [left, right], ReelStyle())
    assert plan.mode == "fit"
    assert plan.x == 0
    assert plan.x + plan.w >= right.x + right.w
    assert plan.w > 202


def test_split_screen_swaps_in_the_full_frame_instead_of_resizing_the_crop():
    from reel_generator.crop import CropPlan, crop_plan_for_faces

    plan = crop_plan_for_faces(
        640,
        360,
        [Face(114, 65, 181, 181), Face(390, 80, 142, 142)],
        ReelStyle(),
    )
    assert isinstance(plan, CropPlan)
    graph = filtergraph(
        Path("clip.ass"),
        plan,
        ReelStyle(),
        stamp=False,
        stamp_start=0,
        stamp_end=1,
    )
    assert "scale=1080:-2" in graph
    assert "crop=640" not in graph
    assert "between(t\\,0.000\\,1.000)" in graph


def test_one_face_stays_a_narrow_cover():
    from reel_generator.crop import crop_plan_for_faces

    face = Face(400, 60, 90, 90)
    plan = crop_plan_for_faces(640, 360, [face], ReelStyle())
    assert plan.mode == "cover"
    assert plan.w < 250
    assert plan.x <= face.x
    assert plan.x + plan.w >= face.x + face.w


def test_nested_face_boxes_count_as_one_person():
    from reel_generator.crop import distinct_faces

    faces = distinct_faces([Face(340, 50, 90, 90), Face(355, 45, 110, 110)])
    assert len(faces) == 1


def test_captions_follow_words_actually_heard_and_drop_the_cut_word():
    from reel_generator.heard import captions_from_heard

    planned = [
        Word("Dokonca", 10.0, 10.4, "Erik"),
        Word("dve", 10.4, 10.8, "Erik"),
        Word("opatrenia", 10.8, 11.3, "Erik"),
    ]
    heard = [
        Word("dve", 10.50, 10.9, ""),
        Word("opatrenia", 10.9, 11.4, ""),
    ]
    captions = captions_from_heard(planned, heard, 10.4, 12.0)
    assert [word.text for word in captions] == ["dve", "opatrenia"]
    assert captions[0].start == 10.50


def test_words_from_transcript_offsets_both_backends():
    from reel_generator.heard import words_from_transcript

    chunks = words_from_transcript(
        {
            "chunks": [
                {"text": " Mne", "timestamp": (1.0, 1.4)},
                {"text": "  ", "timestamp": (1.4, 1.6)},
                {"text": "členovia", "timestamp": (1.5, None)},
            ]
        },
        10.0,
    )
    assert [(word.text, word.start, word.end) for word in chunks] == [
        ("Mne", 11.0, 11.4),
        ("členovia", 11.5, 11.8),
    ]
    segments = words_from_transcript(
        {"segments": [{"words": [{"word": " dvere", "start": 0.5, "end": 0.9}]}]},
        5.0,
    )
    assert segments[0].text == "dvere"
    assert segments[0].start == 5.5


def test_trim_to_speech_skips_silence_and_starts_on_the_voice(tmp_path: Path):
    import numpy as np
    import soundfile as sf

    from reel_generator.heard import trim_to_speech

    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    audio[int(0.40 * sr) : int(0.70 * sr)] = 0.08
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), audio, sr)
    trimmed = trim_to_speech(
        [
            Word("ticho", 100.0, 100.30, ""),
            Word("slovo", 100.2, 100.90, ""),
        ],
        wav,
        100.0,
    )
    assert [word.text for word in trimmed] == ["slovo"]
    assert trimmed[0].start == pytest.approx(100.40, abs=0.03)
    assert trimmed[0].end == pytest.approx(100.70, abs=0.03)


def test_transcribe_span_hears_with_local_whisper_when_mlx_is_missing(tmp_path: Path, monkeypatch):
    import numpy as np
    import soundfile as sf

    import reel_generator.heard as heard

    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    audio[int(0.40 * sr) : int(0.70 * sr)] = 0.08
    wav = tmp_path / "align.wav"
    sf.write(str(wav), audio, sr)
    monkeypatch.setattr(heard, "_extract_wav", lambda *args, **kwargs: wav)

    def missing_mlx(path, language):
        raise ImportError("No module named 'mlx_whisper'")

    def local_whisper(path, language):
        return {"chunks": [{"text": " slovo", "timestamp": (0.2, 0.9)}]}

    monkeypatch.setattr(heard, "_mlx_transcribe", missing_mlx)
    monkeypatch.setattr(heard, "_transformers_transcribe", local_whisper)
    words = heard.transcribe_span(Path("video.mp4"), 100.0, 101.0, tmp_path)
    assert [word.text for word in words] == ["slovo"]
    assert words[0].start == pytest.approx(100.40, abs=0.03)


def test_false_fact_pads_before_and_blinks_only_on_the_claim(tmp_path: Path):
    result = generate(
        "620752",
        LIE,
        output_dir=tmp_path,
        align="proportional",
        dry_run=True,
        frame_with_llm=False,
        hear=False,
    )
    assert result.clip_start <= result.stamp_at - 2.8
    assert result.clip_end >= result.stamp_at + 3.0
    command = " ".join(result.command)
    assert "mod(" in command
    start = float(re.search(r"gte\(t\\,([0-9.]+)\)", command).group(1))
    end = float(re.search(r"lte\(t\\,([0-9.]+)\)", command).group(1))
    assert start >= 2.5
    assert end < result.clip_end - result.clip_start - 0.3


def test_small_face_drift_keeps_the_same_crop():
    from reel_generator.crop import spans_along_track

    track = [
        (0.0, [Face(300, 50, 80, 100)]),
        (0.5, [Face(312, 50, 80, 100)]),
        (1.0, [Face(306, 48, 80, 100)]),
    ]
    spans = spans_along_track(track, [(0.0, 1.5)], 640, 360, 0.0, 1.5)
    assert len(spans) == 1
    assert spans[0].w < 250


def test_crop_follows_a_driving_face_and_holds_through_a_miss():
    from reel_generator.crop import spans_along_track

    track = [
        (10.0, [Face(320, 40, 100, 120)]),
        (10.5, []),
        (11.0, [Face(400, 40, 100, 120)]),
    ]
    spans = spans_along_track(track, [(10.0, 11.5)], 640, 360, 10.0, 1.5)
    assert spans[0].x > 240
    assert spans[-1].x > spans[0].x
    assert all(span.w < 250 for span in spans)


def test_small_faces_in_a_wide_shot_use_the_full_frame():
    from reel_generator.crop import window_for_faces

    x, width = window_for_faces(640, 360, [Face(80, 70, 55, 55), Face(470, 60, 60, 60)])
    assert (x, width) == (0, 640)


def test_dry_run_writes_ass_stamp_and_command(tmp_path: Path):
    result = generate(
        "620752",
        DODGE,
        output_dir=tmp_path,
        align="proportional",
        dry_run=True,
        frame_with_llm=False,
    )
    assert result.stamp_text == "UHNUTIE"
    assert result.ass_path.is_file()
    assert result.stamp_path.is_file()
    assert result.stamp_path.stat().st_size > 100
    assert "overlay=" in " ".join(result.command)
    assert result.rendered is False
    assert result.clip_start < result.stamp_at < result.clip_end
    text = result.ass_path.read_text(encoding="utf-8")
    assert "Dialogue:" in text
    assert "mod(" in " ".join(result.command)
    starts = []
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        clock = line.split(",", 2)[1]
        hours, minutes, rest = clock.split(":")
        starts.append(int(hours) * 3600 + int(minutes) * 60 + float(rest))
    starts.sort()
    assert max(starts) - min(starts) > 25
    for moment in starts:
        packed = sum(1 for other in starts if moment <= other < moment + 0.2)
        assert packed <= 6
