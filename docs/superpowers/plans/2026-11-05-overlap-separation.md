# Overlap Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When two people talk at once, give each of them their own transcript line with their own words, transcribed by the same Slovak Whisper model, and measure the gain on a hand-made gold set before enabling it.

**Architecture:** Diarization switches to `pyannote/speaker-diarization-community-1` and keeps both its overlap-aware and exclusive outputs (new `.diarization.v2.json` cache). A new module `src/overlap.py` finds overlap regions, enrolls each speaker's voice from clean speech, runs ClearerVoice MossFormer2 two-speaker separation on each region and assigns the streams to speakers by embedding similarity. `src/transcription.py` is refactored so Whisper transcribes a list of audio chunks (clean slices, fallback slices and separated streams) that each carry their speaker. `src/overlap_eval.py` plus two `src/eval.py` commands build and score the gold set.

**Tech Stack:** Python 3.11, pyannote.audio 4.0.7, torch 2.13.0 (CPU/MPS), transformers (Whisper), ClearerVoice `MossFormer2_SS_16K` (model class only, loaded directly), soundfile, numpy, click, pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-overlap-separation-design.md`

## Global Constraints

- Tests are offline only: no model download, no network, no real audio files. Run with `.venv/bin/python -m pytest -q` (the system interpreter lacks dependencies).
- Installing ClearerVoice must not change the installed versions of torch (2.13.0), torchaudio, transformers, numpy, soundfile or librosa. Install with `--no-deps` exactly as in Task 1.
- `overlap_separation_enabled` defaults to `False`. It is only switched on by the user after the gold gate passes; no task flips it.
- Transcript line format stays `Label [MM:SS]: text`, and cross-talk lines keep the exact prefix `(cez seba) `.
- Old caches (`*.diarization.json`), transcripts and reports are never read as overlap-aware, rewritten or deleted.
- Separation and speaker embedding run on CPU. Whisper keeps using `settings.whisper_device`.
- Match the surrounding code: sparse comments, only for constraints the code cannot show.

## Prerequisite

`src/interpretation.py` and `tests/test_interpretation.py` have uncommitted changes from earlier work. Ask the user to commit them (or confirm you may) before Task 1, so each task below produces a focused commit. Confirm the suite is green first:

Run: `.venv/bin/python -m pytest -q`
Expected: `426 passed`

## File Structure

| File | Responsibility |
| --- | --- |
| `requirements.txt` (modify) | Document the `--no-deps` ClearerVoice install |
| `config.py` (modify) | Diarization model default, six overlap settings, drop `crosstalk_min_overlap` |
| `src/overlap.py` (create) | `Region`, region finding, span trimming, enrollment selection, lead speaker, stream assignment, region processing, model wrappers, `load_overlap_tools` |
| `src/transcription.py` (modify) | `Diarization` + v2 cache, `load_audio`, `WhisperASR`, `clean_spans`, `build_segments`, `merge_segments`, `load_or_diarize`, rewired `run_transcription` |
| `src/overlap_eval.py` (create) | Gold candidates, token scoring, label mapping, row scoring, gate, eval orchestration |
| `src/eval.py` (modify) | CLI commands `overlap-candidates` and `overlap` |
| `tests/conftest.py` (modify) | Fake separator, fake embedder, fake ASR fixtures |
| `tests/test_config.py` (modify) | New defaults |
| `tests/test_overlap.py` (create) | Everything in `src/overlap.py` except real models |
| `tests/test_transcription.py` (modify) | Diarization parsing, cache, segment building, merging, wiring |
| `tests/test_overlap_eval.py` (create) | Scoring, mapping, candidates, orchestration with fakes, CLI help |
| `docs/superpowers/specs/2026-11-05-overlap-separation-design.md` (modify) | Record decisions made while planning (Task 10) |

---

### Task 1: ClearerVoice install and settings

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py:79-92`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.diarization_model_id` (default `"pyannote/speaker-diarization-community-1"`), `Settings.overlap_separation_enabled: bool = False`, `overlap_min_seconds: float = 0.5`, `overlap_pad_seconds: float = 0.25`, `overlap_assign_min_sim: float = 0.3`, `enroll_max_seconds: float = 30.0`, `enroll_min_seconds: float = 3.0`. `crosstalk_min_overlap` stays until Task 6 removes it.

- [ ] **Step 1: Snapshot installed packages**

Run: `.venv/bin/pip freeze > /tmp/freeze_before.txt && wc -l /tmp/freeze_before.txt`
Expected: a line count, no error.

- [ ] **Step 2: Install ClearerVoice without its dependency pins**

`einops` (0.8.2) and `PyYAML` are already installed; these four packages are all the MossFormer2 separation model needs when its class is imported directly.

Run: `.venv/bin/pip install --no-deps clearvoice==0.1.2 yamlargparse==1.31.1 rotary-embedding-torch==0.8.3 torchinfo==1.8.0`
Expected: `Successfully installed clearvoice-0.1.2 rotary-embedding-torch-0.8.3 torchinfo-1.8.0 yamlargparse-1.31.1` (order may differ).

- [ ] **Step 3: Verify nothing else changed**

Run: `.venv/bin/pip freeze > /tmp/freeze_after.txt; diff /tmp/freeze_before.txt /tmp/freeze_after.txt`
Expected: exactly four added lines (`> clearvoice==0.1.2`, `> rotary-embedding-torch==0.8.3`, `> torchinfo==1.8.0`, `> yamlargparse==1.31.1`) and no `<` lines. If any `<` line appears, stop and report it to the user.

- [ ] **Step 4: Verify the model class imports**

Run: `.venv/bin/python -c "from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K; import torch; print('ok', torch.__version__)"`
Expected: `ok 2.13.0`

- [ ] **Step 5: Write the failing settings test**

Append to `tests/test_config.py`:

```python
def test_overlap_settings_defaults() -> None:
    s = _settings()
    assert s.diarization_model_id == "pyannote/speaker-diarization-community-1"
    assert s.overlap_separation_enabled is False
    assert s.overlap_min_seconds == 0.5
    assert s.overlap_pad_seconds == 0.25
    assert s.overlap_assign_min_sim == 0.3
    assert s.enroll_max_seconds == 30.0
    assert s.enroll_min_seconds == 3.0
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_overlap_settings_defaults -v`
Expected: FAIL (`AttributeError` or assertion on `diarization_model_id`).

- [ ] **Step 7: Implement the settings**

In `config.py`, change the diarization default and add the overlap block after `crosstalk_min_overlap`:

```python
    whisper_device: str = "cpu"
    whisper_model_id: str = "kinit/whisper-large-v3-sk"
    diarization_model_id: str = "pyannote/speaker-diarization-community-1"
```

```python
    # Flag a line as cross-talk when a second speaker overlaps at least this
    # fraction of the segment's duration.
    crosstalk_min_overlap: float = 0.35
    # Overlap separation: two-speaker regions are split into one stream per
    # speaker and transcribed separately. Off until the overlap gold gate passes.
    overlap_separation_enabled: bool = False
    overlap_min_seconds: float = 0.5
    overlap_pad_seconds: float = 0.25
    overlap_assign_min_sim: float = 0.3
    enroll_max_seconds: float = 30.0
    enroll_min_seconds: float = 3.0
```

Append to `requirements.txt`:

```
# Overlap separation (optional, used only with OVERLAP_SEPARATION_ENABLED=true).
# clearvoice pins torch/numpy/soundfile/librosa versions we must not change, so
# install it and the three packages its separation model imports with --no-deps:
#   pip install --no-deps clearvoice==0.1.2 yamlargparse==1.31.1 rotary-embedding-torch==0.8.3 torchinfo==1.8.0
```

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/python -m pytest -q`
Expected: `427 passed`

- [ ] **Step 9: Commit**

```bash
git add config.py requirements.txt tests/test_config.py
git commit -m "feat: overlap separation settings and community-1 diarization default"
```

---

### Task 2: Overlap geometry

**Files:**
- Create: `src/overlap.py`
- Create: `tests/test_overlap.py`

**Interfaces:**
- Produces (in `src/overlap.py`):
  - `SR = 16000`
  - `Turn = tuple[float, float, str]`
  - `@dataclass(frozen=True) class Region: start: float; end: float; speakers: frozenset[str]; group: int`
  - `find_overlap_regions(turns: list[Turn], *, min_seconds: float, merge_gap: float = 0.3, max_chunk_seconds: float = 8.0) -> list[Region]`
  - `trim_spans(spans: list[Turn], regions: list[Region]) -> list[Turn]`
  - `select_enrollment(turns: list[Turn], *, max_seconds: float, min_seconds: float) -> dict[str, list[tuple[float, float]]]`
  - `lead_speaker(region: Region, turns: list[Turn]) -> str`

`max_chunk_seconds=8.0` exists because the MossFormer2 model handles at most about 10 s in one pass; longer regions are split into equal chunks sharing one `group`, and each chunk is assigned separately, which also keeps stream order from swapping between chunks.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overlap.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_overlap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.overlap'`.

- [ ] **Step 3: Implement the geometry**

Create `src/overlap.py`:

```python
"""Overlap regions, voice enrollment and separated-stream assignment."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SR = 16000
Turn = tuple[float, float, str]


@dataclass(frozen=True)
class Region:
    start: float
    end: float
    speakers: frozenset[str]
    group: int


def _overlap_spans(turns: list[Turn]) -> list[tuple[float, float, frozenset[str]]]:
    events: list[tuple[float, int, str]] = []
    for start, end, speaker in turns:
        if end > start:
            events.append((start, 1, speaker))
            events.append((end, -1, speaker))
    events.sort(key=lambda ev: (ev[0], ev[1]))
    active: dict[str, int] = {}
    spans: list[tuple[float, float, frozenset[str]]] = []
    prev_t: float | None = None
    for t, kind, speaker in events:
        if prev_t is not None and t > prev_t:
            live = frozenset(k for k, v in active.items() if v > 0)
            if len(live) >= 2:
                spans.append((prev_t, t, live))
        active[speaker] = active.get(speaker, 0) + kind
        prev_t = t
    return spans


def find_overlap_regions(
    turns: list[Turn],
    *,
    min_seconds: float,
    merge_gap: float = 0.3,
    max_chunk_seconds: float = 8.0,
) -> list[Region]:
    merged: list[tuple[float, float, frozenset[str]]] = []
    for start, end, speakers in _overlap_spans(turns):
        if merged and merged[-1][2] == speakers and start - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], end, speakers)
        else:
            merged.append((start, end, speakers))

    regions: list[Region] = []
    kept = [m for m in merged if m[1] - m[0] >= min_seconds]
    for group, (start, end, speakers) in enumerate(kept):
        n = max(1, math.ceil((end - start) / max_chunk_seconds - 1e-9))
        step = (end - start) / n
        for i in range(n):
            chunk_end = end if i == n - 1 else start + (i + 1) * step
            regions.append(Region(start + i * step, chunk_end, speakers, group))
    return regions


def _subtract(start: float, end: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for c0, c1 in cuts:
        nxt: list[tuple[float, float]] = []
        for p0, p1 in pieces:
            if c1 <= p0 or c0 >= p1:
                nxt.append((p0, p1))
                continue
            if c0 > p0:
                nxt.append((p0, c0))
            if c1 < p1:
                nxt.append((c1, p1))
        pieces = nxt
    return pieces


def trim_spans(spans: list[Turn], regions: list[Region]) -> list[Turn]:
    cuts = [(r.start, r.end) for r in regions]
    out: list[Turn] = []
    for start, end, speaker in spans:
        out.extend((p0, p1, speaker) for p0, p1 in _subtract(start, end, cuts))
    return out


def select_enrollment(
    turns: list[Turn], *, max_seconds: float, min_seconds: float
) -> dict[str, list[tuple[float, float]]]:
    cuts = [(s, e) for s, e, _ in _overlap_spans(turns)]
    pieces: dict[str, list[tuple[float, float]]] = {}
    for start, end, speaker in turns:
        pieces.setdefault(speaker, []).extend(_subtract(start, end, cuts))

    out: dict[str, list[tuple[float, float]]] = {}
    for speaker, spans in pieces.items():
        chosen: list[tuple[float, float]] = []
        total = 0.0
        for p0, p1 in sorted(spans, key=lambda p: (-(p[1] - p[0]), p[0])):
            if total >= max_seconds:
                break
            take = min(p1 - p0, max_seconds - total)
            chosen.append((p0, p1 if take >= p1 - p0 else p0 + take))
            total += take
        if total >= min_seconds:
            out[speaker] = chosen
    return out


def lead_speaker(region: Region, turns: list[Turn]) -> str:
    active = [
        (start, speaker)
        for start, end, speaker in turns
        if speaker in region.speakers and start <= region.start + 1e-6 and end > region.start
    ]
    if active:
        return min(active)[1]
    return sorted(region.speakers)[0]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_overlap.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/overlap.py tests/test_overlap.py
git commit -m "feat: overlap regions, span trimming and enrollment selection"
```

---

### Task 3: Stream assignment and region processing

**Files:**
- Modify: `src/overlap.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_overlap.py`

**Interfaces:**
- Consumes: `Region`, `SR`, `Turn`, `select_enrollment`, `lead_speaker` from Task 2; `Settings` fields from Task 1.
- Produces (in `src/overlap.py`):
  - `class Embedder(Protocol): def embed(self, audio: np.ndarray) -> np.ndarray`
  - `class Separator(Protocol): def separate(self, audio: np.ndarray) -> list[np.ndarray]`
  - `@dataclass class OverlapTools: separator: Separator | None; embedder: Embedder`
  - `@dataclass class RegionResult: region: Region; lead: str; streams: dict[str, np.ndarray]; reason: str | None` (`reason` is `None` when separated, else one of `"too_many_speakers"`, `"no_enrollment"`, `"separator_unavailable"`, `"separator_error"`, `"low_similarity"`)
  - `cosine(a: np.ndarray, b: np.ndarray) -> float`
  - `assign_streams(sims: np.ndarray, speakers: list[str], min_sim: float) -> dict[str, int]` (rows are streams, columns are `speakers`; returns speaker to stream index)
  - `enroll(audio: np.ndarray, spans: dict[str, list[tuple[float, float]]], embedder: Embedder) -> dict[str, np.ndarray]`
  - `process_regions(audio: np.ndarray, regions: list[Region], turns: list[Turn], settings: Settings, tools: OverlapTools) -> list[RegionResult]`
  - `log_summary(results: list[RegionResult]) -> str` (logs and returns the line)
- Produces (in `tests/conftest.py`): fixtures `fake_embedder`, `fake_separator`, `voice_audio`.

- [ ] **Step 1: Add the fakes to `tests/conftest.py`**

Append:

```python
import numpy as np

SR = 16000


class FakeEmbedder:
    """Voice A is positive audio, voice B negative; silence embeds to zero."""

    def embed(self, audio: np.ndarray) -> np.ndarray:
        m = float(np.mean(audio)) if len(audio) else 0.0
        if m > 1e-6:
            return np.array([1.0, 0.0])
        if m < -1e-6:
            return np.array([0.0, 1.0])
        return np.array([0.0, 0.0])


class FakeSeparator:
    """Returns voice B then voice A, so assignment has to swap them."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def separate(self, audio: np.ndarray) -> list[np.ndarray]:
        self.calls.append(len(audio))
        return [np.full(len(audio), -0.5, dtype=np.float32), np.full(len(audio), 0.5, dtype=np.float32)]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_separator() -> FakeSeparator:
    return FakeSeparator()


@pytest.fixture
def voice_audio() -> np.ndarray:
    """20 s: A alone 0-4 and 6-10, A+B mixed 4-6, B alone 12-16, silence elsewhere."""
    audio = np.zeros(20 * SR, dtype=np.float32)
    audio[0 : 4 * SR] = 0.5
    audio[6 * SR : 10 * SR] = 0.5
    audio[12 * SR : 16 * SR] = -0.5
    return audio
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_overlap.py`:

```python
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
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_overlap.py -v`
Expected: FAIL with `ImportError: cannot import name 'OverlapTools'`.

- [ ] **Step 4: Implement assignment and processing**

In `src/overlap.py`, extend the imports:

```python
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from config import Settings
```

Append:

```python
class Embedder(Protocol):
    def embed(self, audio: np.ndarray) -> np.ndarray: ...


class Separator(Protocol):
    def separate(self, audio: np.ndarray) -> list[np.ndarray]: ...


@dataclass
class OverlapTools:
    separator: Separator | None
    embedder: Embedder


@dataclass
class RegionResult:
    region: Region
    lead: str
    streams: dict[str, np.ndarray] = field(default_factory=dict)
    reason: str | None = None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def assign_streams(sims: np.ndarray, speakers: list[str], min_sim: float) -> dict[str, int]:
    straight = sims[0, 0] + sims[1, 1]
    crossed = sims[0, 1] + sims[1, 0]
    pairs = [(0, 0), (1, 1)] if straight >= crossed else [(0, 1), (1, 0)]
    return {speakers[col]: row for row, col in pairs if sims[row, col] >= min_sim}


def _slice(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    return audio[max(0, int(start * SR)) : min(len(audio), int(end * SR))]


def enroll(
    audio: np.ndarray, spans: dict[str, list[tuple[float, float]]], embedder: Embedder
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for speaker, pieces in spans.items():
        vecs = [embedder.embed(_slice(audio, s, e)) for s, e in pieces]
        out[speaker] = np.mean(np.stack(vecs), axis=0)
    return out


def _peak_normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if peak == 0.0 else (x * (0.9 / peak)).astype(np.float32)


def _separate_one(
    audio: np.ndarray,
    region: Region,
    enrollments: dict[str, np.ndarray],
    settings: Settings,
    tools: OverlapTools,
) -> tuple[dict[str, np.ndarray], str | None]:
    if len(region.speakers) != 2:
        return {}, "too_many_speakers"
    speakers = sorted(region.speakers)
    if any(s not in enrollments for s in speakers):
        return {}, "no_enrollment"
    if tools.separator is None:
        return {}, "separator_unavailable"

    pad = settings.overlap_pad_seconds
    s0 = max(0, int((region.start - pad) * SR))
    e0 = min(len(audio), int((region.end + pad) * SR))
    try:
        raw = tools.separator.separate(audio[s0:e0])
    except Exception:
        logger.exception("Separator failed on region %.2f-%.2f", region.start, region.end)
        return {}, "separator_error"

    offset = int(region.start * SR) - s0
    length = int(region.end * SR) - int(region.start * SR)
    streams = [_peak_normalize(np.asarray(x[offset : offset + length], dtype=np.float32)) for x in raw[:2]]
    vecs = [tools.embedder.embed(x) for x in streams]
    sims = np.array([[cosine(v, enrollments[s]) for s in speakers] for v in vecs])
    assigned = assign_streams(sims, speakers, settings.overlap_assign_min_sim)
    if not assigned:
        return {}, "low_similarity"
    return {speaker: streams[idx] for speaker, idx in assigned.items()}, None


def process_regions(
    audio: np.ndarray,
    regions: list[Region],
    turns: list[Turn],
    settings: Settings,
    tools: OverlapTools,
) -> list[RegionResult]:
    if not regions:
        return []
    spans = select_enrollment(
        turns, max_seconds=settings.enroll_max_seconds, min_seconds=settings.enroll_min_seconds
    )
    enrollments = enroll(audio, spans, tools.embedder)
    results: list[RegionResult] = []
    for region in regions:
        streams, reason = _separate_one(audio, region, enrollments, settings, tools)
        results.append(RegionResult(region, lead_speaker(region, turns), streams, reason))
    return results


def log_summary(results: list[RegionResult]) -> str:
    separated = sum(1 for r in results if r.reason is None)
    reasons = Counter(r.reason for r in results if r.reason is not None)
    line = f"Overlap regions: {len(results)} found, {separated} separated"
    if reasons:
        line += ", fallback " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
    logger.info(line)
    return line
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_overlap.py -v`
Expected: 22 passed.

- [ ] **Step 6: Commit**

```bash
git add src/overlap.py tests/test_overlap.py tests/conftest.py
git commit -m "feat: assign separated streams to enrolled speakers per overlap region"
```

---

### Task 4: Model wrappers

**Files:**
- Modify: `src/overlap.py`
- Test: `tests/test_overlap.py`

**Interfaces:**
- Consumes: `OverlapTools`, `SR` from Tasks 2-3.
- Produces (in `src/overlap.py`):
  - `class MossFormerSeparator: REPO = "alibabasglab/MossFormer2_SS_16K"; __init__(self, device: str = "cpu"); separate(self, audio: np.ndarray) -> list[np.ndarray]`
  - `class PyannoteEmbedder: __init__(self, token: str | None, device: str = "cpu"); embed(self, audio: np.ndarray) -> np.ndarray`
  - `load_overlap_tools(settings: Settings) -> OverlapTools` (separator is `None` with one warning when MossFormer2 cannot load)

Verified while planning: `last_best_checkpoint.pt` loads with `weights_only=True`, is a dict with keys `model`, `optimizer`, `epoch`, `step`, and the 1099 weights under `model` have no `module.` prefix. The community-1 embedding model is a 256-dim `WeSpeakerResNet34` that accepts clips down to 0.5 s.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_overlap.py`:

```python
def test_load_overlap_tools_without_separator(monkeypatch, caplog, fake_embedder) -> None:
    import src.overlap as ov

    class NoSeparator:
        def __init__(self, device: str = "cpu") -> None:
            raise ImportError("No module named 'clearvoice'")

    monkeypatch.setattr(ov, "MossFormerSeparator", NoSeparator)
    monkeypatch.setattr(ov, "PyannoteEmbedder", lambda token, device="cpu": fake_embedder)
    with caplog.at_level(logging.WARNING, logger="src.overlap"):
        tools = ov.load_overlap_tools(_settings())
    assert tools.separator is None
    assert tools.embedder is fake_embedder
    assert "clearvoice" in caplog.text


def test_load_overlap_tools_with_separator(monkeypatch, fake_embedder, fake_separator) -> None:
    import src.overlap as ov

    monkeypatch.setattr(ov, "MossFormerSeparator", lambda device="cpu": fake_separator)
    monkeypatch.setattr(ov, "PyannoteEmbedder", lambda token, device="cpu": fake_embedder)
    tools = ov.load_overlap_tools(_settings())
    assert tools.separator is fake_separator
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_overlap.py -k load_overlap_tools -v`
Expected: FAIL with `AttributeError: <module 'src.overlap'> does not have the attribute 'MossFormerSeparator'`.

- [ ] **Step 3: Implement the wrappers**

Append to `src/overlap.py`:

```python
class MossFormerSeparator:
    REPO = "alibabasglab/MossFormer2_SS_16K"

    def __init__(self, device: str = "cpu") -> None:
        from types import SimpleNamespace

        import torch
        from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
        from huggingface_hub import hf_hub_download

        args = SimpleNamespace(
            encoder_embedding_dim=512,
            mossformer_sequence_dim=512,
            num_mossformer_layer=24,
            encoder_kernel_size=16,
            num_spks=2,
        )
        model = MossFormer2_SS_16K(args).model
        path = hf_hub_download(self.REPO, "last_best_checkpoint.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        missing, _unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing:
            raise RuntimeError(f"MossFormer2 checkpoint is missing {len(missing)} weights")
        self._model = model.to(device).eval()
        self._device = device
        self._torch = torch

    def separate(self, audio: np.ndarray) -> list[np.ndarray]:
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None].to(self._device)
        with torch.inference_mode():
            outputs = self._model(x)
        return [o[0].detach().cpu().numpy() for o in outputs]


class PyannoteEmbedder:
    def __init__(self, token: str | None, device: str = "cpu") -> None:
        import torch
        from pyannote.audio import Inference, Model

        model = Model.from_pretrained(
            "pyannote/speaker-diarization-community-1", subfolder="embedding", token=token
        )
        self._inference = Inference(model, window="whole", device=torch.device(device))
        self._torch = torch

    def embed(self, audio: np.ndarray) -> np.ndarray:
        wav = self._torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        return np.asarray(self._inference({"waveform": wav, "sample_rate": SR}), dtype=np.float32)


def load_overlap_tools(settings: Settings) -> OverlapTools:
    embedder = PyannoteEmbedder(settings.huggingface_token)
    try:
        separator: Separator | None = MossFormerSeparator()
    except Exception as exc:
        logger.warning("Overlap separation disabled for this run: %s", exc)
        separator = None
    return OverlapTools(separator=separator, embedder=embedder)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_overlap.py -v`
Expected: 24 passed.

- [ ] **Step 5: Smoke-test the real models on an artificial mixture**

This is a manual check, not a test: it downloads models. It mixes 4 s of a clean A turn with 4 s of a clean B turn from a real episode and checks the separated streams land on the right speakers.

Run:

```bash
.venv/bin/python - <<'EOF'
import json
import numpy as np, soundfile as sf
from config import get_settings
from src.overlap import MossFormerSeparator, PyannoteEmbedder, cosine, select_enrollment, enroll, assign_streams, SR

s = get_settings()
audio, sr = sf.read("data/audio/604147.wav", dtype="float32")
assert sr == SR
turns = [tuple(t) for t in json.load(open("data/transcripts/604147.diarization.json"))]
spans = select_enrollment(turns, max_seconds=30.0, min_seconds=3.0)
a, b = sorted(spans, key=lambda k: -sum(e - s0 for s0, e in spans[k]))[:2]
emb = PyannoteEmbedder(s.huggingface_token)
ref = enroll(audio, {a: spans[a][1:], b: spans[b][1:]}, emb)
(a0, _), (b0, _) = spans[a][0], spans[b][0]
mix = audio[int(a0*SR):int(a0*SR)+4*SR] + audio[int(b0*SR):int(b0*SR)+4*SR]
streams = MossFormerSeparator().separate(mix)
sims = np.array([[cosine(emb.embed(x), ref[spk]) for spk in (a, b)] for x in streams])
print("sims", sims.round(2), "assigned", assign_streams(sims, [a, b], 0.3))
EOF
```

Expected: two separated streams, a `sims` matrix whose larger values sit on one diagonal, and `assigned` mapping both speakers to different stream indices. If both speakers are not assigned, stop and report the printed matrix to the user before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/overlap.py tests/test_overlap.py
git commit -m "feat: MossFormer2 separator and community-1 embedder wrappers"
```

---

### Task 5: Diarization with both outputs and the v2 cache

**Files:**
- Modify: `src/transcription.py:39-85`
- Test: `tests/test_transcription.py`

**Interfaces:**
- Consumes: `Turn` from `src/overlap.py`.
- Produces (in `src/transcription.py`):
  - `@dataclass class Diarization: model: str; turns: list[Turn]; exclusive_turns: list[Turn]`
  - `diarization_from_output(output, model: str) -> Diarization`
  - `diarize(audio_path: Path, settings: Settings) -> Diarization`
  - `diarization_cache_path(transcript_dir: Path, episode_id: str) -> Path` (returns `transcript_dir / f"{episode_id}.diarization.v2.json"`)
  - `save_diarization(path: Path, diar: Diarization) -> None`
  - `load_diarization(path: Path) -> Diarization | None`

`run_transcription` is not touched in this task; Task 7 rewires it. To keep `run_transcription` working in between, this task changes its two diarization lines to use `.exclusive_turns` (Step 3).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transcription.py`:

```python
from types import SimpleNamespace

from src.transcription import (
    Diarization,
    diarization_cache_path,
    diarization_from_output,
    load_diarization,
    save_diarization,
)


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
    (tmp_path / "ep1.diarization.json").write_text("[[0.0, 1.0, \"S0\"]]", encoding="utf-8")
    assert load_diarization(diarization_cache_path(tmp_path, "ep1")) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transcription.py -v`
Expected: FAIL with `ImportError: cannot import name 'Diarization'`.

- [ ] **Step 3: Implement**

In `src/transcription.py`, add `import json` to the top imports and `from src.overlap import Turn`. Replace `diarize` (lines 39-85) with:

```python
@dataclass
class Diarization:
    model: str
    turns: list[Turn]
    exclusive_turns: list[Turn]


def _annotation_turns(annotation) -> list[Turn]:
    return [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def diarization_from_output(output, model: str) -> Diarization:
    regular = getattr(output, "speaker_diarization", None)
    if regular is None:
        turns = _annotation_turns(output)
        return Diarization(model, turns, list(turns))
    exclusive = getattr(output, "exclusive_speaker_diarization", None) or regular
    return Diarization(model, _annotation_turns(regular), _annotation_turns(exclusive))


def diarize(audio_path: Path, settings: Settings) -> Diarization:
    """Overlap-aware and exclusive speaker turns from pyannote."""
    model_id = settings.diarization_model_id
    if not settings.huggingface_token:
        msg = (
            "HUGGINGFACE_TOKEN missing — diarization cannot run, so the whole "
            "transcript would collapse to a single speaker. Accept terms at "
            f"https://huggingface.co/{model_id} and set the token."
        )
        if settings.require_diarization:
            raise RuntimeError(msg)
        logger.warning("%s Falling back to a single speaker (Speaker A).", msg)
        return Diarization(model_id, [], [])

    from pyannote.audio import Pipeline

    logger.info("Loading diarization model %s", model_id)
    try:
        pipeline = Pipeline.from_pretrained(model_id, token=settings.huggingface_token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=settings.huggingface_token)

    import torch

    pipeline.to(torch.device(_resolve_device(settings)))

    logger.info("Running diarization on %s", audio_path)
    diar = diarization_from_output(pipeline(str(audio_path)), model_id)
    logger.info(
        "Found %d speaker turns (%d exclusive)", len(diar.turns), len(diar.exclusive_turns)
    )
    return diar


def diarization_cache_path(transcript_dir: Path, episode_id: str) -> Path:
    return transcript_dir / f"{episode_id}.diarization.v2.json"


def save_diarization(path: Path, diar: Diarization) -> None:
    payload = {"model": diar.model, "turns": diar.turns, "exclusive_turns": diar.exclusive_turns}
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_diarization(path: Path) -> Diarization | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))

    def turns(key: str) -> list[Turn]:
        return [(float(a), float(b), str(s)) for a, b, s in raw[key]]

    return Diarization(raw["model"], turns("turns"), turns("exclusive_turns"))
```

In `run_transcription`, keep the old cache logic for now but adapt it to the new return type, replacing

```python
        diar_turns = diarize(audio_path, settings)
```

with

```python
        diar_turns = diarize(audio_path, settings).exclusive_turns
```

and remove the now-duplicate `import json` inside `run_transcription`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (`455 passed`).

- [ ] **Step 5: Commit**

```bash
git add src/transcription.py tests/test_transcription.py
git commit -m "feat: keep overlap-aware and exclusive diarization in a v2 cache"
```

---

### Task 6: Chunk-based ASR, segment building and merging

**Files:**
- Modify: `src/transcription.py` (replace `transcribe`, `_assign_speaker`, `merge_diarization_and_asr`; extend `Segment`)
- Modify: `config.py` (remove `crosstalk_min_overlap`)
- Modify: `tests/conftest.py` (fake ASR)
- Test: `tests/test_transcription.py`

**Interfaces:**
- Consumes: `Diarization` (Task 5); `Region`, `SR`, `Turn`, `OverlapTools`, `find_overlap_regions`, `trim_spans`, `process_regions`, `log_summary` (Tasks 2-3).
- Produces (in `src/transcription.py`):
  - `Segment` gains `region: int | None = None` and `rank: int = 0`
  - `MIN_SLICE_SECONDS = 0.3`
  - `load_audio(audio_path: Path) -> np.ndarray` (mono float32, raises on non-16 kHz)
  - `class ASR(Protocol): def transcribe(self, chunks: list[np.ndarray]) -> list[str]`
  - `class WhisperASR: __init__(self, settings: Settings, batch_size: int | None = None); transcribe(self, chunks: list[np.ndarray]) -> list[str]; close(self) -> None`
  - `clean_spans(exclusive_turns: list[Turn], total_s: float, regions: list[Region]) -> list[Turn]`
  - `build_segments(audio: np.ndarray, diar: Diarization, settings: Settings, asr: ASR, *, tools: OverlapTools | None = None, window: tuple[float, float] | None = None) -> list[Segment]` (raw labels, sorted by `(start, rank)`)
  - `merge_segments(segments: list[Segment], max_gap_seconds: float = 2.0) -> list[Segment]` (relabels to `Speaker A..`)
- Removes: `transcribe`, `_assign_speaker`, `merge_diarization_and_asr`, `_overlap`, `Settings.crosstalk_min_overlap`.
- Produces (in `tests/conftest.py`): fixture `fake_asr` (object with `.transcribe(chunks)` returning `c0, c1, ...` and recording chunk lengths in `.lengths`).

- [ ] **Step 1: Add the fake ASR to `tests/conftest.py`**

Append:

```python
class FakeASR:
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def transcribe(self, chunks: list[np.ndarray]) -> list[str]:
        out = []
        for chunk in chunks:
            out.append(f"c{len(self.lengths)}")
            self.lengths.append(len(chunk))
        return out


@pytest.fixture
def fake_asr() -> FakeASR:
    return FakeASR()
```

- [ ] **Step 2: Replace the old merge tests with failing new ones**

In `tests/test_transcription.py`, replace the import block and every test that uses `_assign_speaker` or `merge_diarization_and_asr` (the five tests from `test_assign_speaker_dominant_no_crosstalk` through `test_format_transcript_marks_crosstalk`) with the following. Keep the two `_window_starts` tests and the Task 5 tests.

```python
import numpy as np

from config import Settings
from src.overlap import OverlapTools
from src.transcription import (
    Segment,
    _window_starts,
    build_segments,
    clean_spans,
    format_transcript,
    merge_segments,
)

SR = 16000


def _settings(**kw) -> Settings:
    base = dict(enroll_min_seconds=1.0, overlap_min_seconds=0.5, overlap_pad_seconds=0.25)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _rows(segments):
    return [(s.speaker, s.start, s.end, s.crosstalk) for s in segments]


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


def test_clean_spans_merge_chunk_and_trim() -> None:
    from src.overlap import Region

    turns = [(0.0, 10.0, "S0"), (10.3, 40.0, "S0"), (40.0, 40.2, "S1")]
    regions = [Region(4.0, 6.0, frozenset({"S0", "S1"}), 0)]
    assert clean_spans(turns, 45.0, regions) == [
        (0.0, 4.0, "S0"),
        (6.0, 34.0, "S0"),
        (34.0, 40.0, "S0"),
    ]


def test_flag_off_matches_exclusive_turn_slicing(fake_asr) -> None:
    from src.transcription import Diarization

    audio = np.zeros(20 * SR, dtype=np.float32)
    diar = Diarization("m", [(0.0, 10.0, "S0"), (8.0, 20.0, "S1")], [(0.0, 10.0, "S0"), (10.0, 20.0, "S1")])
    raw = build_segments(audio, diar, _settings(), fake_asr)
    assert _rows(raw) == [("S0", 0.0, 10.0, False), ("S1", 10.0, 20.0, False)]
    assert format_transcript(merge_segments(raw)) == "Speaker A [00:00]: c0\nSpeaker B [00:10]: c1\n"


def test_no_diarization_falls_back_to_30s_windows(fake_asr) -> None:
    from src.transcription import Diarization

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
    from src.transcription import Diarization

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
    from src.transcription import Diarization

    tools = OverlapTools(separator=None, embedder=fake_embedder)
    raw = build_segments(voice_audio, Diarization("m", **OVERLAP_DIAR), _settings(), fake_asr, tools=tools)
    assert _rows(raw) == [
        ("A", 0.0, 4.0, False),
        ("A", 4.0, 6.0, True),
        ("A", 6.0, 10.0, False),
        ("B", 12.0, 16.0, False),
    ]


def test_window_limits_segments(voice_audio, fake_separator, fake_embedder, fake_asr) -> None:
    from src.transcription import Diarization

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
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transcription.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_segments'`.

- [ ] **Step 4: Implement**

In `src/transcription.py`:

Top imports become:

```python
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from config import Settings, get_settings
from src.gcs import upload_file
from src.overlap import (
    SR,
    OverlapTools,
    Region,
    Turn,
    find_overlap_regions,
    log_summary,
    process_regions,
    trim_spans,
)
```

`Segment` becomes:

```python
@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_00"
    crosstalk: bool = False  # another speaker talks at the same time
    region: int | None = None
    rank: int = 0
```

Delete `transcribe`, `_overlap`, `_assign_speaker` and `merge_diarization_and_asr`. Add after `_chunk_spans`:

```python
MIN_SLICE_SECONDS = 0.3


def load_audio(audio_path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise RuntimeError(f"Expected 16 kHz WAV, got {sr} Hz — re-extract audio")
    return audio


class ASR(Protocol):
    def transcribe(self, chunks: list[np.ndarray]) -> list[str]: ...


class WhisperASR:
    """Batched Slovak Whisper over in-memory audio chunks.

    This model does not reliably emit Whisper timestamp tokens, and word-level
    timestamps via cross-attention DTW are ~7x slower on MPS, so each chunk is
    one diarization slice or separated stream and its speaker is known upfront.
    """

    def __init__(self, settings: Settings, batch_size: int | None = None) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self._torch = torch
        self.device = _resolve_device(settings)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        model_id = settings.whisper_model_id
        logger.info("Loading Whisper model %s on %s (dtype=%s)", model_id, self.device, self.dtype)
        load_kwargs: dict = {}
        if settings.huggingface_token and not Path(model_id).exists():
            load_kwargs["token"] = settings.huggingface_token
        self.processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=self.dtype, low_cpu_mem_usage=True, **load_kwargs
        )
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size or settings.asr_batch_size

    def transcribe(self, chunks: list[np.ndarray]) -> list[str]:
        torch = self._torch
        texts: list[str] = []
        bs = self.batch_size
        for b in range(0, len(chunks), bs):
            batch = chunks[b : b + bs]
            inputs = self.processor(
                batch, sampling_rate=SR, return_tensors="pt", return_attention_mask=True
            )
            features = inputs.input_features.to(device=self.device, dtype=self.dtype)
            mask = inputs.attention_mask.to(device=self.device)
            with torch.inference_mode():
                predicted = self.model.generate(
                    features,
                    attention_mask=mask,
                    language="slovak",
                    task="transcribe",
                    no_repeat_ngram_size=3,
                )
            texts.extend(t.strip() for t in self.processor.batch_decode(predicted, skip_special_tokens=True))
            done = min(b + bs, len(chunks))
            if done % (bs * 5) == 0 or done == len(chunks):
                logger.info("ASR progress %d/%d (%.0f%%)", done, len(chunks), 100.0 * done / len(chunks))
        return texts

    def close(self) -> None:
        del self.model
        if self.device == "mps":
            self._torch.mps.empty_cache()


def clean_spans(exclusive_turns: list[Turn], total_s: float, regions: list[Region]) -> list[Turn]:
    spans: list[Turn] = []
    for start, end, speaker in trim_spans(_merge_turns(exclusive_turns), regions):
        for s, e in _chunk_spans(start, min(end, total_s)):
            if e - s >= MIN_SLICE_SECONDS:
                spans.append((s, e, speaker))
    return spans


def _clip(spans: list[Turn], window: tuple[float, float] | None) -> list[Turn]:
    if window is None:
        return spans
    w0, w1 = window
    out: list[Turn] = []
    for s, e, speaker in spans:
        s, e = max(s, w0), min(e, w1)
        if e - s >= MIN_SLICE_SECONDS:
            out.append((s, e, speaker))
    return out


def build_segments(
    audio: np.ndarray,
    diar: Diarization,
    settings: Settings,
    asr: ASR,
    *,
    tools: OverlapTools | None = None,
    window: tuple[float, float] | None = None,
) -> list[Segment]:
    """Speaker-tagged ASR segments: clean slices, fallback slices, separated streams."""
    total_s = len(audio) / SR
    pending: list[tuple[np.ndarray, Segment]] = []

    def slice_of(s: float, e: float) -> np.ndarray:
        return audio[int(s * SR) : int(e * SR)]

    if not diar.exclusive_turns:
        logger.warning("No diarization turns — falling back to coarse 30s windows")
        spans = [(s, min(total_s, s + 30.0), "SPEAKER_00") for s in _window_starts(total_s, 30.0, 30.0)]
        for s, e, speaker in _clip(spans, window):
            pending.append((slice_of(s, e), Segment(s, e, "", speaker)))
    else:
        regions: list[Region] = []
        results = []
        if tools is not None:
            regions = find_overlap_regions(diar.turns, min_seconds=settings.overlap_min_seconds)
            if window is not None:
                regions = [r for r in regions if r.start >= window[0] and r.end <= window[1]]
            results = process_regions(audio, regions, diar.turns, settings, tools)
            log_summary(results)

        for s, e, speaker in _clip(clean_spans(diar.exclusive_turns, total_s, regions), window):
            pending.append((slice_of(s, e), Segment(s, e, "", speaker)))

        for res in results:
            r = res.region
            if res.reason is None:
                for speaker, stream in res.streams.items():
                    rank = 0 if speaker == res.lead else 1
                    pending.append(
                        (stream, Segment(r.start, r.end, "", speaker, True, r.group, rank))
                    )
                continue
            for s, e, speaker in diar.exclusive_turns:
                s, e = max(s, r.start), min(e, r.end)
                if e - s >= MIN_SLICE_SECONDS:
                    rank = 0 if speaker == res.lead else 1
                    pending.append((slice_of(s, e), Segment(s, e, "", speaker, True, r.group, rank)))

    texts = asr.transcribe([chunk for chunk, _ in pending])
    segments: list[Segment] = []
    for (_, seg), text in zip(pending, texts):
        if text:
            seg.text = text
            segments.append(seg)
    segments.sort(key=lambda seg: (seg.start, seg.rank))
    logger.info("ASR produced %d segments", len(segments))
    return segments


def merge_segments(segments: list[Segment], max_gap_seconds: float = 2.0) -> list[Segment]:
    """Relabel to Speaker A.. and merge consecutive same-speaker segments."""
    label_map: dict[str, str] = {}
    merged: list[Segment] = []
    for seg in segments:
        label = _speaker_label(seg.speaker, label_map)
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.speaker == label
            and prev.crosstalk == seg.crosstalk
            and prev.region == seg.region
            and seg.start - prev.end <= max_gap_seconds
        ):
            prev.text = f"{prev.text} {seg.text}".strip()
            prev.end = max(prev.end, seg.end)
        else:
            merged.append(
                Segment(seg.start, seg.end, seg.text, label, seg.crosstalk, seg.region, seg.rank)
            )
    return merged
```

Temporarily make `run_transcription` use the new pieces (Task 7 finishes the rewiring). Replace its body from `asr_segments = transcribe(...)` through `text = format_transcript(merged)` with:

```python
    diar = Diarization(settings.diarization_model_id, diar_turns, diar_turns)
    asr = WhisperASR(settings)
    try:
        raw = build_segments(load_audio(audio_path), diar, settings, asr)
    finally:
        asr.close()
    merged = merge_segments(raw)
    text = format_transcript(merged)
```

In `config.py`, delete the `crosstalk_min_overlap` setting and its two comment lines.

- [ ] **Step 5: Confirm nothing else used the removed names**

Run: `rg -n "crosstalk_min_overlap|merge_diarization_and_asr|_assign_speaker|from src.transcription import .*transcribe\b" --glob '*.py' .`
Expected: no matches.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/transcription.py config.py tests/conftest.py tests/test_transcription.py
git commit -m "feat: transcribe speaker-tagged chunks and keep cross-talk lines separate"
```

---

### Task 7: Wire `run_transcription`

**Files:**
- Modify: `src/transcription.py` (`run_transcription`, new `load_or_diarize`)
- Test: `tests/test_transcription.py`

**Interfaces:**
- Consumes: everything from Tasks 4-6; `load_overlap_tools` from `src/overlap.py`.
- Produces: `load_or_diarize(audio_path: Path, episode_id: str, settings: Settings) -> Diarization` (reads the v2 cache when its `model` equals `settings.diarization_model_id`, else diarizes and caches non-empty results). `run_transcription` signature unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transcription.py`:

```python
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
    monkeypatch.setattr(tr, "WhisperASR", lambda settings: SimpleNamespace(transcribe=fake_asr.transcribe, close=lambda: None))
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_transcription.py -k run_transcription -v`
Expected: FAIL (`load_overlap_tools` is not an attribute of `src.transcription`, and the old cache is still read).

- [ ] **Step 3: Implement**

In `src/transcription.py`, add `load_overlap_tools` to the `src.overlap` import list. Replace `run_transcription` with:

```python
def load_or_diarize(audio_path: Path, episode_id: str, settings: Settings) -> Diarization:
    cache = diarization_cache_path(settings.transcript_dir, episode_id)
    diar = load_diarization(cache)
    if diar is not None and diar.model == settings.diarization_model_id:
        logger.info("Loaded cached diarization (%d turns) from %s", len(diar.turns), cache)
        return diar
    diar = diarize(audio_path, settings)
    if diar.turns:
        save_diarization(cache, diar)
        logger.info("Cached diarization -> %s", cache)
    return diar


def run_transcription(audio_path: Path, episode_id: str, settings: Settings | None = None) -> Path:
    """Diarize + ASR + merge; write transcript file and return its path."""
    settings = settings or get_settings()
    settings.ensure_dirs()

    diar = load_or_diarize(audio_path, episode_id, settings)
    tools = None
    if settings.overlap_separation_enabled and diar.turns:
        tools = load_overlap_tools(settings)

    audio = load_audio(audio_path)
    asr = WhisperASR(settings)
    try:
        raw = build_segments(audio, diar, settings, asr, tools=tools)
    finally:
        asr.close()
    merged = merge_segments(raw)
    text = format_transcript(merged)

    out_path = settings.transcript_dir / f"{episode_id}.txt"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote transcript (%d lines) -> %s", len(merged), out_path)
    upload_file(out_path, object_name=f"transcripts/{out_path.name}", settings=settings)
    return out_path
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/transcription.py tests/test_transcription.py
git commit -m "feat: run_transcription uses the v2 cache and optional overlap separation"
```

---

### Task 8: Gold candidates and scoring

**Files:**
- Create: `src/overlap_eval.py`
- Create: `tests/test_overlap_eval.py`

**Interfaces:**
- Consumes: `Region`, `SR`, `Turn`, `lead_speaker`, `find_overlap_regions`, `OverlapTools` (`src/overlap.py`); `Diarization`, `Segment`, `build_segments`, `ASR` (`src/transcription.py`); `parse_lines`, `strip_markers`, `Line` (`src/transcript_lines.py`).
- Produces (in `src/overlap_eval.py`):
  - `CONTEXT_SECONDS = 2.0`
  - `tokens(text: str) -> list[str]`
  - `matched(gold: list[str], hyp: list[str], min_block: int = 1) -> int`
  - `score_region(gold: dict[str, str], dominant: str, hyp: dict[str, str]) -> dict` → `{"dominant": {"gold": int, "found": int}, "interrupter": {"gold": int, "found": int}, "misattributed": int}`
  - `summarize(scores: list[dict]) -> dict` → `{"dominant_recall": float, "interrupter_recall": float, "misattributed": int, "regions": int}`
  - `gate(base: dict, new: dict) -> dict` → `{"interrupter_gain": float, "dominant_drop": float, "misattributed_delta": int, "passed": bool}`
  - `line_spans(lines: list[Line]) -> list[tuple[float, float, str, str]]` (start, end, label, text)
  - `map_labels(spans: list[tuple[float, float, str, str]], turns: list[Turn]) -> dict[str, str]`
  - `hyp_from_lines(spans, mapping: dict[str, str], start: float, end: float) -> dict[str, str]`
  - `hyp_from_segments(segments: list[Segment], start: float, end: float) -> dict[str, str]`
  - `build_candidates(episode: str, diar: Diarization, audio: np.ndarray, transcript: str, regions: list[Region], clip_dir: Path, top: int) -> list[dict]`
  - `merge_gold(existing: list[dict], new: list[dict]) -> list[dict]`

Gold entry shape (written by `build_candidates`, filled in by the user):

```json
{
  "episode": "604147",
  "start": 741.32,
  "end": 743.9,
  "speakers": ["SPEAKER_00", "SPEAKER_02"],
  "dominant": "SPEAKER_00",
  "clip": "data/eval/overlap_clips/604147/604147_741.32.wav",
  "transcript_lines": ["Speaker A [12:21]: ..."],
  "gold": {"SPEAKER_00": "", "SPEAKER_02": ""}
}
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overlap_eval.py`:

```python
"""Unit tests for the overlap gold set: candidates, scoring and the gate."""

from __future__ import annotations

import json

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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_overlap_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.overlap_eval'`.

- [ ] **Step 3: Implement**

Create `src/overlap_eval.py`:

```python
"""Overlap gold set: candidate regions, token scoring and the enable gate."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from src.overlap import SR, Region, Turn, lead_speaker
from src.transcript_lines import Line, parse_lines, strip_markers
from src.transcription import Diarization, Segment

CONTEXT_SECONDS = 2.0
LINE_TOLERANCE = 1.0
LAST_LINE_SECONDS = 30.0
_TOKEN = re.compile(r"\w+", re.UNICODE)

Span = tuple[float, float, str, str]


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(strip_markers(text))]


def matched(gold: list[str], hyp: list[str], min_block: int = 1) -> int:
    blocks = SequenceMatcher(None, gold, hyp, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= min_block)


def score_region(gold: dict[str, str], dominant: str, hyp: dict[str, str]) -> dict:
    interrupter = next(s for s in gold if s != dominant)
    out: dict = {}
    for role, speaker in (("dominant", dominant), ("interrupter", interrupter)):
        g = tokens(gold[speaker])
        out[role] = {"gold": len(g), "found": matched(g, tokens(hyp.get(speaker, "")))}
    out["misattributed"] = sum(
        matched(tokens(gold[g_spk]), tokens(text), min_block=2)
        for g_spk in gold
        for h_spk, text in hyp.items()
        if h_spk != g_spk
    )
    return out


def summarize(scores: list[dict]) -> dict:
    def recall(role: str) -> float:
        g = sum(s[role]["gold"] for s in scores)
        return round(sum(s[role]["found"] for s in scores) / g, 4) if g else 0.0

    return {
        "dominant_recall": recall("dominant"),
        "interrupter_recall": recall("interrupter"),
        "misattributed": sum(s["misattributed"] for s in scores),
        "regions": len(scores),
    }


def gate(base: dict, new: dict) -> dict:
    gain = round(new["interrupter_recall"] - base["interrupter_recall"], 4)
    drop = round(base["dominant_recall"] - new["dominant_recall"], 4)
    delta = new["misattributed"] - base["misattributed"]
    return {
        "interrupter_gain": gain,
        "dominant_drop": drop,
        "misattributed_delta": delta,
        "passed": gain >= 0.25 and drop <= 0.05 and delta <= 0,
    }


def line_spans(lines: list[Line]) -> list[Span]:
    out: list[Span] = []
    for i, ln in enumerate(lines):
        start = float(ln.seconds)
        end = float(lines[i + 1].seconds) if i + 1 < len(lines) else start + LAST_LINE_SECONDS
        out.append((start, end, ln.speaker, ln.text))
    return out


def map_labels(spans: list[Span], turns: list[Turn]) -> dict[str, str]:
    totals: dict[str, dict[str, float]] = {}
    for s, e, label, _ in spans:
        for t0, t1, raw in turns:
            ov = min(e, t1) - max(s, t0)
            if ov > 0:
                totals.setdefault(label, {}).setdefault(raw, 0.0)
                totals[label][raw] += ov
    return {label: max(by_raw, key=by_raw.get) for label, by_raw in totals.items()}


def _join(parts: dict[str, list[str]]) -> dict[str, str]:
    return {speaker: " ".join(texts) for speaker, texts in sorted(parts.items())}


def hyp_from_lines(spans: list[Span], mapping: dict[str, str], start: float, end: float) -> dict[str, str]:
    parts: dict[str, list[str]] = {}
    for s, e, label, text in spans:
        if e > start - LINE_TOLERANCE and s < end + LINE_TOLERANCE and label in mapping:
            parts.setdefault(mapping[label], []).append(text)
    return _join(parts)


def hyp_from_segments(segments: list[Segment], start: float, end: float) -> dict[str, str]:
    parts: dict[str, list[str]] = {}
    for seg in segments:
        if seg.end >= start and seg.start <= end:
            parts.setdefault(seg.speaker, []).append(seg.text)
    return _join(parts)


def build_candidates(
    episode: str,
    diar: Diarization,
    audio: np.ndarray,
    transcript: str,
    regions: list[Region],
    clip_dir: Path,
    top: int,
) -> list[dict]:
    import soundfile as sf

    spans = line_spans(parse_lines(transcript))
    pairs = [r for r in regions if len(r.speakers) == 2]
    chosen = sorted(pairs, key=lambda r: -(r.end - r.start))[:top]
    clip_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for r in sorted(chosen, key=lambda r: r.start):
        c0 = max(0, int((r.start - CONTEXT_SECONDS) * SR))
        c1 = min(len(audio), int((r.end + CONTEXT_SECONDS) * SR))
        clip = clip_dir / f"{episode}_{r.start:.2f}.wav"
        sf.write(str(clip), audio[c0:c1], SR)
        speakers = sorted(r.speakers)
        out.append(
            {
                "episode": episode,
                "start": round(r.start, 2),
                "end": round(r.end, 2),
                "speakers": speakers,
                "dominant": lead_speaker(r, diar.turns),
                "clip": str(clip),
                "transcript_lines": [
                    f"{label} [{int(s) // 60:02d}:{int(s) % 60:02d}]: {text}"
                    for s, e, label, text in spans
                    if e > r.start - LINE_TOLERANCE and s < r.end + LINE_TOLERANCE
                ],
                "gold": {s: "" for s in speakers},
            }
        )
    return out


def merge_gold(existing: list[dict], new: list[dict]) -> list[dict]:
    seen = {(e["episode"], e["start"]) for e in existing}
    return existing + [n for n in new if (n["episode"], n["start"]) not in seen]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_overlap_eval.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/overlap_eval.py tests/test_overlap_eval.py
git commit -m "feat: overlap gold candidates, token scoring and enable gate"
```

---

### Task 9: Eval orchestration and CLI commands

**Files:**
- Modify: `src/overlap_eval.py` (add `run_overlap_eval`)
- Modify: `src/eval.py` (two commands)
- Test: `tests/test_overlap_eval.py`

**Interfaces:**
- Consumes: Task 8 functions; `build_segments`, `Diarization`, `ASR` (`src/transcription.py`); `OverlapTools`.
- Produces:
  - `run_overlap_eval(gold: list[dict], *, transcripts: dict[str, str], audio: dict[str, np.ndarray], diar: dict[str, Diarization], settings: Settings, asr: ASR, tools: OverlapTools) -> dict` → `{"rows": {"existing": summary, "exclusive": summary, "separated": summary}, "gate": gate(...), "regions": [...]}`. Only entries whose `gold` texts are all non-empty are scored; each `regions` item has `episode`, `start`, `end` and the per-row `score_region` dicts.
  - CLI: `python -m src.eval overlap-candidates EPISODE [--top 10] [--gold data/eval/overlap_gold.json] [--clips data/eval/overlap_clips]`
  - CLI: `python -m src.eval overlap [--gold data/eval/overlap_gold.json] [--out data/eval/overlap_result.json]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_overlap_eval.py`:

```python
from click.testing import CliRunner

from config import Settings
from src.overlap import OverlapTools
from src.overlap_eval import run_overlap_eval


def test_run_overlap_eval_scores_three_rows(voice_audio, fake_separator, fake_embedder, fake_asr) -> None:
    diar = Diarization(
        "m",
        [(0.0, 10.0, "A"), (4.0, 6.0, "B"), (12.0, 16.0, "B")],
        [(0.0, 10.0, "A"), (12.0, 16.0, "B")],
    )
    gold = [
        {"episode": "ep", "start": 4.0, "end": 6.0, "speakers": ["A", "B"], "dominant": "A",
         "gold": {"A": "c0 c1", "B": "c2"}},
        {"episode": "ep", "start": 9.0, "end": 9.5, "speakers": ["A", "B"], "dominant": "A",
         "gold": {"A": "", "B": ""}},
    ]
    settings = Settings(_env_file=None, enroll_min_seconds=1.0)
    result = run_overlap_eval(
        gold,
        transcripts={"ep": "Speaker A [00:00]: c0 c1 c2\nSpeaker B [00:12]: nič\n"},
        audio={"ep": voice_audio},
        diar={"ep": diar},
        settings=settings,
        asr=fake_asr,
        tools=OverlapTools(separator=fake_separator, embedder=fake_embedder),
    )
    assert set(result["rows"]) == {"existing", "exclusive", "separated"}
    assert all(row["regions"] == 1 for row in result["rows"].values())
    assert result["rows"]["existing"]["interrupter_recall"] == 0.0
    assert result["rows"]["existing"]["misattributed"] == 0
    assert set(result["gate"]) == {"interrupter_gain", "dominant_drop", "misattributed_delta", "passed"}
    assert [r["start"] for r in result["regions"]] == [4.0]


def test_cli_commands_registered() -> None:
    from src.eval import cli

    runner = CliRunner()
    assert runner.invoke(cli, ["overlap-candidates", "--help"]).exit_code == 0
    assert runner.invoke(cli, ["overlap", "--help"]).exit_code == 0
```

The `existing` row scores 0 interrupter recall because the gold `B` text `c2` sits only in a line labelled `Speaker A`; `misattributed` stays 0 there because only blocks of two or more tokens count and `c2` is one token.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_overlap_eval.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_overlap_eval'`.

- [ ] **Step 3: Implement the orchestration**

Append to `src/overlap_eval.py` (extend the `src.transcription` import with `ASR, build_segments` and add `from src.overlap import OverlapTools`; add `from typing import TYPE_CHECKING` with `if TYPE_CHECKING: from config import Settings`):

```python
def run_overlap_eval(
    gold: list[dict],
    *,
    transcripts: dict[str, str],
    audio: dict[str, np.ndarray],
    diar: dict[str, Diarization],
    settings: Settings,
    asr: ASR,
    tools: OverlapTools,
) -> dict:
    filled = [g for g in gold if g["gold"] and all(t.strip() for t in g["gold"].values())]
    rows: dict[str, list[dict]] = {"existing": [], "exclusive": [], "separated": []}
    regions: list[dict] = []
    spans_by_ep = {ep: line_spans(parse_lines(text)) for ep, text in transcripts.items()}
    maps = {ep: map_labels(spans_by_ep[ep], diar[ep].exclusive_turns) for ep in transcripts}

    for g in filled:
        ep, start, end = g["episode"], g["start"], g["end"]
        window = (start - CONTEXT_SECONDS, end + CONTEXT_SECONDS)
        exclusive = build_segments(audio[ep], diar[ep], settings, asr, window=window)
        separated = build_segments(audio[ep], diar[ep], settings, asr, tools=tools, window=window)
        scores = {
            "existing": score_region(g["gold"], g["dominant"], hyp_from_lines(spans_by_ep[ep], maps[ep], start, end)),
            "exclusive": score_region(g["gold"], g["dominant"], hyp_from_segments(exclusive, start, end)),
            "separated": score_region(g["gold"], g["dominant"], hyp_from_segments(separated, start, end)),
        }
        for name, score in scores.items():
            rows[name].append(score)
        regions.append({"episode": ep, "start": start, "end": end, **scores})

    summaries = {name: summarize(scores) for name, scores in rows.items()}
    return {
        "rows": summaries,
        "gate": gate(summaries["exclusive"], summaries["separated"]),
        "regions": regions,
    }
```

- [ ] **Step 4: Add the CLI commands**

Append to `src/eval.py` (before any `if __name__ == "__main__":` block, after the last command):

```python
@cli.command("overlap-candidates")
@click.argument("episode")
@click.option("--top", default=10, type=int, help="Longest two-speaker regions to export.")
@click.option("--gold", "gold_path", type=click.Path(path_type=Path), default=Path("data/eval/overlap_gold.json"))
@click.option("--clips", "clips_dir", type=click.Path(path_type=Path), default=Path("data/eval/overlap_clips"))
def overlap_candidates(episode: str, top: int, gold_path: Path, clips_dir: Path) -> None:
    """Export overlap regions of an episode as clips plus gold entries to fill in."""
    from config import get_settings
    from src.overlap import find_overlap_regions
    from src.overlap_eval import build_candidates, merge_gold
    from src.transcription import load_audio, load_or_diarize

    settings = get_settings()
    audio_path = settings.audio_dir / f"{episode}.wav"
    diar = load_or_diarize(audio_path, episode, settings)
    regions = find_overlap_regions(diar.turns, min_seconds=settings.overlap_min_seconds)
    transcript = (settings.transcript_dir / f"{episode}.txt").read_text(encoding="utf-8")
    new = build_candidates(episode, diar, load_audio(audio_path), transcript, regions, clips_dir / episode, top)

    existing = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.exists() else []
    merged = merge_gold(existing, new)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(f"{len(regions)} overlap regions, exported {len(new)}; gold now has {len(merged)} entries")
    click.echo(f"Fill in 'gold' per speaker in {gold_path}, listening to each 'clip'.")


@cli.command()
@click.option("--gold", "gold_path", type=click.Path(path_type=Path, exists=True), default=Path("data/eval/overlap_gold.json"))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path("data/eval/overlap_result.json"))
def overlap(gold_path: Path, out_path: Path) -> None:
    """Score existing, exclusive and separated transcripts on the overlap gold set."""
    from config import get_settings
    from src.overlap import load_overlap_tools
    from src.overlap_eval import run_overlap_eval
    from src.transcription import WhisperASR, load_audio, load_or_diarize

    settings = get_settings()
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    episodes = sorted({g["episode"] for g in gold})
    audio, diar, transcripts = {}, {}, {}
    for ep in episodes:
        audio_path = settings.audio_dir / f"{ep}.wav"
        audio[ep] = load_audio(audio_path)
        diar[ep] = load_or_diarize(audio_path, ep, settings)
        transcripts[ep] = (settings.transcript_dir / f"{ep}.txt").read_text(encoding="utf-8")

    tools = load_overlap_tools(settings)
    asr = WhisperASR(settings)
    try:
        result = run_overlap_eval(
            gold, transcripts=transcripts, audio=audio, diar=diar, settings=settings, asr=asr, tools=tools
        )
    finally:
        asr.close()
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    click.echo(f"{'row':<11}{'regions':>8}{'dominant':>10}{'interrupter':>13}{'misattr':>9}")
    for name, row in result["rows"].items():
        click.echo(
            f"{name:<11}{row['regions']:>8}{row['dominant_recall']:>10.2f}"
            f"{row['interrupter_recall']:>13.2f}{row['misattributed']:>9}"
        )
    g = result["gate"]
    verdict = "PASS" if g["passed"] else "FAIL"
    click.echo(
        f"GATE {verdict}: interrupter +{g['interrupter_gain']:.2f} (need >= 0.25), "
        f"dominant -{g['dominant_drop']:.2f} (max 0.05), misattributed {g['misattributed_delta']:+d} (max 0)"
    )
    click.echo(f"Written: {out_path}")
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/overlap_eval.py src/eval.py tests/test_overlap_eval.py
git commit -m "feat: overlap-candidates and overlap eval commands"
```

---

### Task 10: Record planning decisions and produce the first candidates

**Files:**
- Modify: `docs/superpowers/specs/2026-11-05-overlap-separation-design.md`

- [ ] **Step 1: Add a "Decisions made while planning" section to the spec**

Append to the spec:

```markdown
## Decisions made while planning

- ClearerVoice is installed with `--no-deps` (`clearvoice==0.1.2`,
  `yamlargparse==1.31.1`, `rotary-embedding-torch==0.8.3`,
  `torchinfo==1.8.0`). A normal install would upgrade torch 2.13 to 2.15 and
  pin numpy 1.26.4 and soundfile 0.12.1. Only the MossFormer2 model class is
  imported; the `ClearVoice` wrapper (and its OpenCV, librosa and video
  dependencies) is not used. Weights come from
  `alibabasglab/MossFormer2_SS_16K/last_best_checkpoint.pt`, loaded with
  `weights_only=True`.
- Regions longer than 8 s are split into equal chunks sharing one group,
  because the model handles about 10 s per pass. Each chunk is assigned on its
  own, which keeps stream order from swapping between chunks.
- Overlap spans merge only when the same speakers are active, so a short
  three-speaker moment inside a two-speaker crossing falls back alone.
- Enrollment uses clean pieces of turns (turn time minus every overlap, even
  overlaps shorter than `overlap_min_seconds`) rather than only whole turns
  without overlap.
- Separation and embedding run on CPU; Whisper keeps `whisper_device`.
- The v2 cache is ignored when its `model` differs from
  `diarization_model_id`, and an empty diarization is not cached.
- `crosstalk_min_overlap` is removed: speakers now come from the slice or
  stream that produced the text, so no overlap fraction is computed.
```

- [ ] **Step 2: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, no warnings about missing modules.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-11-05-overlap-separation-design.md
git commit -m "docs: record overlap separation planning decisions"
```

- [ ] **Step 4: Export the first gold candidates (manual, downloads models)**

Pick two recent episodes that have both `data/audio/<id>.wav` and `data/transcripts/<id>.txt` (for example `604147` and `625729`). Re-diarizing an hour of audio with community-1 takes several minutes per episode.

Run: `.venv/bin/python -m src.eval overlap-candidates 604147 && .venv/bin/python -m src.eval overlap-candidates 625729`
Expected: two lines per episode like `N overlap regions, exported 10; gold now has M entries`, and WAV clips under `data/eval/overlap_clips/<id>/`.

Then stop and hand over to the user: they listen to each clip and fill in `gold` per speaker in `data/eval/overlap_gold.json`. Scoring (`.venv/bin/python -m src.eval overlap`) and enabling `OVERLAP_SEPARATION_ENABLED=true` in `.env` happen only after that, and only if the printed gate says `PASS`.

## Spec coverage

- Diarization switch and v2 cache: Tasks 1, 5, 7.
- Overlap regions, enrollment, separation, assignment, fallback reasons, summary log: Tasks 2-4.
- Transcription inside/outside regions, trimming, fallback slices, no-merge rule, same-start ordering: Task 6.
- Transcript shape unchanged: Task 6 tests (`format_transcript` output).
- Configuration table: Task 1 (and removal of `crosstalk_min_overlap` in Task 6).
- Optional dependency without version changes: Task 1 Steps 1-4.
- Evaluation commands, three rows, recall, misattribution, gate: Tasks 8-9.
- Testing list: Tasks 2-9; "flag off reproduces today's output" is `test_flag_off_matches_exclusive_turn_slicing` plus `test_no_diarization_falls_back_to_30s_windows`.
- Rollout (flag off, new episodes only, gold first): Global Constraints and Task 10 Step 4.
