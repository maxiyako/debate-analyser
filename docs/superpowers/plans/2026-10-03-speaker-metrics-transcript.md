# Speaker Mapping, Raw Metrics, Dynamic Scoring, Safe Transcript Correction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `docs/superpowers/specs/2026-10-03-speaker-metrics-transcript-design.md`: map diarization labels to real people in code, report exact per-speaker metrics and a claim funnel, normalize fouls by real word counts and dodges by challenging questions, and replace the one-shot LLM transcript rewrite with audited, guarded edits that can never be the basis of an accusation.

**Architecture:** Two leaf modules (`src/transcript_lines.py` for line parsing, `src/report_models.py` for new report sections) feed four new focused modules: `src/speakers.py` (label → person), `src/correction.py` (LLM proposes edits, code applies them), `src/questions.py` (question audit), plus a funnel builder in `src/selection.py`. `src/scoring.py` consumes the new report sections; `src/validation.py` gains an accusation guard; `run_analysis` wires it all and `correct_transcript` is deleted. Every new LLM call is a `Callable[[str], Model]` injected for offline tests, with a `default_*_llm(settings)` factory built on `src.llm.generate_json`.

**Tech Stack:** Python 3.11, pydantic v2, `google-genai` via `src.llm.generate_json`, CrewAI (unchanged crews), `click`, `pytest`.

## Global Constraints

- Tests run offline with injected fakes: `python3 -m pytest -q` from the repo root. Baseline before Task 1: `110 passed`.
- **Import-cycle rule:** `src/transcript_lines.py` imports nothing from `src`. `src/report_models.py` imports nothing from `src`. `src/speakers.py`, `src/correction.py`, `src/questions.py` import at module level only from `src.report_models` and `src.transcript_lines`; `src.validation` and `src.llm` are imported inside functions. `src.validation` may import `src.correction` and `src.transcript_lines` at module level.
- Thresholds (verbatim from spec): clip label < 1.5 % words; main label ≥ 10 % words (or top `len(roster) + 1` when a roster is known); debate start = first window of 10 lines of main labels only; speaker-map confidence floor 0.7; correction chunks of 40 lines with 5 context lines each side; spelling/word-boundary similarity ≥ 0.6; raw-quote similarity for accusations ≥ 0.85; word floor 500; substantive turn ≥ 15 words; interjection < 5 words; question ≥ 4 words; answer < 15 words = interrupted; dodge prior 2 questions / 0.3 mass; partial answer weight 0.5.
- Names: clip and unknown voices are named `Záznam`; default moderator name `Moderátor`.
- `Unverified`/`Contested` never penalize. Any doubt in the question audit or accusation guard resolves in the speaker's favour (`answered`, `Unverified`).
- Code comments, docstrings, commit messages in English. LLM prompts: English instructions; text that reaches the report (`reason`) in Slovak.

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `src/transcript_lines.py` | create | `Line`, `parse_lines`, `format_lines`, `strip_markers`, `ts_seconds` |
| `src/report_models.py` | create | `SpeakerMap*`, `EditType`, `TranscriptEdit`, `EditResult`, `TranscriptQuality`, `QuestionKind`, `QuestionOutcome`, `QuestionItem`, `ClaimFunnel` |
| `src/speakers.py` | create | heuristics, LLM assignment + validation, `map_speakers`, `apply_speaker_map`, `canonical_speaker` |
| `src/correction.py` | create | edit guards, `apply_corrections`, chunked `propose_corrections`, `correct_transcript_edits` |
| `src/questions.py` | create | candidates, classification + validation, counts, rendering, balance finding |
| `src/selection.py` | modify | `build_claim_funnel`, `refresh_funnel_verdicts` |
| `src/validation.py` | modify | `best_window`, `raw_quote_support`, `enforce_accusation_support` |
| `src/scoring.py` | modify | stats on parsed lines, new score fields, no 1000-word default, status gate, structured disciplines, question-based responsiveness |
| `src/agents.py` | modify | new report fields, new `run_analysis` flow, delete `correct_transcript`, FB prompt rules |
| `main.py` | modify | `--guests`, `--moderator`, `corrections.json`, richer scoreboard print |
| `tests/test_transcript_lines.py`, `tests/test_report_models.py`, `tests/test_speakers.py`, `tests/test_correction.py`, `tests/test_questions.py`, `tests/test_funnel.py`, `tests/test_accusation_guard.py`, `tests/test_wiring_helpers.py` | create | offline unit tests |
| `tests/test_scoring.py` | modify | new behaviour, one rewritten test |

---

### Task 1: Shared transcript line parsing and richer speaking stats

**Files:**
- Create: `src/transcript_lines.py`
- Modify: `src/scoring.py` (imports, `_LINE_RE` removal, `SpeakerStats`, `transcript_speaker_stats`)
- Test: `tests/test_transcript_lines.py`, `tests/test_scoring.py`

**Interfaces:**
- Produces: `Line(no: int, speaker: str, ts: str, text: str, raw_no: int | None, edit_ids: list[int])` with properties `crosstalk: bool`, `seconds: int`, `words: int`; `parse_lines(transcript: str) -> list[Line]`; `format_lines(lines: list[Line]) -> str`; `strip_markers(text: str) -> str`; `ts_seconds(ts: str) -> int`; `LINE_RE`.
- Produces: `transcript_speaker_stats(transcript: str, debate_start: str | None = None) -> dict[str, SpeakerStats]`, `SpeakerStats` gains `substantive_turns: int`, `interjections: int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_transcript_lines.py`:

```python
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
```

Append to `tests/test_scoring.py`:

```python
def test_stats_ignore_markers_and_count_turn_types() -> None:
    transcript = (
        "M [00:00]: Otázka?\n"
        "A [00:05]: (cez seba) " + "slovo " * 20 + "\n"
        "M [00:10]: Áno.\n"
        "A [00:12]: Nie nie nie.\n"
    )
    stats = transcript_speaker_stats(transcript)
    assert stats["a"].words == 23
    assert stats["a"].turns == 2
    assert stats["a"].substantive_turns == 1
    assert stats["a"].interjections == 1
    assert stats["m"].interjections == 2


def test_stats_skip_lines_before_debate_start() -> None:
    transcript = "A [00:10]: zostrih slová\nA [02:00]: jedna dva tri\n"
    assert transcript_speaker_stats(transcript, debate_start="01:58")["a"].words == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_transcript_lines.py tests/test_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.transcript_lines'`.

- [ ] **Step 3: Create `src/transcript_lines.py`**

```python
"""Parse 'Label [MM:SS]: text' transcript lines.

Leaf module: imported by scoring, speakers, correction, questions, and
validation, so it must not import anything from `src`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LINE_RE = re.compile(
    r"^(?P<speaker>[^\[\]]{1,60}?)\s*\[(?P<ts>(?:\d{1,3}:)?\d{1,2}:\d{2})\]\s*:\s*(?P<text>.*)$"
)
_MARKER_RE = re.compile(r"\((?:hovorenie\s+)?cez\s+seba[^)]*\)", re.IGNORECASE)


def ts_seconds(ts: str) -> int:
    total = 0
    for part in ts.split(":"):
        total = total * 60 + int(part)
    return total


def strip_markers(text: str) -> str:
    """Drop cross-talk annotations so they are not counted as spoken words."""
    return " ".join(_MARKER_RE.sub(" ", text or "").split())


@dataclass
class Line:
    no: int
    speaker: str
    ts: str
    text: str
    raw_no: int | None = None
    edit_ids: list[int] = field(default_factory=list)

    @property
    def crosstalk(self) -> bool:
        return bool(_MARKER_RE.search(self.text))

    @property
    def seconds(self) -> int:
        return ts_seconds(self.ts)

    @property
    def words(self) -> int:
        return len(strip_markers(self.text).split())


def parse_lines(transcript: str) -> list[Line]:
    out: list[Line] = []
    for raw in (transcript or "").splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        n = len(out)
        out.append(
            Line(
                no=n,
                speaker=m.group("speaker").strip(),
                ts=m.group("ts"),
                text=m.group("text").strip(),
                raw_no=n,
            )
        )
    return out


def format_lines(lines: list[Line]) -> str:
    return "".join(f"{ln.speaker} [{ln.ts}]: {ln.text}\n" for ln in lines)
```

- [ ] **Step 4: Rewrite stats in `src/scoring.py`**

Delete `_LINE_RE` (lines 23-26). Add the import below the existing `from src.agents import ...` line:

```python
from src.transcript_lines import parse_lines, strip_markers, ts_seconds
```

Add constants next to `_MIN_WORDS_FLOOR`:

```python
_SUBSTANTIVE_TURN_WORDS = 15
_INTERJECTION_WORDS = 5
```

Replace `SpeakerStats` and `transcript_speaker_stats` with:

```python
class SpeakerStats(BaseModel):
    speaker: str
    words: int = 0
    turns: int = 0
    substantive_turns: int = 0
    interjections: int = 0
    interruptions_caused: int = 0


def _close_turn(row: SpeakerStats | None, words: int) -> None:
    if row is None:
        return
    if words >= _SUBSTANTIVE_TURN_WORDS:
        row.substantive_turns += 1
    elif words < _INTERJECTION_WORDS:
        row.interjections += 1


def transcript_speaker_stats(
    transcript: str, debate_start: str | None = None
) -> dict[str, SpeakerStats]:
    """Word/turn counts + interruption proxy per speaker key, from the transcript.

    Lines before `debate_start` (the opening recap) are ignored. Cross-talk
    markers are not counted as words.

    Interruption proxy: speaker B starts a turn while A's previous line did not
    end with terminal punctuation (cut-off mid-sentence). Only counted when the
    transcript is reliably punctuated (most lines end with punctuation),
    otherwise ASR noise would inflate it.
    """
    lines = parse_lines(transcript)
    if debate_start:
        start = ts_seconds(debate_start)
        lines = [ln for ln in lines if ln.seconds >= start]
    if not lines:
        return {}
    texts = [strip_markers(ln.text) for ln in lines]
    punctuated = sum(1 for t in texts if t.endswith(_TERMINAL_PUNCT))
    punct_reliable = punctuated / len(lines) >= 0.5

    stats: dict[str, SpeakerStats] = {}
    prev_key: str | None = None
    prev_text = ""
    turn_row: SpeakerStats | None = None
    turn_words = 0
    for ln, text in zip(lines, texts):
        key = _speaker_key(ln.speaker)
        row = stats.setdefault(key, SpeakerStats(speaker=ln.speaker))
        if len(ln.speaker) > len(row.speaker):
            row.speaker = ln.speaker
        words = len(text.split())
        row.words += words
        if key != prev_key:
            _close_turn(turn_row, turn_words)
            turn_row, turn_words = row, 0
            row.turns += 1
            if (
                punct_reliable
                and prev_key is not None
                and prev_text
                and not prev_text.endswith(_TERMINAL_PUNCT)
            ):
                row.interruptions_caused += 1
        turn_words += words
        prev_key, prev_text = key, text
    _close_turn(turn_row, turn_words)
    return stats
```

- [ ] **Step 5: Run the suite**

Run: `python3 -m pytest -q`
Expected: `115 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/transcript_lines.py src/scoring.py tests/test_transcript_lines.py tests/test_scoring.py
git commit -m "feat: shared transcript line parser; stats skip recap, ignore cross-talk markers, count turn types"
```

---

### Task 2: Report section models

**Files:**
- Create: `src/report_models.py`
- Modify: `src/agents.py` (imports; `TimeShare`, `VerifiedFact`, `AnalysisReport`, `CorrectedTranscript`)
- Test: `tests/test_report_models.py`

**Interfaces:**
- Produces (all in `src.report_models`):
  - `SpeakerRole` (`MODERATOR`, `GUEST`, `CLIP`, `UNKNOWN`), `SpeakerMapEntry(label, name, role, confidence, evidence)`, `SpeakerMap(status, source, debate_start, entries, notes)` with `name_for(label) -> str | None`, `guests() -> list[str]`, `moderator() -> str | None`.
  - `EditType` (`SPELLING`, `WORD_BOUNDARY`, `PROPER_NOUN`, `PUNCTUATION`, `SPLIT_TURN`), `TranscriptEdit(line_no, type, before, after, new_speaker, reason)`, `EditResult(id, edit, applied, rule)`, `TranscriptQuality(applied, rejected_by_rule, log_path)`.
  - `QuestionKind` (`CHALLENGING`, `OPEN`, `PROCEDURAL`, `RHETORICAL`), `QuestionOutcome` (`ANSWERED`, `PARTIAL`, `DODGED`, `INTERRUPTED`), `QuestionItem(id, timestamp, addressee, question, answer_words, answer_text [excluded from dumps], kind, outcome, evidence, reason)`.
  - `ClaimFunnel(speaker, extracted, non_empirical, dropped_by_selection, selected_for_check, removed_ungrounded, final_facts, checked, unverified, contested)`.
- Produces (in `src.agents`): `TimeShare.turns: int`, `VerifiedFact.quote_raw: str`, `VerifiedFact.timestamp: str`, `VerifiedFact.transcript_edits: list[int]`, `AnalysisReport.speaker_map: SpeakerMap | None`, `AnalysisReport.question_audit: list[QuestionItem]`, `AnalysisReport.claim_funnel: list[ClaimFunnel]`, `AnalysisReport.transcript_quality: TranscriptQuality | None`, `CorrectedTranscript.log: list[EditResult]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_report_models.py`:

```python
"""Offline tests for report section models."""

from __future__ import annotations

from src.agents import AnalysisReport, CorrectedTranscript, TimeShare, Verdict, VerifiedFact
from src.report_models import QuestionItem, SpeakerMap, SpeakerMapEntry, SpeakerRole


def _map() -> SpeakerMap:
    return SpeakerMap(
        status="ok",
        source="cli",
        entries=[
            SpeakerMapEntry(label="Speaker A", name="Moderátor", role=SpeakerRole.MODERATOR, confidence=1),
            SpeakerMapEntry(label="Speaker H", name="Erik Tomáš", role=SpeakerRole.GUEST, confidence=0.9),
            SpeakerMapEntry(label="Speaker D", name="Marián Viskupič", role=SpeakerRole.GUEST, confidence=0.9),
            SpeakerMapEntry(label="Speaker B", name="Záznam", role=SpeakerRole.CLIP, confidence=1),
        ],
    )


def test_speaker_map_helpers() -> None:
    smap = _map()
    assert smap.name_for("Speaker H") == "Erik Tomáš"
    assert smap.name_for("Speaker Z") is None
    assert smap.guests() == ["Erik Tomáš", "Marián Viskupič"]
    assert smap.moderator() == "Moderátor"


def test_analysis_report_carries_new_sections() -> None:
    dumped = AnalysisReport().model_dump(mode="json")
    assert dumped["speaker_map"] is None
    assert dumped["question_audit"] == []
    assert dumped["claim_funnel"] == []
    assert dumped["transcript_quality"] is None


def test_question_answer_text_not_serialized() -> None:
    q = QuestionItem(id=1, timestamp="01:00", addressee="A", question="Q?", answer_text="long")
    assert "answer_text" not in q.model_dump()


def test_new_fact_and_timeshare_fields_default() -> None:
    fact = VerifiedFact(claim="c", speaker="s", verdict=Verdict.TRUE)
    assert fact.quote_raw == "" and fact.timestamp == "" and fact.transcript_edits == []
    assert TimeShare(speaker="A").turns == 0
    assert CorrectedTranscript(text="x").log == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_report_models.py -q`
Expected: FAIL — `No module named 'src.report_models'`.

- [ ] **Step 3: Create `src/report_models.py`**

```python
"""Report sections produced in code: speaker map, transcript edits, question
audit, claim funnel.

Imported by `src.agents` at module level, so this module must not import
`src.agents`, `src.validation`, or anything that does.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SpeakerRole(str, Enum):
    MODERATOR = "moderator"
    GUEST = "guest"
    CLIP = "clip"
    UNKNOWN = "unknown"


class SpeakerMapEntry(BaseModel):
    label: str
    name: str
    role: SpeakerRole
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)


class SpeakerMap(BaseModel):
    status: Literal["ok", "partial", "failed"] = "failed"
    source: Literal["cli", "llm", "heuristic"] = "heuristic"
    debate_start: str | None = None
    entries: list[SpeakerMapEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def name_for(self, label: str) -> str | None:
        for e in self.entries:
            if e.label == label:
                return e.name
        return None

    def guests(self) -> list[str]:
        out: list[str] = []
        for e in self.entries:
            if e.role == SpeakerRole.GUEST and e.name not in out:
                out.append(e.name)
        return out

    def moderator(self) -> str | None:
        return next((e.name for e in self.entries if e.role == SpeakerRole.MODERATOR), None)


class EditType(str, Enum):
    SPELLING = "spelling"
    WORD_BOUNDARY = "word_boundary"
    PROPER_NOUN = "proper_noun"
    PUNCTUATION = "punctuation"
    SPLIT_TURN = "split_turn"


class TranscriptEdit(BaseModel):
    line_no: int
    type: EditType
    before: str = Field(
        default="",
        description=(
            "Exact text copied from the line. For split_turn: the first words "
            "spoken by the second speaker (the cut anchor)."
        ),
    )
    after: str = ""
    new_speaker: str | None = None
    reason: str = ""


class EditResult(BaseModel):
    id: int
    edit: TranscriptEdit
    applied: bool
    rule: str = Field(default="", description="Guard that rejected the edit; empty when applied.")


class TranscriptQuality(BaseModel):
    applied: int = 0
    rejected_by_rule: dict[str, int] = Field(default_factory=dict)
    log_path: str = ""


class QuestionKind(str, Enum):
    CHALLENGING = "challenging"
    OPEN = "open"
    PROCEDURAL = "procedural"
    RHETORICAL = "rhetorical"


class QuestionOutcome(str, Enum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    DODGED = "dodged"
    INTERRUPTED = "interrupted"


class QuestionItem(BaseModel):
    id: int
    timestamp: str
    addressee: str
    question: str
    answer_words: int = 0
    answer_text: str = Field(default="", exclude=True)
    kind: QuestionKind | None = None
    outcome: QuestionOutcome | None = None
    evidence: list[str] = Field(default_factory=list)
    reason: str = ""


class ClaimFunnel(BaseModel):
    speaker: str
    extracted: int = 0
    non_empirical: int = 0
    dropped_by_selection: int = 0
    selected_for_check: int = 0
    removed_ungrounded: int = 0
    final_facts: int = 0
    checked: int = 0
    unverified: int = 0
    contested: int = 0
```

- [ ] **Step 4: Extend models in `src/agents.py`**

Add after the existing `from src.briefing import ...`/settings imports at the top of the file:

```python
from src.report_models import (
    ClaimFunnel,
    EditResult,
    QuestionItem,
    SpeakerMap,
    TranscriptQuality,
)
```

`TimeShare` becomes:

```python
class TimeShare(BaseModel):
    speaker: str
    approximate_share_percent: float = 0.0
    turns: int = 0
```

In `VerifiedFact`, after `rationale: str = ""` add:

```python
    quote_raw: str = Field(
        default="",
        description="The matching window of the original ASR transcript (before correction).",
    )
    timestamp: str = ""
    transcript_edits: list[int] = Field(
        default_factory=list,
        description="Ids of applied transcript edits on the quoted line(s).",
    )
```

In `AnalysisReport`, after `briefing` add:

```python
    speaker_map: SpeakerMap | None = None
    question_audit: list[QuestionItem] = Field(default_factory=list)
    claim_funnel: list[ClaimFunnel] = Field(default_factory=list)
    transcript_quality: TranscriptQuality | None = None
```

In `CorrectedTranscript`, after `notes` add:

```python
    log: list[EditResult] = Field(
        default_factory=list,
        description="Every proposed transcript edit with its guard decision.",
    )
```

- [ ] **Step 5: Run the suite**

Run: `python3 -m pytest -q`
Expected: `119 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/report_models.py src/agents.py tests/test_report_models.py
git commit -m "feat: report models for speaker map, transcript edits, question audit, claim funnel"
```

---

### Task 3: Deterministic speaker heuristics, relabeling, canonical names

**Files:**
- Create: `src/speakers.py`
- Test: `tests/test_speakers.py`

**Interfaces:**
- Consumes: `parse_lines`, `Line`, `LINE_RE` (Task 1); `SpeakerMap`, `SpeakerMapEntry`, `SpeakerRole` (Task 2).
- Produces: `LabelProfile`, `label_profiles(lines) -> dict[str, LabelProfile]`, `heuristic_roles(lines, n_main: int | None = None) -> dict[str, SpeakerRole]`, `detect_debate_start(lines, roles) -> str | None`, `apply_speaker_map(transcript: str, smap: SpeakerMap) -> str`, `canonical_speaker(name: str, names: list[str]) -> str | None`, constants `CLIP_NAME = "Záznam"`, `DEFAULT_MODERATOR = "Moderátor"`.

- [ ] **Step 1: Write the failing tests** — `tests/test_speakers.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_speakers.py -q`
Expected: FAIL — `No module named 'src.speakers'`.

- [ ] **Step 3: Create `src/speakers.py` (deterministic part)**

```python
"""Map diarization labels (Speaker A..) to real people.

The mapping is decided once (heuristics + one small LLM call, validated in
code) and applied to the transcript in code, so names never depend on an LLM
rewriting the transcript.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.report_models import SpeakerMap, SpeakerMapEntry, SpeakerRole
from src.transcript_lines import LINE_RE, Line

logger = logging.getLogger(__name__)

CLIP_NAME = "Záznam"
DEFAULT_MODERATOR = "Moderátor"
_CLIP_SHARE = 0.015
_MAIN_SHARE = 0.10
_START_WINDOW = 10
_SPEAKER_ID = re.compile(r"\bspeaker\s+[a-z0-9]+\b", re.IGNORECASE)
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class LabelProfile:
    label: str
    words: int = 0
    turns: int = 0
    first_no: int = 0
    share: float = 0.0

    @property
    def words_per_turn(self) -> float:
        return self.words / self.turns if self.turns else 0.0


def label_profiles(lines: list[Line]) -> dict[str, LabelProfile]:
    profiles: dict[str, LabelProfile] = {}
    prev: str | None = None
    for ln in lines:
        p = profiles.get(ln.speaker)
        if p is None:
            p = profiles[ln.speaker] = LabelProfile(label=ln.speaker, first_no=ln.no)
        p.words += ln.words
        if ln.speaker != prev:
            p.turns += 1
        prev = ln.speaker
    total = sum(p.words for p in profiles.values()) or 1
    for p in profiles.values():
        p.share = p.words / total
    return profiles


def heuristic_roles(lines: list[Line], n_main: int | None = None) -> dict[str, SpeakerRole]:
    """Moderator = main label with the fewest words per turn; other main labels are guests.

    Share of questions does not work: guests ask more questions than the
    moderator in some debates (617000, 605345).
    """
    profiles = label_profiles(lines)
    ranked = sorted(profiles.values(), key=lambda p: -p.words)
    if n_main:
        main = [p for p in ranked[:n_main] if p.share >= _CLIP_SHARE]
    else:
        main = [p for p in ranked if p.share >= _MAIN_SHARE]
    roles = {
        p.label: SpeakerRole.CLIP if p.share < _CLIP_SHARE else SpeakerRole.UNKNOWN
        for p in profiles.values()
    }
    if main:
        moderator = min(main, key=lambda p: (p.words_per_turn, p.first_no))
        for p in main:
            roles[p.label] = SpeakerRole.MODERATOR if p is moderator else SpeakerRole.GUEST
    return roles


def detect_debate_start(lines: list[Line], roles: dict[str, SpeakerRole]) -> str | None:
    """First line after which the opening recap (many short clip voices) is over."""
    main = {
        label
        for label, role in roles.items()
        if role in (SpeakerRole.MODERATOR, SpeakerRole.GUEST)
    }
    for i in range(len(lines) - _START_WINDOW + 1):
        if all(ln.speaker in main for ln in lines[i : i + _START_WINDOW]):
            return lines[i].ts
    return None


def apply_speaker_map(transcript: str, smap: SpeakerMap) -> str:
    """Replace line labels with mapped names; the spoken text is untouched."""
    out: list[str] = []
    for raw in (transcript or "").splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            out.append(raw)
            continue
        label = m.group("speaker").strip()
        name = smap.name_for(label) or label
        out.append(f"{name} [{m.group('ts')}]: {m.group('text').strip()}")
    return "\n".join(out) + ("\n" if out else "")


def _tokens(name: str) -> list[str]:
    return [t.casefold() for t in _TOKEN.findall(name or "")]


def canonical_speaker(name: str, names: list[str]) -> str | None:
    """Resolve an agent-written speaker name to exactly one roster name.

    Exact (case-insensitive) match, then a unique surname match, then a unique
    5-letter surname stem (Slovak inflection: 'Tomáša' -> 'Tomáš'). 'Speaker X'
    labels only ever match exactly.
    """
    if not name:
        return None
    folded = name.casefold().strip()
    for n in names:
        if n.casefold() == folded:
            return n
    if _SPEAKER_ID.search(name):
        return None
    toks = _tokens(name)
    if not toks:
        return None
    surname = toks[-1]
    hits = [n for n in names if surname in _tokens(n)]
    if len(hits) == 1:
        return hits[0]
    stem = surname[:5]
    if len(stem) < 4:
        return None
    hits = [n for n in names if any(t.startswith(stem) for t in _tokens(n))]
    return hits[0] if len(hits) == 1 else None
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_speakers.py -q`
Expected: all pass (the 620752 test is skipped if the local file is missing).

- [ ] **Step 5: Commit**

```bash
git add src/speakers.py tests/test_speakers.py
git commit -m "feat: speaker heuristics, debate start detection, label relabeling, canonical names"
```

---

### Task 4: LLM speaker assignment with validation and fallback

**Files:**
- Modify: `src/speakers.py`
- Test: `tests/test_speakers.py`

**Interfaces:**
- Consumes: Task 3 functions; `src.validation.normalize`, `src.validation.quote_grounded` (function-level import); `src.llm.generate_json`.
- Produces: `LabelAssignment(label, name, confidence, evidence)`, `SpeakerAssignment(guests_from_intro, assignments)`, `label_context(lines, label, moderator_label) -> list[Line]`, `build_speaker_prompt(intro, body, roles, roster, moderator_name, briefing_text="") -> str`, `validate_assignment(assign, body, roles, roster, moderator_name) -> tuple[list[SpeakerMapEntry], list[str], bool]`, `map_speakers(transcript, *, guests=None, moderator=None, llm=None, briefing_text="") -> SpeakerMap`, `default_speaker_llm(settings) -> Callable[[str], SpeakerAssignment]`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_speakers.py`:

```python
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


def test_llm_failure_falls_back_to_heuristic() -> None:
    def boom(prompt: str) -> SpeakerAssignment:
        raise RuntimeError("vertex down")

    smap = map_speakers(synthetic(), guests=GUESTS, llm=boom)
    assert smap.source == "heuristic"
    assert smap.status == "partial"
    assert smap.name_for("G1") == "Erik Tomáš"
    assert smap.name_for("G2") == "Marián Viskupič"
    assert smap.name_for("M") == "Moderátor"


def test_roster_from_intro_when_no_cli_names() -> None:
    def fake(prompt: str) -> SpeakerAssignment:
        a = _good(prompt)
        a.guests_from_intro = ["Erik Tomáš", "Marián Viskupič", "Robert Fico"]
        return a

    smap = map_speakers(synthetic(), llm=fake)
    assert smap.status == "ok"
    assert smap.source == "llm"
    assert smap.guests() == ["Erik Tomáš", "Marián Viskupič"]


def test_prompt_lists_only_main_labels() -> None:
    lines = parse_lines(synthetic())
    roles = heuristic_roles(lines)
    prompt = build_speaker_prompt(lines, lines, roles, GUESTS, "Moderátor")
    assert "LABEL G1" in prompt and "LABEL M" in prompt
    assert "LABEL X" not in prompt
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_speakers.py -q`
Expected: FAIL — `ImportError: cannot import name 'LabelAssignment'`.

- [ ] **Step 3: Add the LLM part to `src/speakers.py`**

Extend imports at the top:

```python
from typing import Callable

from pydantic import BaseModel, Field

from src.transcript_lines import LINE_RE, Line, parse_lines, ts_seconds
```

Add constants next to the others:

```python
_MIN_CONFIDENCE = 0.7
_MIN_EVIDENCE_CHARS = 15
_SELF_ID = re.compile(
    r"\b(ja ako|my v|za stranu|naša strana|moja strana|ako minister|ako poslan)",
    re.IGNORECASE,
)
_TS_PREFIX = re.compile(r"^\s*\[[\d:]+\]\s*")
```

Append:

```python
class LabelAssignment(BaseModel):
    label: str
    name: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class SpeakerAssignment(BaseModel):
    guests_from_intro: list[str] = Field(default_factory=list)
    assignments: list[LabelAssignment] = Field(default_factory=list)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "…"


def label_context(lines: list[Line], label: str, moderator_label: str | None) -> list[Line]:
    """The label's lines plus the moderator line right before each of its turns."""
    ctx: list[Line] = []
    seen: set[int] = set()
    for i, ln in enumerate(lines):
        if ln.speaker != label:
            continue
        prev = lines[i - 1] if i else None
        if prev is not None and moderator_label and prev.speaker == moderator_label and prev.no not in seen:
            ctx.append(prev)
            seen.add(prev.no)
        if ln.no not in seen:
            ctx.append(ln)
            seen.add(ln.no)
    return ctx


def _moderator_label(roles: dict[str, SpeakerRole]) -> str | None:
    return next((lbl for lbl, r in roles.items() if r == SpeakerRole.MODERATOR), None)


def _main_labels(roles: dict[str, SpeakerRole]) -> list[str]:
    return [lbl for lbl, r in roles.items() if r in (SpeakerRole.MODERATOR, SpeakerRole.GUEST)]


def build_speaker_prompt(
    intro: list[Line],
    body: list[Line],
    roles: dict[str, SpeakerRole],
    roster: list[str],
    moderator_name: str,
    briefing_text: str = "",
) -> str:
    profiles = label_profiles(body)
    mod_label = _moderator_label(roles)
    roster_line = (
        ", ".join(roster)
        if roster
        else "(empty: first list in guests_from_intro the invited guests the "
        "moderator introduces, full names in nominative)"
    )
    parts = [
        "You map anonymous diarization labels of a Slovak TV debate to real people.",
        f"GUEST ROSTER: {roster_line}",
        f"MODERATOR NAME: {moderator_name}",
        "Return one assignment per LABEL below: a guest from the roster, or the "
        "moderator name. Use only: self-identification (party, office, 'ja ako "
        "minister'), the moderator addressing someone right before the label "
        "speaks (the addressed person usually answers next), and greetings after "
        "being introduced ('ďakujem za pozvanie').",
        "evidence: 1-3 excerpts copied verbatim from the lines shown for that "
        "label, each prefixed with its [MM:SS]. confidence: 0-1.",
    ]
    if briefing_text:
        parts.append(f"BACKGROUND (who is who; not evidence):\n{_clip(briefing_text, 2500)}")
    parts.append(
        "INTRO:\n" + "\n".join(f"{ln.speaker} [{ln.ts}]: {_clip(ln.text, 400)}" for ln in intro[:25])
    )
    for label in _main_labels(roles):
        p = profiles.get(label)
        if p is None:
            continue
        ctx = label_context(body, label, None if label == mod_label else mod_label)
        own = [ln for ln in ctx if ln.speaker == label]
        picked = own[:8] + [ln for ln in own[8:] if _SELF_ID.search(ln.text)][:5]
        before = [ln for ln in ctx if ln.speaker != label][:5]
        block = [
            f"LABEL {label} ({100 * p.share:.1f}% words, {p.turns} turns, "
            f"candidate role: {roles[label].value})"
        ]
        block += [f"  [{ln.ts}] {_clip(ln.text, 300)}" for ln in picked]
        block += [f"  moderator before it [{ln.ts}]: {_clip(ln.text, 200)}" for ln in before]
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def _roster_from_intro(names: list[str], lines: list[Line]) -> list[str]:
    """Keep only intro names whose surname stem actually occurs in the opening lines."""
    head = " ".join(ln.text for ln in lines[:40]).casefold()
    out: list[str] = []
    for n in names:
        toks = _tokens(n)
        if toks and toks[-1][:5] in head and n not in out:
            out.append(n)
    return out


def validate_assignment(
    assign: SpeakerAssignment,
    body: list[Line],
    roles: dict[str, SpeakerRole],
    roster: list[str],
    moderator_name: str,
) -> tuple[list[SpeakerMapEntry], list[str], bool]:
    from src.validation import normalize, quote_grounded

    notes: list[str] = []
    ok = True
    mod_label = _moderator_label(roles)
    main = set(_main_labels(roles))
    entries: dict[str, SpeakerMapEntry] = {}
    for a in assign.assignments:
        if a.label not in main:
            notes.append(f"Speaker map: ignored assignment for non-main label {a.label}")
            continue
        if a.name == moderator_name:
            role = SpeakerRole.MODERATOR
        elif a.name in roster:
            role = SpeakerRole.GUEST
        else:
            notes.append(f"Speaker map: {a.label} -> '{a.name}' is not in the roster")
            ok = False
            continue
        if (role == SpeakerRole.MODERATOR) != (a.label == mod_label):
            notes.append(f"Speaker map: LLM role for {a.label} contradicts the turn-taking heuristic")
            ok = False
        ctx = label_context(body, a.label, None if a.label == mod_label else mod_label)
        context = "\n".join(ln.text for ln in ctx)
        grounded = []
        for ev in a.evidence:
            text = _TS_PREFIX.sub("", ev)
            if len(normalize(text)) >= _MIN_EVIDENCE_CHARS and quote_grounded(text, context):
                grounded.append(ev)
        confidence = a.confidence if grounded else 0.0
        if not grounded:
            notes.append(f"Speaker map: no grounded evidence for {a.label} -> {a.name}")
        if confidence < _MIN_CONFIDENCE:
            ok = False
        entries[a.label] = SpeakerMapEntry(
            label=a.label, name=a.name, role=role, confidence=confidence, evidence=grounded[:3]
        )
    guest_names = [e.name for e in entries.values() if e.role == SpeakerRole.GUEST]
    dup = sorted({n for n in guest_names if guest_names.count(n) > 1})
    if dup:
        notes.append(f"Speaker map: guest assigned to several main labels: {', '.join(dup)}")
        ok = False
    missing = [n for n in roster if n not in guest_names]
    if missing:
        notes.append(f"Speaker map: roster guests without a label: {', '.join(missing)}")
        ok = False
    unassigned = sorted(main - set(entries))
    if unassigned:
        notes.append(f"Speaker map: main labels left unnamed: {', '.join(unassigned)}")
        ok = False
    return list(entries.values()), notes, ok


def _heuristic_entries(
    body: list[Line], roles: dict[str, SpeakerRole], roster: list[str], moderator_name: str
) -> list[SpeakerMapEntry]:
    """Fallback: guests named in roster order by who speaks first after the recap."""
    first: dict[str, int] = {}
    for ln in body:
        first.setdefault(ln.speaker, ln.no)
    guests = sorted(
        (lbl for lbl, r in roles.items() if r == SpeakerRole.GUEST),
        key=lambda lbl: first.get(lbl, 10**9),
    )
    entries = [
        SpeakerMapEntry(label=lbl, name=moderator_name, role=SpeakerRole.MODERATOR, confidence=0.5)
        for lbl, r in roles.items()
        if r == SpeakerRole.MODERATOR
    ]
    for lbl, name in zip(guests, roster):
        entries.append(SpeakerMapEntry(label=lbl, name=name, role=SpeakerRole.GUEST, confidence=0.3))
    for lbl in guests[len(roster):]:
        entries.append(SpeakerMapEntry(label=lbl, name=lbl, role=SpeakerRole.GUEST, confidence=0.0))
    return entries


def map_speakers(
    transcript: str,
    *,
    guests: list[str] | None = None,
    moderator: str | None = None,
    llm: Callable[[str], SpeakerAssignment] | None = None,
    briefing_text: str = "",
) -> SpeakerMap:
    lines = parse_lines(transcript)
    roster = list(guests or [])
    roles = heuristic_roles(lines, n_main=len(roster) + 1 if roster else None)
    start = detect_debate_start(lines, roles)
    start_s = ts_seconds(start) if start else 0
    body = [ln for ln in lines if ln.seconds >= start_s]
    moderator_name = moderator or DEFAULT_MODERATOR
    notes: list[str] = []
    entries: list[SpeakerMapEntry] = []
    ok = False
    source = "cli" if roster else "llm"

    if llm is not None:
        try:
            assign = llm(build_speaker_prompt(lines, body, roles, roster, moderator_name, briefing_text))
            if not roster:
                roster = _roster_from_intro(assign.guests_from_intro, lines)
            entries, v_notes, ok = validate_assignment(assign, body, roles, roster, moderator_name)
            notes.extend(v_notes)
        except Exception as exc:  # noqa: BLE001 - fall back to heuristics, never abort the run
            logger.warning("Speaker mapping LLM failed: %s", exc)
            notes.append(f"Speaker map LLM failed: {exc}"[:300])
            entries = []
    if not entries:
        entries = _heuristic_entries(body, roles, roster, moderator_name)
        source = "heuristic"
        ok = False
    if not roster:
        notes.append("Speaker map: no guest names known (no --guests, none found in the intro)")
        ok = False

    mapped = {e.label for e in entries}
    for label, role in roles.items():
        if label not in mapped and role in (SpeakerRole.CLIP, SpeakerRole.UNKNOWN):
            entries.append(
                SpeakerMapEntry(
                    label=label,
                    name=CLIP_NAME,
                    role=role,
                    confidence=1.0 if role == SpeakerRole.CLIP else 0.5,
                )
            )
    named_guest = any(e.role == SpeakerRole.GUEST and e.name in roster for e in entries)
    status = "ok" if ok else ("partial" if named_guest else "failed")
    return SpeakerMap(
        status=status, source=source, debate_start=start, entries=entries, notes=notes
    )


def default_speaker_llm(settings) -> Callable[[str], SpeakerAssignment]:
    from src.llm import generate_json

    return lambda prompt: generate_json(
        prompt, SpeakerAssignment, settings, label="speaker_map"
    )
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_speakers.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/speakers.py tests/test_speakers.py
git commit -m "feat: validated LLM speaker assignment with heuristic fallback"
```

---

### Task 5: Transcript edit guards and application

**Files:**
- Create: `src/correction.py`
- Test: `tests/test_correction.py`

**Interfaces:**
- Consumes: `Line`, `parse_lines` (Task 1); `EditType`, `TranscriptEdit`, `EditResult` (Task 2).
- Produces: `number_tokens(s: str) -> list[str]`, `negation_tokens(s: str) -> set[str]`, `check_edit(edit, line_text, allowed_names: set[str], speaker_names: set[str]) -> str` (`""` = safe, else rule name: `not_found`, `noop`, `number`, `negation`, `punctuation`, `word_count`, `similarity`, `proper_noun`, `split`), `apply_corrections(lines, edits, allowed_names, speaker_names) -> tuple[list[Line], list[EditResult]]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_correction.py`:

```python
"""Offline tests for guarded transcript correction (src/correction.py)."""

from __future__ import annotations

import pytest

from src.correction import apply_corrections, check_edit
from src.report_models import EditType, TranscriptEdit
from src.transcript_lines import parse_lines

NAMES = {"Erik Tomáš", "Marián Viskupič", "Milan Majerský", "HLAS"}
SPEAKERS = {"Moderátor", "Erik Tomáš", "Marián Viskupič"}
S, WB, PN, P = EditType.SPELLING, EditType.WORD_BOUNDARY, EditType.PROPER_NOUN, EditType.PUNCTUATION


def e(kind: EditType, before: str, after: str = "") -> TranscriptEdit:
    return TranscriptEdit(line_no=0, type=kind, before=before, after=after)


@pytest.mark.parametrize(
    "edit,line,rule",
    [
        (e(S, "ľudi", "ľudí"), "pre ľudi", ""),
        (e(WB, "kpointe", "k pointe"), "príde aj kpointe", ""),
        (e(WB, "napl nela", "naplnila"), "HLas napl nela", ""),
        (e(S, "čí", "či"), "čí áno", ""),
        (e(PN, "HLas", "HLAS"), "strana HLas", ""),
        (e(P, "áno ale", "áno, ale"), "áno ale", ""),
        (e(S, "40", "140"), "zo 40 na 135", "number"),
        (e(S, "tri", "štyri"), "tri roky", "number"),
        (e(S, "podporili", "nepodporili"), "oni podporili", "negation"),
        (e(S, "neni", "je"), "to neni pravda", "negation"),
        (e(S, "Armádsky", "Pán Majerský"), "Armádsky, povedal", "similarity"),
        (e(PN, "Armádsky", "Pán Majerský"), "Armádsky, povedal", "proper_noun"),
        (e(S, "Ako to vidíte vy,", ""), "Ako to vidíte vy, pán", "word_count"),
        (e(P, "spoločnost", "spoločnosť."), "spoločnost", "punctuation"),
        (e(S, "xyz", "xy"), "abc", "not_found"),
    ],
)
def test_check_edit(edit: TranscriptEdit, line: str, rule: str) -> None:
    assert check_edit(edit, line, NAMES, SPEAKERS) == rule


def test_apply_records_log_and_edit_ids() -> None:
    lines = parse_lines("Erik Tomáš [00:12]: Strana HLas presadila zo 40 na 135 eur pre ľudi.\n")
    edits = [
        TranscriptEdit(line_no=0, type=PN, before="HLas", after="HLAS"),
        TranscriptEdit(line_no=0, type=S, before="40", after="140"),
        TranscriptEdit(line_no=0, type=S, before="ľudi", after="ľudí"),
        TranscriptEdit(line_no=7, type=S, before="a", after="b"),
    ]
    out, log = apply_corrections(lines, edits, NAMES, SPEAKERS)
    assert out[0].text == "Strana HLAS presadila zo 40 na 135 eur pre ľudí."
    assert [(r.id, r.applied, r.rule) for r in log] == [
        (0, True, ""), (1, False, "number"), (2, True, ""), (3, False, "line"),
    ]
    assert out[0].edit_ids == [0, 2]
    assert lines[0].text.startswith("Strana HLas")  # input is not mutated


def test_split_turn_preserves_text_and_alternates_speakers() -> None:
    text = (
        "Moderátor [01:58]: Vítam Erika Tomáša. Ďakujem za pozvanie a všetkým "
        "prajem peknú nedeľu. A rovnako vítam Mariána Viskupiča.\n"
    )
    lines = parse_lines(text)
    edits = [
        TranscriptEdit(line_no=0, type=EditType.SPLIT_TURN, before="A rovnako", new_speaker="Moderátor"),
        TranscriptEdit(line_no=0, type=EditType.SPLIT_TURN, before="Ďakujem za pozvanie", new_speaker="Erik Tomáš"),
    ]
    out, log = apply_corrections(lines, edits, NAMES, SPEAKERS)
    assert [ln.speaker for ln in out] == ["Moderátor", "Erik Tomáš", "Moderátor"]
    assert " ".join(ln.text for ln in out) == lines[0].text
    assert all(ln.raw_no == 0 and ln.ts == "01:58" for ln in out)
    assert [ln.no for ln in out] == [0, 1, 2]
    assert out[1].edit_ids == [1] and out[0].edit_ids == []
    assert all(r.applied for r in log)


def test_split_rejected_for_unknown_speaker_or_line_start() -> None:
    lines = parse_lines("Moderátor [01:58]: Dobrý deň, vitajte.\n")
    edits = [
        TranscriptEdit(line_no=0, type=EditType.SPLIT_TURN, before="vitajte", new_speaker="Robert Fico"),
        TranscriptEdit(line_no=0, type=EditType.SPLIT_TURN, before="Dobrý deň", new_speaker="Erik Tomáš"),
    ]
    out, log = apply_corrections(lines, edits, NAMES, SPEAKERS)
    assert len(out) == 1
    assert [r.rule for r in log] == ["split", "split"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_correction.py -q`
Expected: FAIL — `No module named 'src.correction'`.

- [ ] **Step 3: Create `src/correction.py`**

```python
"""Auditable transcript correction: the LLM proposes small edits, code decides.

The raw ASR transcript stays the source of truth. An edit is applied only if
it cannot change what was said: numbers, negation, and content words are
invariant; only spelling, word boundaries, known proper nouns, punctuation,
and turn splits are allowed. Every decision is logged.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import replace

from src.report_models import EditResult, EditType, TranscriptEdit
from src.transcript_lines import Line

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
NUMERAL_WORDS = frozenset({
    "jeden", "jedna", "jedno", "dva", "dve", "tri", "štyri", "päť", "šesť",
    "sedem", "osem", "deväť", "desať", "jedenásť", "dvanásť", "dvadsať",
    "tridsať", "štyridsať", "päťdesiat", "šesťdesiat", "sedemdesiat",
    "osemdesiat", "deväťdesiat", "sto", "dvesto", "tisíc", "milión",
    "milióny", "miliónov", "miliarda", "miliardy", "miliárd", "polovica",
    "tretina", "štvrtina",
})
NEGATION_WORDS = frozenset({
    "nie", "nikdy", "nič", "ani", "nikto", "nikde", "nijako",
    "žiaden", "žiadny", "žiadna", "žiadne", "žiadnu", "žiadneho",
})
_NEG_ALIASES = {"neni": "nie"}
_ORTHO = (EditType.SPELLING, EditType.WORD_BOUNDARY)
_SIMILARITY_MIN = 0.6


def _words(s: str) -> list[str]:
    return [_NEG_ALIASES.get(w, w) for w in _WORD.findall((s or "").casefold())]


def number_tokens(s: str) -> list[str]:
    return sorted(_NUM.findall(s or "")) + sorted(w for w in _words(s) if w in NUMERAL_WORDS)


def negation_tokens(s: str) -> set[str]:
    return {w for w in _words(s) if w in NEGATION_WORDS}


def _negation_flip(before: str, after: str) -> bool:
    if negation_tokens(before) != negation_tokens(after):
        return True
    wb, wa = set(_words(before)), set(_words(after))
    return any("ne" + w in wa for w in wb) or any("ne" + w in wb for w in wa)


def _similar(before: str, after: str) -> bool:
    b, a = before.casefold(), after.casefold()
    if len(b) == len(a) and sum(x != y for x, y in zip(b, a)) <= 1:
        return True
    return difflib.SequenceMatcher(None, b, a).ratio() >= _SIMILARITY_MIN


def check_edit(
    edit: TranscriptEdit, line_text: str, allowed_names: set[str], speaker_names: set[str]
) -> str:
    """'' when the edit is safe to apply, else the name of the guard it breaks."""
    if edit.type == EditType.SPLIT_TURN:
        pos = line_text.find(edit.before) if edit.before else -1
        if edit.new_speaker not in speaker_names or pos <= 0:
            return "split"
        return ""
    if not edit.before or edit.before not in line_text:
        return "not_found"
    if edit.before == edit.after:
        return "noop"
    if number_tokens(edit.before) != number_tokens(edit.after):
        return "number"
    if _negation_flip(edit.before, edit.after):
        return "negation"
    if edit.type == EditType.PUNCTUATION:
        return "" if _words(edit.before) == _words(edit.after) else "punctuation"
    if abs(len(edit.before.split()) - len(edit.after.split())) > 1:
        return "word_count"
    if edit.type in _ORTHO and not _similar(edit.before, edit.after):
        return "similarity"
    if edit.type == EditType.PROPER_NOUN:
        allowed = {w for name in allowed_names for w in _words(name)}
        before_words = set(_words(edit.before))
        if any(w not in allowed for w in _words(edit.after) if w not in before_words):
            return "proper_noun"
    return ""


def apply_corrections(
    lines: list[Line],
    edits: list[TranscriptEdit],
    allowed_names: set[str],
    speaker_names: set[str],
) -> tuple[list[Line], list[EditResult]]:
    """Apply guarded edits; returns new lines (input untouched) and the full log."""
    work = [replace(ln, edit_ids=list(ln.edit_ids)) for ln in lines]
    log: list[EditResult] = []
    splits: dict[int, list[int]] = {}
    for edit in edits:
        eid = len(log)
        if not 0 <= edit.line_no < len(work):
            log.append(EditResult(id=eid, edit=edit, applied=False, rule="line"))
            continue
        ln = work[edit.line_no]
        rule = check_edit(edit, ln.text, allowed_names, speaker_names)
        log.append(EditResult(id=eid, edit=edit, applied=not rule, rule=rule))
        if rule:
            continue
        ln.edit_ids.append(eid)
        if edit.type == EditType.SPLIT_TURN:
            splits.setdefault(ln.no, []).append(eid)
        else:
            ln.text = ln.text.replace(edit.before, edit.after, 1)

    out: list[Line] = []
    for ln in work:
        cuts: list[tuple[int, int]] = []
        for eid in splits.get(ln.no, []):
            pos = ln.text.find(log[eid].edit.before)
            if pos <= 0 or any(p == pos for p, _ in cuts):
                log[eid] = log[eid].model_copy(update={"applied": False, "rule": "split"})
                ln.edit_ids.remove(eid)
            else:
                cuts.append((pos, eid))
        if not cuts:
            out.append(ln)
            continue
        cuts.sort()
        split_ids = {eid for _, eid in cuts}
        base_ids = [i for i in ln.edit_ids if i not in split_ids]
        bounds = [0, *(p for p, _ in cuts), len(ln.text)]
        speakers = [ln.speaker, *(log[eid].edit.new_speaker for _, eid in cuts)]
        own_ids: list[list[int]] = [[], *([eid] for _, eid in cuts)]
        for j, speaker in enumerate(speakers):
            out.append(
                replace(
                    ln,
                    speaker=speaker,
                    text=ln.text[bounds[j] : bounds[j + 1]].strip(),
                    edit_ids=[*base_ids, *own_ids[j]],
                )
            )
    for i, ln in enumerate(out):
        ln.no = i
    return out, log
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_correction.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/correction.py tests/test_correction.py
git commit -m "feat: guarded transcript edits (numbers, negation, similarity, names, turn splits)"
```

---

### Task 6: Chunked LLM edit proposals and end-to-end correction

**Files:**
- Modify: `src/correction.py`
- Test: `tests/test_correction.py`

**Interfaces:**
- Consumes: Task 5; `format_lines`, `parse_lines` (Task 1); `TranscriptQuality` (Task 2); `src.llm.generate_json`.
- Produces: `CorrectionBatch(edits: list[TranscriptEdit])`, `CHUNK_LINES = 40`, `CONTEXT_LINES = 5`, `build_correction_prompt(lines, start, end, allowed_names, speaker_names) -> str`, `propose_corrections(lines, *, llm, allowed_names, speaker_names) -> tuple[list[TranscriptEdit], list[str]]`, `CorrectionOutcome(text, lines, log, quality, notes)`, `correct_transcript_edits(named_text, *, llm, allowed_names, speaker_names) -> CorrectionOutcome`, `default_correction_llm(settings)`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_correction.py`:

```python
from src.correction import (  # noqa: E402
    CorrectionBatch,
    correct_transcript_edits,
    propose_corrections,
)


def _many(n: int) -> str:
    return "".join(f"Moderátor [{i // 60:02d}:{i % 60:02d}]: riadok {i} pre ľudi\n" for i in range(n))


def test_propose_chunks_and_drops_out_of_range_edits() -> None:
    lines = parse_lines(_many(85))
    prompts: list[str] = []

    def fake(prompt: str) -> CorrectionBatch:
        prompts.append(prompt)
        n = len(prompts) - 1
        return CorrectionBatch(edits=[
            TranscriptEdit(line_no=n * 40, type=S, before="ľudi", after="ľudí"),
            TranscriptEdit(line_no=999, type=S, before="x", after="y"),
        ])

    edits, notes = propose_corrections(lines, llm=fake, allowed_names=set(), speaker_names={"Moderátor"})
    assert len(prompts) == 3
    assert [x.line_no for x in edits] == [0, 40, 80]
    assert "Edit only lines #40..#79" in prompts[1]
    assert "#35 Moderátor" in prompts[1]  # 5 context lines before the chunk
    assert notes == []


def test_failed_chunk_is_noted_and_skipped() -> None:
    lines = parse_lines(_many(85))
    calls = {"n": 0}

    def fake(prompt: str) -> CorrectionBatch:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("timeout")
        return CorrectionBatch(edits=[])

    _, notes = propose_corrections(lines, llm=fake, allowed_names=set(), speaker_names=set())
    assert calls["n"] == 3
    assert len(notes) == 1 and "#40-#79" in notes[0]


def test_correct_transcript_edits_end_to_end() -> None:
    named = (
        "Moderátor [00:10]: Vitajte pre ľudi.\n"
        "Erik Tomáš [00:12]: Strana HLas presadila zo 40 na 135 eur.\n"
    )

    def fake(prompt: str) -> CorrectionBatch:
        return CorrectionBatch(edits=[
            TranscriptEdit(line_no=0, type=S, before="ľudi", after="ľudí"),
            TranscriptEdit(line_no=1, type=PN, before="HLas", after="HLAS"),
            TranscriptEdit(line_no=1, type=S, before="40", after="140"),
        ])

    out = correct_transcript_edits(
        named, llm=fake, allowed_names={"HLAS"}, speaker_names={"Moderátor", "Erik Tomáš"}
    )
    assert out.text == (
        "Moderátor [00:10]: Vitajte pre ľudí.\n"
        "Erik Tomáš [00:12]: Strana HLAS presadila zo 40 na 135 eur.\n"
    )
    assert out.quality.applied == 2
    assert out.quality.rejected_by_rule == {"number": 1}
    assert [ln.raw_no for ln in out.lines] == [0, 1]
    assert out.lines[1].edit_ids == [1]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_correction.py -q`
Expected: FAIL — `ImportError: cannot import name 'CorrectionBatch'`.

- [ ] **Step 3: Add to `src/correction.py`**

Extend imports:

```python
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable

from pydantic import BaseModel, Field

from src.report_models import EditResult, EditType, TranscriptEdit, TranscriptQuality
from src.transcript_lines import Line, format_lines, parse_lines
```

Append:

```python
CHUNK_LINES = 40
CONTEXT_LINES = 5

_RULES = (
    "You fix ASR errors in a Slovak TV debate transcript. Return ONLY a list "
    "of small edits; never rewrite lines.\n"
    "Allowed edit types:\n"
    "- spelling: a misspelled word or missing diacritics ('ľudi' -> 'ľudí').\n"
    "- word_boundary: wrongly split or merged words ('kpointe' -> 'k pointe').\n"
    "- proper_noun: a misheard name, ONLY to a name from NAMES.\n"
    "- punctuation: commas and sentence ends; words stay identical.\n"
    "- split_turn: one line contains two speakers. `before` = the first words "
    "spoken by the second speaker, copied exactly; `new_speaker` from SPEAKERS. "
    "Use one split_turn per speaker change.\n"
    "Never: fix grammar or style of spoken language, paraphrase, add missing "
    "words, delete repetitions or filler words, change numbers or negation. "
    "If unsure, propose nothing.\n"
    "`before` must be copied exactly from the line and kept short (the wrong "
    "word with at most one neighbour). `line_no` is the number after '#'.\n"
)


class CorrectionBatch(BaseModel):
    edits: list[TranscriptEdit] = Field(default_factory=list)


def build_correction_prompt(
    lines: list[Line],
    start: int,
    end: int,
    allowed_names: set[str],
    speaker_names: set[str],
) -> str:
    lo, hi = max(0, start - CONTEXT_LINES), min(len(lines), end + CONTEXT_LINES)
    body = "\n".join(f"#{ln.no} {ln.speaker} [{ln.ts}]: {ln.text}" for ln in lines[lo:hi])
    return (
        f"{_RULES}\n"
        f"NAMES: {', '.join(sorted(allowed_names)) or '(none)'}\n"
        f"SPEAKERS: {', '.join(sorted(speaker_names)) or '(none)'}\n\n"
        f"Edit only lines #{start}..#{end - 1}; other lines are context.\n\n{body}"
    )


def propose_corrections(
    lines: list[Line],
    *,
    llm: Callable[[str], CorrectionBatch],
    allowed_names: set[str],
    speaker_names: set[str],
) -> tuple[list[TranscriptEdit], list[str]]:
    edits: list[TranscriptEdit] = []
    notes: list[str] = []
    for start in range(0, len(lines), CHUNK_LINES):
        end = min(len(lines), start + CHUNK_LINES)
        try:
            batch = llm(build_correction_prompt(lines, start, end, allowed_names, speaker_names))
        except Exception as exc:  # noqa: BLE001 - a failed chunk keeps its raw text
            logger.warning("Correction chunk %d-%d failed: %s", start, end - 1, exc)
            notes.append(f"Transcript correction chunk #{start}-#{end - 1} failed: {exc}"[:300])
            continue
        edits.extend(x for x in batch.edits if start <= x.line_no < end)
    return edits, notes


@dataclass
class CorrectionOutcome:
    text: str
    lines: list[Line]
    log: list[EditResult]
    quality: TranscriptQuality
    notes: list[str]


def correct_transcript_edits(
    named_text: str,
    *,
    llm: Callable[[str], CorrectionBatch],
    allowed_names: set[str],
    speaker_names: set[str],
) -> CorrectionOutcome:
    lines = parse_lines(named_text)
    edits, notes = propose_corrections(
        lines, llm=llm, allowed_names=allowed_names, speaker_names=speaker_names
    )
    out, log = apply_corrections(lines, edits, allowed_names, speaker_names)
    quality = TranscriptQuality(
        applied=sum(1 for r in log if r.applied),
        rejected_by_rule=dict(Counter(r.rule for r in log if not r.applied)),
    )
    return CorrectionOutcome(
        text=format_lines(out), lines=out, log=log, quality=quality, notes=notes
    )


def default_correction_llm(settings) -> Callable[[str], CorrectionBatch]:
    from src.llm import generate_json

    return lambda prompt: generate_json(
        prompt, CorrectionBatch, settings, label="transcript_correction"
    )
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_correction.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/correction.py tests/test_correction.py
git commit -m "feat: chunked LLM edit proposals with guarded application and quality summary"
```

---

### Task 7: Question candidates

**Files:**
- Create: `src/questions.py`
- Test: `tests/test_questions.py`

**Interfaces:**
- Consumes: `Line`, `strip_markers` (Task 1); `QuestionItem` (Task 2).
- Produces: `MIN_QUESTION_WORDS = 4`, `MIN_ANSWER_WORDS = 15`, `extract_question_candidates(lines, moderator: str, guests: list[str], start_s: int = 0) -> list[QuestionItem]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_questions.py`:

```python
"""Offline tests for the question audit (src/questions.py)."""

from __future__ import annotations

from src.questions import extract_question_candidates
from src.transcript_lines import parse_lines

GUESTS = ["Erik Tomáš", "Marián Viskupič"]
LINES = parse_lines(
    "Moderátor [00:10]: Dobrý deň.\n"
    "Moderátor [00:12]: Pán Viskupič, zvýšite dane, áno alebo nie?\n"
    "Erik Tomáš [00:15]: (cez seba) To nie je pravda.\n"
    "Marián Viskupič [00:18]: "
    + "Dane nezvýšime, to vám garantujem úplne jasne a zreteľne " * 2 + "\n"
    "Moderátor [00:30]: Áno?\n"
    "Marián Viskupič [00:31]: A ešte jedna vec k tomu.\n"
    "Moderátor [00:40]: A vy, koľko to bude stáť?\n"
    "Erik Tomáš [00:45]: Uvidíme.\n"
    "Moderátor [00:50]: Ďalšia téma, prečo ste hlasovali proti?\n"
    "Marián Viskupič [00:55]: " + "slovo " * 20 + "\n"
)


def test_candidates_addressee_and_answer_span() -> None:
    items = extract_question_candidates(LINES, "Moderátor", GUESTS)
    assert [(i.id, i.timestamp, i.addressee) for i in items] == [
        (1, "00:12", "Marián Viskupič"),
        (2, "00:40", "Erik Tomáš"),
        (3, "00:50", "Marián Viskupič"),
    ]
    # The short "Áno?" does not end the answer span.
    assert items[0].answer_words == 24
    assert items[1].answer_words == 1
    assert items[0].question == "Pán Viskupič, zvýšite dane, áno alebo nie?"


def test_candidates_respect_debate_start() -> None:
    items = extract_question_candidates(LINES, "Moderátor", GUESTS, start_s=45)
    assert [i.timestamp for i in items] == ["00:50"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_questions.py -q`
Expected: FAIL — `No module named 'src.questions'`.

- [ ] **Step 3: Create `src/questions.py`**

```python
"""Question audit: which moderator questions demanded a concrete answer, and
whether the addressed guest gave one.

Candidates and answer spans are found in code; an LLM classifies them; code
validates every dodge against the answer text. Doubt resolves to `answered`.
"""

from __future__ import annotations

import logging
import re

from src.report_models import QuestionItem
from src.transcript_lines import Line, strip_markers

logger = logging.getLogger(__name__)

MIN_QUESTION_WORDS = 4
MIN_ANSWER_WORDS = 15
_QUESTION = re.compile(r"[^.!?]*\?")
_VOCATIVE = re.compile(r"\b(?:pán|pani|pane)\s+([^\W\d_]+)", re.IGNORECASE)


def _questions(text: str) -> list[str]:
    return [
        q.strip()
        for q in _QUESTION.findall(strip_markers(text))
        if len(q.split()) >= MIN_QUESTION_WORDS
    ]


def _addressee(text: str, guests: list[str]) -> str | None:
    """The guest addressed by name ('pán Viskupič'); the last such vocative wins."""
    for m in reversed(list(_VOCATIVE.finditer(text))):
        stem = m.group(1).casefold()[:5]
        if len(stem) < 4:
            continue
        hits = [g for g in guests if any(t.casefold().startswith(stem) for t in g.split())]
        if len(hits) == 1:
            return hits[0]
    return None


def extract_question_candidates(
    lines: list[Line], moderator: str, guests: list[str], start_s: int = 0
) -> list[QuestionItem]:
    """One candidate per moderator turn that contains a real question."""
    items: list[QuestionItem] = []
    for i, ln in enumerate(lines):
        if ln.speaker != moderator or ln.seconds < start_s:
            continue
        qs = _questions(ln.text)
        if not qs:
            continue
        addressee = _addressee(ln.text, guests) or next(
            (nl.speaker for nl in lines[i + 1 :] if nl.speaker in guests), None
        )
        if addressee is None:
            continue
        answer: list[str] = []
        for nl in lines[i + 1 :]:
            if nl.speaker == moderator and _questions(nl.text):
                break
            if nl.speaker == addressee and not nl.crosstalk:
                answer.append(strip_markers(nl.text))
        text = " ".join(answer)
        items.append(
            QuestionItem(
                id=len(items) + 1,
                timestamp=ln.ts,
                addressee=addressee,
                question=" ".join(qs)[:600],
                answer_words=len(text.split()),
                answer_text=text,
            )
        )
    return items
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_questions.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/questions.py tests/test_questions.py
git commit -m "feat: question candidates with addressee and answer span"
```

---

### Task 8: Question classification, validation, counts

**Files:**
- Modify: `src/questions.py`
- Test: `tests/test_questions.py`

**Interfaces:**
- Consumes: Task 7; `QuestionKind`, `QuestionOutcome`, `SpeakerMap` (Task 2); `ts_seconds`, `Line` (Task 1); `src.validation.normalize`, `quote_grounded` (function-level); `src.llm.generate_json`.
- Produces: `ClassifiedQuestion(id, kind, outcome, evidence, reason)`, `QuestionClassification(items)`, `build_question_prompt(items) -> str`, `classify_questions(items, *, llm) -> list[str]` (mutates items, returns notes), `question_counts(items) -> dict[str, int]` with keys `received`, `challenging`, `interrupted`, `dodged`, `partial`, `render_dodges(items) -> list[str]`, `question_balance_finding(items, guests) -> str | None`, `run_question_audit(lines, smap, *, llm) -> tuple[list[QuestionItem], list[str]]`, `default_question_llm(settings)`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_questions.py`:

```python
from src.questions import (  # noqa: E402
    ClassifiedQuestion,
    QuestionClassification,
    classify_questions,
    question_balance_finding,
    question_counts,
    render_dodges,
    run_question_audit,
)
from src.report_models import (  # noqa: E402
    QuestionItem,
    QuestionKind,
    QuestionOutcome,
    SpeakerMap,
    SpeakerMapEntry,
    SpeakerRole,
)

CH, OPEN = QuestionKind.CHALLENGING, QuestionKind.OPEN
DODGED, ANSWERED, INTERRUPTED = QuestionOutcome.DODGED, QuestionOutcome.ANSWERED, QuestionOutcome.INTERRUPTED


def _all_dodged(prompt: str) -> QuestionClassification:
    return QuestionClassification(items=[
        ClassifiedQuestion(id=1, kind=CH, outcome=DODGED, reason="odbočil",
                           evidence=["[00:18] Dane nezvýšime, to vám garantujem"]),
        ClassifiedQuestion(id=2, kind=CH, outcome=DODGED, evidence=["Uvidíme"]),
        ClassifiedQuestion(id=3, kind=CH, outcome=DODGED,
                           evidence=["Toto sa v odpovedi nikde nenachádza vôbec"]),
    ])


def test_classification_is_validated_against_the_answer() -> None:
    items = extract_question_candidates(LINES, "Moderátor", GUESTS)
    notes = classify_questions(items, llm=_all_dodged)
    assert [i.outcome for i in items] == [DODGED, INTERRUPTED, ANSWERED]
    assert items[0].evidence[0].startswith("[00:12] otázka:")
    assert "garantujem" in items[0].evidence[1]
    assert len(notes) == 1 and "#3" in notes[0]
    assert question_counts(items) == {
        "received": 3, "challenging": 3, "interrupted": 1, "dodged": 1, "partial": 0,
    }
    assert render_dodges(items) == [
        "[00:12] Pán Viskupič, zvýšite dane, áno alebo nie? → dodged: odbočil"
    ]


def test_non_challenging_and_unclassified_are_not_counted() -> None:
    items = extract_question_candidates(LINES, "Moderátor", GUESTS)

    def fake(prompt: str) -> QuestionClassification:
        return QuestionClassification(items=[
            ClassifiedQuestion(id=1, kind=OPEN, outcome=DODGED, evidence=["x"]),
        ])

    notes = classify_questions(items, llm=fake)
    assert items[0].kind == OPEN and items[0].outcome is None
    assert items[1].kind == OPEN and items[2].kind == OPEN
    assert len(notes) == 2
    assert question_counts(items)["challenging"] == 0


def test_question_balance_finding() -> None:
    items = [
        QuestionItem(id=i, timestamp="00:00", addressee="A" if i < 6 else "B", question="q", kind=CH)
        for i in range(7)
    ]
    finding = question_balance_finding(items, ["A", "B"])
    assert finding is not None and "A: 6" in finding and "B: 1" in finding
    assert question_balance_finding(items[:2] + items[6:], ["A", "B"]) is None


def _smap() -> SpeakerMap:
    return SpeakerMap(status="ok", entries=[
        SpeakerMapEntry(label="a", name="Moderátor", role=SpeakerRole.MODERATOR),
        SpeakerMapEntry(label="b", name="Erik Tomáš", role=SpeakerRole.GUEST),
        SpeakerMapEntry(label="c", name="Marián Viskupič", role=SpeakerRole.GUEST),
    ])


def test_run_question_audit_and_failure() -> None:
    items, notes = run_question_audit(LINES, _smap(), llm=_all_dodged)
    assert len(items) == 3 and len(notes) == 1

    def boom(prompt: str) -> QuestionClassification:
        raise RuntimeError("vertex down")

    items, notes = run_question_audit(LINES, _smap(), llm=boom)
    assert items == [] and "failed" in notes[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_questions.py -q`
Expected: FAIL — `ImportError: cannot import name 'ClassifiedQuestion'`.

- [ ] **Step 3: Add to `src/questions.py`**

Extend imports:

```python
from typing import Callable

from pydantic import BaseModel, Field

from src.report_models import QuestionItem, QuestionKind, QuestionOutcome, SpeakerMap
from src.transcript_lines import Line, strip_markers, ts_seconds
```

Append:

```python
_MIN_EVIDENCE_CHARS = 15
_TS_PREFIX = re.compile(r"^\s*\[[\d:]+\]\s*")
_BALANCE_MIN_GAP = 4

_PROMPT = (
    "Classify each moderator question from a Slovak TV debate and judge the "
    "addressed guest's answer.\n"
    "kind:\n"
    "- challenging: a direct question on substance that requires a concrete "
    "answer: a position (yes/no), a number, responsibility, a deadline, or a "
    "confrontation with a contradiction or a fact.\n"
    "- open: a soft open invitation ('Ako to vidíte?').\n"
    "- procedural: running the debate ('Dokončíte?', 'Môžeme ísť ďalej?').\n"
    "- rhetorical: no answer expected.\n"
    "outcome (only for challenging):\n"
    "- answered: the answer addresses the core of the question, even if it is "
    "unpleasant, hedged, or combative.\n"
    "- partial: answers part of it but omits the core (the number, the yes/no, "
    "the responsibility).\n"
    "- dodged: switches topic, counter-attacks, or refuses without a "
    "substantive answer.\n"
    "- interrupted: the guest had no real chance to answer.\n"
    "A tough, uncomfortable, or critical answer is answered, not dodged. Be "
    "symmetric across coalition and opposition.\n"
    "For partial and dodged copy 1-2 verbatim excerpts from ANSWER that show "
    "the evasion into `evidence`. `reason`: one sentence in Slovak.\n"
)


class ClassifiedQuestion(BaseModel):
    id: int
    kind: QuestionKind
    outcome: QuestionOutcome | None = None
    evidence: list[str] = Field(default_factory=list)
    reason: str = ""


class QuestionClassification(BaseModel):
    items: list[ClassifiedQuestion] = Field(default_factory=list)


def build_question_prompt(items: list[QuestionItem]) -> str:
    blocks = [
        f"#{it.id} [{it.timestamp}] to {it.addressee}\n"
        f"QUESTION: {it.question}\n"
        f"ANSWER ({it.answer_words} words): {it.answer_text[:1500]}"
        for it in items
    ]
    return _PROMPT + "\n\n" + "\n\n".join(blocks)


def classify_questions(
    items: list[QuestionItem], *, llm: Callable[[str], QuestionClassification]
) -> list[str]:
    """Merge the LLM classification into `items`; every dodge must quote the answer."""
    from src.validation import normalize, quote_grounded

    notes: list[str] = []
    if not items:
        return notes
    by_id = {c.id: c for c in llm(build_question_prompt(items)).items}
    for it in items:
        c = by_id.get(it.id)
        if c is None:
            it.kind, it.outcome, it.reason = QuestionKind.OPEN, None, "neklasifikované"
            notes.append(f"Question audit: #{it.id} [{it.timestamp}] not classified; not counted")
            continue
        it.kind, it.reason = c.kind, c.reason
        if it.kind != QuestionKind.CHALLENGING:
            it.outcome = None
            continue
        if it.answer_words < MIN_ANSWER_WORDS:
            it.outcome = QuestionOutcome.INTERRUPTED
            continue
        outcome = c.outcome or QuestionOutcome.ANSWERED
        if outcome in (QuestionOutcome.PARTIAL, QuestionOutcome.DODGED):
            quotes = [_TS_PREFIX.sub("", ev) for ev in c.evidence]
            grounded = [
                q for q in quotes
                if len(normalize(q)) >= _MIN_EVIDENCE_CHARS and quote_grounded(q, it.answer_text)
            ]
            if not grounded:
                notes.append(
                    f"Question audit: #{it.id} [{it.timestamp}] {outcome.value} without a "
                    "grounded answer quote -> answered"
                )
                outcome = QuestionOutcome.ANSWERED
            else:
                it.evidence = [
                    f"[{it.timestamp}] otázka: {it.question[:200]}",
                    *(f"odpoveď: {q[:200]}" for q in grounded[:2]),
                ]
        it.outcome = outcome
    return notes


def question_counts(items: list[QuestionItem]) -> dict[str, int]:
    challenging = [i for i in items if i.kind == QuestionKind.CHALLENGING]
    return {
        "received": len(items),
        "challenging": len(challenging),
        "interrupted": sum(1 for i in challenging if i.outcome == QuestionOutcome.INTERRUPTED),
        "dodged": sum(1 for i in challenging if i.outcome == QuestionOutcome.DODGED),
        "partial": sum(1 for i in challenging if i.outcome == QuestionOutcome.PARTIAL),
    }


def render_dodges(items: list[QuestionItem]) -> list[str]:
    return [
        f"[{i.timestamp}] {i.question[:160]} → {i.outcome.value}: {i.reason}"
        for i in items
        if i.outcome in (QuestionOutcome.PARTIAL, QuestionOutcome.DODGED)
    ]


def question_balance_finding(items: list[QuestionItem], guests: list[str]) -> str | None:
    counts = {
        g: sum(1 for i in items if i.addressee == g and i.kind == QuestionKind.CHALLENGING)
        for g in guests
    }
    if len(counts) < 2:
        return None
    hi, lo = max(counts.values()), min(counts.values())
    if hi - lo >= _BALANCE_MIN_GAP and hi >= 2 * max(lo, 1):
        return "Nepomer podstatných otázok moderátora: " + ", ".join(
            f"{g}: {n}" for g, n in counts.items()
        )
    return None


def run_question_audit(
    lines: list[Line], smap: SpeakerMap, *, llm: Callable[[str], QuestionClassification]
) -> tuple[list[QuestionItem], list[str]]:
    moderator, guests = smap.moderator(), smap.guests()
    if not moderator or not guests:
        return [], ["Question audit skipped: no moderator or guests in the speaker map"]
    start_s = ts_seconds(smap.debate_start) if smap.debate_start else 0
    items = extract_question_candidates(lines, moderator, guests, start_s)
    try:
        notes = classify_questions(items, llm=llm)
    except Exception as exc:  # noqa: BLE001 - responsiveness becomes "no data"
        logger.warning("Question audit failed: %s", exc)
        return [], [f"Question audit failed: {exc}"[:300]]
    return items, notes


def default_question_llm(settings) -> Callable[[str], QuestionClassification]:
    from src.llm import generate_json

    return lambda prompt: generate_json(
        prompt, QuestionClassification, settings, label="question_audit"
    )
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_questions.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/questions.py tests/test_questions.py
git commit -m "feat: question classification with grounded dodge evidence and counts"
```

---

### Task 9: Accusation guard against transcript-dependent verdicts

**Files:**
- Modify: `src/validation.py`
- Test: `tests/test_accusation_guard.py`

**Interfaces:**
- Consumes: `apply_corrections`, `number_tokens`, `negation_tokens` (Task 5); `Line`, `parse_lines` (Task 1); `EditResult`, `EditType` (Task 2).
- Produces: `RAW_MIN_SIMILARITY = 0.85`, `best_window(quote, text) -> tuple[float, str]`, `raw_quote_support(quote, raw_text) -> tuple[float, str]`, `enforce_accusation_support(facts, corrected_lines, raw_lines, log) -> list[str]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_accusation_guard.py`:

```python
"""Offline tests: an accusation never rests on a corrected or misattributed quote."""

from __future__ import annotations

from src.agents import Severity, Verdict, VerifiedFact
from src.correction import apply_corrections
from src.report_models import EditType, TranscriptEdit
from src.transcript_lines import parse_lines
from src.validation import enforce_accusation_support

RAW = parse_lines(
    "Moderátor [00:10]: Podporili ste to?\n"
    "Erik Tomáš [00:12]: Strana HLas presadila zvýšenie výživného zo 40 na 135 eur pre ľudi.\n"
    "Marián Viskupič [00:20]: (cez seba) Opozícia nepodporila zvýšenie príspevku pri narodení.\n"
    "Marián Viskupič [00:25]: Vláda zobrala každej rodine 858 eur cez konsolidáciu.\n"
)
CORRECTED, LOG = apply_corrections(
    RAW,
    [
        TranscriptEdit(line_no=1, type=EditType.PROPER_NOUN, before="HLas", after="HLAS"),
        TranscriptEdit(line_no=1, type=EditType.SPELLING, before="ľudi", after="ľudí"),
    ],
    {"HLAS"},
    {"Moderátor", "Erik Tomáš", "Marián Viskupič"},
)


def _fact(speaker: str, quote: str, verdict: Verdict = Verdict.FALSE) -> VerifiedFact:
    return VerifiedFact(
        claim="c", speaker=speaker, quote=quote, verdict=verdict,
        severity=Severity.MATERIAL, sources=["https://nrsr.sk/x"],
    )


def _run(fact: VerifiedFact) -> list[str]:
    return enforce_accusation_support([fact], CORRECTED, RAW, LOG)


def test_quote_from_another_speaker_is_downgraded() -> None:
    fact = _fact("Marián Viskupič", "zvýšenie výživného zo 40 na 135 eur")
    notes = _run(fact)
    assert fact.verdict == Verdict.UNVERIFIED and fact.severity is None
    assert "priradenie" in notes[0]


def test_crosstalk_only_quote_is_downgraded() -> None:
    fact = _fact("Marián Viskupič", "Opozícia nepodporila zvýšenie príspevku pri narodení")
    _run(fact)
    assert fact.verdict == Verdict.UNVERIFIED


def test_quote_depending_on_proper_noun_edit_is_downgraded() -> None:
    fact = _fact("Erik Tomáš", "Strana HLAS presadila zvýšenie výživného")
    notes = _run(fact)
    assert fact.verdict == Verdict.UNVERIFIED
    assert "opravy prepisu" in notes[0]


def test_spelling_only_quote_keeps_verdict_and_gets_raw_window() -> None:
    fact = _fact("Erik Tomáš", "zo 40 na 135 eur pre ľudí")
    notes = _run(fact)
    assert notes == []
    assert fact.verdict == Verdict.FALSE
    assert fact.quote_raw == "zo 40 na 135 eur pre ľudi."
    assert fact.timestamp == "00:12"
    assert fact.transcript_edits == [0, 1]


def test_loose_quote_below_raw_similarity_is_downgraded() -> None:
    fact = _fact(
        "Marián Viskupič",
        "Vláda zobrala každej rodine 858 eur cez konsolidáciu a ešte oveľa viac peňazí",
    )
    notes = _run(fact)
    assert fact.verdict == Verdict.UNVERIFIED
    assert "nezhoduje" in notes[0]


def test_true_verdicts_are_annotated_not_downgraded() -> None:
    fact = _fact("Marián Viskupič", "zvýšenie výživného zo 40 na 135 eur", Verdict.TRUE)
    assert _run(fact) == []
    assert fact.verdict == Verdict.TRUE
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_accusation_guard.py -q`
Expected: FAIL — `ImportError: cannot import name 'enforce_accusation_support'`.

- [ ] **Step 3: Add to `src/validation.py`**

Add imports after `from src.agents import ...`:

```python
from src.correction import negation_tokens, number_tokens
from src.report_models import EditResult, EditType
from src.transcript_lines import Line
```

Add constants after `URL_CHECK_TIMEOUT`:

```python
RAW_MIN_SIMILARITY = 0.85
_ACCUSATION = (Verdict.FALSE, Verdict.MISLEADING)
_RISKY_EDITS = (EditType.PROPER_NOUN, EditType.SPLIT_TURN)
```

Append at the end of the file:

```python
def best_window(quote: str, text: str) -> tuple[float, str]:
    """Most similar run of words in `text`, as long as the quote (ratio, original words)."""
    q = normalize(quote)
    toks = (text or "").split()
    if not q or not toks:
        return 0.0, ""
    n = len(q.split())
    norm = [normalize(t) for t in toks]
    best = (0.0, "")
    for i in range(max(1, len(toks) - n + 1)):
        cand = " ".join(w for w in norm[i : i + n] if w)
        ratio = difflib.SequenceMatcher(None, q, cand).ratio()
        if ratio > best[0]:
            best = (ratio, " ".join(toks[i : i + n]))
    return best


def raw_quote_support(quote: str, raw_text: str) -> tuple[float, str]:
    """Similarity of a (possibly ellipsis-joined) quote to the raw ASR text."""
    parts = [p for p in _ELLIPSIS_RE.split(quote) if len(normalize(p)) >= MIN_QUOTE_LEN]
    scored = [best_window(p, raw_text) for p in (parts or [quote])]
    return min(s for s, _ in scored), " … ".join(w for _, w in scored)


def _downgrade(fact: VerifiedFact, why: str, notes: list[str]) -> None:
    notes.append(f'Downgraded {fact.verdict.value}->Unverified ({why}): "{fact.claim[:120]}"')
    fact.verdict = Verdict.UNVERIFIED
    fact.severity = None
    fact.rationale = f"{fact.rationale} [{why}]".strip()


def enforce_accusation_support(
    facts: list[VerifiedFact],
    corrected_lines: list[Line],
    raw_lines: list[Line],
    log: list[EditResult],
) -> list[str]:
    """Annotate every fact with its raw ASR window; downgrade False/Misleading
    verdicts whose quote is misattributed, cross-talk only, loosely quoted, or
    dependent on a transcript correction."""
    notes: list[str] = []
    applied = {r.id: r for r in log if r.applied}
    for fact in facts:
        accusation = fact.verdict in _ACCUSATION
        reliable = [ln for ln in corrected_lines if ln.speaker == fact.speaker and not ln.crosstalk]
        parts = [p for p in _ELLIPSIS_RE.split(fact.quote or "") if p.strip()] or [fact.quote or ""]
        hits = [ln for ln in reliable if any(quote_grounded(p, ln.text) for p in parts)]
        if not hits or not quote_grounded(fact.quote or "", "\n".join(ln.text for ln in hits)):
            if accusation:
                _downgrade(fact, "nepotvrdené priradenie rečníka", notes)
            continue
        fact.timestamp = hits[0].ts
        raw_text = "\n".join(
            raw_lines[ln.raw_no].text
            for ln in hits
            if ln.raw_no is not None and ln.raw_no < len(raw_lines)
        )
        similarity, window = raw_quote_support(fact.quote, raw_text)
        fact.quote_raw = window
        fact.transcript_edits = sorted({i for ln in hits for i in ln.edit_ids if i in applied})
        if not accusation:
            continue
        quote_norm = normalize(fact.quote)
        risky = [
            applied[i]
            for i in fact.transcript_edits
            if applied[i].edit.type in _RISKY_EDITS
            and (
                applied[i].edit.type == EditType.SPLIT_TURN
                or normalize(applied[i].edit.after) in quote_norm
            )
        ]
        if similarity < RAW_MIN_SIMILARITY:
            _downgrade(fact, "citácia sa nezhoduje s pôvodným prepisom", notes)
        elif risky:
            _downgrade(fact, "verdikt závisí od opravy prepisu", notes)
        elif set(number_tokens(fact.quote)) - set(number_tokens(window)) or (
            negation_tokens(fact.quote) != negation_tokens(window)
        ):
            _downgrade(fact, "číslo alebo zápor v citácii chýba v pôvodnom prepise", notes)
    return notes
```

- [ ] **Step 4: Run the suite**

Run: `python3 -m pytest -q`
Expected: all pass (no import cycle: `validation → correction → report_models/transcript_lines`).

- [ ] **Step 5: Commit**

```bash
git add src/validation.py tests/test_accusation_guard.py
git commit -m "feat: accusation guard — raw ASR window, attribution, edit dependence"
```

---

### Task 10: Claim funnel

**Files:**
- Modify: `src/selection.py`
- Test: `tests/test_funnel.py`

**Interfaces:**
- Consumes: `ClaimFunnel` (Task 2), `canonical_speaker` (Task 3).
- Produces: `build_claim_funnel(extracted: list[ExtractedClaim], kept: list[ExtractedClaim], final_facts: list[VerifiedFact], names: list[str]) -> list[ClaimFunnel]`, `refresh_funnel_verdicts(funnel: list[ClaimFunnel], facts: list[VerifiedFact], names: list[str]) -> None`.

- [ ] **Step 1: Write the failing tests** — `tests/test_funnel.py`:

```python
"""Offline tests for the per-speaker claim funnel (src/selection.py)."""

from __future__ import annotations

from src.agents import Checkability, ExtractedClaim, Verdict, VerifiedFact
from src.selection import build_claim_funnel, refresh_funnel_verdicts

NAMES = ["Erik Tomáš", "Marián Viskupič"]


def claim(i: int, speaker: str, check: Checkability = Checkability.EMPIRICAL) -> ExtractedClaim:
    return ExtractedClaim(id=i, claim=f"tvrdenie {i}", speaker=speaker, quote="q", checkability=check)


def test_funnel_counts_and_invariants() -> None:
    claims = [
        claim(1, "Erik Tomáš"),
        claim(2, "Tomáš"),
        claim(3, "Erik Tomáš", Checkability.OPINION),
        claim(4, "Marián Viskupič"),
        claim(5, "Marián Viskupič"),
    ]
    kept = [claims[0], claims[1], claims[3]]
    facts = [
        VerifiedFact(claim="a", speaker="Erik Tomáš", verdict=Verdict.TRUE),
        VerifiedFact(claim="b", speaker="Marián Viskupič", verdict=Verdict.UNVERIFIED),
    ]
    rows = {r.speaker: r for r in build_claim_funnel(claims, kept, facts, NAMES)}
    erik, marian = rows["Erik Tomáš"], rows["Marián Viskupič"]
    assert (erik.extracted, erik.non_empirical, erik.selected_for_check, erik.dropped_by_selection) == (3, 1, 2, 0)
    assert (erik.final_facts, erik.removed_ungrounded, erik.checked) == (1, 1, 1)
    assert (marian.extracted, marian.selected_for_check, marian.dropped_by_selection) == (2, 1, 1)
    assert (marian.unverified, marian.checked) == (1, 0)
    for r in rows.values():
        assert r.extracted == r.non_empirical + r.dropped_by_selection + r.selected_for_check
        assert r.selected_for_check == r.removed_ungrounded + r.final_facts


def test_refresh_reflects_later_downgrades() -> None:
    facts = [VerifiedFact(claim="a", speaker="Erik Tomáš", verdict=Verdict.FALSE)]
    rows = build_claim_funnel([claim(1, "Erik Tomáš")], [claim(1, "Erik Tomáš")], facts, NAMES)
    facts[0].verdict = Verdict.UNVERIFIED
    refresh_funnel_verdicts(rows, facts, NAMES)
    assert (rows[0].checked, rows[0].unverified) == (0, 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_funnel.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_claim_funnel'`.

- [ ] **Step 3: Add to `src/selection.py`**

Change the agents import and add two imports:

```python
from src.agents import Checkability, ClaimUsage, ExtractedClaim, Verdict, VerifiedFact
from src.report_models import ClaimFunnel
from src.speakers import canonical_speaker
```

Append:

```python
_CHECKED = (Verdict.TRUE, Verdict.FALSE, Verdict.MISLEADING)


def refresh_funnel_verdicts(
    funnel: list[ClaimFunnel], facts: list[VerifiedFact], names: list[str]
) -> None:
    """Recount verdict buckets; the judge may downgrade after the funnel is built."""
    rows = {r.speaker: r for r in funnel}
    for r in funnel:
        r.checked = r.unverified = r.contested = 0
    for f in facts:
        r = rows.get(canonical_speaker(f.speaker, names) or f.speaker)
        if r is None:
            continue
        if f.verdict in _CHECKED:
            r.checked += 1
        elif f.verdict == Verdict.UNVERIFIED:
            r.unverified += 1
        elif f.verdict == Verdict.CONTESTED:
            r.contested += 1


def build_claim_funnel(
    extracted: list[ExtractedClaim],
    kept: list[ExtractedClaim],
    final_facts: list[VerifiedFact],
    names: list[str],
) -> list[ClaimFunnel]:
    """Per speaker: extracted → non-empirical / dropped / selected → final facts."""
    rows: dict[str, ClaimFunnel] = {}

    def row(speaker: str) -> ClaimFunnel:
        key = canonical_speaker(speaker, names) or speaker
        return rows.setdefault(key, ClaimFunnel(speaker=key))

    kept_ids = {c.id for c in kept}
    for c in extracted:
        r = row(c.speaker)
        r.extracted += 1
        if c.checkability != Checkability.EMPIRICAL:
            r.non_empirical += 1
        elif c.id in kept_ids:
            r.selected_for_check += 1
        else:
            r.dropped_by_selection += 1
    for f in final_facts:
        row(f.speaker).final_facts += 1
    for r in rows.values():
        r.removed_ungrounded = max(0, r.selected_for_check - r.final_facts)
    funnel = list(rows.values())
    refresh_funnel_verdicts(funnel, final_facts, names)
    return funnel
```

- [ ] **Step 4: Run the suite**

Run: `python3 -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/selection.py tests/test_funnel.py
git commit -m "feat: per-speaker claim funnel from extraction to final verdicts"
```

---

### Task 11: Scoring refactor

**Files:**
- Modify: `src/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: Tasks 1-3, 8, 10. Reads `report.speaker_map`, `report.question_audit`, `report.claim_funnel`.
- Produces: `DisciplineResult` gains `inputs: dict[str, int]`, `weights: dict[str, float]`, `penalty_points: float | None`, `normalization_words: int | None`, `rate_per_1000: float | None`, `rate: float | None`. `SpeakerScore` gains `transcript_share_percent`, `turns`, `substantive_turns`, `interjections`, `interruptions_caused`, `questions_received`, `challenging_questions`, `questions_dodged`, `questions_partial`, `questions_interrupted`, `claims_extracted`, `claims_selected`. `DebateVerdict.scoring_status: Literal["ok", "degraded", "no_transcript"]`. `score_report(report, transcript=None)` signature unchanged.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_scoring.py`, and add to its imports `apply_deterministic_moderator_metrics` from `src.scoring` plus:

```python
from src.report_models import (
    ClaimFunnel,
    QuestionItem,
    QuestionKind,
    QuestionOutcome,
    SpeakerMap,
)
```

```python
def test_unmatched_transcript_is_degraded_not_assumed_1000() -> None:
    transcript = (
        "Speaker H [02:00]: " + "slovo " * 50 + "\n"
        "Speaker D [02:10]: " + "slovo " * 50 + "\n"
    )
    speakers = [
        SpeakerTactics(speaker="Erik Tomáš", manipulation=["m"], civility={"score": 9}),
        SpeakerTactics(speaker="Marián Viskupič", civility={"score": 2}),
    ]
    verdict = score_report(_report(speakers), transcript=transcript)
    assert verdict.scoring_status == "degraded"
    assert verdict.winner == ""
    row = _row(verdict, "Erik Tomáš")
    assert row.words == 0
    assert _discipline(row, "manipulation").score is None
    assert "manipulation" not in verdict.discipline_winners
    assert verdict.discipline_winners["civility"] == "Erik Tomáš"
    assert not any("1000 words" in n for n in verdict.method_notes)


def test_partial_speaker_map_is_degraded() -> None:
    report = _report(["A", "B"])
    report.speaker_map = SpeakerMap(status="partial", source="heuristic")
    transcript = "A [00:01]: " + "slovo " * 600 + "\nB [00:05]: " + "slovo " * 600 + "\n"
    assert score_report(report, transcript=transcript).scoring_status == "degraded"


def test_manipulation_rate_is_reproducible_from_fields() -> None:
    transcript = "A [00:01]: " + "slovo " * 2000 + "\n"
    speakers = [SpeakerTactics(speaker="A", manipulation=["m1", "m2"], logical_fallacies=["f"])]
    verdict = score_report(_report(speakers), transcript=transcript)
    d = _discipline(_row(verdict, "A"), "manipulation")
    assert d.inputs == {"manipulation": 2, "fallacies": 1}
    assert d.weights == {"manipulation": 8.0, "fallacies": 6.0}
    assert d.penalty_points == 22.0
    assert d.normalization_words == 2000
    assert d.rate_per_1000 == 11.0
    assert d.score == 89.0
    assert verdict.scoring_status == "ok"


def test_word_share_among_guests_and_turns() -> None:
    transcript = (
        "Moderátor [00:00]: " + "a " * 100 + "\n"
        "A [00:10]: " + "b " * 300 + "\n"
        "B [00:20]: " + "c " * 100 + "\n"
    )
    verdict = score_report(_report(["Moderátor", "A", "B"]), transcript=transcript)
    a = _row(verdict, "A")
    assert a.words == 300
    assert a.word_share_percent == 75.0
    assert a.transcript_share_percent == 60.0
    assert (a.turns, a.substantive_turns) == (1, 1)
    assert sum(r.word_share_percent for r in verdict.scoreboard) == 100.0


def _q(i: int, outcome, kind=QuestionKind.CHALLENGING) -> QuestionItem:
    return QuestionItem(id=i, timestamp="01:00", addressee="A", question=f"otázka {i}?",
                        kind=kind, outcome=outcome, reason="r")


def test_responsiveness_per_challenging_question() -> None:
    report = _report(["A"])
    report.question_audit = [
        _q(1, QuestionOutcome.INTERRUPTED),
        _q(2, QuestionOutcome.DODGED), _q(3, QuestionOutcome.DODGED),
        _q(4, QuestionOutcome.PARTIAL),
        *[_q(i, QuestionOutcome.ANSWERED) for i in range(5, 9)],
        _q(9, None, QuestionKind.OPEN),
    ]
    verdict = score_report(report, transcript="A [00:01]: " + "slovo " * 600 + "\n")
    row = _row(verdict, "A")
    d = _discipline(row, "responsiveness")
    assert d.inputs == {"challenging": 8, "interrupted": 1, "dodged": 2, "partial": 1}
    assert d.rate == 0.311
    assert abs(d.score - 68.89) < 0.01
    assert (row.questions_received, row.challenging_questions, row.questions_dodged) == (9, 8, 2)
    assert "2x vyhýbanie sa otázke" in row.badges
    assert len(d.evidence) == 3


def test_responsiveness_none_without_challenging_questions() -> None:
    report = _report(["A"])
    report.question_audit = [_q(1, None, QuestionKind.OPEN)]
    verdict = score_report(report, transcript="A [00:01]: " + "slovo " * 600 + "\n")
    assert _discipline(_row(verdict, "A"), "responsiveness").score is None


def test_claim_funnel_is_copied_and_refreshed() -> None:
    report = _report(["A"], [_fact("A", Verdict.TRUE)])
    report.claim_funnel = [ClaimFunnel(speaker="A", extracted=5, selected_for_check=3, final_facts=1)]
    verdict = score_report(report, transcript="A [00:01]: " + "slovo " * 600 + "\n")
    row = _row(verdict, "A")
    assert (row.claims_extracted, row.claims_selected, row.checked_claims) == (5, 3, 1)
    assert report.claim_funnel[0].checked == 1
    assert "z 3 vybraných (extrahovaných 5)" in _discipline(row, "truthfulness").detail


def test_moderator_time_distribution_uses_names_turns_and_start() -> None:
    report = _report(["Erik Tomáš"])
    report.speaker_map = SpeakerMap(status="ok", source="cli", debate_start="01:00")
    transcript = (
        "Záznam [00:10]: " + "x " * 50 + "\n"
        "Moderátor [01:00]: Otázka?\n"
        "Erik Tomáš [01:05]: " + "y " * 20 + "\n"
        "Moderátor [01:30]: Ďalej?\n"
    )
    apply_deterministic_moderator_metrics(report, transcript)
    dist = {t.speaker: t for t in report.moderator_audit.equal_time_distribution}
    assert "Záznam" not in dist
    assert dist["Moderátor"].turns == 2
    assert dist["Erik Tomáš"].approximate_share_percent == 90.9
```

Replace `test_close_scores_are_not_a_win` with:

```python
def test_close_scores_are_not_a_win() -> None:
    # Civility 6 vs 5 is a 10-point discipline gap but under 5 final points
    # once clean manipulation (100 each) is weighed in. Not a debate winner.
    transcript = "A [00:01]: " + "slovo " * 1000 + "\nB [00:05]: " + "slovo " * 1000 + "\n"
    speakers = [
        SpeakerTactics(speaker="A", civility={"score": 6}),
        SpeakerTactics(speaker="B", civility={"score": 5}),
    ]
    verdict = score_report(_report(speakers), transcript=transcript)
    assert verdict.margin is not None and verdict.margin < 5
    assert verdict.winner == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_scoring.py -q`
Expected: FAIL — `ImportError: cannot import name 'ClaimFunnel'` is fine only if Task 2 is missing; with Task 2 done, failures are `AttributeError: 'DebateVerdict' object has no attribute 'scoring_status'` and similar.

- [ ] **Step 3: Rewrite `src/scoring.py`**

Update the module docstring's second bullet to:

```python
"""Deterministic fair-play scoring from an AnalysisReport (no LLM).

Design goals:
- Normalize per opportunity so speaking time doesn't bias the winner:
  fact accuracy is rated per CHECKED claim, manipulation per 1000 spoken
  words (real transcript word counts), question dodging per challenging
  question the speaker had room to answer.
- Every discipline is reported with its value, the raw counts behind it,
  and the supporting evidence (quotes/claims), so the verdict is auditable.
- Unverified/Contested facts never penalize a speaker — failure to verify
  is our problem, not theirs.
- When speaker identity is uncertain the verdict is "degraded": no winner.
"""
```

Imports:

```python
from typing import Literal

from pydantic import BaseModel, Field

from src.agents import AnalysisReport, Severity, TimeShare, Verdict
from src.questions import question_counts, render_dodges
from src.report_models import QuestionItem
from src.selection import refresh_funnel_verdicts
from src.speakers import canonical_speaker
from src.transcript_lines import parse_lines, strip_markers, ts_seconds
```

`_NON_CONTESTANT` becomes:

```python
_NON_CONTESTANT = re.compile(
    r"\b(moder[aá]tor|moderator|zostrih|dokr[uú]tka|z[aá]znam)\b",
    re.IGNORECASE,
)
```

Replace the foul constants block (`_MANIPULATION_WEIGHT` … `_MAX_EVIDENCE`) with:

```python
# Manipulation weights (score points per foul per 1000 spoken words).
_MANIPULATION_WEIGHT = 8.0
_FALLACY_WEIGHT = 6.0
# Floor for per-1000-words normalization: protects against exploding foul
# rates when only a few words were matched to a speaker.
_MIN_WORDS_FLOOR = 500
# Responsiveness: dodge mass per challenging question, shrunk toward a mild
# prior so 1 dodge of 1 question is not 0 and 1 answer of 1 is not 100.
_PARTIAL_WEIGHT = 0.5
_DODGE_PRIOR_QUESTIONS = 2.0
_DODGE_PRIOR_MASS = 0.3
# Normalized by words or questions; not named when speaker identity is uncertain.
_IDENTITY_SENSITIVE = ("manipulation", "responsiveness")
_MAX_EVIDENCE = 5
```

Models:

```python
class DisciplineResult(BaseModel):
    discipline: str
    label: str = ""
    score: float | None = Field(
        default=None, description="0-100; None = no data for this discipline."
    )
    detail: str = ""
    evidence: list[str] = Field(default_factory=list)
    inputs: dict[str, int] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    penalty_points: float | None = None
    normalization_words: int | None = None
    rate_per_1000: float | None = None
    rate: float | None = None


class SpeakerScore(BaseModel):
    speaker: str
    score: float = 100.0
    words: int = 0
    word_share_percent: float = Field(
        default=0.0, description="Share of words among scored guests (guests sum to 100)."
    )
    transcript_share_percent: float = Field(
        default=0.0, description="Share of all words after the opening recap, moderator included."
    )
    turns: int = 0
    substantive_turns: int = 0
    interjections: int = 0
    interruptions_caused: int = 0
    questions_received: int = 0
    challenging_questions: int = 0
    questions_dodged: int = 0
    questions_partial: int = 0
    questions_interrupted: int = 0
    claims_extracted: int = 0
    claims_selected: int = 0
    disciplines: list[DisciplineResult] = Field(default_factory=list)
    badges: list[str] = Field(default_factory=list)
    # Raw counters (kept for badges / red flag / downstream display).
    fabrication_count: int = 0
    manipulation_count: int = 0
    civility_score: float = 5.0
    incivility_count: int = 0
    checked_claims: int = 0
    true_count: int = 0
    false_count: int = 0
    misleading_count: int = 0
    unverified_count: int = 0
    contested_count: int = 0


class DebateVerdict(BaseModel):
    winner: str = ""
    margin: float | None = Field(
        default=None,
        description="Points between 1st and 2nd. None when fewer than two contestants.",
    )
    red_flag_speaker: str = ""
    discipline_winners: dict[str, str] = Field(default_factory=dict)
    scoring_status: Literal["ok", "degraded", "no_transcript"] = "ok"
    scoreboard: list[SpeakerScore] = Field(default_factory=list)
    method_notes: list[str] = Field(default_factory=list)
```

`apply_deterministic_moderator_metrics` becomes:

```python
def apply_deterministic_moderator_metrics(
    report: AnalysisReport, transcript: str
) -> list[str]:
    """Replace LLM-estimated time shares/interruptions with computed values."""
    start = report.speaker_map.debate_start if report.speaker_map else None
    stats = transcript_speaker_stats(transcript, start)
    if not stats:
        return []
    total = sum(s.words for s in stats.values())
    if not total:
        return []
    report.moderator_audit.equal_time_distribution = [
        TimeShare(
            speaker=s.speaker,
            approximate_share_percent=round(100.0 * s.words / total, 1),
            turns=s.turns,
        )
        for s in sorted(stats.values(), key=lambda r: -r.words)
    ]
    interruptions = [
        f"{s.speaker}: {s.interruptions_caused}"
        for s in sorted(stats.values(), key=lambda r: -r.interruptions_caused)
        if s.interruptions_caused > 0
    ]
    if interruptions:
        report.moderator_audit.findings.append(
            "Prerušenia (deterministický proxy — vstup do reči pred dokončením "
            "vety): " + ", ".join(interruptions)
        )
    clips = [
        s
        for s in stats.values()
        if 100.0 * s.words / total < 1.5 and not is_non_contestant(s.speaker)
    ]
    if clips:
        report.moderator_audit.findings.append(
            "Pod 1,5 % slov — prehrávka alebo neoznačený hlas, nie hosť debaty: "
            + ", ".join(
                f"{s.speaker} ({100.0 * s.words / total:.1f}%)"
                for s in sorted(clips, key=lambda r: -r.words)
            )
        )
    return [
        "Moderator time shares and turns computed deterministically from "
        "transcript word counts after the opening recap (LLM estimate replaced)."
    ]
```

`_resolve_existing` becomes:

```python
def _resolve_existing(
    name: str, speakers: dict[str, SpeakerScore]
) -> SpeakerScore | None:
    """Find the scoreboard row for a name written by any agent; never creates one."""
    key = _speaker_key(name)
    if key in speakers:
        return speakers[key]
    by_name = {row.speaker: row for row in speakers.values()}
    hit = canonical_speaker(name, list(by_name))
    return by_name[hit] if hit else None
```

Replace `score_report` entirely with:

```python
def score_report(
    report: AnalysisReport, transcript: str | None = None
) -> DebateVerdict:
    """Compute per-discipline and final fair-play scores, winner, red flag.

    Pass the (corrected, named) transcript; manipulation is normalized by the
    real word count of each guest. Without it, manipulation has no score.
    """
    method_notes: list[str] = []
    speakers: dict[str, SpeakerScore] = {}

    def ensure(name: str) -> SpeakerScore:
        key = _speaker_key(name)
        if key not in speakers:
            speakers[key] = SpeakerScore(speaker=name)
        elif len(name) > len(speakers[key].speaker):
            speakers[key].speaker = name  # prefer the longer display name
        return speakers[key]

    skipped: list[str] = []
    for s in report.behavioral_analysis.speakers:
        if is_non_contestant(s.speaker):
            skipped.append(s.speaker)
            continue
        ensure(s.speaker)
    if skipped:
        method_notes.append(
            "Non-contestants excluded from the scoreboard (moderator, insert, "
            "recap): " + ", ".join(skipped)
        )

    # --- Raw speaking metrics ------------------------------------------------
    smap = report.speaker_map
    debate_start = smap.debate_start if smap else None
    stats = transcript_speaker_stats(transcript, debate_start) if transcript else {}
    total_words = sum(st.words for st in stats.values())
    for st in stats.values():
        row = _resolve_existing(st.speaker, speakers)
        if row is None:
            continue  # moderator, clip, or unmatched label — not scored
        row.words += st.words
        row.turns += st.turns
        row.substantive_turns += st.substantive_turns
        row.interjections += st.interjections
        row.interruptions_caused += st.interruptions_caused
    guest_words = sum(r.words for r in speakers.values())
    for row in speakers.values():
        if guest_words:
            row.word_share_percent = round(100.0 * row.words / guest_words, 1)
        if total_words:
            row.transcript_share_percent = round(100.0 * row.words / total_words, 1)

    unmatched = [r.speaker for r in speakers.values() if not r.words]
    if not transcript:
        status = "no_transcript"
        method_notes.append("No transcript given: manipulation has no score.")
    elif unmatched or (smap is not None and smap.status != "ok"):
        status = "degraded"
        why: list[str] = []
        if smap is not None and smap.status != "ok":
            why.append(f"speaker map status '{smap.status}'")
        if unmatched:
            why.append("no transcript words matched: " + ", ".join(unmatched))
        method_notes.append(
            "Scoring degraded (" + "; ".join(why) + "): no overall winner and no "
            "manipulation/responsiveness winner is named."
        )
    else:
        status = "ok"
        method_notes.append(
            "Manipulation normalized per 1000 spoken words (transcript word "
            "counts); responsiveness per challenging question; fact accuracy "
            "per checked claim."
        )

    # --- Claim funnel ----------------------------------------------------------
    roster = [r.speaker for r in speakers.values()]
    if report.claim_funnel:
        refresh_funnel_verdicts(report.claim_funnel, report.facts, roster)
        for f in report.claim_funnel:
            row = _resolve_existing(f.speaker, speakers)
            if row is not None:
                row.claims_extracted += f.extracted
                row.claims_selected += f.selected_for_check

    # --- Facts and questions per roster row -----------------------------------
    facts_by_row: dict[int, list] = {}
    for fact in report.facts:
        if not fact.speaker:
            continue
        row = _resolve_existing(fact.speaker, speakers)
        if row is not None:
            facts_by_row.setdefault(id(row), []).append(fact)
    questions_by_row: dict[int, list[QuestionItem]] = {}
    for q in report.question_audit:
        row = _resolve_existing(q.addressee, speakers)
        if row is not None:
            questions_by_row.setdefault(id(row), []).append(q)

    floored: list[str] = []
    for s in report.behavioral_analysis.speakers:
        if is_non_contestant(s.speaker):
            continue
        row = ensure(s.speaker)
        row_facts = facts_by_row.get(id(row), [])
        disciplines: list[DisciplineResult] = []

        # --- Truthfulness: weighted error per CHECKED claim -----------------
        checked = [
            f
            for f in row_facts
            if f.verdict in (Verdict.TRUE, Verdict.FALSE, Verdict.MISLEADING)
            # Legacy unverified shape: Misleading with no severity and no sources.
            and not (
                f.verdict == Verdict.MISLEADING
                and f.severity is None
                and not f.sources
            )
        ]
        row.checked_claims = len(checked)
        row.true_count = sum(1 for f in checked if f.verdict == Verdict.TRUE)
        row.false_count = sum(1 for f in checked if f.verdict == Verdict.FALSE)
        row.misleading_count = sum(1 for f in checked if f.verdict == Verdict.MISLEADING)
        row.unverified_count = sum(
            1 for f in row_facts if f.verdict == Verdict.UNVERIFIED
        ) + sum(
            1
            for f in row_facts
            if f.verdict == Verdict.MISLEADING and f.severity is None and not f.sources
        )
        row.contested_count = sum(1 for f in row_facts if f.verdict == Verdict.CONTESTED)
        row.fabrication_count = sum(
            1
            for f in checked
            if f.verdict == Verdict.FALSE and f.severity == Severity.FABRICATION
        )
        problematic = [f for f in checked if f.verdict != Verdict.TRUE]
        if checked:
            # Salience 3 weighs 1.0; a central attack (5) outweighs trivia (1).
            weights = [max(1, min(5, int(f.salience or 3))) / 3.0 for f in checked]
            err = sum(
                w * _ERROR_WEIGHTS.get((f.verdict, f.severity), 0.5)
                for f, w in zip(checked, weights)
                if f.verdict != Verdict.TRUE
            )
            mass = sum(weights)
            truth_score = max(
                0.0,
                100.0 * (1.0 - (err + _TRUTH_PRIOR_ERROR) / (mass + _TRUTH_PRIOR_CLAIMS)),
            )
        else:
            truth_score = None
        sample = (
            f" z {row.claims_selected} vybraných (extrahovaných {row.claims_extracted})"
            if row.claims_selected
            else ""
        )
        disciplines.append(
            DisciplineResult(
                discipline="truthfulness",
                label=_DISCIPLINE_LABELS["truthfulness"],
                score=truth_score,
                detail=(
                    f"overené: {row.checked_claims}{sample} (pravda: {row.true_count}, "
                    f"nepravda: {row.false_count}, zavádzajúce: "
                    f"{row.misleading_count}); neoverené: {row.unverified_count}, "
                    f"sporné: {row.contested_count}; "
                    f"malá vzorka stiahnutá k prioru "
                    f"({_TRUTH_PRIOR_CLAIMS:.0f} pseudo-tvrdenia)"
                ),
                evidence=[_fmt_fact(f) for f in problematic[:_MAX_EVIDENCE]],
                inputs={
                    "checked": row.checked_claims,
                    "false": row.false_count,
                    "misleading": row.misleading_count,
                },
            )
        )

        # --- Manipulation & fallacies: fouls per 1000 real words ------------
        n_manip = len(s.manipulation)
        n_fall = len(s.logical_fallacies)
        row.manipulation_count = n_manip
        points = _MANIPULATION_WEIGHT * n_manip + _FALLACY_WEIGHT * n_fall
        if row.words:
            norm = max(row.words, _MIN_WORDS_FLOOR)
            if norm != row.words:
                floored.append(row.speaker)
            rate_1000: float | None = points / norm * 1000.0
            manip_score: float | None = max(0.0, 100.0 - rate_1000)
            manip_detail = (
                f"manipulácie: {n_manip}, logické chyby: {n_fall} (penalta "
                f"{points:.0f} b. / {norm} slov × 1000 = {rate_1000:.1f})"
            )
        else:
            norm = None
            rate_1000 = manip_score = None
            manip_detail = (
                f"manipulácie: {n_manip}, logické chyby: {n_fall}; chýbajú "
                "štatistiky reči, bez skóre"
            )
        disciplines.append(
            DisciplineResult(
                discipline="manipulation",
                label=_DISCIPLINE_LABELS["manipulation"],
                score=manip_score,
                detail=manip_detail,
                evidence=[*s.manipulation, *s.logical_fallacies][:_MAX_EVIDENCE],
                inputs={"manipulation": n_manip, "fallacies": n_fall},
                weights={"manipulation": _MANIPULATION_WEIGHT, "fallacies": _FALLACY_WEIGHT},
                penalty_points=points,
                normalization_words=norm,
                rate_per_1000=round(rate_1000, 2) if rate_1000 is not None else None,
            )
        )

        # --- Responsiveness: dodges per challenging question ----------------
        qs = questions_by_row.get(id(row), [])
        qc = question_counts(qs)
        row.questions_received = qc["received"]
        row.challenging_questions = qc["challenging"]
        row.questions_dodged = qc["dodged"]
        row.questions_partial = qc["partial"]
        row.questions_interrupted = qc["interrupted"]
        n_room = qc["challenging"] - qc["interrupted"]
        if n_room > 0:
            dodge_mass = qc["dodged"] + _PARTIAL_WEIGHT * qc["partial"]
            dodge_rate: float | None = (dodge_mass + _DODGE_PRIOR_MASS) / (
                n_room + _DODGE_PRIOR_QUESTIONS
            )
            resp_score: float | None = 100.0 * (1.0 - dodge_rate)
            resp_detail = (
                f"podstatné otázky: {qc['challenging']} (prerušené: "
                f"{qc['interrupted']}), vyhnutia: {qc['dodged']}, čiastočné: "
                f"{qc['partial']}; ({dodge_mass:.1f} + {_DODGE_PRIOR_MASS}) / "
                f"({n_room} + {_DODGE_PRIOR_QUESTIONS:.0f}) = {dodge_rate:.3f}"
            )
        else:
            dodge_rate = resp_score = None
            resp_detail = "žiadne podstatné priame otázky s priestorom na odpoveď"
        disciplines.append(
            DisciplineResult(
                discipline="responsiveness",
                label=_DISCIPLINE_LABELS["responsiveness"],
                score=resp_score,
                detail=resp_detail,
                evidence=render_dodges(qs)[:_MAX_EVIDENCE],
                inputs={
                    "challenging": qc["challenging"],
                    "interrupted": qc["interrupted"],
                    "dodged": qc["dodged"],
                    "partial": qc["partial"],
                },
                weights={"dodged": 1.0, "partial": _PARTIAL_WEIGHT},
                rate=round(dodge_rate, 3) if dodge_rate is not None else None,
            )
        )

        # --- Civility (sociological, not naive politeness) -------------------
        row.civility_score = s.civility.score
        row.incivility_count = len(s.civility.incivility)
        disciplines.append(
            DisciplineResult(
                discipline="civility",
                label=_DISCIPLINE_LABELS["civility"],
                score=max(0.0, min(100.0, 10.0 * s.civility.score)),
                detail=(
                    f"civility skóre: {s.civility.score}/10, porušenia noriem: "
                    f"{row.incivility_count}, korektné momenty: "
                    f"{len(s.civility.fair_conduct)}"
                ),
                evidence=[*s.civility.incivility, *s.civility.fair_conduct][:_MAX_EVIDENCE],
            )
        )

        row.disciplines = disciplines

        # Final score: weighted mean over disciplines with data.
        avail = [(d, _DISCIPLINE_WEIGHTS[d.discipline]) for d in disciplines if d.score is not None]
        weight_sum = sum(w for _, w in avail)
        row.score = (
            round(sum(d.score * w for d, w in avail) / weight_sum, 1) if weight_sum else 100.0
        )

        # Badges (raw-count based, for the gamified post).
        if row.questions_dodged >= 2:
            row.badges.append(f"{row.questions_dodged}x vyhýbanie sa otázke")
        if n_manip >= 2:
            row.badges.append(f"{n_manip}x manipulácia")
        if n_fall >= 2:
            row.badges.append(f"{n_fall}x logická chyba")
        if row.false_count >= 2:
            row.badges.append(f"fact-check {row.false_count}x nepravda")
        if s.civility.score <= 3.0:
            row.badges.append("neslušné/nekorektné vystupovanie")
        elif s.civility.score >= 8.0:
            row.badges.append("korektné a vecné vystupovanie")

    if floored:
        method_notes.append(
            f"Word floor {_MIN_WORDS_FLOOR} applied to manipulation for: " + ", ".join(floored)
        )
    method_notes.append(
        "Final score = weighted mean of disciplines: "
        + ", ".join(f"{k} {v:.0%}" for k, v in _DISCIPLINE_WEIGHTS.items())
        + " (weights renormalized when a discipline has no data). "
        "Unverified/Contested claims carry no penalty. "
        f"Truthfulness shrinks toward a prior of {_TRUTH_PRIOR_CLAIMS:.0f} "
        "pseudo-claims; responsiveness toward a prior of "
        f"{_DODGE_PRIOR_QUESTIONS:.0f} pseudo-questions. "
        "This is a fair-play index (accuracy and fouls), not a measure of "
        "who persuaded the audience. "
        f"A winner is named only when the lead is at least {_WIN_MARGIN:.0f} points; "
        f"a discipline winner only when the lead is at least {_DISCIPLINE_WIN_MARGIN:.0f}."
    )

    scoreboard = sorted(speakers.values(), key=lambda r: r.score, reverse=True)
    margin: float | None = None
    winner = ""
    if len(scoreboard) >= 2:
        margin = round(scoreboard[0].score - scoreboard[1].score, 1)
    if status == "degraded":
        pass
    elif len(scoreboard) == 1:
        winner = scoreboard[0].speaker
    elif margin is not None:
        if margin >= _WIN_MARGIN:
            winner = scoreboard[0].speaker
        else:
            method_notes.append(
                f"No winner: {scoreboard[0].speaker} leads "
                f"{scoreboard[1].speaker} by {margin:.1f}, under the "
                f"{_WIN_MARGIN:.0f}-point margin."
            )

    discipline_winners: dict[str, str] = {}
    for disc in _DISCIPLINE_WEIGHTS:
        if status == "degraded" and disc in _IDENTITY_SENSITIVE:
            continue
        rows = [
            (r, d)
            for r in scoreboard
            for d in r.disciplines
            if d.discipline == disc and d.score is not None
        ]
        if not rows:
            continue
        rows.sort(key=lambda rd: rd[1].score, reverse=True)
        best = rows[0][1].score
        second = rows[1][1].score if len(rows) > 1 else None
        if second is None or best - second >= _DISCIPLINE_WIN_MARGIN:
            discipline_winners[disc] = rows[0][0].speaker

    red_flag = ""
    if scoreboard:
        red = max(
            scoreboard,
            key=lambda r: (r.fabrication_count + r.manipulation_count, -r.score),
        )
        if red.fabrication_count + red.manipulation_count > 0:
            red_flag = red.speaker

    return DebateVerdict(
        winner=winner,
        margin=margin,
        red_flag_speaker=red_flag,
        discipline_winners=discipline_winners,
        scoring_status=status,
        scoreboard=scoreboard,
        method_notes=method_notes,
    )
```

Delete the now-unused `_DEFAULT_WORDS`, `_DODGE_WEIGHT`, and `_NAME_TOKEN` (check with `rg "_NAME_TOKEN|_DEFAULT_WORDS|_DODGE_WEIGHT" src` — must return nothing). Keep `_SPEAKER_ID` (used by `_speaker_key`).

- [ ] **Step 4: Run the suite**

Run: `python3 -m pytest -q`
Expected: all pass. If `test_tied_discipline_has_no_winner` or `test_moderator_cannot_win` fail, check that `status == "no_transcript"` does not suppress the winner.

- [ ] **Step 5: Commit**

```bash
git add src/scoring.py tests/test_scoring.py
git commit -m "feat: scoring on real word counts, question-based responsiveness, degraded gate, structured disciplines"
```

---

### Task 12: Wire the pipeline, CLI, and publisher

**Files:**
- Modify: `src/agents.py` (`run_analysis`, delete `correct_transcript`, add `_allowed_names`, `_canonicalize_speakers`, FB prompt)
- Modify: `main.py`
- Test: `tests/test_wiring_helpers.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_analysis(transcript, settings=None, debate_date=None, guests: list[str] | None = None, moderator: str | None = None) -> tuple[AnalysisReport, CorrectedTranscript]`; `_allowed_names(smap, briefing) -> set[str]`; `_canonicalize_speakers(report, names) -> None`; CLI `--guests "A;B"`, `--moderator NAME`; file `data/transcripts/{ep}.corrections.json`.

- [ ] **Step 1: Write the failing tests** — `tests/test_wiring_helpers.py`:

```python
"""Offline tests for run_analysis helpers."""

from __future__ import annotations

from src.agents import (
    AnalysisReport,
    BehavioralAnalysis,
    SpeakerTactics,
    Verdict,
    VerifiedFact,
    _allowed_names,
    _canonicalize_speakers,
)
from src.briefing import DebateBriefing, EntityEntry, Participant
from src.report_models import SpeakerMap, SpeakerMapEntry, SpeakerRole


def test_allowed_names_merge_map_and_briefing() -> None:
    smap = SpeakerMap(entries=[SpeakerMapEntry(label="H", name="Erik Tomáš", role=SpeakerRole.GUEST)])
    briefing = DebateBriefing(
        participants=[Participant(name="Marián Viskupič", party="SaS")],
        entity_index=[EntityEntry(reference="minister financií", person="Ladislav Kamenický")],
    )
    names = _allowed_names(smap, briefing)
    assert {"Erik Tomáš", "Marián Viskupič", "SaS", "Ladislav Kamenický"} <= names
    assert _allowed_names(smap, None) == {"Erik Tomáš"}


def test_canonicalize_speakers_in_facts_and_behavior() -> None:
    report = AnalysisReport(
        behavioral_analysis=BehavioralAnalysis(speakers=[SpeakerTactics(speaker="Tomáš")]),
        facts=[VerifiedFact(claim="c", speaker="Erika Tomáša", verdict=Verdict.TRUE)],
    )
    _canonicalize_speakers(report, ["Erik Tomáš", "Marián Viskupič"])
    assert report.facts[0].speaker == "Erik Tomáš"
    assert report.behavioral_analysis.speakers[0].speaker == "Erik Tomáš"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_wiring_helpers.py -q`
Expected: FAIL — `ImportError: cannot import name '_allowed_names'`.

- [ ] **Step 3: Delete `correct_transcript` and add helpers in `src/agents.py`**

Delete the whole `correct_transcript` function (currently `src/agents.py:1028-1102`). Keep `_transcript_block`. Add in its place:

```python
def _allowed_names(smap: SpeakerMap, briefing: "DebateBriefing | None") -> set[str]:
    """Proper nouns the transcript corrector may write: roster, parties, briefing people."""
    names = {e.name for e in smap.entries}
    if briefing is not None:
        for p in briefing.participants:
            names.update([p.name, p.party])
        names.update(e.person for e in briefing.entity_index)
        names.update(g.term for g in briefing.glossary)
    return {n for n in names if n}


def _canonicalize_speakers(report: AnalysisReport, names: list[str]) -> None:
    """Rewrite agent-written speaker names to the speaker-map roster."""
    from src.speakers import canonical_speaker

    if not names:
        return
    for f in report.facts:
        f.speaker = canonical_speaker(f.speaker, names) or f.speaker
    for s in report.behavioral_analysis.speakers:
        s.speaker = canonical_speaker(s.speaker, names) or s.speaker
```

If `DebateBriefing` is not yet imported at module level in `agents.py`, the string annotation keeps it lazy; do not add a module-level import of `src.briefing` if one does not exist.

- [ ] **Step 4: Rewrite the start of `run_analysis`**

Replace the signature, docstring, and everything from `logger.info("Correcting transcript speaker attribution")` through the end of the briefing `elif settings.briefing_enabled:` block with:

```python
def run_analysis(
    transcript: str,
    settings: Settings | None = None,
    debate_date: "date | None" = None,
    guests: list[str] | None = None,
    moderator: str | None = None,
) -> tuple[AnalysisReport, CorrectedTranscript]:
    """Map speakers, correct the transcript by audited edits, then analyze.

    Phases: 0) briefing → speaker map → edit-based correction → A) behavioral/
    moderator/extraction + question audit → selection → B) grounded checker +
    specialists → B2) manager → C) critic → deterministic reconcile, validation,
    accusation guard, claim funnel. Returns (report, corrected).
    """
    settings = settings or get_settings()
    if not settings.vertex_ready():
        raise RuntimeError("GCP_PROJECT_ID is not configured")

    from src.correction import correct_transcript_edits, default_correction_llm
    from src.reconcile import merge_checklists, parse_task_output, reconcile_checks
    from src.speakers import apply_speaker_map, default_speaker_llm, map_speakers
    from src.transcript_lines import parse_lines

    pipeline_notes: list[str] = []

    # Phase 0: political briefing (background only — never evidence). Labels
    # are irrelevant to it, so it runs on the raw transcript and its
    # participants then help the speaker mapper.
    briefing: DebateBriefing | None = None
    briefing_text = ""
    if settings.briefing_enabled and debate_date is not None:
        from src.briefing import render_briefing, run_briefing

        try:
            briefing, briefing_notes = run_briefing(transcript, debate_date, settings)
            briefing_text = render_briefing(briefing)
            pipeline_notes.extend(briefing_notes)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Briefing failed; continuing with legacy extraction")
            pipeline_notes.append(
                f"Briefing failed ({exc}); used legacy salience-based selection."
            )
            briefing = None
            briefing_text = ""
    elif settings.briefing_enabled:
        pipeline_notes.append("Briefing skipped: no --debate-date given.")

    logger.info("Mapping diarization labels to speakers")
    smap = map_speakers(
        transcript,
        guests=guests,
        moderator=moderator,
        llm=default_speaker_llm(settings),
        briefing_text=briefing_text,
    )
    pipeline_notes.append(
        f"Speaker map: status {smap.status}, source {smap.source}, "
        f"debate start {smap.debate_start or 'unknown'}: "
        + ", ".join(f"{e.label}={e.name}" for e in smap.entries)
    )
    pipeline_notes.extend(smap.notes)
    named = apply_speaker_map(transcript, smap)

    logger.info("Correcting transcript by guarded edits")
    outcome = correct_transcript_edits(
        named,
        llm=default_correction_llm(settings),
        allowed_names=_allowed_names(smap, briefing),
        speaker_names={e.name for e in smap.entries},
    )
    pipeline_notes.extend(outcome.notes)
    working = outcome.text if outcome.text.strip() else named
    corrected = CorrectedTranscript(
        text=working,
        notes=[
            f"Transcript edits applied: {outcome.quality.applied}; rejected by "
            f"rule: {outcome.quality.rejected_by_rule}"
        ],
        log=outcome.log,
    )
```

(This removes the old word-drift guard and the old briefing block that used `working`; the new briefing block above replaces it. Keep the briefing `logger.info("Briefing: ...")` line inside the `try` if you want the log line — it is unchanged.)

- [ ] **Step 5: Add the question audit after Phase A**

Directly after the line `moderator_raw = getattr(a_tasks["moderator"].output, "raw", "") or ""` insert:

```python
    from src.questions import default_question_llm, run_question_audit

    question_audit, q_notes = run_question_audit(
        parse_lines(working), smap, llm=default_question_llm(settings)
    )
    pipeline_notes.extend(q_notes)
```

- [ ] **Step 6: Replace the tail of `run_analysis`**

Replace everything from `report.facts = facts` to the final `return report, corrected` with:

```python
    report.facts = facts
    report.briefing = briefing
    report.speaker_map = smap
    report.question_audit = question_audit
    report.transcript_quality = outcome.quality
    guest_names = smap.guests()
    _canonicalize_speakers(report, guest_names)

    from src.questions import question_balance_finding, render_dodges

    for s in report.behavioral_analysis.speakers:
        s.question_dodging = render_dodges(
            [q for q in question_audit if q.addressee == s.speaker]
        )

    notes = [*pipeline_notes, *reconcile_notes, *manager_notes]
    if notes:
        report.critic_notes = [*report.critic_notes, *notes]
    logger.info(
        "Reconciled %d facts from extract=%s grounded=%s specialists=%s (seen_urls=%d)",
        len(facts),
        len(kept_claims),
        len(grounded.checks) if grounded else 0,
        len(specialist_checks.checks) if specialist_checks else 0,
        len(seen_urls),
    )

    # Deterministic moderator metrics: time shares (word counts) and an
    # interruption proxy computed from the transcript beat LLM estimates.
    from src.scoring import apply_deterministic_moderator_metrics

    moderator_notes = apply_deterministic_moderator_metrics(report, working)
    if moderator_notes:
        report.critic_notes = [*report.critic_notes, *moderator_notes]
    balance = question_balance_finding(question_audit, guest_names)
    if balance:
        report.moderator_audit.findings.append(balance)

    from src.selection import build_claim_funnel
    from src.validation import enforce_accusation_support, validate_report

    report = validate_report(report, working, allowed_urls=seen_urls)
    accusation_notes = enforce_accusation_support(
        report.facts, parse_lines(working), parse_lines(named), outcome.log
    )
    report.critic_notes = [*report.critic_notes, *accusation_notes]
    report.claim_funnel = build_claim_funnel(
        extracted.claims if extracted else [], kept_claims, report.facts, guest_names
    )
    return report, corrected
```

- [ ] **Step 7: Publisher prompt rules**

In `generate_facebook_post`, change the report dump to exclude bulky code-made sections:

```python
    report_json = json.dumps(
        report.model_dump(
            mode="json", exclude={"briefing", "speaker_map", "transcript_quality"}
        ),
        ensure_ascii=False,
        indent=2,
    )
```

and add these two rules right after the `"- Víťaza, red flag a skóre ber VÝLUČNE z DebateVerdict — neurčuj ich sám.\n"` line:

```python
            "- Ak DebateVerdict.scoring_status nie je 'ok', víťaza nevyhlasuj a "
            "jednou vetou uveď, že rečníkov sa nepodarilo spoľahlivo priradiť "
            "k prepisu.\n"
            "- Pri rečníkoch môžeš uviesť podiel slov (word_share_percent), počet "
            "prehovorov (turns), podstatné otázky a vyhnutia "
            "(challenging_questions, questions_dodged) — iba z DebateVerdict.\n"
```

- [ ] **Step 8: CLI in `main.py`**

Add two options after `--debate-date`:

```python
@click.option(
    "--guests",
    default=None,
    help='Invited guests, ";"-separated, e.g. "Erik Tomáš;Marián Viskupič". Names the diarization labels.',
)
@click.option("--moderator", default=None, help="Moderator name (default: Moderátor).")
```

Add `guests: str | None, moderator: str | None` to `main(...)`'s parameters. Replace the `run_analysis(...)` call with:

```python
        guest_list = [g.strip() for g in guests.split(";") if g.strip()] if guests else None
        report, corrected = run_analysis(
            transcript_text,
            settings=settings,
            debate_date=debate_day,
            guests=guest_list,
            moderator=moderator,
        )
```

After the block that writes and uploads `corrected_path`, add:

```python
    corrections_path = settings.transcript_dir / f"{ep_id}.corrections.json"
    corrections_path.write_text(
        json.dumps(
            [r.model_dump(mode="json") for r in corrected.log], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    upload_file(
        corrections_path,
        object_name=f"transcripts/{corrections_path.name}",
        settings=settings,
    )
    if report.transcript_quality is not None:
        report.transcript_quality.log_path = f"transcripts/{corrections_path.name}"
```

Replace the per-row `click.echo` in the scoreboard print with:

```python
        click.echo(
            f"  {row.speaker}: {row.score:.1f} (slová {row.words}, podiel "
            f"{row.word_share_percent:.1f}%, prehovory {row.turns}, podstatné "
            f"otázky {row.challenging_questions}, vyhnutia {row.questions_dodged}, "
            f"tvrdenia {row.checked_claims}/{row.claims_selected}/{row.claims_extracted}) [{parts}]"
        )
```

and print the status right after `click.echo("Scoreboard:")`:

```python
    click.echo(f"  Stav skórovania: {verdict.scoring_status}")
```

- [ ] **Step 9: Run the suite and an import smoke test**

Run: `python3 -m pytest -q && python3 -c "import main, src.agents, src.scoring, src.validation, src.selection"`
Expected: all tests pass; the import prints nothing. Also `rg "correct_transcript\(" src main.py` must return nothing.

- [ ] **Step 10: Commit**

```bash
git add src/agents.py main.py tests/test_wiring_helpers.py
git commit -m "feat: wire speaker map, guarded correction, question audit, accusation guard, claim funnel"
```

---

### Task 13: Verify on 620752

Needs Vertex credentials; costs one full analysis run. No code changes unless a check fails.

- [ ] **Step 1: Back up the current report**

```bash
cp data/reports/620752.json data/reports/620752.pre-speakermap.json
```

- [ ] **Step 2: Run from the existing transcript**

```bash
python3 main.py --url https://www.stvr.sk/televizia/archiv/14036/620752 \
  --transcript data/transcripts/620752.txt --episode-id 620752 \
  --debate-date 2026-09-27 --guests "Erik Tomáš;Marián Viskupič"
```

Expected console: `Stav skórovania: ok`, both guests with non-zero `slová`.

- [ ] **Step 3: Check the numbers**

```bash
python3 - <<'EOF'
import json
r = json.load(open("data/reports/620752.json"))
v = r["verdict"]
print(r["speaker_map"]["status"], r["speaker_map"]["debate_start"],
      [(e["label"], e["name"]) for e in r["speaker_map"]["entries"] if e["role"] != "clip"])
for s in v["scoreboard"]:
    print(s["speaker"], s["words"], s["word_share_percent"], s["turns"],
          s["challenging_questions"], s["questions_dodged"],
          s["claims_extracted"], s["claims_selected"], s["checked_claims"])
print([t["speaker"] for t in r["moderator_audit"]["equal_time_distribution"]])
print(r["transcript_quality"])
print(sum(1 for q in r["question_audit"] if q["outcome"] == "dodged"), "dodged")
EOF
```

Pass criteria:
- `speaker_map.status == "ok"`, `Speaker H = Erik Tomáš`, `Speaker D = Marián Viskupič`, `Speaker A = Moderátor`, `debate_start == "01:58"`.
- Tomáš words within ±5 % of 6278, Viskupič within ±5 % of 4537 (from `620752.empty-facts.bak.json`).
- `equal_time_distribution` lists names, not `Speaker X`.
- `transcript_quality.rejected_by_rule` has no surprise categories; open `data/transcripts/620752.corrections.json` and read 20 random applied edits: none changes meaning.
- Read every `dodged` item in `question_audit`: each has a question quote and an answer quote, and none is a tough-but-real answer.

- [ ] **Step 4: Record the result**

If all pass, delete the backup or keep it for comparison (it stays untracked under `data/`). If a criterion fails, open a fix task with the failing evidence before touching thresholds.
