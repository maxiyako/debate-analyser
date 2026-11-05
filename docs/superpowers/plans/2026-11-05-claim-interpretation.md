# Claim Interpretation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before fact-checking, record for every selected claim exactly what the speaker asserted (qualifiers kept, referents resolved, drift from the quote corrected), and make checkers, the manager and the fidelity gate judge that reading.

**Architecture:** A new module `src/interpretation.py` runs one batched LLM pass after claim selection (Phase A2). Code guards (`apply_readings`) accept or reject each reading and rewrite drifted claim text. New optional fields on `ExtractedClaim` flow into the checker prompts through `_claims_json`. Related claims are rendered as read-only blocks for specialists and manager rounds. Phase B and B2 move out of `run_analysis` into `check_claims`, so `python -m src.eval interpret-replay` can re-check gold claims without a full pipeline run.

**Tech Stack:** Python 3.11, pydantic v2, CrewAI (checkers, manager), `google-genai` through `src.llm.generate_json` (interpretation, gate, replay classifier), click (eval CLI), pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-claim-interpretation-design.md`

## Global Constraints

- Tests are offline only. Never call a live LLM or the network from a test. Run with `.venv/bin/python -m pytest -q` (the system interpreter lacks dependencies).
- All free text the LLM writes into reports (interpretation, claim, rationale, notes inside prompts' output) is Slovak. Code-side notes in `critic_notes` stay English, like the existing ones.
- The interpretation pass receives from the briefing only participants (name, party, office), `entity_index`, and glossary term + definition. Never the timeline, disputes, stakes, or glossary `status`.
- Every new pydantic field has a default. Stored reports in `data/reports/*.json` must keep loading.
- `interpretation_enabled` defaults to `False` until Task 11 passes the replay acceptance.
- The interpretation LLM uses `settings.judge_model or settings.gemini_model`, like the fidelity gate.
- Provenance enforcement, quote grounding, `accusation_backed`, `enforce_accusation_support` and the fidelity gate's downgrade-only contract must not be weakened.
- Match the surrounding code: sparse comments, only for constraints the code cannot show.

## Prerequisite

The sub-fact and fidelity-gate work from the previous session (`src/fidelity.py`, `src/reconcile.py`, `src/agents.py`, `src/eval.py`, `config.py` and their tests) is still uncommitted. Ask the user to commit it, or confirm you may commit it as `feat: sub-fact statuses and accusation fidelity gate`, before Task 1, so every task below produces a focused commit. Confirm the suite is green first:

Run: `.venv/bin/python -m pytest -q`
Expected: `385 passed`

## File Structure

| File | Responsibility |
| --- | --- |
| `config.py` (modify) | Three settings: `interpretation_enabled`, `interpretation_window`, `interpretation_chunk` |
| `src/agents.py` (modify) | `Referent` model, new `ExtractedClaim` and `VerifiedFact` fields, reading rules in `_CHECK_RULES`, consistency rule in `_MANAGER_RULES`, `_related_block`, `related` parameters on prompt builders, `check_claims` (Phase B + B2), wiring in `run_analysis` |
| `src/reconcile.py` (modify) | `_build_fact` copies the new fields onto `VerifiedFact` |
| `src/interpretation.py` (create) | Reading models, window location, number guard, `apply_readings`, prompt, batched pass, flag-guarded entry point |
| `src/fidelity.py` (modify) | Recorded reading in the gate prompt, `siblings` parameter on `audit_accusations` |
| `src/eval.py` (modify) | Replay: rebuild claims from stored facts, category classifier, `replay_report`, `interpret_replay`, CLI `interpret-replay` |
| `tests/test_config.py` (modify) | Setting defaults |
| `tests/test_claim_fields.py` (create) | New fields, `_build_fact` copying, legacy report loading |
| `tests/test_interpretation.py` (create) | Window, number guard, `apply_readings`, prompt, chunking, entry point |
| `tests/test_related_context.py` (create) | Reading rules, `_related_block`, checker and manager prompt blocks, `check_claims` signature |
| `tests/test_fidelity.py` (modify) | Recorded reading in prompt, `siblings` parameter |
| `tests/test_eval.py` (modify) | Replay harness with fakes |
| `docs/superpowers/specs/2026-11-05-claim-interpretation-design.md` (modify) | Record decisions made while planning (Task 10) and replay results (Task 11) |

---

### Task 1: Settings

**Files:**
- Modify: `config.py:57` (after `fidelity_enabled`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.interpretation_enabled: bool = False`, `Settings.interpretation_window: int = 6`, `Settings.interpretation_chunk: int = 20`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_interpretation_off_by_default_with_window_and_chunk() -> None:
    s = _settings()
    assert s.interpretation_enabled is False
    assert s.interpretation_window == 6
    assert s.interpretation_chunk == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_interpretation_off_by_default_with_window_and_chunk -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'interpretation_enabled'`

- [ ] **Step 3: Add the settings**

In `config.py`, directly after `fidelity_enabled: bool = True`:

```python
    # Claim interpretation (Phase A2): one batched pass after selection records
    # what each claim asserts, read from the transcript around its quote;
    # checkers judge that reading. Uses judge_model.
    interpretation_enabled: bool = False
    interpretation_window: int = 6  # transcript lines before and after the quote
    interpretation_chunk: int = 20  # claims per interpretation call
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add claim interpretation settings"
```

---

### Task 2: Claim and fact fields

**Files:**
- Modify: `src/agents.py` (move `ClaimCategory` above `VerifiedFact`; add `ReferentOrigin`, `Referent`; extend `ExtractedClaim`, `VerifiedFact`)
- Modify: `src/reconcile.py:364-384` (`_build_fact`)
- Test: `tests/test_claim_fields.py` (create)

**Interfaces:**
- Produces in `src.agents`:
  - `class ReferentOrigin(str, Enum)`: `TRANSCRIPT = "transcript"`, `BRIEFING = "briefing"`
  - `class Referent(BaseModel)`: `text: str`, `resolved: str`, `origin: ReferentOrigin`
  - `ExtractedClaim` new fields: `claim_extracted: str = ""`, `interpretation: str = ""`, `not_asserted: list[str] = []`, `refutation_would_be: str = ""`, `referents: list[Referent] = []`, `related_claim_ids: list[int] = []`
  - `VerifiedFact` new fields: `interpretation: str = ""`, `not_asserted: list[str] = []`, `claim_extracted: str = ""`, `category: ClaimCategory | None = None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_claim_fields.py`:

```python
"""New claim/fact fields carry the interpretation from extraction to the report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents import (
    AnalysisReport,
    ClaimCategory,
    ExtractedClaim,
    ExtractedClaims,
    Referent,
    ReferentOrigin,
    VerifiedFact,
    Verdict,
)
from src.reconcile import reconcile_checks

REPORTS = Path(__file__).resolve().parents[1] / "data" / "reports"


def test_new_fields_default_empty() -> None:
    c = ExtractedClaim(id=1, claim="c", speaker="X", quote="q")
    assert (c.claim_extracted, c.interpretation, c.refutation_would_be) == ("", "", "")
    assert c.not_asserted == [] and c.referents == [] and c.related_claim_ids == []
    f = VerifiedFact(claim="c", speaker="X", verdict=Verdict.TRUE)
    assert f.interpretation == "" and f.not_asserted == [] and f.claim_extracted == ""
    assert f.category is None


def test_build_fact_copies_reading_and_category() -> None:
    claim = ExtractedClaim(
        id=1,
        claim="Zrušenie transakčnej dane pre živnostníkov prešlo hlasmi opozície a SNS.",
        claim_extracted="Zrušenie transakčnej dane prešlo hlasmi opozície a SNS.",
        speaker="Michal Šimečka",
        quote="o transakčnej dani a jej zrušení pre živnostníkov",
        category=ClaimCategory.HISTORY_POLITICS,
        interpretation="Výnimka z transakčnej dane pre živnostníkov prešla s hlasmi opozície a SNS.",
        not_asserted=["úplné zrušenie transakčnej dane"],
        referents=[Referent(text="to", resolved="výnimka pre živnostníkov", origin=ReferentOrigin.TRANSCRIPT)],
    )
    facts, _ = reconcile_checks(ExtractedClaims(claims=[claim]), None, None, set())
    f = facts[0]
    assert f.claim == claim.claim
    assert f.claim_extracted == claim.claim_extracted
    assert f.interpretation == claim.interpretation
    assert f.not_asserted == ["úplné zrušenie transakčnej dane"]
    assert f.category == ClaimCategory.HISTORY_POLITICS


def test_stored_reports_still_load() -> None:
    from src.eval import report_files

    if not REPORTS.exists():
        pytest.skip("no local reports")
    paths = report_files(REPORTS)
    if not paths:
        pytest.skip("no local reports")
    for p in paths:
        AnalysisReport.model_validate(json.loads(p.read_text(encoding="utf-8")))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_claim_fields.py -v`
Expected: FAIL with `ImportError: cannot import name 'Referent' from 'src.agents'`

- [ ] **Step 3: Move `ClaimCategory` above `VerifiedFact`**

`VerifiedFact` (around `src/agents.py:185`) will reference `ClaimCategory`, which is defined later (around line 291). Cut the whole `class ClaimCategory(str, Enum): ...` block (docstring and four members, unchanged) and paste it directly above `class VerifiedFact(BaseModel):`.

- [ ] **Step 4: Add fields to `VerifiedFact`**

In `VerifiedFact`, directly after `transcript_edits: list[int] = Field(...)`:

```python
    interpretation: str = Field(
        default="",
        description="What the speaker asserted, as recorded before checking (set in code).",
    )
    not_asserted: list[str] = Field(
        default_factory=list,
        description="Readings the speaker did not make (set in code).",
    )
    claim_extracted: str = Field(
        default="",
        description="Original extracted claim text when the interpretation rewrote it (set in code).",
    )
    category: ClaimCategory | None = None
```

- [ ] **Step 5: Add `Referent` and the `ExtractedClaim` fields**

Directly above `class ExtractedClaim(BaseModel):`:

```python
class ReferentOrigin(str, Enum):
    TRANSCRIPT = "transcript"
    BRIEFING = "briefing"


class Referent(BaseModel):
    text: str = Field(description='The words as spoken, e.g. "my".')
    resolved: str = Field(description="Who or what they refer to, in Slovak.")
    origin: ReferentOrigin
```

In `ExtractedClaim`, directly after the `time_window` field:

```python
    claim_extracted: str = Field(
        default="",
        description="Set in code by the interpretation pass; leave empty.",
    )
    interpretation: str = Field(
        default="",
        description="Set in code by the interpretation pass; leave empty.",
    )
    not_asserted: list[str] = Field(
        default_factory=list,
        description="Set in code by the interpretation pass; leave empty.",
    )
    refutation_would_be: str = Field(
        default="",
        description="Set in code by the interpretation pass; leave empty.",
    )
    referents: list[Referent] = Field(
        default_factory=list,
        description="Set in code by the interpretation pass; leave empty.",
    )
    related_claim_ids: list[int] = Field(
        default_factory=list,
        description="Set in code by the interpretation pass; leave empty.",
    )
```

- [ ] **Step 6: Copy the fields in `_build_fact`**

In `src/reconcile.py`, `_build_fact` returns `VerifiedFact(...)`. Add four keyword arguments after `rationale=rationale,`:

```python
        interpretation=claim.interpretation,
        not_asserted=list(claim.not_asserted),
        claim_extracted=claim.claim_extracted,
        category=claim.category,
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_claim_fields.py -v && .venv/bin/python -m pytest -q`
Expected: new tests PASS (the legacy-load test may SKIP on a machine without `data/reports`); full suite green.

- [ ] **Step 8: Commit**

```bash
git add src/agents.py src/reconcile.py tests/test_claim_fields.py
git commit -m "feat: carry claim interpretation fields from extraction to facts"
```

---

### Task 3: Readings, window location, number guard, `apply_readings`

**Files:**
- Create: `src/interpretation.py`
- Test: `tests/test_interpretation.py` (create)

**Interfaces:**
- Consumes: `ExtractedClaim`, `Referent`, `ReferentOrigin` (Task 2); `DebateBriefing` from `src.briefing`; `Line`, `format_lines` from `src.transcript_lines`; `best_window`, `DEFAULT_MIN_RATIO` from `src.validation`
- Produces in `src.interpretation`:
  - `class DriftKind(str, Enum)`: `added_qualifier`, `dropped_qualifier`, `wrong_referent`, `changed_metric`, `changed_period`
  - `class Drift(BaseModel)`: `kind: DriftKind`, `detail: str = ""`
  - `class ClaimReading(BaseModel)`: `claim_id: int`, `interpretation: str = ""`, `drift: list[Drift]`, `claim: str = ""`, `referents: list[Referent]`, `not_asserted: list[str]`, `refutation_would_be: str = ""`, `related_claim_ids: list[int]`
  - `class ClaimReadings(BaseModel)`: `readings: list[ClaimReading]`
  - `locate_window(lines: list[Line], claim: ExtractedClaim, size: int) -> str`
  - `referent_context(briefing: DebateBriefing | None) -> str`
  - `digit_tokens(text: str) -> set[str]`
  - `unsupported_numbers(reading: ClaimReading, claim: ExtractedClaim, window: str, ref_context: str, debate_date: date | None) -> list[str]`
  - `apply_readings(claims: list[ExtractedClaim], readings: list[ClaimReading], windows: dict[int, str], ref_context: str, debate_date: date | None = None) -> list[str]`
  - `clear_interpretation(claims: list[ExtractedClaim], *, restore_claim: bool = False) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_interpretation.py`:

```python
"""Offline tests for the claim interpretation pass (src/interpretation.py).

Checkers refuted claims the speaker never made: full abolition of a tax
instead of the exemption for živnostníci, a ministry instead of the
government. The pass records the asserted proposition before any search;
code decides which readings reach the checkers.
"""

from __future__ import annotations

from datetime import date

from src.agents import ExtractedClaim, Referent, ReferentOrigin
from src.briefing import (
    DebateBriefing,
    Dispute,
    EntityEntry,
    GlossaryEntry,
    Participant,
    TimelineEvent,
)
from src.interpretation import (
    ClaimReading,
    Drift,
    DriftKind,
    apply_readings,
    clear_interpretation,
    digit_tokens,
    locate_window,
    referent_context,
    unsupported_numbers,
)
from src.transcript_lines import parse_lines

TRANSCRIPT = """\
Moderátor [08:40]: Hovorme o daniach.
Tomáš Taraba [08:50]: Opozícia nič nepresadila.
Moderátor [08:58]: Pán Šimečka?
Michal Šimečka [09:04]: Keď sa hlasovalo o transakčnej dani a jej zrušení pre živnostníkov, rovnako hlasovala opozícia za.
Michal Šimečka [09:12]: A prešlo to preto, že aj SNS hlasovala za.
Moderátor [09:20]: Ďakujem.
Tomáš Taraba [09:25]: To nie je pravda.
"""
LINES = parse_lines(TRANSCRIPT)
DEBATE = date(2026, 9, 27)


def claim(**kw) -> ExtractedClaim:
    base = dict(
        id=1,
        claim="Zrušenie transakčnej dane prešlo v parlamente hlasmi opozície a SNS.",
        speaker="Michal Šimečka",
        quote="o transakčnej dani a jej zrušení pre živnostníkov, rovnako hlasovala opozícia za",
    )
    base.update(kw)
    return ExtractedClaim(**base)


def reading(**kw) -> ClaimReading:
    base = dict(
        claim_id=1,
        interpretation="Výnimka z transakčnej dane pre živnostníkov prešla s hlasmi opozície a SNS.",
        drift=[Drift(kind=DriftKind.DROPPED_QUALIFIER, detail="vypadlo 'pre živnostníkov'")],
        claim="Zrušenie transakčnej dane pre živnostníkov prešlo hlasmi opozície a SNS.",
        not_asserted=["úplné zrušenie transakčnej dane"],
        refutation_would_be="záznam hlasovania o výnimke pre živnostníkov bez hlasov opozície alebo SNS",
    )
    base.update(kw)
    return ClaimReading(**base)


def briefing() -> DebateBriefing:
    return DebateBriefing(
        debate_date="2026-09-27",
        participants=[Participant(name="Michal Šimečka", party="PS", office="poslanec NR SR")],
        entity_index=[EntityEntry(reference="vláda", person="vláda Roberta Fica (od 25. 10. 2023)")],
        glossary=[GlossaryEntry(term="transakčná daň", definition="daň z platobných transakcií", status="výnimka schválená 2025-09-10")],
        live_disputes=[Dispute(title="Dane", factual_question="Kto zrušil daň?")],
        timeline=[TimelineEvent(date="2025-09-10", event="NR SR schválila výnimku")],
    )


# --- window -----------------------------------------------------------------


def test_window_surrounds_the_quoted_line() -> None:
    w = locate_window(LINES, claim(), size=1)
    assert "Michal Šimečka [09:04]" in w
    assert "Moderátor [08:58]" in w and "Michal Šimečka [09:12]" in w
    assert "[08:50]" not in w and "[09:20]" not in w


def test_window_for_quote_spanning_two_lines_covers_both() -> None:
    c = claim(quote="rovnako hlasovala opozícia za. A prešlo to preto, že aj SNS hlasovala za")
    w = locate_window(LINES, c, size=1)
    assert "Michal Šimečka [09:04]" in w and "Michal Šimečka [09:12]" in w


def test_window_is_empty_without_a_match() -> None:
    assert locate_window(LINES, claim(quote="qqq zzz xxx www vvv kkk"), size=2) == ""


def test_window_is_clipped_at_transcript_edges() -> None:
    c = claim(speaker="Moderátor", quote="Hovorme o daniach.")
    w = locate_window(LINES, c, size=3)
    assert w.startswith("Moderátor [08:40]")
    assert "[09:12]" not in w


# --- briefing context -----------------------------------------------------------


def test_referent_context_has_only_offices_entities_and_glossary_terms() -> None:
    ctx = referent_context(briefing())
    assert "Michal Šimečka" in ctx and "poslanec NR SR" in ctx
    assert '"vláda" = vláda Roberta Fica (od 25. 10. 2023)' in ctx
    assert "transakčná daň: daň z platobných transakcií" in ctx
    assert "2025-09-10" not in ctx  # glossary status and timeline are facts, not referents
    assert "Kto zrušil daň?" not in ctx
    assert referent_context(None) == ""


# --- number guard ---------------------------------------------------------------


def test_digit_tokens_normalize_decimal_comma_and_leading_zeros() -> None:
    assert digit_tokens("11,5 mld. v 09/2026") == {"11.5", "9", "2026"}
    assert digit_tokens("bez čísel") == set()


def test_numbers_from_quote_window_and_time_window_are_allowed() -> None:
    c = claim(quote="za tie 3 balíky, keď sa vybralo cez 11,5 mld.", time_window="2023-10")
    r = reading(interpretation="Za 3 balíky sa vybralo 11.5 mld. od 2023 a dlh rástol od 10/2023.", not_asserted=[])
    assert unsupported_numbers(r, c, "Moderátor [09:20]: Ďakujem.", "", DEBATE) == []


def test_number_from_nowhere_is_reported() -> None:
    r = reading(refutation_would_be="súčet balíkov 7,4 mld.")
    assert unsupported_numbers(r, claim(), "", "", DEBATE) == ["7.4"]


def test_briefing_referent_digits_need_the_briefing() -> None:
    ctx = referent_context(briefing())
    ok = reading(
        interpretation="Dlh vzrástol od nástupu vlády v roku 2023.",
        referents=[Referent(text="keď preberali", resolved="nástup vlády Roberta Fica (25. 10. 2023)", origin=ReferentOrigin.BRIEFING)],
        not_asserted=[],
    )
    assert unsupported_numbers(ok, claim(), "", ctx, DEBATE) == []
    bad = ok.model_copy(update={"referents": [Referent(text="keď preberali", resolved="október 2020", origin=ReferentOrigin.BRIEFING)]})
    assert unsupported_numbers(bad, claim(), "", ctx, DEBATE) == ["2020", "2023"]


def test_debate_date_digits_are_allowed() -> None:
    r = reading(interpretation="K 27. 9. 2026 platí výnimka.", not_asserted=[])
    assert unsupported_numbers(r, claim(), "", "", DEBATE) == []


# --- apply_readings ---------------------------------------------------------------


def test_drift_rewrites_claim_and_keeps_original() -> None:
    c = claim()
    notes = apply_readings([c], [reading()], {1: ""}, "", DEBATE)
    assert c.claim == "Zrušenie transakčnej dane pre živnostníkov prešlo hlasmi opozície a SNS."
    assert c.claim_extracted == "Zrušenie transakčnej dane prešlo v parlamente hlasmi opozície a SNS."
    assert c.interpretation.startswith("Výnimka z transakčnej dane")
    assert c.not_asserted == ["úplné zrušenie transakčnej dane"]
    assert c.refutation_would_be.startswith("záznam hlasovania")
    assert any("rewrote claim 1" in n and "dropped_qualifier" in n for n in notes)


def test_no_drift_keeps_claim_text() -> None:
    c = claim()
    apply_readings([c], [reading(drift=[], claim="iný text")], {1: ""}, "", DEBATE)
    assert c.claim.startswith("Zrušenie transakčnej dane prešlo v parlamente")
    assert c.claim_extracted == ""
    assert c.interpretation


def test_drift_without_claim_text_keeps_claim_and_notes_it() -> None:
    c = claim()
    notes = apply_readings([c], [reading(claim="")], {1: ""}, "", DEBATE)
    assert c.claim_extracted == "" and c.claim.startswith("Zrušenie transakčnej dane prešlo v")
    assert c.interpretation
    assert any("gave no claim text" in n for n in notes)


def test_rejected_reading_leaves_claim_untouched() -> None:
    c = claim()
    notes = apply_readings([c], [reading(refutation_would_be="súčet 7,4 mld.")], {1: ""}, "", DEBATE)
    assert c.interpretation == "" and c.claim_extracted == ""
    assert any("rejected for claim 1" in n and "7.4" in n for n in notes)


def test_unknown_missing_and_related_ids() -> None:
    a, b = claim(id=1), claim(id=2, claim="Iné tvrdenie.")
    notes = apply_readings(
        [a, b],
        [reading(related_claim_ids=[1, 2, 99]), reading(claim_id=7)],
        {1: "", 2: ""},
        "",
        DEBATE,
    )
    assert a.related_claim_ids == [2]
    assert b.interpretation == ""
    assert any("unknown claim 7" in n for n in notes)
    assert any("no reading for claim 2" in n for n in notes)


def test_protected_fields_never_change() -> None:
    c = claim(time_reference="v septembri", time_window="2025-09", consequence=5)
    before = (c.quote, c.speaker, c.category, c.checkability, c.consequence, c.time_reference, c.time_window)
    apply_readings([c], [reading()], {1: ""}, "", DEBATE)
    after = (c.quote, c.speaker, c.category, c.checkability, c.consequence, c.time_reference, c.time_window)
    assert before == after


def test_clear_interpretation_blanks_and_optionally_restores() -> None:
    c = claim()
    apply_readings([c], [reading()], {1: ""}, "", DEBATE)
    clear_interpretation([c], restore_claim=True)
    assert c.claim == "Zrušenie transakčnej dane prešlo v parlamente hlasmi opozície a SNS."
    assert (c.claim_extracted, c.interpretation, c.refutation_would_be) == ("", "", "")
    assert c.not_asserted == [] and c.referents == [] and c.related_claim_ids == []

    d = claim(claim_extracted="extractor filled this", interpretation="x")
    clear_interpretation([d])
    assert d.claim.startswith("Zrušenie") and d.claim_extracted == "" and d.interpretation == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_interpretation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.interpretation'`

- [ ] **Step 3: Write `src/interpretation.py`**

```python
"""Claim interpretation (Phase A2): what exactly did the speaker assert?

Extraction rewords claims and checkers never see the transcript, so a verdict
can refute a stronger or different statement than the one made. One batched
pass reads each selected claim against the transcript around its quote and
records the asserted proposition, drift from the quote, resolved referents,
readings the speaker did not make, and what evidence would refute it. Code
guards decide which readings reach the checkers.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from enum import Enum

from pydantic import BaseModel, Field

from src.agents import ExtractedClaim, Referent, ReferentOrigin
from src.briefing import DebateBriefing
from src.transcript_lines import Line, format_lines
from src.validation import DEFAULT_MIN_RATIO, best_window

logger = logging.getLogger(__name__)

_PROBE_WORDS = 12
_DIGITS = re.compile(r"\d+(?:[.,]\d+)?")


class DriftKind(str, Enum):
    ADDED_QUALIFIER = "added_qualifier"
    DROPPED_QUALIFIER = "dropped_qualifier"
    WRONG_REFERENT = "wrong_referent"
    CHANGED_METRIC = "changed_metric"
    CHANGED_PERIOD = "changed_period"


class Drift(BaseModel):
    kind: DriftKind
    detail: str = Field(default="", description="One Slovak sentence naming the drift.")


class ClaimReading(BaseModel):
    claim_id: int = Field(description="Must match the claim id.")
    interpretation: str = Field(
        default="",
        description="The exact proposition the speaker asserted, in Slovak, with the "
        "speaker's qualifiers kept and referents resolved; nothing added.",
    )
    drift: list[Drift] = Field(
        default_factory=list,
        description="Every difference between the extracted claim and the quote/transcript.",
    )
    claim: str = Field(
        default="",
        description="Faithful Slovak claim text; required when drift is not empty.",
    )
    referents: list[Referent] = Field(default_factory=list)
    not_asserted: list[str] = Field(
        default_factory=list,
        description="Stronger or different readings a checker might test that the speaker did NOT make.",
    )
    refutation_would_be: str = Field(
        default="",
        description="What evidence would refute the interpretation: metric, period, actor, stage.",
    )
    related_claim_ids: list[int] = Field(
        default_factory=list,
        description="Ids of other listed claims about the same fact or dispute.",
    )


class ClaimReadings(BaseModel):
    readings: list[ClaimReading] = Field(default_factory=list)


def locate_window(lines: list[Line], claim: ExtractedClaim, size: int) -> str:
    """Transcript lines around the line where the quote starts; empty when not found."""
    probe = " ".join((claim.quote or "").split()[:_PROBE_WORDS])
    if not probe:
        return ""
    own = [i for i, ln in enumerate(lines) if ln.speaker == claim.speaker]
    for pool in (own, range(len(lines))):
        best_i, best = -1, 0.0
        for i in pool:
            score, _ = best_window(probe, lines[i].text)
            if score > best:
                best, best_i = score, i
        if best_i >= 0 and best >= DEFAULT_MIN_RATIO:
            return format_lines(lines[max(0, best_i - size) : best_i + size + 1])
    return ""


def referent_context(briefing: DebateBriefing | None) -> str:
    """Who holds which office, role references and terms: enough to resolve referents, no events."""
    if briefing is None:
        return ""
    out = ["ÚČASTNÍCI A FUNKCIE k dátumu debaty (iba na určenie, koho alebo čo rečník myslí; NIE dôkaz):"]
    for p in briefing.participants:
        bits = ", ".join(b for b in (p.party, p.office) if b)
        out.append(f"- {p.name}" + (f" ({bits})" if bits else ""))
    for e in briefing.entity_index:
        out.append(f'- "{e.reference}" = {e.person}')
    if briefing.glossary:
        out.append("POJMY:")
        out.extend(f"- {g.term}: {g.definition}" for g in briefing.glossary)
    return "\n".join(out)


def _norm_number(tok: str) -> str:
    tok = tok.replace(",", ".")
    return tok if "." in tok else str(int(tok))


def digit_tokens(text: str) -> set[str]:
    return {_norm_number(m.group()) for m in _DIGITS.finditer(text or "")}


def unsupported_numbers(
    reading: ClaimReading,
    claim: ExtractedClaim,
    window: str,
    ref_context: str,
    debate_date: date | None,
) -> list[str]:
    """Digit tokens in the reading that neither the transcript nor a briefing referent backs."""
    allowed = (
        digit_tokens(claim.quote)
        | digit_tokens(window)
        | digit_tokens(claim.time_window)
        | digit_tokens(debate_date.isoformat() if debate_date else "")
    )
    briefing_digits = digit_tokens(ref_context)
    for r in reading.referents:
        toks = digit_tokens(r.resolved)
        if r.origin == ReferentOrigin.BRIEFING and toks <= briefing_digits:
            allowed |= toks
    texts = [
        reading.interpretation,
        reading.claim,
        reading.refutation_would_be,
        *reading.not_asserted,
        *(r.resolved for r in reading.referents),
    ]
    bad: set[str] = set()
    for text in texts:
        bad |= digit_tokens(text) - allowed
    return sorted(bad)


def apply_readings(
    claims: list[ExtractedClaim],
    readings: list[ClaimReading],
    windows: dict[int, str],
    ref_context: str,
    debate_date: date | None = None,
) -> list[str]:
    """Copy accepted readings onto the claims, in place; returns notes."""
    notes: list[str] = []
    by_id = {c.id: c for c in claims}
    handled: set[int] = set()
    for r in readings:
        c = by_id.get(r.claim_id)
        if c is None:
            notes.append(f"Interpretation: ignored reading for unknown claim {r.claim_id}.")
            continue
        if r.claim_id in handled:
            notes.append(f"Interpretation: ignored duplicate reading for claim {r.claim_id}.")
            continue
        handled.add(r.claim_id)
        bad = unsupported_numbers(r, c, windows.get(c.id, ""), ref_context, debate_date)
        if bad:
            notes.append(
                f"Interpretation rejected for claim {c.id}: numbers not in quote, "
                f"transcript or briefing referents ({', '.join(bad)})."
            )
            continue
        if not r.interpretation.strip():
            notes.append(f"Interpretation: empty reading for claim {c.id}; checked as extracted.")
            continue
        c.interpretation = r.interpretation.strip()
        c.not_asserted = [s.strip() for s in r.not_asserted if s.strip()]
        c.refutation_would_be = r.refutation_would_be.strip()
        c.referents = list(r.referents)
        c.related_claim_ids = sorted({i for i in r.related_claim_ids if i in by_id and i != c.id})
        if not r.drift:
            continue
        kinds = "; ".join(f"{d.kind.value}: {d.detail}" for d in r.drift)
        if r.claim.strip():
            c.claim_extracted = c.claim
            c.claim = r.claim.strip()
            notes.append(
                f'Interpretation rewrote claim {c.id} ({kinds}): '
                f'"{c.claim_extracted[:100]}" -> "{c.claim[:100]}"'
            )
        else:
            notes.append(
                f"Interpretation found drift in claim {c.id} but gave no claim text; "
                f"claim kept ({kinds})."
            )
    for c in claims:
        if c.id not in handled:
            notes.append(f"Interpretation: no reading for claim {c.id}; checked as extracted.")
    return notes


def clear_interpretation(claims: list[ExtractedClaim], *, restore_claim: bool = False) -> None:
    """Blank the code-owned reading fields; optionally undo a claim rewrite."""
    for c in claims:
        if restore_claim and c.claim_extracted:
            c.claim = c.claim_extracted
        c.claim_extracted = ""
        c.interpretation = ""
        c.not_asserted = []
        c.refutation_would_be = ""
        c.referents = []
        c.related_claim_ids = []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_interpretation.py -v`
Expected: all PASS. If a window test fails on the score threshold, lower `_PROBE_WORDS` (for example to 8) rather than lowering `DEFAULT_MIN_RATIO`. The ratio is shared with quote grounding.

- [ ] **Step 5: Commit**

```bash
git add src/interpretation.py tests/test_interpretation.py
git commit -m "feat: claim readings with window location and number guard"
```

---

### Task 4: Interpretation prompt, batched pass, entry point

**Files:**
- Modify: `src/interpretation.py`
- Test: `tests/test_interpretation.py`

**Interfaces:**
- Consumes: everything Task 3 produced; `Settings` (Task 1); `generate_json(prompt, schema, settings, *, label, model)` from `src.llm`
- Produces in `src.interpretation`:
  - `build_interpretation_prompt(chunk: list[ExtractedClaim], windows: dict[int, str], siblings: list[ExtractedClaim], ref_context: str, debate_date: date | None) -> str`
  - `interpret_claims(claims: list[ExtractedClaim], lines: list[Line], *, llm: Callable[[str], ClaimReadings], briefing: DebateBriefing | None = None, debate_date: date | None = None, window: int = 6, chunk: int = 20) -> list[str]`
  - `run_interpretation(claims: list[ExtractedClaim], lines: list[Line], settings: Settings, *, briefing: DebateBriefing | None = None, debate_date: date | None = None, llm: Callable[[str], ClaimReadings] | None = None) -> list[str]`
  - `default_interpretation_llm(settings: Settings) -> Callable[[str], ClaimReadings]`

- [ ] **Step 1: Write the failing tests**

Add to the import block at the top of `tests/test_interpretation.py`:

```python
from config import Settings
from src.interpretation import (
    ClaimReadings,
    build_interpretation_prompt,
    interpret_claims,
    run_interpretation,
)
```

Append to `tests/test_interpretation.py`:

```python
# --- prompt and batched pass --------------------------------------------------------


def test_prompt_carries_claim_quote_window_siblings_and_referents_only() -> None:
    c = claim(time_reference="v septembri", time_window="2025-09")
    sib = claim(id=2, speaker="Tomáš Taraba", claim="Opozícia nič nepresadila.", quote="Opozícia nič nepresadila.")
    ctx = referent_context(briefing())
    p = build_interpretation_prompt([c], {1: locate_window(LINES, c, 1)}, [c, sib], ctx, DEBATE)
    assert "CLAIM 1" in p and c.claim in p and c.quote in p
    assert "Michal Šimečka [09:04]" in p
    assert "v septembri" in p and "2025-09" in p
    assert "#2 Tomáš Taraba: Opozícia nič nepresadila." in p
    assert "2026-09-27" in p
    assert '"vláda" = vláda Roberta Fica' in p
    assert "NR SR schválila výnimku" not in p and "Kto zrušil daň?" not in p
    assert "not_asserted" in p and "refutation_would_be" in p


def test_claims_are_sent_in_chunks_with_full_sibling_list() -> None:
    claims = [claim(id=i, claim=f"Tvrdenie {i}.") for i in range(1, 46)]
    prompts: list[str] = []

    def llm(prompt: str) -> ClaimReadings:
        prompts.append(prompt)
        return ClaimReadings()

    interpret_claims(claims, LINES, llm=llm, debate_date=DEBATE, window=1, chunk=20)
    assert len(prompts) == 3
    assert all("#45 Michal Šimečka: Tvrdenie 45." in p for p in prompts)
    assert "CLAIM 21" in prompts[1] and "CLAIM 20" not in prompts[1]


def test_failed_chunk_is_noted_and_others_still_apply() -> None:
    claims = [claim(id=1), claim(id=2)]
    calls = iter([RuntimeError("quota"), ClaimReadings(readings=[reading(claim_id=2)])])

    def llm(_prompt: str) -> ClaimReadings:
        out = next(calls)
        if isinstance(out, Exception):
            raise out
        return out

    notes = interpret_claims(claims, LINES, llm=llm, debate_date=DEBATE, window=1, chunk=1)
    assert claims[0].interpretation == "" and claims[1].interpretation
    assert any("quota" in n for n in notes)


def _on(**kw) -> Settings:
    return Settings(_env_file=None, interpretation_enabled=True, **kw)


def test_run_interpretation_is_a_no_op_when_disabled() -> None:
    def boom(_prompt: str) -> ClaimReadings:
        raise AssertionError("must not be called")

    c = claim()
    assert run_interpretation([c], LINES, Settings(_env_file=None), llm=boom) == []
    assert c.interpretation == ""


def test_run_interpretation_applies_readings_when_enabled() -> None:
    c = claim()
    notes = run_interpretation(
        [c], LINES, _on(), debate_date=DEBATE, llm=lambda _p: ClaimReadings(readings=[reading()])
    )
    assert c.claim_extracted and c.interpretation
    assert any("rewrote claim 1" in n for n in notes)


def test_run_interpretation_failure_restores_claims(monkeypatch) -> None:
    import src.interpretation as mod

    def half_done(claims, *_a, **_kw):
        claims[0].claim_extracted = claims[0].claim
        claims[0].claim = "prepísané"
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "interpret_claims", half_done)
    c = claim()
    notes = run_interpretation([c], LINES, _on(), llm=lambda _p: ClaimReadings())
    assert c.claim.startswith("Zrušenie transakčnej dane prešlo v")
    assert c.claim_extracted == ""
    assert any("Interpretation failed" in n and "boom" in n for n in notes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_interpretation.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_interpretation_prompt'`

- [ ] **Step 3: Add the prompt, pass and entry point**

In `src/interpretation.py`, extend the imports:

```python
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from config import Settings
```

Append to `src/interpretation.py`:

```python
_INSTRUCTIONS = """\
You record what speakers in a Slovak political debate asserted, before anyone
fact-checks them. You do NOT check facts, you do NOT say whether a claim is
true, and you do NOT use your own knowledge of the facts.

For every CLAIM below, read the QUOTE as a listener in the studio would
understand it from the TRANSCRIPT AROUND THE QUOTE and from what the speaker
said in the other listed claims. Return one reading per claim:

- interpretation: the exact proposition the speaker asserted, in Slovak. Keep
  the speaker's own qualifiers and scope ("pre živnostníkov", "za posledné 3
  roky", "my", "približne"). Add none ("automaticky", "iba", "všetci",
  "definitívne", "úplne"). Resolve who or what "my", "oni", "to", "vtedy",
  role names and similar words refer to.
- drift: compare the EXTRACTED CLAIM with the quote and the transcript. List
  every added qualifier, dropped qualifier, wrong referent, changed metric
  (e.g. production start instead of allocation, ministry instead of
  government) and changed period. Empty when the claim is faithful.
- claim: when drift is not empty, a faithful one-sentence Slovak claim text.
- referents: each resolution as {text, resolved, origin}. origin is
  "transcript" when the transcript settles it, "briefing" when only the
  PARTICIPANTS AND OFFICES list does. Use that list for nothing else.
- not_asserted: stronger or different readings a checker might wrongly test,
  which the speaker did NOT make (e.g. full abolition when the speaker said an
  exemption; the packages' own total when the speaker spoke of debt growth).
- refutation_would_be: what evidence would show the interpretation false:
  the metric, period, actor and stage it must be about. Describe the kind of
  evidence; do not state any fact or figure you know.
- related_claim_ids: ids of other listed claims about the same fact, vote,
  figure or dispute.

Write no number, date or figure that is not in the quote, the transcript
around it, the claim's time window, or the PARTICIPANTS AND OFFICES list.
"""


def _claim_block(c: ExtractedClaim, window: str) -> str:
    time = ""
    if c.time_reference or c.time_window:
        time = f"TIME: {c.time_reference} -> {c.time_window}\n"
    return (
        f"CLAIM {c.id} — {c.speaker}\n"
        f"EXTRACTED CLAIM: {c.claim}\n"
        f"QUOTE: {c.quote}\n"
        f"{time}"
        f"TRANSCRIPT AROUND THE QUOTE:\n{window or '(unavailable)'}\n"
    )


def build_interpretation_prompt(
    chunk: list[ExtractedClaim],
    windows: dict[int, str],
    siblings: list[ExtractedClaim],
    ref_context: str,
    debate_date: date | None,
) -> str:
    sibs = "\n".join(f"- #{c.id} {c.speaker}: {c.claim}" for c in siblings)
    ref = f"DEBATE DATE: {debate_date.isoformat()}\n\n" if debate_date else ""
    return (
        f"{_INSTRUCTIONS}\n"
        f"{ref}"
        f"PARTICIPANTS AND OFFICES:\n{ref_context or '(none)'}\n\n"
        f"ALL CLAIMS OF THIS DEBATE:\n{sibs}\n\n"
        + "\n".join(_claim_block(c, windows.get(c.id, "")) for c in chunk)
    )


def interpret_claims(
    claims: list[ExtractedClaim],
    lines: list[Line],
    *,
    llm: Callable[[str], ClaimReadings],
    briefing: DebateBriefing | None = None,
    debate_date: date | None = None,
    window: int = 6,
    chunk: int = 20,
) -> list[str]:
    """Read every claim in chunks and apply the accepted readings in place."""
    ref_context = referent_context(briefing)
    windows = {c.id: locate_window(lines, c, window) for c in claims}
    readings: list[ClaimReading] = []
    notes: list[str] = []
    for start in range(0, len(claims), max(1, chunk)):
        part = claims[start : start + max(1, chunk)]
        prompt = build_interpretation_prompt(part, windows, claims, ref_context, debate_date)
        try:
            readings.extend(llm(prompt).readings)
        except Exception as exc:  # noqa: BLE001 - one failed chunk must not sink the rest
            logger.warning("Interpretation failed for claims %d-%d: %s", part[0].id, part[-1].id, exc)
            notes.append(
                f"Interpretation call failed for claims {part[0].id}-{part[-1].id} ({exc}); "
                "they are checked as extracted."
            )
    notes.extend(apply_readings(claims, readings, windows, ref_context, debate_date))
    return notes


def run_interpretation(
    claims: list[ExtractedClaim],
    lines: list[Line],
    settings: "Settings",
    *,
    briefing: DebateBriefing | None = None,
    debate_date: date | None = None,
    llm: Callable[[str], ClaimReadings] | None = None,
) -> list[str]:
    """Phase A2 entry point: flag-guarded, and never fails the run."""
    if not settings.interpretation_enabled or not claims:
        return []
    try:
        return interpret_claims(
            claims,
            lines,
            llm=llm or default_interpretation_llm(settings),
            briefing=briefing,
            debate_date=debate_date,
            window=settings.interpretation_window,
            chunk=settings.interpretation_chunk,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Interpretation failed; checking claims as extracted")
        clear_interpretation(claims, restore_claim=True)
        return [f"Interpretation failed ({exc}); claims checked as extracted."]


def default_interpretation_llm(settings: "Settings") -> Callable[[str], ClaimReadings]:
    from src.llm import generate_json

    model = settings.judge_model or settings.gemini_model
    return lambda prompt: generate_json(
        prompt, ClaimReadings, settings, label="interpretation", model=model
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_interpretation.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/interpretation.py tests/test_interpretation.py
git commit -m "feat: batched claim interpretation pass behind a flag"
```

---

### Task 5: Checker reading rules and related-claims block

**Files:**
- Modify: `src/agents.py` (`_CHECK_RULES` around line 1415, `_check_task_description` around line 1504, `build_check_crew` around line 1639)
- Test: `tests/test_related_context.py` (create)

**Interfaces:**
- Consumes: `ExtractedClaim.interpretation`, `.related_claim_ids` (Task 2)
- Produces in `src.agents`:
  - `_READING_RULES: str` (prepended into `_CHECK_RULES`)
  - `_related_block(claims: list[ExtractedClaim], pool: list[ExtractedClaim], facts_by_id: dict[int, VerifiedFact] | None = None) -> str`
  - `_check_task_description(header, claims, debate_date=None, related: str = "") -> str`
  - `build_check_crew(claims, settings, seen_urls, debate_date=None, pool: list[ExtractedClaim] | None = None)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_related_context.py`:

```python
"""Checkers and the manager see the recorded reading and related claims."""

from __future__ import annotations

from src.agents import (
    _CHECK_RULES,
    ExtractedClaim,
    Severity,
    Verdict,
    VerifiedFact,
    _check_task_description,
    _related_block,
)


def c(cid: int, related: list[int] | None = None, interpretation: str = "") -> ExtractedClaim:
    return ExtractedClaim(
        id=cid,
        claim=f"Tvrdenie {cid}.",
        speaker="Michal Šimečka",
        quote="q",
        related_claim_ids=related or [],
        interpretation=interpretation,
    )


def test_check_rules_bind_the_reading() -> None:
    for word in ("interpretation", "not_asserted", "refutation_would_be", "RELATED CLAIMS"):
        assert word in _CHECK_RULES
    assert "judge the quote" in _CHECK_RULES


def test_related_block_lists_only_outside_related_claims() -> None:
    pool = [c(1, [2, 3]), c(2, interpretation="Výnimka prešla."), c(3), c(4)]
    block = _related_block([pool[0], pool[2]], pool)
    assert block.startswith("RELATED CLAIMS (read-only")
    assert "#2 Michal Šimečka: Tvrdenie 2." in block
    assert "výklad: Výnimka prešla." in block
    assert "#3" not in block  # claim 3 is checked in this task, not context
    assert "#4" not in block
    assert _related_block([pool[3]], pool) == ""


def test_related_block_with_verdicts() -> None:
    pool = [c(1, [2]), c(2)]
    fact = VerifiedFact(
        claim="Tvrdenie 2.",
        speaker="Michal Šimečka",
        verdict=Verdict.FALSE,
        severity=Severity.MATERIAL,
        rationale="x" * 400,
    )
    block = _related_block([pool[0]], pool, {2: fact})
    assert block.startswith("RELATED CLAIMS AND CURRENT VERDICTS")
    assert "aktuálny verdikt: False / material." in block
    assert "x" * 300 in block and "x" * 301 not in block


def test_task_description_appends_related_block() -> None:
    d = _check_task_description("HEADER", [c(1)], None, related="RELATED CLAIMS (read-only): -")
    assert d.rstrip().endswith("RELATED CLAIMS (read-only): -")
    assert "RELATED CLAIMS (read-only)" not in _check_task_description("HEADER", [c(1)], None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_related_context.py -v`
Expected: FAIL with `ImportError: cannot import name '_related_block'`

- [ ] **Step 3: Add `_READING_RULES` and prepend it to `_CHECK_RULES`**

Directly above `_CHECK_RULES = (` in `src/agents.py`:

```python
_READING_RULES = (
    "READING OF THE CLAIM: a claim may carry `interpretation`, `not_asserted`, "
    "`refutation_would_be`, `referents` and `related_claim_ids`, recorded before "
    "any search from the transcript around the quote. Order of authority: the "
    "`quote` is what was said; the `interpretation` is the binding reading of "
    "the quote; `context` is only background on the debate moment. If you "
    "believe the interpretation misreads the quote, judge the quote and say so "
    "in the rationale. Build queries from the interpretation, the quote and "
    "`refutation_would_be`; entities named in `referents` count as present in "
    "the claim. Sub-facts decompose the interpretation. A `not_asserted` item "
    "is a reading the speaker did NOT make: never test it, never mark a "
    "sub-fact refuted or distorted because of it, and evidence against it is "
    "unusable. False or Misleading requires evidence of the shape "
    "`refutation_would_be` describes (same metric, period, actor and stage). "
    "RELATED CLAIMS, when listed, are read-only context: return no verdict for "
    "them.\n"
)
```

Then change the first line of `_CHECK_RULES` from:

```python
_CHECK_RULES = (
    "When you call a search tool, build the query from the claim text, its "
```

to:

```python
_CHECK_RULES = (
    _READING_RULES
    + "When you call a search tool, build the query from the claim text, its "
```

The rest of the `_CHECK_RULES` expression is a chain of adjacent string literals ending in `+ _SK_OUTPUT`; adjacent literals concatenate before `+`, so the result is `_READING_RULES + (all old text) + _SK_OUTPUT`.

- [ ] **Step 4: Add `_related_block` and the `related` argument**

Directly above `def _check_task_description(`:

```python
def _related_block(
    claims: list[ExtractedClaim],
    pool: list[ExtractedClaim],
    facts_by_id: "dict[int, VerifiedFact] | None" = None,
) -> str:
    """Read-only block of claims related to `claims` that are not among them."""
    own = {c.id for c in claims}
    by_id = {c.id: c for c in pool}
    ids = sorted({i for c in claims for i in c.related_claim_ids if i not in own and i in by_id})
    if not ids:
        return ""
    rows = []
    for i in ids:
        rc = by_id[i]
        row = f"- #{i} {rc.speaker}: {rc.claim}"
        if rc.interpretation:
            row += f"\n  výklad: {rc.interpretation}"
        fact = (facts_by_id or {}).get(i)
        if fact is not None:
            sev = f" / {fact.severity.value}" if fact.severity else ""
            row += f"\n  aktuálny verdikt: {fact.verdict.value}{sev}. {fact.rationale[:300]}"
        rows.append(row)
    header = "RELATED CLAIMS AND CURRENT VERDICTS" if facts_by_id is not None else "RELATED CLAIMS"
    return f"{header} (read-only context, return no verdict for them):\n" + "\n".join(rows)
```

Change `_check_task_description` to:

```python
def _check_task_description(
    header: str,
    claims: list[ExtractedClaim],
    debate_date: "date | None" = None,
    related: str = "",
) -> str:
    from datetime import date as _date

    ref = (debate_date or _date.today()).isoformat()
    return (
        f"DEBATE DATE (reference 'now'): {ref}. The speakers' statements refer to "
        f"this date; judge source freshness against it. A source or event dated "
        f"AFTER {ref} is anachronistic (it happened after the debate): it CANNOT "
        "be used as evidence and MUST NOT make a claim False. When a claim carries "
        "a time_reference/time_window, verify the event WITHIN that window relative "
        f"to {ref}; if the closest matching event has a different date or number, "
        "the verdict is Misleading (imprecise), NOT False for 'no such event'.\n\n"
        f"{header}\n{_CHECK_RULES}\n\n"
        "CLAIMS TO VERIFY (ExtractedClaims JSON):\n"
        f"{_claims_json(claims)}"
        + (f"\n\n{related}" if related else "")
    )
```

- [ ] **Step 5: Pass related claims in `build_check_crew`**

Change the signature:

```python
def build_check_crew(
    claims: list[ExtractedClaim],
    settings: Settings,
    seen_urls: set[str],
    debate_date: "date | None" = None,
    pool: list[ExtractedClaim] | None = None,
):
```

After `from src.tools_susr import build_susr_data_tool`, add:

```python
    pool = pool or claims
```

In `t_grounded`, change `description=_check_task_description(..., claims, debate_date,)` to pass `related=_related_block(claims, pool)`:

```python
    t_grounded = Task(
        description=_check_task_description(
            "Verify EVERY claim below with the vertex_grounded_search tool "
            "(Google Search grounded via Vertex AI). Prefer official data and "
            "high-FCRI Slovak mainstream sources in citations; do not rely on "
            "disinfo/gray-zone sites as proof.",
            claims,
            debate_date,
            related=_related_block(claims, pool),
        ),
```

In the specialist loop:

```python
        task = Task(
            description=_check_task_description(
                spec["header"], cat_claims, debate_date, related=_related_block(cat_claims, pool)
            ),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_related_context.py tests/test_subfacts.py tests/test_grounded_prompt.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/agents.py tests/test_related_context.py
git commit -m "feat: checkers judge the recorded reading and see related claims"
```

---

### Task 6: Manager consistency rule and related verdicts

**Files:**
- Modify: `src/agents.py` (`_MANAGER_RULES` around line 1745, `build_manager_crew` around line 1793)
- Test: `tests/test_related_context.py`

**Interfaces:**
- Consumes: `_related_block` (Task 5)
- Produces: `build_manager_crew(claims, prelim_facts_json, settings, seen_urls, debate_date=None, round_no=1, related: str = "")`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_related_context.py`:

```python
def test_manager_rules_require_consistency_with_related_verdicts() -> None:
    from src.agents import _MANAGER_RULES

    assert "CONSISTENCY" in _MANAGER_RULES
    assert "RELATED CLAIMS AND CURRENT VERDICTS" in _MANAGER_RULES


def test_manager_crew_accepts_related_block() -> None:
    import inspect

    from src.agents import build_manager_crew

    assert inspect.signature(build_manager_crew).parameters["related"].default == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_related_context.py -v`
Expected: the two new tests FAIL (`assert 'CONSISTENCY' in ...`, `KeyError: 'related'`)

- [ ] **Step 3: Add the rule**

In `_MANAGER_RULES`, after the step-5 ESCALATE text that ends with `"the rationale. Otherwise escalate=false.\n\n"`, insert:

```python
    "6) CONSISTENCY — RELATED CLAIMS AND CURRENT VERDICTS, when listed, are "
    "this debate's verdicts on other claims about the same fact. If your "
    "verdict contradicts one of them, reconcile the two or explain the "
    "difference in the rationale. Return no ManagerReview for them.\n\n"
```

- [ ] **Step 4: Add the `related` parameter**

Change the signature of `build_manager_crew`:

```python
def build_manager_crew(
    claims: list[ExtractedClaim],
    prelim_facts_json: str,
    settings: Settings,
    seen_urls: set[str],
    debate_date: "date | None" = None,
    round_no: int = 1,
    related: str = "",
):
```

and its task description:

```python
    task = Task(
        description=(
            _check_task_description(
                f"MANAGER REVIEW — ROUND {round_no}.\n{_MANAGER_RULES}",
                claims,
                debate_date,
            )
            + "\n\nFIRST-PASS RESULTS (verdict, sources, rationale per claim_id):\n"
            + prelim_facts_json
            + (f"\n\n{related}" if related else "")
        ),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_related_context.py tests/test_subfacts.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/agents.py tests/test_related_context.py
git commit -m "feat: manager rounds see related claims with current verdicts"
```

---

### Task 7: `check_claims` and wiring in `run_analysis`

**Files:**
- Modify: `src/agents.py` (new `check_claims` above `run_analysis`; `run_analysis` around lines 1938-2245)
- Test: `tests/test_related_context.py`

**Interfaces:**
- Consumes: `build_check_crew(..., pool=)` (Task 5), `build_manager_crew(..., related=)` (Task 6), `_related_block` (Task 5), `run_interpretation`, `clear_interpretation` (Tasks 3-4)
- Produces in `src.agents`: `check_claims(claims: list[ExtractedClaim], settings: Settings, debate_date: date | None = None, *, pool: list[ExtractedClaim] | None = None, pool_facts: dict[int, VerifiedFact] | None = None) -> tuple[list[VerifiedFact], set[str], list[str]]`. It returns one fact per claim, in claim order, plus `seen_urls` and the reconcile and manager notes.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_related_context.py`:

```python
def test_check_claims_signature_and_empty_input() -> None:
    import inspect

    from config import Settings
    from src.agents import check_claims

    params = inspect.signature(check_claims).parameters
    assert list(params)[:3] == ["claims", "settings", "debate_date"]
    assert params["pool"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["pool_facts"].kind is inspect.Parameter.KEYWORD_ONLY
    assert check_claims([], Settings(_env_file=None)) == ([], set(), [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_related_context.py::test_check_claims_signature_and_empty_input -v`
Expected: FAIL with `ImportError: cannot import name 'check_claims'`

- [ ] **Step 3: Add `check_claims`**

Directly above `def run_analysis(` in `src/agents.py`:

```python
def check_claims(
    claims: list[ExtractedClaim],
    settings: Settings,
    debate_date: "date | None" = None,
    *,
    pool: list[ExtractedClaim] | None = None,
    pool_facts: "dict[int, VerifiedFact] | None" = None,
) -> tuple[list[VerifiedFact], set[str], list[str]]:
    """Phase B + B2: grounded checker and specialists, reconcile, manager rounds.

    `pool` holds every claim that related_claim_ids may point to (default:
    `claims`); `pool_facts` holds verdicts for pool claims not checked here.
    Returns one fact per claim in claim order, the provenance URLs, and notes.
    """
    from src.reconcile import (
        apply_manager_reviews,
        merge_checklists,
        parse_task_output,
        prelim_facts_json,
        reconcile_checks,
    )

    seen_urls: set[str] = set()
    if not claims:
        return [], seen_urls, []
    pool = pool or claims

    b_result, b_tasks = kickoff_with_retry(
        lambda: build_check_crew(claims, settings, seen_urls, debate_date=debate_date, pool=pool),
        "check",
    )
    TRACKER.add_crew("analysis_check", b_result)
    grounded = parse_task_output(b_tasks["grounded"], ClaimCheckList)
    specialist_checks = merge_checklists(
        [parse_task_output(t, ClaimCheckList) for t in b_tasks["specialists"]]
    )
    facts, notes = reconcile_checks(
        ExtractedClaims(claims=claims), grounded, specialist_checks, seen_urls
    )
    logger.info(
        "Reconciled %d facts from extract=%d grounded=%d specialists=%d (seen_urls=%d)",
        len(facts),
        len(claims),
        len(grounded.checks) if grounded else 0,
        len(specialist_checks.checks) if specialist_checks else 0,
        len(seen_urls),
    )

    if facts and settings.factcheck_manager_rounds > 0:
        # reconcile_checks emits one fact per claim, in claim order.
        facts_by_id = {c.id: f for c, f in zip(claims, facts)}
        claim_by_id = {c.id: c for c in claims}
        review_ids = [c.id for c in claims]
        for round_no in range(1, settings.factcheck_manager_rounds + 1):
            round_claims = [claim_by_id[i] for i in review_ids]
            related = _related_block(round_claims, pool, {**(pool_facts or {}), **facts_by_id})
            m_result, m_task = kickoff_with_retry(
                lambda: build_manager_crew(
                    round_claims,
                    prelim_facts_json(round_claims, facts_by_id),
                    settings,
                    seen_urls,
                    debate_date=debate_date,
                    round_no=round_no,
                    related=related,
                ),
                f"manager r{round_no}",
            )
            TRACKER.add_crew(f"analysis_manager_r{round_no}", m_result)
            reviews = parse_task_output(m_task, ManagerReviewList)
            escalated, m_notes = apply_manager_reviews(
                round_claims, facts_by_id, reviews, seen_urls
            )
            notes.extend(f"[round {round_no}] {n}" for n in m_notes)
            logger.info(
                "Manager round %d: reviewed %d claims, escalated %d",
                round_no,
                len(round_claims),
                len(escalated),
            )
            review_ids = escalated
            if not review_ids:
                break
    return facts, seen_urls, notes
```

- [ ] **Step 4: Rewire `run_analysis`**

Make these five edits in `run_analysis`.

(a) In the docstring, change `moderator/extraction + question audit -> selection -> B) grounded checker +` to `moderator/extraction + question audit -> selection -> A2) interpretation -> B) grounded checker +`.

(b) Change the import line `from src.reconcile import merge_checklists, parse_task_output, reconcile_checks` to:

```python
    from src.interpretation import clear_interpretation, run_interpretation
    from src.reconcile import parse_task_output
```

(c) Directly after the `retry_empty_extraction` block (after `pipeline_notes.extend(retry_notes)`), add:

```python
    if extracted:
        # The extractor sees these fields in its schema; only code may fill them.
        clear_interpretation(extracted.claims)
```

(d) Replace the whole Phase B block:

```python
    # Phase B: grounded checker (all claims) + per-category specialists.
    seen_urls: set[str] = set()
    grounded = None
    specialist_checks = None
    if kept_claims:
        b_result, b_tasks = kickoff_with_retry(
            ...
        )
        TRACKER.add_crew("analysis_check", b_result)
        grounded = parse_task_output(b_tasks["grounded"], ClaimCheckList)
        specialist_checks = merge_checklists(
            [parse_task_output(t, ClaimCheckList) for t in b_tasks["specialists"]]
        )
```

with:

```python
    # Phase A2: record what each selected claim asserts before any search.
    pipeline_notes.extend(
        run_interpretation(
            kept_claims, working_lines, settings, briefing=briefing, debate_date=debate_date
        )
    )

    # Phase B + B2: grounded checker, specialists, reconcile, manager rounds.
    facts, seen_urls, check_notes = check_claims(kept_claims, settings, debate_date)
```

(e) Delete everything from `kept_extracted = ExtractedClaims(claims=kept_claims)` through the end of the manager `for round_no ...` loop (the `reconcile_checks` call, the `manager_notes` list and the whole Phase B2 block). Then change:

```python
    notes = [*pipeline_notes, *reconcile_notes, *manager_notes]
```

to:

```python
    notes = [*pipeline_notes, *check_notes]
```

and delete the `logger.info("Reconciled %d facts from extract=%s grounded=%s specialists=%s (seen_urls=%d)", ...)` call that follows it (it moved into `check_claims`). `report.facts = facts`, `validate_report(report, working, allowed_urls=seen_urls)` and the rest stay as they are.

- [ ] **Step 5: Verify no stale names remain**

Run: `rg -n "reconcile_notes|manager_notes|specialist_checks|grounded\b" src/agents.py`
Expected: matches only inside `check_claims` and `build_check_crew` (no use in `run_analysis`).

- [ ] **Step 6: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/agents.py tests/test_related_context.py
git commit -m "refactor: move fact checking into check_claims and wire interpretation"
```

---

### Task 8: Fidelity gate reads the recorded reading

**Files:**
- Modify: `src/fidelity.py` (`_INSTRUCTIONS`, `build_fidelity_prompt`, `audit_accusations`)
- Test: `tests/test_fidelity.py`

**Interfaces:**
- Consumes: `VerifiedFact.interpretation`, `.not_asserted` (Task 2)
- Produces: `audit_accusations(facts, lines, *, llm, siblings: list[VerifiedFact] | None = None) -> list[str]`. `siblings` defaults to `facts` and is what the prompt lists as other verdicts.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fidelity.py`:

```python
def test_prompt_carries_recorded_reading_when_present() -> None:
    f = fact()
    f.interpretation = "Návrh prešiel prvým čítaním a KDH zaň hlasovalo."
    f.not_asserted = ["návrh bol definitívne schválený"]
    p = build_fidelity_prompt(f, "", [f])
    assert "READING RECORDED BEFORE CHECKING (may be wrong)" in p
    assert f.interpretation in p and "návrh bol definitívne schválený" in p
    assert "READING RECORDED BEFORE CHECKING" not in build_fidelity_prompt(fact(), "", [])


def test_siblings_can_differ_from_audited_facts() -> None:
    target = fact()
    other = fact(Verdict.TRUE, ts="11:59", claim="Iné tvrdenie o daniach.")
    prompts: list[str] = []

    def llm(prompt: str) -> FidelityOutput:
        prompts.append(prompt)
        return out(Reading.SAME)

    audit_accusations([target], LINES, llm=llm, siblings=[target, other])
    assert len(prompts) == 1 and "Iné tvrdenie o daniach." in prompts[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_fidelity.py -v`
Expected: the two new tests FAIL (missing reading text; `TypeError: ... unexpected keyword argument 'siblings'`)

- [ ] **Step 3: Implement**

In `_INSTRUCTIONS`, before the final paragraph that starts `Pick same whenever the evidence`, add:

```
A READING RECORDED BEFORE CHECKING, when given, was written from the
transcript before any search and may be wrong: still write your own
`asserted`. When what the evidence refutes matches one of its "not
asserted" items, the reading is stronger_reading or different_referent.

```

In `build_fidelity_prompt`, compute the block and insert it after the `CLAIM AS EXTRACTED` line:

```python
    recorded = ""
    if fact.interpretation:
        recorded = (
            "READING RECORDED BEFORE CHECKING (may be wrong):\n"
            f"  interpretation: {fact.interpretation}\n"
            f"  not asserted: {'; '.join(fact.not_asserted) or '(none)'}\n"
        )
    return (
        f"{_INSTRUCTIONS}\n"
        f"TRANSCRIPT AROUND THE QUOTE:\n{window or '(unavailable)'}\n\n"
        f"SPEAKER: {fact.speaker}\n"
        f"QUOTE: {fact.quote}\n"
        f"CLAIM AS EXTRACTED: {fact.claim}\n"
        f"{recorded}"
        f"VERDICT: {fact.verdict.value}"
        f"{' / ' + fact.severity.value if fact.severity else ''}\n"
        f"RATIONALE: {fact.rationale[:1500]}\n\n"
        f"OTHER VERDICTS IN THIS DEBATE:\n{siblings or '(none)'}\n"
    )
```

Change `audit_accusations`:

```python
def audit_accusations(
    facts: list[VerifiedFact],
    lines: list[Line],
    *,
    llm: Callable[[str], FidelityOutput],
    siblings: list[VerifiedFact] | None = None,
) -> list[str]:
    """Downgrade, in place, every accusation whose rationale misses the assertion."""
    notes: list[str] = []
    context = facts if siblings is None else siblings
    for fact in facts:
        if fact.verdict not in _ACCUSATIONS:
            continue
        prompt = build_fidelity_prompt(fact, transcript_window(lines, fact), context)
```

The rest of the loop is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_fidelity.py tests/test_eval.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/fidelity.py tests/test_fidelity.py
git commit -m "feat: fidelity gate weighs the reading recorded before checking"
```

---

### Task 9: Replay harness `interpret-replay`

**Files:**
- Modify: `src/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `interpret_claims` (Task 4), `check_claims` (Task 7), `audit_accusations(..., siblings=)` (Task 8), `ExtractedClaim`, `ClaimCategory`, `VerifiedFact`, `DebateBriefing`
- Produces in `src.eval`:
  - `class CategoryAssignment(BaseModel)`: `claim_id: int`, `category: ClaimCategory`
  - `class CategoryList(BaseModel)`: `items: list[CategoryAssignment]`
  - `rebuild_claims(facts: list[VerifiedFact]) -> list[ExtractedClaim]` (id = index + 1)
  - `assign_categories(claims: list[ExtractedClaim], facts: list[VerifiedFact], llm: Callable[[str], CategoryList]) -> None`
  - `Recheck = Callable[[list[ExtractedClaim], list[ExtractedClaim], dict[int, VerifiedFact], date | None], list[VerifiedFact]]`
  - `replay_report(report: str, cases: list[dict], reports_dir: Path, transcripts_dir: Path, dates: dict[str, str], *, interpret_llm, category_llm, recheck: Recheck, gate_llm, no_check: bool = False, window: int = 6, chunk: int = 20, run: int = 1) -> dict` returning `{"cases": [...], "stale": [...], "notes": [...]}`
  - `interpret_replay(gold: list[dict], reports_dir: Path, transcripts_dir: Path, dates: dict[str, str], *, interpret_llm, category_llm, recheck, gate_llm, labels: set[str], only: set[str] | None = None, runs: int = 1, no_check: bool = False, window: int = 6, chunk: int = 20) -> dict` returning `{"summary": {label: {"n", "accusation"}}, "cases", "stale", "notes"}`
  - CLI `python -m src.eval interpret-replay`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval.py`:

```python
# --- interpret-replay ---------------------------------------------------------------


def _replay_fixture(tmp_path: Path):
    reports, transcripts = tmp_path / "r", tmp_path / "t"
    reports.mkdir()
    transcripts.mkdir()
    facts = [
        {**_fact("False"), "claim": "Zrušenie dane prešlo.", "speaker": "Michal Šimečka",
         "quote": "o transakčnej dani a jej zrušení pre živnostníkov", "timestamp": "09:04"},
        {**_fact("Misleading"), "claim": "Druhé tvrdenie.", "timestamp": "09:20"},
        {**_fact("True"), "claim": "Tretie tvrdenie."},
    ]
    _write_report(reports, "100", facts, debate_date="2026-09-27")
    (transcripts / "100.corrected.txt").write_text(
        "Michal Šimečka [09:04]: Keď sa hlasovalo o transakčnej dani a jej zrušení pre živnostníkov, hlasovali sme za.\n"
        "X [09:20]: q\n",
        encoding="utf-8",
    )
    gold = [
        {"report": "100", "index": 0, "claim_prefix": "Zrušenie dane", "label": "downgrade"},
        {"report": "100", "index": 1, "claim_prefix": "Druhé", "label": "keep"},
        {"report": "100", "index": 2, "claim_prefix": "Zmenené", "label": "keep"},  # stale
        {"report": "100", "index": 1, "claim_prefix": "Druhé", "label": "doubtful"},
    ]
    return gold, reports, transcripts


def _interpret_llm(_prompt: str):
    from src.interpretation import ClaimReading, ClaimReadings, Drift, DriftKind

    return ClaimReadings(
        readings=[
            ClaimReading(
                claim_id=1,
                interpretation="Výnimka z dane pre živnostníkov prešla.",
                drift=[Drift(kind=DriftKind.DROPPED_QUALIFIER, detail="pre živnostníkov")],
                claim="Zrušenie dane pre živnostníkov prešlo.",
                not_asserted=["úplné zrušenie dane"],
                related_claim_ids=[2],
            )
        ]
    )


def _category_llm(_prompt: str):
    from src.agents import ClaimCategory
    from src.eval import CategoryAssignment, CategoryList

    return CategoryList(items=[CategoryAssignment(claim_id=i, category=ClaimCategory.HISTORY_POLITICS) for i in (1, 2, 3)])


def test_interpret_replay_rechecks_targets_and_scores_labels(tmp_path: Path) -> None:
    from src.agents import Verdict, VerifiedFact
    from src.eval import interpret_replay
    from src.fidelity import FidelityOutput, Reading

    gold, reports, transcripts = _replay_fixture(tmp_path)
    calls: list[tuple[list[int], list[int], list[int]]] = []

    def recheck(targets, pool, pool_facts, debate):
        calls.append(([c.id for c in targets], [c.id for c in pool], sorted(pool_facts)))
        assert debate.isoformat() == "2026-09-27"
        assert all(c.category.value == "history_politics" for c in targets)
        verdicts = {1: Verdict.UNVERIFIED, 2: Verdict.MISLEADING}
        return [
            VerifiedFact(claim=c.claim, speaker=c.speaker, quote=c.quote, verdict=verdicts[c.id],
                         severity="material" if verdicts[c.id] == Verdict.MISLEADING else None,
                         sources=["https://a.sk/x"], interpretation=c.interpretation)
            for c in targets
        ]

    gate_prompts: list[str] = []

    def gate(prompt: str) -> FidelityOutput:
        gate_prompts.append(prompt)
        return FidelityOutput(asserted="a", refuted="r", reading=Reading.SAME)

    out = interpret_replay(
        gold, reports, transcripts, {},
        interpret_llm=_interpret_llm, category_llm=_category_llm, recheck=recheck, gate_llm=gate,
        labels={"downgrade", "keep"},
    )
    assert calls == [([1, 2], [1, 2, 3], [1, 2, 3])]
    assert out["stale"] == ["100#2"]
    assert out["summary"]["downgrade"] == {"n": 1, "accusation": 0}
    assert out["summary"]["keep"] == {"n": 1, "accusation": 1}
    first = next(c for c in out["cases"] if c["index"] == 0)
    assert first["before"] == "False" and first["after"] == "Unverified"
    assert first["claim_after"] == "Zrušenie dane pre živnostníkov prešlo."
    assert first["interpretation"].startswith("Výnimka")
    assert first["not_asserted"] == ["úplné zrušenie dane"]
    assert len(gate_prompts) == 1  # only the re-checked accusation is audited
    assert "Tretie tvrdenie." in gate_prompts[0]  # stored siblings are context


def test_interpret_replay_no_check_reads_only(tmp_path: Path) -> None:
    from src.eval import interpret_replay

    gold, reports, transcripts = _replay_fixture(tmp_path)

    def never(*_a, **_kw):
        raise AssertionError("must not be called")

    out = interpret_replay(
        gold, reports, transcripts, {},
        interpret_llm=_interpret_llm, category_llm=never, recheck=never, gate_llm=never,
        labels={"downgrade"}, no_check=True,
    )
    assert [c["index"] for c in out["cases"]] == [0]
    assert out["cases"][0]["after"] is None
    assert out["cases"][0]["claim_after"] == "Zrušenie dane pre živnostníkov prešlo."


def test_interpret_replay_only_and_runs(tmp_path: Path) -> None:
    from src.eval import interpret_replay

    gold, reports, transcripts = _replay_fixture(tmp_path)
    out = interpret_replay(
        gold, reports, transcripts, {},
        interpret_llm=_interpret_llm, category_llm=_category_llm, recheck=None, gate_llm=None,
        labels={"downgrade", "keep"}, only={"100#0"}, runs=2, no_check=True,
    )
    assert [(c["index"], c["run"]) for c in out["cases"]] == [(0, 1), (0, 2)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: the three new tests FAIL with `ImportError: cannot import name 'interpret_replay'`

- [ ] **Step 3: Implement the replay functions**

In `src/eval.py`, extend the imports:

```python
from pydantic import BaseModel

from src.agents import ClaimCategory, ExtractedClaim, Verdict, VerifiedFact
```

(replacing the existing `from src.agents import VerifiedFact`).

Add after `fidelity_eval`:

```python
_ACCUSATIONS = (Verdict.FALSE, Verdict.MISLEADING)


class CategoryAssignment(BaseModel):
    claim_id: int
    category: ClaimCategory


class CategoryList(BaseModel):
    items: list[CategoryAssignment] = []


Recheck = Callable[
    [list[ExtractedClaim], list[ExtractedClaim], dict[int, VerifiedFact], date | None],
    list[VerifiedFact],
]


def rebuild_claims(facts: list[VerifiedFact]) -> list[ExtractedClaim]:
    """Stored facts as claims (id = index + 1); time fields are not stored and stay empty."""
    return [
        ExtractedClaim(
            id=i + 1,
            claim=f.claim,
            speaker=f.speaker,
            quote=f.quote,
            salience=f.salience,
            consequence=f.salience,
            category=f.category or ClaimCategory.CURRENT_EVENTS,
        )
        for i, f in enumerate(facts)
    ]


def assign_categories(
    claims: list[ExtractedClaim],
    facts: list[VerifiedFact],
    llm: Callable[[str], CategoryList],
) -> None:
    """Classify claims whose stored fact has no category, for specialist routing."""
    missing = [c for c, f in zip(claims, facts) if f.category is None]
    if not missing:
        return
    prompt = (
        "Assign each claim one category for specialist fact-checker routing:\n"
        "- economy_finance: budget, deficit, debt, consolidation, taxes, salaries of "
        "officials, public spending.\n"
        "- statistics: official statistical indicators (inflation/CPI, wages, "
        "unemployment, GDP, demographics).\n"
        "- history_politics: parliamentary votes, laws, past governments, past "
        "statements or positions.\n"
        "- current_events: recent events, quote attributions, media affairs.\n\n"
        + "\n".join(f"{c.id}. {c.claim}" for c in missing)
    )
    by_id = {c.id: c for c in missing}
    for item in llm(prompt).items:
        if item.claim_id in by_id:
            by_id[item.claim_id].category = item.category


def replay_report(
    report: str,
    cases: list[dict],
    reports_dir: Path,
    transcripts_dir: Path,
    dates: dict[str, str],
    *,
    interpret_llm,
    category_llm,
    recheck: Recheck | None,
    gate_llm,
    no_check: bool = False,
    window: int = 6,
    chunk: int = 20,
    run: int = 1,
) -> dict:
    """Re-interpret one stored report and re-check its gold cases."""
    from src.briefing import DebateBriefing
    from src.interpretation import interpret_claims
    from src.transcript_lines import parse_lines

    payload = json.loads((reports_dir / f"{report}.json").read_text(encoding="utf-8"))
    facts = [VerifiedFact.model_validate(f) for f in payload.get("facts", [])]
    debate = parse_iso_date(payload.get("debate_date") or dates.get(base_id(report) or "", ""))
    briefing = (
        DebateBriefing.model_validate(payload["briefing"]) if payload.get("briefing") else None
    )
    tpath = transcripts_dir / f"{report}.corrected.txt"
    lines = parse_lines(tpath.read_text(encoding="utf-8")) if tpath.exists() else []

    valid: list[dict] = []
    stale: list[str] = []
    for case in cases:
        i = case["index"]
        if not (0 <= i < len(facts)) or not facts[i].claim.startswith(case["claim_prefix"]):
            stale.append(f"{report}#{i}")
        else:
            valid.append(case)

    claims = rebuild_claims(facts)
    notes = interpret_claims(
        claims, lines, llm=interpret_llm, briefing=briefing, debate_date=debate,
        window=window, chunk=chunk,
    )
    after: dict[int, VerifiedFact] = {}
    if valid and not no_check:
        assert recheck is not None
        assign_categories(claims, facts, category_llm)
        targets = [claims[c["index"]] for c in valid]
        pool_facts = {i + 1: f for i, f in enumerate(facts)}
        new_facts = recheck(targets, claims, pool_facts, debate)
        for case, nf in zip(valid, new_facts):
            nf.timestamp = facts[case["index"]].timestamp
            after[case["index"]] = nf
        siblings = [after.get(i, f) for i, f in enumerate(facts)]
        notes.extend(audit_accusations(list(after.values()), lines, llm=gate_llm, siblings=siblings))

    rows = []
    for case in valid:
        i = case["index"]
        c, nf = claims[i], after.get(i)
        rows.append(
            {
                **case,
                "run": run,
                "claim_after": c.claim,
                "interpretation": c.interpretation,
                "not_asserted": c.not_asserted,
                "refutation_would_be": c.refutation_would_be,
                "before": facts[i].verdict.value,
                "after": nf.verdict.value if nf else None,
                "rationale": nf.rationale if nf else "",
            }
        )
    return {"cases": rows, "stale": stale, "notes": notes}


def interpret_replay(
    gold: list[dict],
    reports_dir: Path,
    transcripts_dir: Path,
    dates: dict[str, str],
    *,
    interpret_llm,
    category_llm,
    recheck: Recheck | None,
    gate_llm,
    labels: set[str],
    only: set[str] | None = None,
    runs: int = 1,
    no_check: bool = False,
    window: int = 6,
    chunk: int = 20,
) -> dict:
    """Re-interpret and re-check gold cases; count which stay accusations per label."""
    picked = [
        c
        for c in gold
        if c["label"] in labels and (not only or f"{c['report']}#{c['index']}" in only)
    ]
    by_report: dict[str, list[dict]] = {}
    for case in picked:
        by_report.setdefault(case["report"], []).append(case)

    cases: list[dict] = []
    stale: list[str] = []
    notes: list[str] = []
    for run in range(1, runs + 1):
        for report, report_cases in by_report.items():
            r = replay_report(
                report, report_cases, reports_dir, transcripts_dir, dates,
                interpret_llm=interpret_llm, category_llm=category_llm, recheck=recheck,
                gate_llm=gate_llm, no_check=no_check, window=window, chunk=chunk, run=run,
            )
            cases.extend(r["cases"])
            notes.extend(f"[run {run}] {report}: {n}" for n in r["notes"])
            if run == 1:
                stale.extend(r["stale"])

    accusations = {v.value for v in _ACCUSATIONS}
    summary = {
        label: {
            "n": sum(1 for c in cases if c["label"] == label),
            "accusation": sum(
                1 for c in cases if c["label"] == label and c["after"] in accusations
            ),
        }
        for label in sorted(labels)
    }
    return {"summary": summary, "cases": cases, "stale": stale, "notes": notes}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: all PASS

- [ ] **Step 5: Add the CLI command**

Add before `if __name__ == "__main__":` in `src/eval.py`:

```python
@cli.command("interpret-replay")
@click.option("--gold", "gold_path", type=click.Path(path_type=Path, exists=True), default=Path("data/eval/fidelity_gold.json"))
@click.option("--reports", "reports_dir", type=click.Path(path_type=Path, exists=True), default=Path("data/reports"))
@click.option("--transcripts", "transcripts_dir", type=click.Path(path_type=Path, exists=True), default=Path("data/transcripts"))
@click.option("--dates", "dates_path", type=click.Path(path_type=Path), default=Path("data/debate_dates.json"))
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path("data/eval/interpret_replay.json"))
@click.option("--labels", default="downgrade,keep", help="Comma-separated gold labels to replay.")
@click.option("--only", multiple=True, help="Gold case(s) as REPORT#INDEX; default all with --labels.")
@click.option("--runs", default=1, type=int, help="Repeat the replay to expose variance.")
@click.option("--no-check", is_flag=True, help="Only re-interpret; print readings, no re-check.")
def interpret_replay_cmd(
    gold_path: Path,
    reports_dir: Path,
    transcripts_dir: Path,
    dates_path: Path,
    out_path: Path,
    labels: str,
    only: tuple[str, ...],
    runs: int,
    no_check: bool,
) -> None:
    """Re-interpret and re-check gold accusations without a full pipeline run."""
    from config import get_settings
    from src.agents import check_claims
    from src.costs import TRACKER
    from src.fidelity import default_fidelity_llm
    from src.interpretation import default_interpretation_llm
    from src.llm import generate_json

    settings = get_settings()
    TRACKER.reset()
    model = settings.judge_model or settings.gemini_model

    def category_llm(prompt: str) -> CategoryList:
        return generate_json(prompt, CategoryList, settings, label="replay_category", model=model)

    def recheck(targets, pool, pool_facts, debate):
        facts, _urls, _notes = check_claims(targets, settings, debate, pool=pool, pool_facts=pool_facts)
        return facts

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    result = interpret_replay(
        gold,
        reports_dir,
        transcripts_dir,
        load_dates(dates_path),
        interpret_llm=default_interpretation_llm(settings),
        category_llm=category_llm,
        recheck=recheck,
        gate_llm=default_fidelity_llm(settings),
        labels={s.strip() for s in labels.split(",") if s.strip()},
        only=set(only) or None,
        runs=runs,
        no_check=no_check,
        window=settings.interpretation_window,
        chunk=settings.interpretation_chunk,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for c in result["cases"]:
        click.echo(
            f"[run {c['run']}] {c['report']}#{c['index']} {c['label']:<9} "
            f"{c['before']} -> {c['after'] or '-'}\n"
            f"    claim: {c['claim_after']}\n"
            f"    reading: {c['interpretation'] or '(none)'}\n"
            f"    not asserted: {'; '.join(c['not_asserted']) or '(none)'}"
        )
    for label, s in result["summary"].items():
        click.echo(f"{label:<10} n={s['n']:>3} still_accusation={s['accusation']:>3}")
    if result["stale"]:
        click.echo("Stale gold cases (report changed): " + ", ".join(result["stale"]))
    click.echo(TRACKER.summary(model))
    click.echo(f"Written: {out_path}")
```

- [ ] **Step 6: Check the CLI loads**

Run: `.venv/bin/python -m src.eval interpret-replay --help`
Expected: the help text lists `--labels`, `--only`, `--runs`, `--no-check`; exit 0.

- [ ] **Step 7: Run the suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS

```bash
git add src/eval.py tests/test_eval.py
git commit -m "feat: interpret-replay re-checks gold accusations with readings"
```

---

### Task 10: Record planning decisions in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-11-05-claim-interpretation-design.md`

Three decisions made while planning change details of the spec. Record them so the spec matches the code.

- [ ] **Step 1: Number guard source**

In the section "`apply_readings` — guards in code", item 2, replace "or the `resolved` text of a briefing-origin referent. A briefing-origin referent's `resolved` text is itself accepted only if each of its digit tokens occurs in the rendered briefing." with:

```
the debate date, or the `resolved` text of a briefing-origin referent. A
briefing-origin referent's `resolved` text is itself accepted only if each of
its digit tokens occurs in the referent context the pass received
(participants, `entity_index`, glossary terms and definitions). Glossary
`status` is left out of that context because it states facts.
```

- [ ] **Step 2: Replay step 4**

In "Evaluation and replay", replace step 4 with:

```
4. Re-check only the target claims through `check_claims` (grounded checker
   plus specialists, reconcile, manager rounds), copy each stored fact's
   `timestamp` onto its new fact, then run the fidelity gate on the re-checked
   facts with the report's other stored facts as siblings
   (`audit_accusations(..., siblings=...)`). `enforce_accusation_support` is
   skipped: stored transcripts no longer carry the edit-to-line mapping it
   needs, and the targets' quotes, which are unchanged, already passed it.
```

- [ ] **Step 3: Extractor schema and `check_claims`**

At the end of "### Phase A2 — Interpretation pass", add:

```
The new `ExtractedClaim` fields appear in the extractor's output schema, so
`run_analysis` clears them right after extraction (`clear_interpretation`);
only the pass may fill them. Phase B and B2 move from `run_analysis` into
`check_claims(claims, settings, debate_date, *, pool, pool_facts)`, which the
replay reuses.
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-11-05-claim-interpretation-design.md
git commit -m "docs: record planning decisions in the interpretation spec"
```

---

### Task 11: Live replay, acceptance, default flip

This task spends money on Vertex: about $0.05 per report for `--no-check`, and $3–5 plus 1–1.5 h for the full replay. **Ask the user before each paid run.**

**Files:**
- Modify: `config.py` (default flip, only if acceptance passes)
- Modify: `tests/test_config.py`
- Modify: `docs/superpowers/specs/2026-11-05-claim-interpretation-design.md` (results)

- [ ] **Step 1: Readings for the three misses**

Run: `.venv/bin/python -m src.eval interpret-replay --no-check --labels downgrade --only ta3-1074913#9 --only 2898725#4 --only 623275#17`
Expected, checked by reading the output:
- ta3-1074913#9: claim mentions živnostníci; `not_asserted` names full abolition.
- 2898725#4: the interpretation is about debt growth since the government took office; `not_asserted` rules out the packages' own total.
- 623275#17: "my" resolves to the government, not the ministry.

If a reading is wrong, adjust `_INSTRUCTIONS` in `src/interpretation.py` and repeat this step. Keep `tests/test_interpretation.py` green after each prompt change.

- [ ] **Step 2: Readings for the keep cases**

Run: `.venv/bin/python -m src.eval interpret-replay --no-check --labels keep`
Expected: no keep-case reading adds a qualifier, or a number the quote does not contain, and no rewrite makes a sound accusation's claim weaker. Fix the prompt and go back to Step 1 if it does.

- [ ] **Step 3: Full replay**

Run: `.venv/bin/python -m src.eval interpret-replay --labels downgrade,keep --out data/eval/interpret_replay.json`
Then: `.venv/bin/python -m src.eval interpret-replay --labels downgrade --runs 2 --only ta3-1074913#9 --only 2898725#4 --only 623275#17 --out data/eval/interpret_replay_misses.json`

Acceptance (from the spec):
- In both runs, all 3 misses end up neither False nor Misleading, or they stay False/Misleading and the user reviews the new reading and rationale and marks them sound.
- `keep` `still_accusation` is at least 32 of 33.
- None of the 12 other `downgrade` cases is False or Misleading.

If it fails, report the failing cases to the user with their readings and rationales and stop. Do not flip the default.

- [ ] **Step 4: Flip the default (only after acceptance)**

In `tests/test_config.py`, rename the test from Task 1 to `test_interpretation_on_by_default_with_window_and_chunk` and change it to `assert s.interpretation_enabled is True`. Run it: it should FAIL. Then in `config.py` set `interpretation_enabled: bool = True`. Run: `.venv/bin/python -m pytest -q`. Everything should PASS.

- [ ] **Step 5: Record results in the spec**

Append a section `## Results` to the spec. Fill it with the numbers from `data/eval/interpret_replay.json` and `interpret_replay_misses.json`: per-label `n` and `still_accusation`, the verdict of each of the 3 misses in each run, cost and wall-clock from the CLI output, and any keep case that was lost, with its reading.

- [ ] **Step 6: Commit**

```bash
git add config.py tests/test_config.py docs/superpowers/specs/2026-11-05-claim-interpretation-design.md
git commit -m "feat: enable claim interpretation by default after replay acceptance"
```
