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
