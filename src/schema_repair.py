"""Turn free-form analyst JSON into the report schema as soon as it is rendered.

The behavioral and moderator agents often return nested objects. The report
schema wants strings. Repair that here, then hand the critic schema-valid JSON.
If the critic still emits empty lists, put the repaired sections back.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src.agents import (
    BehavioralAnalysis,
    Civility,
    ModeratorAudit,
    SpeakerTactics,
    TimeShare,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _load_json(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    except json.JSONDecodeError:
        return None
    return obj


def _labeled(item: dict, *, label_keys: tuple[str, ...], type_keys: tuple[str, ...]) -> str:
    label = next(
        (str(item[k]).strip() for k in label_keys if str(item.get(k) or "").strip()),
        "",
    ).rstrip(" .")
    kind = next(
        (str(item[k]).strip() for k in type_keys if str(item.get(k) or "").strip()),
        "",
    )
    evidence = " ".join(str(item.get("evidence") or "").split()).replace('"', "'")
    head = f"{kind}: {label}" if kind and label else (kind or label)
    if evidence:
        return f'{head}. "{evidence}"' if head else f'"{evidence}"'
    return head


def _entries(
    items: Any,
    *,
    label_keys: tuple[str, ...],
    type_keys: tuple[str, ...] = (),
) -> list[str]:
    out: list[str] = []
    for item in items or []:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            text = _labeled(item, label_keys=label_keys, type_keys=type_keys)
            if text:
                out.append(text)
    return out


def _flatten_speaker(obj: dict) -> SpeakerTactics:
    name = str(obj.get("speaker") or obj.get("speaker_name") or "").strip()
    manipulation = _entries(obj.get("manipulation"), label_keys=("finding",), type_keys=("tactic_type",))
    fallacies = _entries(
        obj.get("logical_fallacies"), label_keys=("finding",), type_keys=("fallacy_type",)
    )
    for item in obj.get("manipulation_and_fallacies") or []:
        if isinstance(item, str) and item.strip():
            manipulation.append(item.strip())
        elif isinstance(item, dict):
            if item.get("fallacy_type"):
                fallacies.append(
                    _labeled(item, label_keys=("finding",), type_keys=("fallacy_type",))
                )
            else:
                manipulation.append(
                    _labeled(item, label_keys=("finding",), type_keys=("tactic_type",))
                )
    civ_raw = obj.get("civility") or {}
    if isinstance(civ_raw, dict):
        try:
            score = float(civ_raw.get("score", 5.0))
        except (TypeError, ValueError):
            score = 5.0
        civility = Civility(
            score=score,
            incivility=_entries(civ_raw.get("incivility"), label_keys=("violation", "finding")),
            fair_conduct=_entries(civ_raw.get("fair_conduct"), label_keys=("behavior", "finding")),
            notes=str(civ_raw.get("notes") or civ_raw.get("rationale") or "").strip(),
        )
    else:
        civility = Civility()
    return SpeakerTactics(
        speaker=name,
        speech_tactics=_entries(
            obj.get("speech_tactics"), label_keys=("finding",), type_keys=("tactic_type",)
        ),
        manipulation=manipulation,
        logical_fallacies=fallacies,
        question_dodging=_entries(
            obj.get("question_dodging"),
            label_keys=("finding",),
            type_keys=("evasion_tactic",),
        ),
        civility=civility,
        notes=str(obj.get("notes") or "").strip(),
    )


def _unwrap_behavioral(data: Any) -> Any:
    if isinstance(data, dict) and isinstance(data.get("behavioral_analysis"), (dict, list)):
        if "speakers" not in data:
            return data["behavioral_analysis"]
    return data


def coerce_behavioral(raw: str) -> BehavioralAnalysis:
    """Validate analyst JSON. On schema failure, flatten nested objects to strings."""
    node = _unwrap_behavioral(_load_json(raw))
    if isinstance(node, dict):
        try:
            return BehavioralAnalysis.model_validate(node)
        except ValidationError:
            speakers_raw = node.get("speakers")
            if not isinstance(speakers_raw, list) and (
                node.get("speaker") or node.get("speaker_name")
            ):
                speakers_raw = [node]
            summary = str(node.get("summary") or "")
    elif isinstance(node, list):
        speakers_raw = node
        summary = ""
    else:
        return BehavioralAnalysis()
    speakers = [
        _flatten_speaker(item) for item in speakers_raw or [] if isinstance(item, dict)
    ]
    speakers = [s for s in speakers if s.speaker]
    return BehavioralAnalysis(speakers=speakers, summary=summary)


def _interruption_text(node: dict) -> str:
    freq = node.get("interruption_frequency")
    if isinstance(freq, str):
        return freq.strip()
    if isinstance(freq, list):
        return _format_interruptions(freq)
    analysis = node.get("interruption_analysis")
    if isinstance(analysis, dict):
        nested = analysis.get("interruption_frequency")
        if isinstance(nested, str):
            return nested.strip()
        if isinstance(nested, list):
            return _format_interruptions(nested)
    return ""


def _format_interruptions(rows: list) -> str:
    bits: list[str] = []
    for row in rows:
        if isinstance(row, str) and row.strip():
            bits.append(row.strip())
        elif isinstance(row, dict):
            who = str(row.get("interrupted_speaker") or row.get("speaker") or "").strip()
            count = row.get("interruptions_by_moderator", row.get("count", ""))
            if who:
                bits.append(f"{who}: {count}")
    return ", ".join(bits)


def _shares(node: dict) -> list[TimeShare]:
    raw = node.get("equal_time_distribution")
    if isinstance(raw, list) and raw and all(isinstance(row, dict) and row.get("speaker") for row in raw):
        shares: list[TimeShare] = []
        for row in raw:
            try:
                pct = float(row.get("approximate_share_percent") or 0)
            except (TypeError, ValueError):
                pct = 0.0
            try:
                turns = int(row.get("turns") or 0)
            except (TypeError, ValueError):
                turns = 0
            shares.append(
                TimeShare(
                    speaker=str(row["speaker"]),
                    approximate_share_percent=pct,
                    turns=turns,
                )
            )
        return shares
    analysis = node.get("speaking_time_analysis")
    if not isinstance(analysis, dict):
        return []
    shares = []
    for row in analysis.get("participants") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("speaker") or "").strip()
        if not name:
            continue
        try:
            pct = float(row.get("share_percentage", row.get("approximate_share_percent", 0)) or 0)
        except (TypeError, ValueError):
            pct = 0.0
        shares.append(TimeShare(speaker=name, approximate_share_percent=pct))
    return shares


def _unwrap_moderator(data: Any) -> Any:
    if isinstance(data, dict) and isinstance(data.get("moderator_audit"), dict):
        return data["moderator_audit"]
    return data


def coerce_moderator(raw: str) -> ModeratorAudit:
    """Validate auditor JSON. Map alternate keys and object findings into the schema."""
    node = _unwrap_moderator(_load_json(raw))
    if not isinstance(node, dict):
        return ModeratorAudit()
    try:
        score = float(node.get("neutrality_score", 5.0))
    except (TypeError, ValueError):
        score = 5.0
    summary = str(node.get("summary") or node.get("neutrality_rationale") or "").strip()
    return ModeratorAudit(
        neutrality_score=score,
        interruption_frequency=_interruption_text(node),
        equal_time_distribution=_shares(node),
        findings=_entries(node.get("findings"), label_keys=("finding",), type_keys=("verdict",)),
        summary=summary,
    )


def _has_content(model: BaseModel) -> bool:
    if isinstance(model, BehavioralAnalysis):
        return bool(model.speakers)
    if isinstance(model, ModeratorAudit):
        return bool(
            model.findings
            or model.summary.strip()
            or model.interruption_frequency.strip()
            or model.equal_time_distribution
            or model.neutrality_score != 5.0
        )
    return False


def for_critic(
    raw: str,
    coerce: Callable[[str], BaseModel],
) -> tuple[str, BaseModel]:
    """Schema JSON for the critic when repair found content; otherwise the raw text."""
    coerced = coerce(raw)
    if _has_content(coerced):
        return coerced.model_dump_json(indent=2), coerced
    return raw, coerced


def restore_empty_sections(
    report,
    behavioral: BehavioralAnalysis,
    moderator: ModeratorAudit,
) -> list[str]:
    """Put repaired sections back when the critic emitted empty-but-valid lists."""
    notes: list[str] = []
    if not report.behavioral_analysis.speakers and behavioral.speakers:
        report.behavioral_analysis.speakers = [
            speaker.model_copy(deep=True) for speaker in behavioral.speakers
        ]
        if not report.behavioral_analysis.summary.strip() and behavioral.summary.strip():
            report.behavioral_analysis.summary = behavioral.summary
        notes.append(
            "Behavioral analysis restored from analyst JSON after critic returned no speakers."
        )
    if not report.moderator_audit.findings and moderator.findings:
        report.moderator_audit.findings = list(moderator.findings)
        if report.moderator_audit.neutrality_score == 5.0 and moderator.neutrality_score != 5.0:
            report.moderator_audit.neutrality_score = moderator.neutrality_score
        if not report.moderator_audit.summary.strip() and moderator.summary.strip():
            report.moderator_audit.summary = moderator.summary
        if (
            not report.moderator_audit.interruption_frequency.strip()
            and moderator.interruption_frequency.strip()
        ):
            report.moderator_audit.interruption_frequency = moderator.interruption_frequency
        notes.append(
            "Moderator findings restored from auditor JSON after critic returned none."
        )
    return notes
