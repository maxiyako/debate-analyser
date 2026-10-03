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

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from src.agents import AnalysisReport, Severity, TimeShare, Verdict
from src.questions import question_counts, render_dodges
from src.report_models import QuestionItem
from src.selection import refresh_funnel_verdicts
from src.speakers import canonical_speaker
from src.transcript_lines import parse_lines, strip_markers, ts_seconds

_SPEAKER_ID = re.compile(r"\bspeaker\s+([a-z0-9]+)\b", re.IGNORECASE)
_TERMINAL_PUNCT = (".", "!", "?", "…", '"', "”", "'", "’")

# Discipline weights for the final score (renormalized over available data).
_DISCIPLINE_WEIGHTS: dict[str, float] = {
    "truthfulness": 0.40,
    "manipulation": 0.25,
    "responsiveness": 0.15,
    "civility": 0.20,
}
# One checked claim is not certainty. Shrink the error rate toward a mild
# prior (two pseudo-claims carrying 0.30 error mass ≈ 85% accurate) so a
# 1/1 speaker cannot post a perfect truthfulness score.
_TRUTH_PRIOR_CLAIMS = 2.0
_TRUTH_PRIOR_ERROR = 0.30
# Below this gap the leader is not a winner. Discipline lists are short and
# civility is one integer, so sub-threshold gaps are noise.
_WIN_MARGIN = 5.0
_DISCIPLINE_WIN_MARGIN = 5.0
# Invited guests only. Moderator, recap, and insert voices are not contestants.
_NON_CONTESTANT = re.compile(
    r"\b(moder[aá]tor|moderator|zostrih|dokr[uú]tka|z[aá]znam)\b",
    re.IGNORECASE,
)
_DISCIPLINE_LABELS: dict[str, str] = {
    "truthfulness": "pravdivosť",
    "manipulation": "manipulácia a fauly",
    "responsiveness": "vecnosť (odpovedanie na otázky)",
    "civility": "slušnosť",
}
# Weighted error per checked claim (0 = fully true, 1 = pure fabrication).
_ERROR_WEIGHTS: dict[tuple[Verdict, Severity | None], float] = {
    (Verdict.FALSE, Severity.FABRICATION): 1.0,
    (Verdict.FALSE, Severity.MATERIAL): 0.7,
    (Verdict.FALSE, Severity.TRIVIAL): 0.2,
    (Verdict.FALSE, None): 0.7,
    (Verdict.MISLEADING, Severity.FABRICATION): 0.8,
    (Verdict.MISLEADING, Severity.MATERIAL): 0.5,
    (Verdict.MISLEADING, Severity.TRIVIAL): 0.1,
    (Verdict.MISLEADING, None): 0.5,
}
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
_DODGE_PRIOR_MASS_SK = str(_DODGE_PRIOR_MASS).replace(".", ",")  # decimal comma
# Normalized by words or questions; not named when speaker identity is uncertain.
_IDENTITY_SENSITIVE = ("manipulation", "responsiveness")
_SUBSTANTIVE_TURN_WORDS = 15
_INTERJECTION_WORDS = 5
_MAX_EVIDENCE = 5


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


def is_non_contestant(name: str) -> bool:
    """Moderator, video insert, and recap labels are not debate contestants."""
    return bool(_NON_CONTESTANT.search(name or ""))


# ---------------------------------------------------------------------------
# Transcript statistics (deterministic, no LLM)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Speaker identity resolution
# ---------------------------------------------------------------------------


def _speaker_key(name: str) -> str:
    """Prefer 'speaker c' id so short and long labels merge."""
    m = _SPEAKER_ID.search(name)
    if m:
        return f"speaker {m.group(1).lower()}"
    return " ".join(name.lower().split())


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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _fmt_fact(fact) -> str:
    sev = f"/{fact.severity.value}" if fact.severity else ""
    quote = f' — „{fact.quote[:120]}"' if fact.quote else ""
    return f"[{fact.verdict.value}{sev}] {fact.claim[:160]}{quote}"


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
    status: Literal["ok", "degraded", "no_transcript"]
    names_in_doubt = smap is not None and smap.status != "ok"
    if not transcript:
        status = "no_transcript"
        method_notes.append(
            "No transcript given: fouls and question dodges could not be counted "
            "per spoken word; no overall winner and no manipulation/responsiveness "
            "winner is named."
        )
    elif unmatched or names_in_doubt:
        status = "degraded"
        why: list[str] = []
        if names_in_doubt:
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
            # Missing salience (older reports) stays at the neutral default.
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
            # Publish score from the published (rounded) rate so they reconcile.
            rate_1000: float | None = round(points / norm * 1000.0, 2)
            manip_score: float | None = max(0.0, round(100.0 - rate_1000, 2))
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
                rate_per_1000=rate_1000,
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
            dodge_rate: float | None = round(
                (dodge_mass + _DODGE_PRIOR_MASS) / (n_room + _DODGE_PRIOR_QUESTIONS), 3
            )
            resp_score: float | None = round(100.0 * (1.0 - dodge_rate), 2)
            resp_detail = (
                f"podstatné otázky: {qc['challenging']} (prerušené: "
                f"{qc['interrupted']}), vyhnutia: {qc['dodged']}, čiastočné: "
                f"{qc['partial']}; ({dodge_mass:.1f} + {_DODGE_PRIOR_MASS_SK}) / "
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
                rate=dodge_rate,
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
    if status != "ok":
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
        if status != "ok" and disc in _IDENTITY_SENSITIVE:
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
    # With a transcript, a row that matched zero words is an identity that was
    # never found in the debate: naming it would pin fouls on a guess.
    candidates = [r for r in scoreboard if r.words] if transcript else scoreboard
    if candidates and not names_in_doubt:
        red = max(
            candidates,
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
