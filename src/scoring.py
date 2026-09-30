"""Deterministic fair-play scoring from an AnalysisReport (no LLM).

Design goals:
- Normalize per opportunity so speaking time doesn't bias the winner:
  fact accuracy is rated per CHECKED claim, behavioral fouls per 1000
  spoken words (computed from the transcript when available).
- Every discipline is reported with its value, the raw counts behind it,
  and the supporting evidence (quotes/claims), so the verdict is auditable.
- Unverified/Contested facts never penalize a speaker — failure to verify
  is our problem, not theirs.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.agents import AnalysisReport, Severity, TimeShare, Verdict

_SPEAKER_ID = re.compile(r"\bspeaker\s+([a-z0-9]+)\b", re.IGNORECASE)
_NAME_TOKEN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)
# 'Speaker X [MM:SS]: text' or 'Real Name [H:MM:SS]: text'
_LINE_RE = re.compile(
    r"^(?P<speaker>[^\[\]]{1,60}?)\s*\[(?:\d{1,3}:)?\d{1,2}:\d{2}\]\s*:\s*(?P<text>.*)$"
)
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
    r"\b(moder[aá]tor|moderator|zostrih|dokr[uú]tka)\b",
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
# Behavioral foul weights (score points per foul per 1000 spoken words).
_MANIPULATION_WEIGHT = 8.0
_FALLACY_WEIGHT = 6.0
_DODGE_WEIGHT = 10.0
_DEFAULT_WORDS = 1000  # assumed when no transcript stats are available
# Floor for per-1000-words normalization: protects against exploding foul
# rates when a speaker's transcript label mismatches and few words matched.
_MIN_WORDS_FLOOR = 500
_MAX_EVIDENCE = 5


class DisciplineResult(BaseModel):
    discipline: str
    label: str = ""
    score: float | None = Field(
        default=None, description="0-100; None = no data for this discipline."
    )
    detail: str = ""
    evidence: list[str] = Field(default_factory=list)


class SpeakerScore(BaseModel):
    speaker: str
    score: float = 100.0
    words: int = 0
    word_share_percent: float = 0.0
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
    interruptions_caused: int = 0


def transcript_speaker_stats(transcript: str) -> dict[str, SpeakerStats]:
    """Word/turn counts + interruption proxy per speaker key, from the transcript.

    Interruption proxy: speaker B starts a turn while A's previous line did not
    end with terminal punctuation (cut-off mid-sentence). Only counted when the
    transcript is reliably punctuated (most lines end with punctuation),
    otherwise ASR noise would inflate it.
    """
    stats: dict[str, SpeakerStats] = {}
    parsed: list[tuple[str, str]] = []
    for line in transcript.splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        parsed.append((m.group("speaker").strip(), m.group("text").strip()))

    if not parsed:
        return {}

    punctuated = sum(1 for _, t in parsed if t.endswith(_TERMINAL_PUNCT))
    punct_reliable = punctuated / len(parsed) >= 0.5

    prev_speaker: str | None = None
    prev_text = ""
    for speaker, text in parsed:
        key = _speaker_key(speaker)
        row = stats.setdefault(key, SpeakerStats(speaker=speaker))
        if len(speaker) > len(row.speaker):
            row.speaker = speaker
        row.words += len(text.split())
        prev_key = _speaker_key(prev_speaker) if prev_speaker is not None else None
        if key != prev_key:  # new speaking turn
            row.turns += 1
            if (
                punct_reliable
                and prev_speaker is not None
                and prev_text
                and not prev_text.endswith(_TERMINAL_PUNCT)
            ):
                row.interruptions_caused += 1
        prev_speaker, prev_text = speaker, text
    return stats


def apply_deterministic_moderator_metrics(
    report: AnalysisReport, transcript: str
) -> list[str]:
    """Replace LLM-estimated time shares/interruptions with computed values."""
    stats = transcript_speaker_stats(transcript)
    if not stats:
        return []
    total = sum(s.words for s in stats.values())
    if not total:
        return []
    report.moderator_audit.equal_time_distribution = [
        TimeShare(
            speaker=s.speaker,
            approximate_share_percent=round(100.0 * s.words / total, 1),
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
        "Moderator time shares computed deterministically from transcript "
        "word counts (LLM estimate replaced)."
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
    """Find the behavioral-roster row for a fact speaker (exact key, then surname).

    Fact and behavioral agents sometimes label the same person differently
    ("Danko" vs "Andrej Danko"); fall back to a unique surname match so
    fact-check penalties aren't silently dropped. Never creates a new row.
    """
    key = _speaker_key(name)
    if key in speakers:
        return speakers[key]
    if _SPEAKER_ID.search(name):
        return None  # "Speaker X" style: only exact-key merge is safe
    toks = _NAME_TOKEN.findall(name.lower())
    if not toks:
        return None
    surname = toks[-1]
    cands = [
        row
        for row in speakers.values()
        if surname in _NAME_TOKEN.findall(row.speaker.lower())
    ]
    return cands[0] if len(cands) == 1 else None


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

    Pass the (corrected) transcript to normalize behavioral fouls per 1000
    spoken words; without it, all speakers are assumed to speak equally.
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

    # Speaking stats: normalize fouls per 1000 words (guests only).
    stats = transcript_speaker_stats(transcript) if transcript else {}
    total_words = sum(st.words for st in stats.values())
    for key, st in stats.items():
        row = speakers.get(key) or _resolve_existing(st.speaker, speakers)
        if row is None:
            continue  # moderator or unmatched label — not scored
        row.words += st.words
    if any(r.words for r in speakers.values()):
        for row in speakers.values():
            if total_words:
                row.word_share_percent = round(100.0 * row.words / total_words, 1)
        method_notes.append(
            "Behavioral fouls normalized per 1000 spoken words (word counts "
            "from transcript); fact accuracy normalized per checked claim."
        )
    else:
        method_notes.append(
            "No transcript speaking stats matched the speaker roster; assumed "
            f"equal speaking time ({_DEFAULT_WORDS} words) for all speakers."
        )

    # Facts per roster row.
    facts_by_row: dict[int, list] = {}
    for fact in report.facts:
        if not fact.speaker:
            continue
        row = _resolve_existing(fact.speaker, speakers)
        if row is None:
            continue
        facts_by_row.setdefault(id(row), []).append(fact)

    for s in report.behavioral_analysis.speakers:
        if is_non_contestant(s.speaker):
            continue
        row = ensure(s.speaker)
        words = max(row.words, _MIN_WORDS_FLOOR) if row.words else _DEFAULT_WORDS
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
        row.misleading_count = sum(
            1 for f in checked if f.verdict == Verdict.MISLEADING
        )
        row.unverified_count = sum(
            1 for f in row_facts if f.verdict == Verdict.UNVERIFIED
        ) + sum(
            1
            for f in row_facts
            if f.verdict == Verdict.MISLEADING
            and f.severity is None
            and not f.sources
        )
        row.contested_count = sum(
            1 for f in row_facts if f.verdict == Verdict.CONTESTED
        )
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
                100.0
                * (1.0 - (err + _TRUTH_PRIOR_ERROR) / (mass + _TRUTH_PRIOR_CLAIMS)),
            )
        else:
            truth_score = None
        disciplines.append(
            DisciplineResult(
                discipline="truthfulness",
                label=_DISCIPLINE_LABELS["truthfulness"],
                score=truth_score,
                detail=(
                    f"overené: {row.checked_claims} (pravda: {row.true_count}, "
                    f"nepravda: {row.false_count}, zavádzajúce: "
                    f"{row.misleading_count}); neoverené: {row.unverified_count}, "
                    f"sporné: {row.contested_count}; "
                    f"malá vzorka stiahnutá k prioru "
                    f"({_TRUTH_PRIOR_CLAIMS:.0f} pseudo-tvrdenia)"
                ),
                evidence=[_fmt_fact(f) for f in problematic[:_MAX_EVIDENCE]],
            )
        )

        # --- Manipulation & fallacies: fouls per 1000 words -----------------
        n_manip = len(s.manipulation)
        n_fall = len(s.logical_fallacies)
        row.manipulation_count = n_manip
        foul_points = _MANIPULATION_WEIGHT * n_manip + _FALLACY_WEIGHT * n_fall
        manip_rate = foul_points / words * 1000.0
        disciplines.append(
            DisciplineResult(
                discipline="manipulation",
                label=_DISCIPLINE_LABELS["manipulation"],
                score=max(0.0, 100.0 - manip_rate),
                detail=(
                    f"manipulácie: {n_manip}, logické chyby: {n_fall} "
                    f"(penalta {manip_rate:.1f} b. na 1000 slov, slov: {words})"
                ),
                evidence=[*s.manipulation, *s.logical_fallacies][:_MAX_EVIDENCE],
            )
        )

        # --- Responsiveness: question dodging per 1000 words -----------------
        n_dodge = len(s.question_dodging)
        dodge_rate = _DODGE_WEIGHT * n_dodge / words * 1000.0
        disciplines.append(
            DisciplineResult(
                discipline="responsiveness",
                label=_DISCIPLINE_LABELS["responsiveness"],
                score=max(0.0, 100.0 - dodge_rate),
                detail=f"vyhýbanie sa otázkam: {n_dodge}x (slov: {words})",
                evidence=list(s.question_dodging)[:_MAX_EVIDENCE],
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
                evidence=[*s.civility.incivility, *s.civility.fair_conduct][
                    :_MAX_EVIDENCE
                ],
            )
        )

        row.disciplines = disciplines

        # Final score: weighted mean over disciplines with data.
        avail = [(d, _DISCIPLINE_WEIGHTS[d.discipline]) for d in disciplines if d.score is not None]
        weight_sum = sum(w for _, w in avail)
        row.score = (
            round(sum(d.score * w for d, w in avail) / weight_sum, 1)
            if weight_sum
            else 100.0
        )

        # Badges (raw-count based, for the gamified post).
        if n_dodge >= 2:
            row.badges.append(f"{n_dodge}x vyhýbanie sa otázke")
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

    method_notes.append(
        "Final score = weighted mean of disciplines: "
        + ", ".join(f"{k} {v:.0%}" for k, v in _DISCIPLINE_WEIGHTS.items())
        + " (weights renormalized when a discipline has no data). "
        "Unverified/Contested claims carry no penalty. "
        f"Truthfulness shrinks toward a prior of {_TRUTH_PRIOR_CLAIMS:.0f} "
        "pseudo-claims so a single checked claim is not a perfect score. "
        "This is a fair-play index (accuracy and fouls), not a measure of "
        "who persuaded the audience. "
        f"A winner is named only when the lead is at least {_WIN_MARGIN:.0f} points; "
        f"a discipline winner only when the lead is at least {_DISCIPLINE_WIN_MARGIN:.0f}."
    )

    scoreboard = sorted(speakers.values(), key=lambda r: r.score, reverse=True)
    margin: float | None = None
    winner = ""
    if len(scoreboard) == 1:
        winner = scoreboard[0].speaker
    elif len(scoreboard) >= 2:
        margin = round(scoreboard[0].score - scoreboard[1].score, 1)
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
        scoreboard=scoreboard,
        method_notes=method_notes,
    )
