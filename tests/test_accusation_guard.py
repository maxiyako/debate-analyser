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


def test_llm_invented_code_fields_are_overwritten() -> None:
    # Misattributed quote: nothing grounded, so invented values must not survive.
    fact = _fact("Marián Viskupič", "zvýšenie výživného zo 40 na 135 eur")
    fact.quote_raw, fact.timestamp, fact.transcript_edits = "invented", "99:99", [7]
    _run(fact)
    assert (fact.quote_raw, fact.timestamp, fact.transcript_edits) == ("", "", [])
    # Grounded quote: replaced with code-derived values.
    ok = _fact("Erik Tomáš", "zo 40 na 135 eur pre ľudí")
    ok.quote_raw, ok.timestamp, ok.transcript_edits = "invented", "99:99", [7]
    _run(ok)
    assert ok.quote_raw == "zo 40 na 135 eur pre ľudi."
    assert ok.timestamp == "00:12"
    assert ok.transcript_edits == [0, 1]


# --- fix round 1 ------------------------------------------------------------

SPEAKERS = {"Moderátor", "Erik Tomáš", "Marián Viskupič"}


def _run_on(text: str, fact: VerifiedFact, edits: list[TranscriptEdit] | None = None) -> list[str]:
    raw = parse_lines(text)
    corrected, log = apply_corrections(raw, edits or [], set(), SPEAKERS)
    return enforce_accusation_support([fact], corrected, raw, log)


def test_ne_prefix_negation_added_by_quote_is_downgraded() -> None:
    text = "Erik Tomáš [00:12]: Premiér to povedal pred kamerami celému národu.\n"
    fact = _fact("Erik Tomáš", "Premiér to nepovedal pred kamerami celému národu")
    _run_on(text, fact)
    assert fact.verdict == Verdict.UNVERIFIED and fact.severity is None


def test_ne_prefix_negation_dropped_by_quote_is_downgraded() -> None:
    text = "Erik Tomáš [00:12]: Premiér to nepovedal pred kamerami celému národu.\n"
    fact = _fact("Erik Tomáš", "Premiér to povedal pred kamerami celému národu")
    _run_on(text, fact)
    assert fact.verdict == Verdict.UNVERIFIED


def test_unchanged_negation_keeps_accusation() -> None:
    text = "Erik Tomáš [00:12]: Premiér nikdy nepovedal pravdu pred kamerami celému národu.\n"
    fact = _fact("Erik Tomáš", "Premiér nikdy nepovedal pravdu pred kamerami celému národu")
    assert _run_on(text, fact) == []
    assert fact.verdict == Verdict.FALSE


def test_split_turn_dependent_quote_is_downgraded() -> None:
    text = "Moderátor [00:10]: Dobrý deň. Vláda zobrala každej rodine 858 eur cez konsolidáciu.\n"
    edits = [
        TranscriptEdit(
            line_no=0, type=EditType.SPLIT_TURN, before="Vláda", new_speaker="Marián Viskupič"
        )
    ]
    fact = _fact("Marián Viskupič", "Vláda zobrala každej rodine 858 eur cez konsolidáciu")
    notes = _run_on(text, fact, edits)
    assert fact.verdict == Verdict.UNVERIFIED
    assert "opravy prepisu" in notes[0]


def test_number_mismatch_with_raw_window_is_downgraded() -> None:
    text = "Marián Viskupič [00:25]: Vláda zobrala každej rodine cez konsolidáciu verejných financií 858 eur.\n"
    fact = _fact("Marián Viskupič", "Vláda zobrala každej rodine cez konsolidáciu verejných financií 585 eur")
    notes = _run_on(text, fact)
    assert fact.verdict == Verdict.UNVERIFIED
    assert "číslo" in notes[0]


def test_empty_quote_on_accusation_is_downgraded_without_crash() -> None:
    fact = _fact("Erik Tomáš", "")
    fact.quote_raw, fact.timestamp, fact.transcript_edits = "x", "99:99", [3]
    notes = _run(fact)
    assert fact.verdict == Verdict.UNVERIFIED and fact.severity is None
    assert len(notes) == 1
    assert (fact.quote_raw, fact.timestamp, fact.transcript_edits) == ("", "", [])


def test_short_quote_is_not_annotated_from_arbitrary_lines() -> None:
    accusation = _fact("Erik Tomáš", "eur")
    _run(accusation)
    assert accusation.verdict == Verdict.UNVERIFIED
    assert (accusation.quote_raw, accusation.timestamp, accusation.transcript_edits) == ("", "", [])
    true_fact = _fact("Erik Tomáš", "eur", Verdict.TRUE)
    assert _run(true_fact) == []
    assert true_fact.verdict == Verdict.TRUE
    assert (true_fact.quote_raw, true_fact.timestamp, true_fact.transcript_edits) == ("", "", [])


def _reference_best_window(quote: str, text: str) -> tuple[float, str]:
    """Unoptimised best_window (no pre-filter): the behaviour to preserve."""
    import difflib

    from src.validation import normalize

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


def test_best_window_prefilter_keeps_results_identical() -> None:
    from src.validation import best_window

    text = (
        "Strana HLas presadila zvýšenie výživného zo 40 na 135 eur pre ľudi. "
        "Vláda zobrala každej rodine 858 eur cez konsolidáciu. Opozícia nepodporila "
        "zvýšenie príspevku pri narodení a vláda to nepovedala pred kamerami."
    )
    quotes = [
        "zo 40 na 135 eur pre ľudí",
        "Vláda zobrala každej rodine 585 eur",
        "nepodporila zvýšenie príspevku",
        "úplne iná veta o niečom inom",
        "eur",
        "",
        "pri narodení a vláda to povedala pred kamerami navyše ešte viac slov než text",
    ]
    for q in quotes:
        assert best_window(q, text) == _reference_best_window(q, text)
    assert best_window("x", "") == (0.0, "")


def test_best_window_matches_reference_on_random_and_tied_inputs() -> None:
    import random

    from src.validation import best_window

    rng = random.Random(7)
    vocab = ["aa", "ab", "ba", "abc", "cab", "eur", "40", "135", "nie", "ne"]
    for _ in range(60):
        text = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 40)))
        quote = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 6)))
        assert best_window(quote, text) == _reference_best_window(quote, text)
    # Identical windows tie: the earliest one wins.
    assert best_window("aa ab", "aa ab x aa ab") == _reference_best_window("aa ab", "aa ab x aa ab")
