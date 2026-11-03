from pathlib import Path

from reel_generator.highlight import Highlight, highlight_from_report, load_report
from reel_generator.scoreboard import (
    board_for_highlight,
    collect_score_events,
    scoreboard_applies,
)
from reel_generator.transcript import load_transcript

DATA = Path(__file__).resolve().parents[2] / "data"
DEBATE = "-H9VRLRwdrI"
LIE = 'facts[speaker="Andrej Danko"][verdict="False"][0]'


def _highlight(kind: str, outcome: str) -> Highlight:
    return Highlight(
        discussion_id="x",
        speaker="A",
        kind=kind,
        index=0,
        source_path="p",
        timestamp_s=1.0,
        timestamp_label="00:01",
        quote="q",
        detail="d",
        outcome=outcome,
        stamp_text="TAG",
        stamp_key="loz",
    )


def test_scoreboard_replaces_the_stamp_only_for_counted_fouls():
    assert scoreboard_applies(_highlight("fact", "False"))
    assert scoreboard_applies(_highlight("fact", "Misleading"))
    assert scoreboard_applies(_highlight("incivility", ""))
    assert scoreboard_applies(_highlight("question_dodging", "dodged"))
    assert scoreboard_applies(_highlight("question_audit", "dodged")) is True
    assert scoreboard_applies(_highlight("fact", "Unverified")) is False
    assert scoreboard_applies(_highlight("logical_fallacies", "")) is False
    assert scoreboard_applies(_highlight("question_dodging", "partial")) is False
    assert scoreboard_applies(_highlight("manipulation", "")) is False


def test_danko_lie_ticks_from_zero_while_grohling_stays_clear():
    report = load_report(DATA / "reports" / f"{DEBATE}.json")
    lines = load_transcript(DATA / "transcripts" / f"{DEBATE}.corrected.txt")
    events = collect_score_events(report, lines, DEBATE)
    item = highlight_from_report(report, LIE, DEBATE, lines)
    board = board_for_highlight(events, item, report)

    assert board.label == "LOŽ"
    assert board.left_name == "Danko"
    assert board.right_name == "Grohling"
    assert board.scoring_side == "left"
    assert (board.left_before, board.left_after) == (0, 1)
    assert (board.right_before, board.right_after) == (0, 0)

    loz = [event for event in events if event.category == "loz"]
    assert len(loz) == 2
    assert sum(1 for event in events if event.category == "misleading") == 3
    assert sum(1 for event in events if event.category == "incivility") == 6
    assert sum(1 for event in events if event.category == "dodge") == 3
