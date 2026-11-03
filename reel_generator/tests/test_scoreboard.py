from pathlib import Path

from reel_generator.highlight import highlight_from_report, load_report
from reel_generator.scoreboard import board_for_highlight, collect_score_events
from reel_generator.transcript import load_transcript

DATA = Path(__file__).resolve().parents[2] / "data"
DEBATE = "-H9VRLRwdrI"
LIE = 'facts[speaker="Andrej Danko"][verdict="False"][0]'


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
