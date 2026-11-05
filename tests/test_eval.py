"""Offline tests for the rejudge harness (src/eval.py)."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval import base_id, load_dates, rejudge_reports, report_files
from src.judge import JudgeAxes, JudgeOutput
from src.tools.page import PageResult


def _write_report(dirpath: Path, stem: str, facts: list[dict], **extra) -> None:
    payload = {"summary": "s", "facts": facts, **extra}
    (dirpath / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def _fact(verdict: str) -> dict:
    return {
        "claim": "Deficit je 6 % HDP.",
        "speaker": "X",
        "quote": "q",
        "verdict": verdict,
        "severity": "material" if verdict == "False" else None,
        "sources": ["https://a.sk/x"],
    }


def _ok(url: str) -> PageResult:
    return PageResult("ok", url, final_url=url, title="t", published="2026-03-01", text="Deficit 6 % HDP")


def _llm(prompt: str) -> JudgeOutput:
    return JudgeOutput(axes=JudgeAxes(support=2, metric_fidelity=2, date_fit=2, coverage=2, independence=1))


def test_base_id_and_report_files(tmp_path: Path) -> None:
    assert base_id("591624") == "591624"
    assert base_id("591624_v2") == "591624"
    assert base_id("602993.corrected") is None
    assert base_id("605345.baseline") is None
    assert base_id("run_591624") is None
    for stem in ("591624", "591624_v2", "602993.corrected", "run_591624"):
        (tmp_path / f"{stem}.json").write_text("{}")
    (tmp_path / "x.log").write_text("")
    assert [p.stem for p in report_files(tmp_path)] == ["591624", "591624_v2"]


def test_load_dates_missing_file(tmp_path: Path) -> None:
    assert load_dates(tmp_path / "nope.json") == {}


def test_rejudge_uses_dates_file_and_payload_and_lists_skipped(tmp_path: Path) -> None:
    _write_report(tmp_path, "100", [_fact("True"), _fact("False")])
    _write_report(tmp_path, "200_v2", [_fact("True")], debate_date="2026-04-19")
    _write_report(tmp_path, "300", [_fact("True")])  # no date anywhere
    out = rejudge_reports(tmp_path, {"100": "2026-04-12"}, fetch=_ok, llm=_llm)
    assert set(out["reports"]) == {"100", "200_v2"}
    assert out["skipped_no_date"] == ["300"]
    assert out["reports"]["100"]["date"] == "2026-04-12"
    assert out["reports"]["200_v2"]["date"] == "2026-04-19"
    assert out["overall"]["judged"] == 3
    assert out["overall"]["mean_quality"] == 95.0  # axes (2,2,2,2,1) -> 95.0


def test_fidelity_eval_scores_gate_against_labels(tmp_path: Path) -> None:
    from src.eval import fidelity_eval
    from src.fidelity import FidelityOutput, Reading

    reports, transcripts = tmp_path / "r", tmp_path / "t"
    reports.mkdir()
    transcripts.mkdir()
    facts = [
        {**_fact("False"), "claim": "Bad one", "timestamp": "00:10"},
        {**_fact("Misleading"), "claim": "Good one", "timestamp": "00:20"},
        {**_fact("False"), "claim": "Unsure one"},
        _fact("True"),
    ]
    _write_report(reports, "100", facts)
    (transcripts / "100.corrected.txt").write_text("X [00:10]: q\nX [00:20]: q\n", encoding="utf-8")
    gold = [
        {"report": "100", "index": 0, "claim_prefix": "Bad", "label": "downgrade"},
        {"report": "100", "index": 1, "claim_prefix": "Good", "label": "keep"},
        {"report": "100", "index": 2, "claim_prefix": "Unsure", "label": "doubtful"},
        {"report": "100", "index": 3, "claim_prefix": "Moved", "label": "keep"},
    ]
    seen: list[str] = []

    def llm(prompt: str) -> FidelityOutput:
        seen.append(prompt)
        bad = "CLAIM AS EXTRACTED: Good" not in prompt
        return FidelityOutput(
            asserted="a", refuted="b", reading=Reading.DIFFERENT if bad else Reading.SAME
        )

    out = fidelity_eval(gold, reports, transcripts, llm=llm)
    assert len(seen) == 3  # True fact is never audited
    assert out["summary"]["downgrade"] == {"n": 1, "downgraded": 1}
    assert out["summary"]["keep"] == {"n": 1, "downgraded": 0}
    assert out["summary"]["doubtful"] == {"n": 1, "downgraded": 1}
    assert out["stale"] == ["100#3"]
    assert "X [00:10]: q" in seen[0]


def test_rejudge_only_filter(tmp_path: Path) -> None:
    _write_report(tmp_path, "100", [_fact("True")])
    _write_report(tmp_path, "101", [_fact("True")])
    out = rejudge_reports(
        tmp_path, {"100": "2026-04-12", "101": "2026-04-12"}, fetch=_ok, llm=_llm, only={"101"}
    )
    assert set(out["reports"]) == {"101"}


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
