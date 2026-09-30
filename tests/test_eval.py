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


def test_rejudge_only_filter(tmp_path: Path) -> None:
    _write_report(tmp_path, "100", [_fact("True")])
    _write_report(tmp_path, "101", [_fact("True")])
    out = rejudge_reports(
        tmp_path, {"100": "2026-04-12", "101": "2026-04-12"}, fetch=_ok, llm=_llm, only={"101"}
    )
    assert set(out["reports"]) == {"101"}
