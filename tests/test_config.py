"""Unit tests for Settings.vertex_ready() and credential wiring."""

from __future__ import annotations

from pathlib import Path

from config import Settings
from src.llm import output_limit_kwargs


def _settings(**kw) -> Settings:
    # _env_file=None isolates the test from the developer's local .env.
    return Settings(_env_file=None, **kw)


def test_output_limit_raises_cap_and_leaves_thinking_dynamic() -> None:
    kw = output_limit_kwargs(_settings())
    assert kw == {"max_tokens": 65536}


def test_fidelity_gate_on_by_default() -> None:
    assert _settings().fidelity_enabled is True


def test_vertex_not_ready_without_project_id() -> None:
    assert _settings(gcp_project_id="").vertex_ready() is False


def test_vertex_ready_with_project_id() -> None:
    assert _settings(gcp_project_id="my-proj").vertex_ready() is True


def test_apply_credentials_noop_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    _settings(gcp_project_id="p", google_application_credentials=None).apply_credentials()
    import os

    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_apply_credentials_exports_path(monkeypatch, tmp_path: Path) -> None:
    key = tmp_path / "sa.json"
    key.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    _settings(
        gcp_project_id="p", google_application_credentials=str(key)
    ).apply_credentials()
    import os

    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(key)


def test_interpretation_off_by_default_with_window_and_chunk() -> None:
    s = _settings()
    assert s.interpretation_enabled is False
    assert s.interpretation_window == 6
    assert s.interpretation_chunk == 20
