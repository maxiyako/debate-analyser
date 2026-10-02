"""Unit tests for Settings.vertex_ready() and credential wiring."""

from __future__ import annotations

from pathlib import Path

from config import Settings


def _settings(**kw) -> Settings:
    # _env_file=None isolates the test from the developer's local .env.
    return Settings(_env_file=None, **kw)


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
