from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LATEST_TRANSCRIPT = REPO_ROOT / "data" / "transcripts" / "604147.txt"


@pytest.fixture(scope="session")
def transcript() -> str:
    """The most recent real debate transcript, used to ground hallucination tests."""
    return LATEST_TRANSCRIPT.read_text(encoding="utf-8")
