from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LATEST_TRANSCRIPT = REPO_ROOT / "data" / "transcripts" / "604147.txt"


@pytest.fixture(scope="session")
def transcript() -> str:
    """The most recent real debate transcript, used to ground hallucination tests."""
    return LATEST_TRANSCRIPT.read_text(encoding="utf-8")


import numpy as np

SR = 16000


class FakeEmbedder:
    """Voice A is positive audio, voice B negative; silence embeds to zero."""

    def embed(self, audio: np.ndarray) -> np.ndarray:
        m = float(np.mean(audio)) if len(audio) else 0.0
        if m > 1e-6:
            return np.array([1.0, 0.0])
        if m < -1e-6:
            return np.array([0.0, 1.0])
        return np.array([0.0, 0.0])


class FakeSeparator:
    """Returns voice B then voice A, so assignment has to swap them."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def separate(self, audio: np.ndarray) -> list[np.ndarray]:
        self.calls.append(len(audio))
        return [np.full(len(audio), -0.5, dtype=np.float32), np.full(len(audio), 0.5, dtype=np.float32)]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_separator() -> FakeSeparator:
    return FakeSeparator()


@pytest.fixture
def voice_audio() -> np.ndarray:
    """20 s: A alone 0-4 and 6-10, A+B mixed 4-6, B alone 12-16, silence elsewhere."""
    audio = np.zeros(20 * SR, dtype=np.float32)
    audio[0 : 4 * SR] = 0.5
    audio[6 * SR : 10 * SR] = 0.5
    audio[12 * SR : 16 * SR] = -0.5
    return audio
