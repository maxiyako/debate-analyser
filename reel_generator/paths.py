"""Locate report, transcript, and raw video for a discussion id.

Directory layout matches the analyzer (`data/raw`, `data/reports`,
`data/transcripts`). This module only reads those folders.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


def find_report(data_dir: Path, discussion_id: str) -> Path:
    path = data_dir / "reports" / f"{discussion_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"report not found: {path}")
    return path


def find_transcript(data_dir: Path, discussion_id: str) -> Path:
    folder = data_dir / "transcripts"
    for name in (f"{discussion_id}.corrected.txt", f"{discussion_id}.txt"):
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"transcript not found for {discussion_id} in {folder} "
        "(expected .corrected.txt or .txt)"
    )


def find_video(data_dir: Path, discussion_id: str) -> Path | None:
    raw = data_dir / "raw"
    if not raw.is_dir():
        return None
    for suffix in VIDEO_SUFFIXES:
        path = raw / f"{discussion_id}{suffix}"
        if path.is_file():
            return path
    matches = [
        path
        for path in raw.glob(f"{discussion_id}.*")
        if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file()
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def expected_video(data_dir: Path, discussion_id: str) -> Path:
    """Path the render command should use, even when the file is not downloaded yet."""
    found = find_video(data_dir, discussion_id)
    if found is not None:
        return found
    return data_dir / "raw" / f"{discussion_id}.mp4"
