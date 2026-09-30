"""Application settings loaded from environment / .env."""

import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GCP / Vertex — no default project: must be provided via env (.env or
    # Cloud Run env vars). Empty means "not configured" (see vertex_ready()).
    gcp_project_id: str = ""
    gcp_location: str = "europe-west1"
    # Optional service-account key file (Cloud Run job / CI). When unset, ADC is used.
    google_application_credentials: str | None = None

    # Gemini
    gemini_model: str = "gemini-2.5-pro"
    # Hard timeout (seconds) on a single Vertex/Gemini call. Without this the
    # underlying HTTP client can hang indefinitely on a stalled response —
    # the process looks alive (idle socket, ~0% CPU) but never progresses.
    llm_timeout_s: float = 900.0

    # Fact pipeline
    max_claims: int = 30  # top-N claims (by salience) forwarded to fact-checkers
    # Fact-check manager (Phase B2): round 1 audits all checked claims;
    # further rounds re-verify only escalated (Unverified/Contested with a
    # concrete lead) claims. 0 disables the manager.
    factcheck_manager_rounds: int = 3

    # Hugging Face
    huggingface_token: str | None = None
    # Fail fast when diarization can't run (no HF token); otherwise the whole
    # transcript collapses to a single speaker and the analysis is meaningless.
    require_diarization: bool = False

    # GCS
    use_gcs: bool = False
    gcs_bucket: str | None = None

    # Local ML
    whisper_device: str = "cpu"
    whisper_model_id: str = "kinit/whisper-large-v3-sk"
    diarization_model_id: str = "pyannote/speaker-diarization-3.1"

    # ASR windowing (overlapping windows + fine segment timestamps for accurate
    # speaker alignment; overlap is deduped from each window's trusted centre).
    asr_window_seconds: float = 30.0
    asr_overlap_seconds: float = 5.0
    # Turn slices transcribed per Whisper generate() batch (throughput vs memory).
    asr_batch_size: int = 4
    # Flag a line as cross-talk when a second speaker overlaps at least this
    # fraction of the segment's duration.
    crosstalk_min_overlap: float = 0.35

    # Paths
    raw_dir: Path = Field(default_factory=lambda: DATA_DIR / "raw")
    audio_dir: Path = Field(default_factory=lambda: DATA_DIR / "audio")
    transcript_dir: Path = Field(default_factory=lambda: DATA_DIR / "transcripts")
    report_dir: Path = Field(default_factory=lambda: DATA_DIR / "reports")
    posts_dir: Path = Field(default_factory=lambda: DATA_DIR / "posts")

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.audio_dir,
            self.transcript_dir,
            self.report_dir,
            self.posts_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def vertex_ready(self) -> bool:
        """True when project id is set (ADC or SA file can still fail at runtime)."""
        return bool(self.gcp_project_id)

    def apply_credentials(self) -> None:
        """Export the SA key path so Vertex/GCS clients (and LiteLLM) pick it up.

        No-op when unset (falls back to Application Default Credentials).
        """
        cred = self.google_application_credentials
        if not cred:
            return
        if not Path(cred).exists():
            logger.warning(
                "GOOGLE_APPLICATION_CREDENTIALS points to a missing file: %s", cred
            )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    settings.apply_credentials()
    return settings
