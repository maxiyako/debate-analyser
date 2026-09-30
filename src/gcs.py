"""Optional Google Cloud Storage helpers. No-ops when USE_GCS=false."""

from __future__ import annotations

import logging
from pathlib import Path

from config import Settings, get_settings

logger = logging.getLogger(__name__)


def _client(settings: Settings):
    from google.cloud import storage

    return storage.Client(project=settings.gcp_project_id)


def upload_file(local_path: Path, object_name: str | None = None, settings: Settings | None = None) -> str | None:
    """Upload a local file to GCS. Returns gs:// URI or None if GCS disabled."""
    settings = settings or get_settings()
    if not settings.use_gcs or not settings.gcs_bucket:
        logger.debug("GCS disabled; skip upload of %s", local_path)
        return None

    object_name = object_name or local_path.name
    client = _client(settings)
    blob = client.bucket(settings.gcs_bucket).blob(object_name)
    blob.upload_from_filename(str(local_path))
    uri = f"gs://{settings.gcs_bucket}/{object_name}"
    logger.info("Uploaded %s -> %s", local_path, uri)
    return uri


def download_file(object_name: str, local_path: Path, settings: Settings | None = None) -> Path | None:
    """Download a GCS object to local_path. Returns path or None if GCS disabled."""
    settings = settings or get_settings()
    if not settings.use_gcs or not settings.gcs_bucket:
        logger.debug("GCS disabled; skip download of %s", object_name)
        return None

    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = _client(settings)
    blob = client.bucket(settings.gcs_bucket).blob(object_name)
    blob.download_to_filename(str(local_path))
    logger.info("Downloaded gs://%s/%s -> %s", settings.gcs_bucket, object_name, local_path)
    return local_path
