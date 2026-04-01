"""GCS upload/download utilities."""

import logging
from pathlib import Path
from typing import Optional

from .config import GCS_BUCKET, GCS_PROJECT

logger = logging.getLogger(__name__)

_client = None


def _get_bucket():
    global _client
    if _client is None:
        from google.cloud import storage

        _client = storage.Client(project=GCS_PROJECT)
    return _client.bucket(GCS_BUCKET)


CONTENT_TYPES = {
    ".json": "application/json",
    ".geojson": "application/geo+json",
    ".tif": "image/tiff",
    ".gpkg": "application/geopackage+sqlite3",
}


def upload_to_gcs(local_path: str | Path, gcs_path: str, content_type: Optional[str] = None):
    """Upload a local file to GCS."""
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    blob = _get_bucket().blob(gcs_path)

    if content_type is None:
        content_type = CONTENT_TYPES.get(local_path.suffix.lower(), "application/octet-stream")

    blob.upload_from_filename(str(local_path), content_type=content_type)
    size_mb = local_path.stat().st_size / 1024 / 1024
    logger.info(f"Uploaded {local_path.name} ({size_mb:.1f} MB) → gs://{GCS_BUCKET}/{gcs_path}")


def download_from_gcs(gcs_path: str, local_path: str | Path):
    """Download a file from GCS to local disk."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    blob = _get_bucket().blob(gcs_path)
    blob.download_to_filename(str(local_path))
    size_mb = local_path.stat().st_size / 1024 / 1024
    logger.info(f"Downloaded gs://{GCS_BUCKET}/{gcs_path} → {local_path} ({size_mb:.1f} MB)")


def list_gcs_prefix(prefix: str) -> list[str]:
    """List all object names under a GCS prefix."""
    return [b.name for b in _get_bucket().list_blobs(prefix=prefix)]


def gcs_exists(gcs_path: str) -> bool:
    """Check if a GCS object exists."""
    return _get_bucket().blob(gcs_path).exists()
