"""Watcher daemon — polls GCS for new data and triggers pipeline stages."""

import json
import logging
import time
from pathlib import Path

from .config import SCANS, scan_prefix

logger = logging.getLogger(__name__)

POLL_INTERVAL = 300  # 5 minutes
STATE_FILE = Path("~/.surveyor_state.json").expanduser()


class TerritorialSurveyor:
    """Watches for new data on GCS and triggers appropriate processing."""

    def __init__(self, data_dir: str = "data", output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.processed: set[str] = set()
        self._load_state()

    def watch(self, poll_interval: int = POLL_INTERVAL):
        """Main watch loop — run indefinitely."""
        logger.info(f"Territorial surveyor started (polling every {poll_interval}s)")

        while True:
            try:
                self._poll()
            except KeyboardInterrupt:
                logger.info("Surveyor stopped by user")
                break
            except Exception as e:
                logger.error(f"Poll error: {e}", exc_info=True)

            time.sleep(poll_interval)

    def _poll(self):
        """Single poll cycle: check for new data, trigger processing."""
        from .gcs import gcs_exists

        for scan_id, scan in SCANS.items():
            prefix = scan_prefix(scan_id)

            # Check if crowns need generating
            crowns_gcs = f"{prefix}/lidar/vectors/crowns.gpkg"
            dsm_gcs = f"{prefix}/lidar/rasters/dsm_0.5m_cog.tif"
            dtm_gcs = f"{prefix}/lidar/rasters/dtm_1m_cog.tif"

            key_crowns = f"crowns_{scan_id}"
            if key_crowns not in self.processed:
                if not gcs_exists(crowns_gcs) and gcs_exists(dsm_gcs) and gcs_exists(dtm_gcs):
                    logger.info(f"[{scan_id}] DSM+DTM present, no crowns → running segmentation")
                    self._run_crowns(scan_id)
                    self.processed.add(key_crowns)

            # Check if snapshot needs building
            snapshot_gcs = f"agent/snapshots/{scan_id}_trees_res13.json"
            key_snap = f"snapshot_{scan_id}"
            if key_snap not in self.processed:
                if not gcs_exists(snapshot_gcs):
                    logger.info(f"[{scan_id}] Snapshot missing → building")
                    self._run_snapshot(scan_id)
                    self.processed.add(key_snap)

        self._save_state()

    def _run_crowns(self, scan_id: str):
        """Download DSM/DTM, run silva segmentation, upload results."""
        from .gcs import download_from_gcs, upload_to_gcs
        from .crowns import run_silva

        prefix = scan_prefix(scan_id)
        local_dir = self.data_dir / f"{scan_id}_crowns"
        local_dir.mkdir(parents=True, exist_ok=True)

        dsm_local = local_dir / "dsm.tif"
        dtm_local = local_dir / "dtm.tif"

        download_from_gcs(f"{prefix}/lidar/rasters/dsm_0.5m_cog.tif", dsm_local)
        download_from_gcs(f"{prefix}/lidar/rasters/dtm_1m_cog.tif", dtm_local)

        crowns = run_silva(str(dsm_local), str(dtm_local), str(local_dir))

        upload_to_gcs(local_dir / "crowns.gpkg", f"{prefix}/lidar/vectors/crowns.gpkg")
        upload_to_gcs(local_dir / "tree_tops.gpkg", f"{prefix}/lidar/vectors/tree_tops.gpkg")

        logger.info(f"[{scan_id}] Crown segmentation done: {len(crowns)} trees")

    def _run_snapshot(self, scan_id: str):
        """Build and upload tree snapshot."""
        from .snapshot import TreeSnapshot

        snapshot = TreeSnapshot(
            scan_id=scan_id,
            output_dir=self.output_dir,
            data_dir=self.data_dir,
        )
        snapshot.build()
        snapshot.upload()

    def _load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                self.processed = set(json.load(f).get("processed", []))

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump({"processed": sorted(self.processed)}, f, indent=2)


def main():
    """Entry point for `python -m surveyor.watch`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    surveyor = TerritorialSurveyor()
    surveyor.watch()


if __name__ == "__main__":
    main()
