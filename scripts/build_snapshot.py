#!/usr/bin/env python3
"""Build tree snapshot at H3 res-13 resolution.

Merges: h3_trees_all.geojson + crowns + DBH + carbon + raster samples
        + temporal derivatives + neighbourhood graph + guild scores + risk

Usage:
    python scripts/build_snapshot.py --scan-id 260227
    python scripts/build_snapshot.py --all
    python scripts/build_snapshot.py --scan-id 260227 --patch ndvi
    python scripts/build_snapshot.py --scan-id 260227 --upload
"""

import argparse
import logging
import sys

from surveyor.config import SCANS, MULTISCAN_IDS
from surveyor.snapshot import TreeSnapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build tree snapshot at H3 res-13")
    parser.add_argument(
        "--scan-id",
        choices=list(SCANS.keys()),
        help="Scan to process (default: latest multiscan)",
    )
    parser.add_argument("--all", action="store_true", help="Process all multiscan scans")
    parser.add_argument("--patch", help="Comma-separated fields to patch (e.g., 'ndvi,risk')")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--data-dir", default="data", help="Input data directory")
    parser.add_argument("--upload", action="store_true", help="Upload to GCS after build")
    args = parser.parse_args()

    if args.all:
        scan_ids = MULTISCAN_IDS
    elif args.scan_id:
        scan_ids = [args.scan_id]
    else:
        scan_ids = [MULTISCAN_IDS[-1]]  # latest
        logger.info(f"No scan specified, using latest: {scan_ids[0]}")

    for scan_id in scan_ids:
        logger.info(f"{'='*60}")
        logger.info(f"Processing scan {scan_id} ({SCANS[scan_id]['label']})")
        logger.info(f"{'='*60}")

        snapshot = TreeSnapshot(
            scan_id=scan_id,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
        )

        if args.patch:
            snapshot.patch(fields=args.patch.split(","))
        else:
            snapshot.build()

        if args.upload:
            snapshot.upload()
            logger.info(f"Uploaded snapshot for {scan_id} to GCS")


if __name__ == "__main__":
    main()
