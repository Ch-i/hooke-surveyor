#!/usr/bin/env python3
"""Run silva2016 crown segmentation for a scan.

Downloads DSM/DTM from GCS if not local, runs CUDA-accelerated segmentation,
optionally uploads results back to GCS.

Usage:
    python scripts/run_crowns.py --scan-id 250525
    python scripts/run_crowns.py --scan-id 250525 --upload
    python scripts/run_crowns.py --dsm path/to/dsm.tif --dtm path/to/dtm.tif -o output/
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Silva crown segmentation (CUDA)")
    parser.add_argument("--scan-id", help="Scan ID to process (downloads from GCS)")
    parser.add_argument("--dsm", help="Local path to DSM raster")
    parser.add_argument("--dtm", help="Local path to DTM raster")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
    parser.add_argument("--min-height", type=float, default=2.0, help="Min tree height (m)")
    parser.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian sigma")
    parser.add_argument("--upload", action="store_true", help="Upload results to GCS")
    args = parser.parse_args()

    if args.scan_id:
        from surveyor.config import SCANS, scan_prefix
        from surveyor.gcs import download_from_gcs

        if args.scan_id not in SCANS:
            logger.error(f"Unknown scan: {args.scan_id}. Known: {list(SCANS.keys())}")
            return

        prefix = scan_prefix(args.scan_id)
        local_dir = Path("data") / f"{args.scan_id}_crowns"
        local_dir.mkdir(parents=True, exist_ok=True)

        dsm_path = local_dir / "dsm.tif"
        dtm_path = local_dir / "dtm.tif"

        if not dsm_path.exists():
            logger.info("Downloading DSM from GCS...")
            download_from_gcs(f"{prefix}/lidar/rasters/dsm_0.5m_cog.tif", dsm_path)
        if not dtm_path.exists():
            logger.info("Downloading DTM from GCS...")
            download_from_gcs(f"{prefix}/lidar/rasters/dtm_1m_cog.tif", dtm_path)

        output_dir = local_dir
    elif args.dsm and args.dtm:
        dsm_path = Path(args.dsm)
        dtm_path = Path(args.dtm)
        output_dir = Path(args.output_dir)
    else:
        parser.error("Provide --scan-id or both --dsm and --dtm")
        return

    from surveyor.crowns import run_silva

    crowns = run_silva(
        str(dsm_path),
        str(dtm_path),
        str(output_dir),
        min_height=args.min_height,
        smooth_sigma=args.smooth_sigma,
    )

    logger.info(f"Results: {len(crowns)} crowns")
    logger.info(f"  Height: {crowns['height_m'].min():.1f}–{crowns['height_m'].max():.1f} m "
                f"(mean {crowns['height_m'].mean():.1f} m)")
    logger.info(f"  Area: {crowns['crown_area_m2'].min():.1f}–{crowns['crown_area_m2'].max():.1f} m² "
                f"(mean {crowns['crown_area_m2'].mean():.1f} m²)")

    if args.upload and args.scan_id:
        from surveyor.gcs import upload_to_gcs

        prefix = scan_prefix(args.scan_id)
        upload_to_gcs(output_dir / "crowns.gpkg", f"{prefix}/lidar/vectors/crowns.gpkg")
        upload_to_gcs(output_dir / "tree_tops.gpkg", f"{prefix}/lidar/vectors/tree_tops.gpkg")
        logger.info("Uploaded crowns + tree_tops to GCS")


if __name__ == "__main__":
    main()
