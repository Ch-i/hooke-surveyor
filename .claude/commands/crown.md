Run silva2016 crown segmentation for a scan using CUDA.

Argument: $ARGUMENTS (scan ID, e.g. "250525". If empty, check which scans need crowns.)

Steps:
1. If no scan ID given, check GCS for which scans have DSM/DTM but no crowns.gpkg — list them and ask which to process
2. Download DSM and DTM COGs from GCS to data/{scan_id}_crowns/
3. Run `python scripts/run_crowns.py --scan-id {scan_id}`
4. Review output: number of crowns detected, height distribution (min/mean/max), total crown area
5. Sanity check: compare tree count to expected (~1000-5000 per scan depending on coverage)
6. If looks good, upload crowns.gpkg and tree_tops.gpkg to GCS: `scans/{scanId}_{season}_{year}/lidar/vectors/`
7. Report results with key statistics

IMPORTANT: Use silva2016 approach (variable-window local maxima + watershed). Never dalponte2016.
IMPORTANT: Check VRAM usage with nvidia-smi before and during processing.
