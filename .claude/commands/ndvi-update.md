Fetch latest satellite NDVI and patch the tree snapshot.

Argument: $ARGUMENTS (optional: "planet" for Planet 3m, "sentinel" for S2 10m, or a file path to a local NDVI raster)

Steps:
1. Identify NDVI source:
   - If local file path given: use that raster directly
   - If "planet" or "sentinel": check for latest imagery in data/ or advise on download
   - If neither: check what NDVI rasters exist on GCS for latest scan
2. Load the NDVI raster (EPSG:27700 or EPSG:3857 — handle both)
3. Sample NDVI at each of the 87,935 tree locations (H3 res-13 centroids)
4. Patch the snapshot:
   - Update the `ndvi` field with new values
   - Recompute `ndvi_trend` using previous NDVI + new reading
   - Recompute `risk_drought` with updated NDVI
5. Run: `python scripts/build_snapshot.py --scan-id {latest} --patch ndvi`
6. Compare before/after: mean NDVI change, trees with significant drops
7. Upload patched snapshot to GCS
8. Report: date of imagery, mean NDVI, change distribution, anomalies flagged
