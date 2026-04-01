Rebuild the full tree snapshot from all data sources.

Argument: $ARGUMENTS (scan ID e.g. "260227", or "all" for every scan. Default: latest scan.)

Steps:
1. Check data/ directory — is the base GeoJSON (h3_trees_all.geojson) present? If not, download from GCS.
2. Check which per-scan data is available locally (crowns, DBH, carbon, rasters). Download missing files from GCS.
3. Run: `python scripts/build_snapshot.py --scan-id {scan_id}` (or `--all`)
4. Verify output:
   - Record count should be ~87,935
   - Required fields present: h3, lat, lon, height_m, status, n_scans
   - Enrichment fields populated: crown_area_m2, species_detected, guild_score, risk_overall
   - No NaN explosions (some NaN is expected for missing scan data)
5. Compare to previous snapshot if one exists: any trees gained/lost? Field coverage improved?
6. If `--upload` or user confirms: push to `agent/snapshots/{scan_id}_trees_res13.json` on GCS
7. Report: record count, field coverage percentages, top anomalies detected, file size
