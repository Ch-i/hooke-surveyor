Check the current state of the territorial surveyor pipeline.

Steps:
1. Check local environment:
   - Python version, cupy/CUDA available, GPU status (nvidia-smi)
   - Installed package version
   - data/ and output/ directory contents
2. Check GCS for each scan (240913, 250525, 260227):
   - Which raster products exist? (DSM, DTM, CHM, NDVI, etc.)
   - Crown segmentation complete? (crowns.gpkg present?)
   - Analysis JSON available? (carbon_estimates, dbh_estimates, species_summary)
3. Check agent/snapshots/ on GCS:
   - Which snapshots exist?
   - When were they last updated?
   - File sizes
4. Check multiscan data:
   - h3_trees_all.geojson present and accessible?
5. Format as a clear status table:

| Scan | Rasters | Crowns | DBH | Carbon | Snapshot | Last updated |
|------|---------|--------|-----|--------|----------|-------------|

6. Flag what's blocking and what can be processed next
