Upload local artifacts to GCS.

Steps:
1. Scan output/ directory for built snapshots and graph files
2. Scan data/ for any crown segmentation outputs that should be uploaded
3. List everything that would be uploaded:
   - File path → GCS destination
   - File size
   - Whether it would overwrite an existing GCS object
4. Ask for confirmation before uploading
5. Upload each file using surveyor.gcs.upload_to_gcs()
6. Verify each upload by checking GCS
7. Report: files uploaded, total size, GCS paths

GCS destinations:
- Snapshots: agent/snapshots/{scan_id}_trees_res13.json
- Graphs: agent/snapshots/{scan_id}_graph.json
- Crowns: scans/{scanId}_{season}_{year}/lidar/vectors/crowns.gpkg
- Tree tops: scans/{scanId}_{season}_{year}/lidar/vectors/tree_tops.gpkg
