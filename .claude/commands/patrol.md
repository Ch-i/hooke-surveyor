Scan the latest tree snapshot for anomalies and report findings.

Steps:
1. Load the most recent snapshot from output/ (or download from GCS if not local)
2. Run anomaly detection against thresholds:
   - NDVI drops > 2σ from population mean
   - Growth deceleration: positive height velocity but negative acceleration
   - Crown contraction > 10%/yr
   - Lost trees (status = "lost") between consecutive scans
   - VCI below drought threshold (0.35)
   - Guild score < 0 (antagonistic neighbourhood)
3. Cluster nearby anomalies spatially (group by H3 res-11 parent block)
4. For each cluster, summarize:
   - Number of affected trees
   - Dominant anomaly type
   - Species affected
   - Probable cause based on neighbourhood context
5. Report:
   - Total anomalies found / 87,935 trees
   - Top 10 clusters by severity (with H3 block IDs)
   - Recommended interventions for worst clusters
   - Any new anomalies since last patrol
