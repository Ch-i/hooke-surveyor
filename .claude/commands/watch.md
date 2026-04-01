Manage the territorial surveyor watcher daemon.

Argument: $ARGUMENTS ("start", "status", or "stop". Default: "status".)

If "start":
1. Check prerequisites: GCS auth configured, CUDA available, base data downloaded
2. Start the watcher: `python -m surveyor.watch`
3. The daemon polls GCS every 5 minutes for:
   - New raw scans needing crown segmentation (DSM+DTM exist but no crowns.gpkg)
   - Missing snapshots that need building
   - Updated raster data that should trigger snapshot patches
4. Report that it's running, what it will watch for

If "status":
1. Check if a watcher process is running (ps aux | grep surveyor.watch)
2. Read ~/.surveyor_state.json for last known state
3. Report: running/stopped, last poll time, known processed items, pending items

If "stop":
1. Find the watcher process and stop it gracefully
2. Report final state
