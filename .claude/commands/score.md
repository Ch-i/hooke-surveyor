Compute and display multi-dimensional scores for trees or blocks.

Argument: $ARGUMENTS (H3 ID for a single tree, H3 res-11 ID for a block, or "summary" for site-wide. Optionally add a weight preset: "carbon", "biodiversity", "timber", "resilience", "restoration")

Steps:
1. Load latest snapshot
2. Determine weight preset (default: "balanced"):
   - balanced: equal weights across all 6 dimensions
   - carbon: emphasizes productivity + soil
   - biodiversity: emphasizes symbiosis + biodiversity
   - timber: emphasizes vitality + productivity
   - resilience: emphasizes resilience + symbiosis
   - restoration: emphasizes biodiversity + soil + symbiosis

3. Compute scores:
   ```python
   from surveyor.scores.engine import compute_scores, WEIGHT_PRESETS
   scores = compute_scores(trees, weights=WEIGHT_PRESETS[preset])
   ```

4. If single tree (H3 res-13):
   - Show all 6 dimension scores + overall
   - Show the tree's snapshot data (height, species, NDVI, neighbours)
   - Explain what's driving each score (what's strong, what's weak)
   - Compare to site average

5. If block (H3 res-11):
   - Aggregate scores across all trees in the block
   - Show dimension breakdown
   - Identify weakest trees in the block (intervention candidates)
   - Compare to adjacent blocks

6. If "summary":
   - Show distribution of overall scores (histogram-style)
   - Top 10 highest-scoring trees (paragons to protect)
   - Bottom 10 lowest-scoring trees (urgent intervention)
   - Score by species (which species are thriving/struggling)
   - Score by zone (geographic patterns)

Report as a clear table with the 6 dimensions + overall.
