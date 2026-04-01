Generate a planting scheme for a target area, informed by scores + corridors + research.

Argument: $ARGUMENTS (H3 block ID, area description like "NW monoculture edge", or "full" for site-wide)

This is the decision engine. Every planting action should be justified by data + research.

Steps:
1. Load latest snapshot and compute scores:
   ```python
   from surveyor.scores.engine import compute_scores, WEIGHT_PRESETS
   scores = compute_scores(trees, weights=WEIGHT_PRESETS["restoration"])
   ```

2. Run physarum to find optimal corridors:
   ```python
   from surveyor.sim.physarum import find_planting_corridors
   corridors = find_planting_corridors(snapshot_records)
   ```

3. Generate planting scheme:
   ```python
   from surveyor.sim.planting import generate_planting_scheme
   scheme = generate_planting_scheme(
       snapshot_records, scores, corridors, species_db, guild_scorer,
       target_area=target_h3_ids, knowledge_dir="knowledge"
   )
   ```

4. Review the scheme critically:
   - Does the species mix make ecological sense? (check guild compatibility)
   - Are the spatial arrangements optimal? (Miyawaki clusters vs syntropic rows)
   - Does it align with research findings in knowledge/techniques/?
   - Does it respect site constraints? (access, deer pressure, campus proximity)
   - Is the planting window realistic?

5. Forecast the scheme:
   ```python
   from surveyor.sim.forecast import run_forecast
   result = run_forecast(snapshot_records, species_db, guild_scorer, interventions, years=30)
   ```

6. Attach citations — every technique should reference supporting papers:
   - "Miyawaki cluster" → cite Miyawaki method papers
   - "N-fixer companion" → cite mycorrhizal/nitrogen cycling papers
   - "Drought-adapted provenance" → cite climate adaptation papers

7. Output the scheme as structured JSON:
   - Per-cluster actions with species, method, priority, rationale, citations
   - Summary: total trees, species mix, estimated carbon gain, timeline
   - Map data: H3 IDs for visualization in the frontend

8. If the user approves, save to output/planting_scheme_{area}_{date}.json

The scheme is a living document — it will be updated as new data arrives, new research is found, and the forest responds to previous interventions.
