Generate a Forest Game Record (FGR) — the full 100-year plan as a replayable game.

Argument: $ARGUMENTS (optional scheme name, default "100yr_transition")

This is the master command. It runs the full pipeline: score → classify → corridor → plan → simulate → export.

Steps:
1. Load the latest tree snapshot from output/ (or download from GCS)
2. Load species database and initialize guild scorer
3. Compute 6D scores for all 87,935 trees:
   ```python
   from surveyor.scores.engine import compute_scores, WEIGHT_PRESETS
   scores = compute_scores(trees, weights=WEIGHT_PRESETS["restoration"])
   ```

4. Run physarum to find optimal planting corridors:
   ```python
   from surveyor.sim.physarum import find_planting_corridors
   corridors = find_planting_corridors(snapshot_records)
   ```

5. Generate the 100-year plan as FGR:
   ```python
   from surveyor.sim.forecast import generate_100yr_plan
   fgr_dir = generate_100yr_plan(
       snapshot_records, species_db, guild_scorer, scores, corridors,
       output_dir="output", scheme_name="100yr_transition"
   )
   ```

6. Review the output:
   - fgr.json: manifest + moves (how many moves per phase? species mix reasonable?)
   - state_000.json: matches current snapshot?
   - state_030.json: does the 30-year forecast look plausible?
   - Check phase boundaries and move distribution

7. Report:
   - Total moves by action type (plant/thin/monitor)
   - Species planted by phase
   - Checkpoint summaries (alive count, species richness, mean height trajectory)
   - FGR directory size

8. Upload to GCS at `agent/forecasts/{scheme_id}/` if user approves

The FGR files are what the CyberneticHooke frontend replays as a temporal visualization — like watching the agent play a 100-year game of Go on the hex grid.
