Run a Game of Life + physarum simulation to forecast forest futures.

Argument: $ARGUMENTS (optional: H3 block ID, area name, or "full" for entire site. Add "+intervention" to test a planting scheme.)

Steps:
1. Load the latest tree snapshot from output/
2. Load species database and guild scorer

3. If specific area: filter snapshot to target H3 cells
   If "full": use all 87,935 trees (may be slow — consider subsampling)

4. Run physarum to find optimal corridors:
   ```python
   from surveyor.sim.physarum import find_planting_corridors
   corridors = find_planting_corridors(snapshot_records)
   ```
   Report: how many cells are in the network, spatial pattern

5. Run GoL forecast — two scenarios:
   ```python
   from surveyor.sim.forecast import run_forecast
   result = run_forecast(
       snapshot_records, species_db, guild_scorer,
       interventions=planting_actions,  # or None for baseline
       years=30, checkpoints=[3, 7, 10, 15, 30]
   )
   ```

6. Compare baseline vs intervention at each checkpoint:
   - Living trees
   - Species richness
   - Mean height
   - Mean health
   - Canopy cover %

7. Report results as a timeline:
   ```
   Year 0  → 45,000 alive, 12 species, 15.2m mean height
   Year 3  → baseline: 44,800 / intervention: 46,200 (+1,400)
   Year 7  → baseline: 44,200 / intervention: 48,500 (+4,300)
   Year 15 → baseline: 43,000 / intervention: 52,800 (+9,800)
   Year 30 → baseline: 41,500 / intervention: 58,200 (+16,700)
   ```

8. If intervention tested: identify which planting actions had the most impact

IMPORTANT: The GoL is a simplified model. Use it for relative comparison between scenarios, not absolute prediction. The value is in understanding WHICH interventions matter most and WHERE.
