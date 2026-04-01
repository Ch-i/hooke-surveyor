Review accumulated knowledge, find gaps, deepen expertise.

This is the reflective phase of the research loop. You are becoming an expert of experts.

Steps:
1. Load current knowledge state:
   ```python
   from surveyor.research.index import get_stats
   stats = get_stats()
   ```

2. Review what you know:
   - Read knowledge/paper_index.json — how many papers per topic?
   - Read knowledge/techniques/ — what methods have you extracted?
   - Read knowledge/species/ — which species have research backing?
   - Read knowledge/bioregion/dorset_maritime.json — are research priorities covered?

3. Identify gaps:
   - Which topics have few papers? Search for more.
   - Which species in species_db/species.json lack research backing? Target them.
   - Which research priorities from the bioregion profile are underserved?
   - Are there techniques you've found that need deeper citation trails?

4. Cross-reference:
   - Do extracted techniques conflict with each other? (e.g., different spacing recommendations)
   - Do species recommendations from papers match the guild matrix?
   - Are there quantitative parameters you can feed into the GoL simulation?
   - Are there novel species or techniques not in the current species database?

5. Synthesize:
   - Write a brief synthesis note for each knowledge domain
   - Update technique confidence levels based on cumulative evidence
   - Flag any paradigm shifts or surprising findings

6. Queue next research:
   - Identify the 5 most valuable searches to run next
   - Prioritize by: knowledge gap severity × management relevance

7. Report:
   - Knowledge coverage by topic (% of research priorities addressed)
   - Key findings that should influence planting decisions
   - Recommended next research targets

You are building a living library of ecological intelligence. Every cycle makes you more capable of stewarding this land.
