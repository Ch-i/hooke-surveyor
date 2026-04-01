Search for scientific papers, extract knowledge, and build expertise.

Argument: $ARGUMENTS (topic keyword like "mycorrhizal", "syntropic", "soil carbon", or "all" for full cycle)

This is the core learning loop. You are building expertise in planetary stewardship.

Steps:
1. If "all" or empty: run a full discovery cycle across all topics in knowledge/topics.json
   ```python
   from surveyor.research.loop import run_cycle
   summary = run_cycle(from_year=2015)
   ```

2. If specific topic: search OpenAlex for that topic
   ```python
   from surveyor.research.scholar import search_openalex
   papers = search_openalex("syntropic agriculture agroforestry", max_results=20, from_year=2015)
   ```

3. Review discovered papers — prioritize by:
   - Citation count (well-established knowledge)
   - Open access (you can read the full text)
   - Bioregion relevance (Dorset maritime, temperate, chalk/clay)
   - Recency (recent papers may reference novel techniques)

4. For the top 3-5 papers with abstracts, ANALYZE THEM:
   - Read the abstract (and full text if available via pdf_url)
   - Extract techniques applicable to Hooke Park
   - Extract species-specific quantitative data
   - Note design principles (spatial arrangement, succession timing, guild structure)
   - Rate confidence (meta-analysis > RCT > observational > case study)
   - Rate bioregion match (0-1, how applicable to Dorset 50.79°N maritime)

5. Save extracted knowledge:
   - Techniques → knowledge/techniques/{slug}.json
   - Species insights → knowledge/species/{species_id}.json
   - Update paper index: knowledge/paper_index.json

6. Follow citation trails from high-value papers:
   ```python
   from surveyor.research.scholar import get_references
   refs = get_references(paper_s2_id)
   ```

7. Report: papers found, analyzed, techniques extracted, knowledge gaps identified

IMPORTANT: You are the expert. Read papers critically. Extract what matters for regenerating Hooke Park. Cross-reference with the bioregion profile at knowledge/bioregion/dorset_maritime.json.
