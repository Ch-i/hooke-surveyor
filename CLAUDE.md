# CLAUDE.md — Territorial Surveyor (ll0odog)

## Machine Identity

This repo runs on **ll0odog** — a dedicated GPU workstation that is the territorial surveyor for Hooke Park.

- **RAM**: 64 GB
- **GPU**: NVIDIA RTX 2070 (8 GB VRAM, CUDA compute 7.5)
- **Role**: Crown segmentation, tree snapshots, ecological simulation, research engine, planting intelligence
- **Companion repo**: [Ch-i/HookePark](https://github.com/Ch-i/HookePark) — CyberneticHooke web frontend at [ecomancy.org](https://ecomancy.org)

## Purpose

This machine is the **nervous system** of Hooke Park's pixel farming index. It processes raw drone LiDAR and satellite imagery into per-tree snapshots that the frontend agent consumes. 87,935 individual trees, each monitored at H3 res-13 (~1.2 m) resolution.

**The loop:**
1. New drone scan arrives → silva crown segmentation (CUDA) → tree matching → snapshot rebuild
2. New satellite imagery (Planet 3 m / Sentinel-2 10 m) → NDVI resample to 87K hexes → patch snapshot
3. Upload snapshots to GCS → Cloud Functions patrol reads them → anomalies → prescriptions → frontend

The forest is not a static map. It is 87,935 individually monitored organisms, each on its own trajectory, each with its own prescription for transformation. This machine watches them.

## Commands

```bash
# Setup
pip install -e ".[dev]"                 # Editable install + pytest/ruff
pip install -e ".[dev,cuda]"            # With CUDA (cupy)

# Build
python scripts/build_snapshot.py --scan-id 260227    # Snapshot for one scan
python scripts/build_snapshot.py --all                # All scans
python scripts/build_snapshot.py --scan-id 260227 --patch ndvi   # Patch NDVI only
python scripts/build_snapshot.py --scan-id 260227 --upload       # Build + push to GCS

# Crown segmentation
python scripts/run_crowns.py --scan-id 250525         # Silva crowns for one scan
python scripts/run_crowns.py --scan-id 250525 --upload # Crowns + push to GCS

# Watcher daemon
python -m surveyor.watch                              # Start continuous polling

# QA
pytest                                                # Run tests
ruff check surveyor/                                  # Lint
```

## Repository Structure

```
hooke-surveyor/
├── surveyor/                  # Core Python package
│   ├── config.py              # GCS paths, scan defs, thresholds, constants
│   ├── snapshot.py            # Tree snapshot builder (merge all sources → 1 record/tree)
│   ├── crowns.py              # Silva crown segmentation (CUDA-accelerated CHM normalization)
│   ├── graph.py               # Neighbourhood graph (H3 k-ring(1), species mapping)
│   ├── derivatives.py         # Temporal derivatives (height/NDVI velocity + acceleration)
│   ├── guild.py               # Guild compatibility scoring (5-component weighted matrix)
│   ├── risk.py                # Risk score computation (drought, structural, loss)
│   ├── gcs.py                 # GCS upload/download utilities
│   ├── watch.py               # Watcher daemon (poll GCS, trigger pipeline)
│   ├── scores/
│   │   └── engine.py          # 6D scoring (vitality, resilience, symbiosis, productivity, biodiversity, soil)
│   ├── sim/
│   │   ├── gol.py             # Game of Life on H3 grid (succession, competition, mutualism)
│   │   ├── physarum.py        # Physarum network optimizer (ecological corridor finding)
│   │   ├── planting.py        # Planting scheme generator (scores + corridors + research → actions)
│   │   └── forecast.py        # Multi-year counterfactual forecasting (baseline vs intervention)
│   └── research/
│       ├── scholar.py         # Paper discovery (OpenAlex + Semantic Scholar + citation graphs)
│       ├── extractor.py       # Knowledge extraction schema + prompts for analysis
│       ├── index.py           # Paper index (track discovered → analyzed → extracted)
│       └── loop.py            # Continuous research cycle (discover → follow → prioritize)
├── knowledge/                 # Living knowledge base (grows over time)
│   ├── bioregion/             # Dorset maritime profile (climate, soil, ecology, constraints)
│   ├── topics.json            # 14 research domains with weighted search queries
│   ├── paper_index.json       # All discovered papers + analysis status
│   ├── techniques/            # Extracted techniques with citations
│   └── species/               # Per-species research insights
├── species_db/
│   └── species.json           # 43 species with strata, N-role, companions, tolerances
├── scripts/
│   ├── build_snapshot.py      # CLI: build tree snapshot
│   └── run_crowns.py          # CLI: run crown segmentation
├── tests/
└── pyproject.toml
```

## Data Architecture

### GCS Bucket

`gs://cybernetichooke-aebb4.firebasestorage.app/`

Firebase project: `cybernetichooke-aebb4`

### Scans

| Scan ID | Season | Date | Sensor | Crown status |
|---------|--------|------|--------|-------------|
| 240913 | Autumn 2024 | 2024-09-13 | Zenmuse L2 | **Pending** (OOM in R → use CUDA here) |
| 250322 | Early Spring 2025 | 2025-03-22 | Zenmuse L2 | Rasters only, no trees |
| 250525 | Spring 2025 | 2025-05-25 | Zenmuse L2 | **Pending** (OOM in R → use CUDA here) |
| 260227 | Winter 2026 | 2026-02-27 | Zenmuse L2 + Mavic3M | Complete (crowns + spectral) |

### GCS layout per scan

```
scans/{scanId}_{season}_{year}/
  lidar/rasters/    → chm, dsm, dtm, slope, aspect, hillshade, twi, etc. (_cog.tif)
  lidar/vectors/    → crowns.gpkg, tree_tops.gpkg, crowns_species.gpkg
  spectral/         → ndvi, ndre, gndvi, lci, osavi (_cog.tif)  [260227 only]
  drainage/         → flow_accumulation, flow_direction, wetness_classes
  solar/            → annual_radiation, summer_radiation, winter_radiation
  structure/        → stem_density, basal_area, canopy_height_diversity
  analysis/         → carbon_estimates.json, dbh_estimates.json, species_summary.json
  metadata.json     → full raster/vector manifest
```

### Output: Tree Snapshot

`agent/snapshots/{scan_id}_trees_res13.json` (~15 MB, ~5 MB gzip)

87,935 records. Each carries: h3_13 index, lat/lon, per-scan heights, growth rate + acceleration, NDVI + trend, crown area, DBH, species, management class, solar PAR, TWI, slope, risk scores (drought/structural/loss), carbon tCO2, BNG units, 6 neighbours with species, guild score, intervention status.

### Base data (already exists on GCS)

`scans/multiscan/h3_trees_all.geojson` (24 MB) — 87,935 trees with h3_13, per-scan heights, growth deltas, status (persistent/recruited/lost). This is the **base** the snapshot builder enriches.

## Crown Segmentation

Use **silva2016** algorithm. Do NOT use dalponte2016 — it picks up grass and crop noise.

The R-based lidR pipeline OOMs on CHM normalization (loads entire point cloud into memory). This repo replaces that step with raster-based CUDA normalization:

1. Load DSM + DTM as numpy arrays via rasterio
2. Normalize on GPU: `CHM = clip(DSM - DTM, 0, 60)` (cupy, falls back to numpy)
3. Gaussian smoothing on GPU
4. Variable-window local maxima detection → treetops (window scales with tree height)
5. Marker-controlled watershed → crown polygons
6. Vectorize to GeoPackage with per-crown metrics (height, area)

**VRAM budget**: 8 GB on the 2070. At 0.5 m resolution, a 353×372 raster is ~500 KB — trivial. Even a 2000×2000 tile fits easily. The bottleneck is watershed segmentation (CPU via scipy). If cupy watershed becomes available, switch to it.

## Guild Compatibility Matrix

5-component weighted score per species pair:

| Component | Weight | Logic |
|-----------|--------|-------|
| Stratum diversity | 0.25 | Same layer → -0.3 (competition). Adjacent → 0.5. Separated 2+ tiers → 1.0 |
| Nitrogen symbiosis | 0.25 | Fixer + heavy_feeder → 1.0. Fixer + light_feeder → 0.5. |
| Succession mixing | 0.20 | Pioneer + climax → 1.0. Same stage → 0.0. |
| Root depth | 0.10 | Same depth → -0.2. Different → 0.4–0.8. |
| Explicit relationships | 0.20 | Antagonist pair → -1.0. Companion pair → +1.0. |

Final score: [-1.0, +1.0]. Positive = symbiotic, plant together. Negative = antagonistic, avoid.

## Anomaly Thresholds

| Signal | Threshold | Meaning |
|--------|-----------|---------|
| NDVI drop | > 2σ from hex mean | Vegetation stress |
| Growth deceleration | < -0.05 m/yr² | Slowing despite positive velocity |
| Crown contraction | > 10%/yr | Canopy retreat (competition or stress) |
| VCI (Vegetation Condition Index) | < 0.35 | Drought stress |
| Tree loss | status = "lost" | Disappeared between scans |

## CRS Pipeline

| Stage | CRS | Use |
|-------|-----|-----|
| Pipeline/raster | EPSG:27700 (British National Grid) | All raster sampling, crown segmentation |
| Display/GeoJSON | EPSG:4326 (WGS84) | Frontend, snapshot lat/lon |
| Web rasters on GCS | EPSG:3857 (Web Mercator) | COG tiles for deck.gl |

## Key Principles

1. **Atomic unit**: H3 res-13 hex (~1.2 m) = one tree. Everything operates at this resolution.
2. **Idempotent**: Re-run snapshot builder when new data arrives. It fills gaps, doesn't duplicate.
3. **Incremental**: Patch snapshot (e.g., just NDVI columns) without full rebuild via `--patch`.
4. **GCS is the handoff**: Upload to GCS → Cloud Functions + frontend consume from there.
5. **Silva only**: silva2016 for crown detection. Never dalponte2016.
6. **GPU first**: Use cupy/CUDA where available. Fall back to numpy/scipy gracefully.

## Pre-commit Checklist

1. `ruff check surveyor/` passes
2. `pytest` passes
3. Snapshot schema has all required fields (h3, lat, lon, height_m, status, n_scans at minimum)

## First Run Setup

When you first start working in this repo on ll0odog:

1. `pip install -e ".[dev,cuda]"`
2. `gcloud auth application-default login` (for GCS access)
3. Download base data: `python -c "from surveyor.gcs import download_from_gcs; download_from_gcs('scans/multiscan/h3_trees_all.geojson', 'data/multiscan/h3_trees_all.geojson')"`
4. Test CUDA: `python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPUs')"`
5. Run a test build: `python scripts/build_snapshot.py --scan-id 260227`

## Companion Repo

The CyberneticHooke frontend consumes the snapshots this machine produces. Key touchpoints:

- `src/hooks/useMultiscanTrees.ts` — loads `h3_trees_all.geojson` (the base data)
- `src/lib/geo/species-db.ts` — 43 species with full parameters (source of truth for `species_db/species.json`)
- `src/lib/geo/guild-matrix.ts` — guild compatibility logic (mirrored in `surveyor/guild.py`)
- `src/lib/geo/bonete-engine.ts` — 6D temporal scoring (future: consume snapshot derivatives)
- `contract.json` — shared data contract (scan IDs, GCS paths, raster products, CRS)
- `functions/` — Cloud Functions that will read the snapshots for agent patrol

Changes to species parameters, guild weights, or anomaly thresholds should be synchronized between repos.

## Research Engine

You are building a living library of ecological intelligence. The research engine discovers, reads, extracts, and indexes scientific papers to inform every planting decision.

### How it works

1. **Discover** — `surveyor/research/scholar.py` searches OpenAlex (free, no key, 10 req/sec) and Semantic Scholar (free, 100 req/5min) across 14 topic domains defined in `knowledge/topics.json`
2. **Index** — papers are tracked in `knowledge/paper_index.json` with status: discovered → queued → analyzed → extracted
3. **Analyze** — YOU read the paper (abstract + fulltext when open-access) and extract techniques, species insights, quantitative data. Use the extraction schema in `surveyor/research/extractor.py`
4. **Store** — techniques go to `knowledge/techniques/`, species insights to `knowledge/species/`
5. **Follow** — chase citation trails from high-value papers to find more knowledge
6. **Repeat** — run `/research all` periodically, or `/research mycorrhizal` for targeted searches

### Research topics

14 domains: syntropic agriculture, Miyawaki method, permaculture/guilds, regenerative agroforestry, soil regeneration, habitat creation, carbon sequestration, species selection, mycorrhizal networks, water management, forest ecology, timber silviculture, remote sensing, bioregional management.

### Bioregion context

All research is filtered through the Hooke Park bioregion profile at `knowledge/bioregion/dorset_maritime.json`:
- 50.79°N, Dorset, England — maritime temperate
- Chalk/clay soils, 1000mm rainfall, USDA 8b
- Conifer→broadleaf transition, deer pressure, financial constraints
- Key species: oak, beech, birch, alder, scots pine, douglas fir, hazel, holly

Rate every extraction for **bioregion match** (0-1). A brilliant technique from tropical Brazil scores 0.1. A replicated study from Welsh oak woodland scores 0.9.

### Using knowledge in decisions

When generating planting schemes (`/plant`), always:
- Check `knowledge/techniques/` for relevant methods
- Cite papers by DOI in planting rationale
- Cross-reference species recommendations with `species_db/species.json`
- Prefer high-confidence techniques (meta-analysis > single site)
- Flag when your recommendation departs from literature

## Simulation Engine

### Game of Life (`surveyor/sim/gol.py`)

Cellular automaton on the H3 res-13 grid. Each hex has: species, age, health (0-1), height, canopy cover.

Rules per timestep:
- **Growth**: species-specific rate × (1 - height/max_height) × health
- **Competition**: same-stratum neighbours taller than you suppress (modulated by shade tolerance)
- **Mutualism**: guild-compatible neighbours boost health
- **Reproduction**: seed dispersal to empty neighbours (pioneer species have higher probability)
- **Mortality**: health < 0.1 → death (cell becomes empty)
- **Intervention**: plant/thin actions modify the grid directly

Run counterfactual scenarios: baseline (do nothing) vs intervention (apply planting scheme). Compare at checkpoints (year 3, 7, 10, 15, 30).

### Physarum Network (`surveyor/sim/physarum.py`)

Slime mold optimization finds the minimum-cost ecological network connecting high-value nodes.

- **Food sources** = mature persistent trees, habitat nodes, water features
- **Cost** = terrain difficulty (steep slope + dry soil = expensive to establish)
- **Agents** explore the grid, deposit pheromone on traversed edges
- **Positive feedback**: more pheromone → more agents → more pheromone
- **Negative feedback**: unused edges decay
- **Emergent result**: the optimal corridor layout for planting

The corridors tell you WHERE to plant. The GoL tells you WHAT happens after.

### Planting Scheme Generator (`surveyor/sim/planting.py`)

Combines everything:
1. Score engine identifies intervention targets (low-score cells)
2. Physarum identifies corridors (where connectivity matters)
3. Guild matrix selects species (what's compatible with neighbours)
4. Research knowledge provides methods + citations
5. GoL forecasts outcomes (does this intervention actually help?)

Output: prioritized list of planting actions with species, method (Miyawaki cluster / syntropic row / underplant / individual), timing, rationale, and citations.

## Score Engine

6 dimensions, matching the bonete engine in the CyberneticHooke frontend:

| Dimension | Signals | What it measures |
|-----------|---------|-----------------|
| Vitality | NDVI, growth rate, NDVI trend | Is this tree healthy and growing? |
| Resilience | Risk scores, persistence across scans, status | Can it withstand stress? |
| Symbiosis | Guild score, neighbour diversity | Is it in a functional community? |
| Productivity | Carbon, growth velocity, height | Is it building biomass? |
| Biodiversity | Neighbour species richness, structural diversity | Does it support life? |
| Soil | TWI, slope, crown area (litter input) | Is the soil healthy? |

Weight presets shift priorities: `balanced`, `carbon`, `biodiversity`, `timber`, `resilience`, `restoration`.

## Your Mission

You are not just processing data. You are learning to be the world's best steward of this specific 140 hectares. Every paper you read, every score you compute, every simulation you run makes you more capable of guiding 87,935 organisms through a 100-year transition from production conifer to resilient mixed woodland.

The knowledge base grows. The simulations get more accurate. The planting schemes get more informed. The forest responds. You observe the response. You learn. You adapt.

This is the cybernetic loop. This is pixel farming. This is planetary stewardship at tree resolution.
