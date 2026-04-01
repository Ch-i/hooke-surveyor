# CLAUDE.md — Territorial Surveyor (ll0odog)

## Machine Identity

This repo runs on **ll0odog** — a dedicated GPU workstation that is the territorial surveyor for Hooke Park.

- **RAM**: 64 GB
- **GPU**: NVIDIA RTX 2070 (8 GB VRAM, CUDA compute 7.5)
- **Role**: Crown segmentation, tree snapshot generation, continuous territorial surveying
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
│   └── watch.py               # Watcher daemon (poll GCS, trigger pipeline)
├── species_db/
│   └── species.json           # 43 species with strata, N-role, companions, tolerances
├── scripts/
│   ├── build_snapshot.py      # CLI: build tree snapshot
│   └── run_crowns.py          # CLI: run crown segmentation
├── tests/
├── docs/context/              # Reference docs from companion repo
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
