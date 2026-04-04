"""metamorphicTerritory — FastAPI service for the planting scheme tile server.

Runs the full metamorphic engine (GoL + physarum corridors + planting),
exports H3 hex polygons as GeoJSON with all properties, then runs
tippecanoe → PMTiles + XYZ tile directory for deck.gl MVTLayer.

Endpoints:
  POST /simulate           — run engine, export tiles
  GET  /tiles/{z}/{x}/{y}.pbf — individual MVT vector tiles
  GET  /scheme.pmtiles     — full PMTiles archive
  GET  /scheme.geojson     — full GeoJSON export
  GET  /cell/{h3_index}    — individual cell detail
  GET  /snapshots          — simulation metrics over time
  GET  /classifications    — block classifications
  GET  /verification       — latest verification results
  GET  /health             — status check
"""

import asyncio
import json
import logging
import threading
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import h3
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("metamorphic")

app = FastAPI(title="metamorphicTerritory", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "output"))
TILES_DIR = DATA_DIR / "tiles"
GEOJSON_PATH = DATA_DIR / "scheme.geojson"
PMTILES_PATH = DATA_DIR / "scheme.pmtiles"
SNAPSHOT_PATH = DATA_DIR / "snapshots.json"

PORT = int(os.environ.get("PORT", "8420"))

# Cache for cell lookup
_cell_index: dict[str, dict] = {}
_simulation_summaries: list[dict] = []
# Cache for new worker outputs
_latest_classifications: dict = {}
_latest_verification: dict = {}


class SimulateRequest(BaseModel):
    lat: float = 50.791
    lng: float = -2.669
    k_ring: int = 18
    resolution: int = 12
    years: int = 20
    scan_id: str = "260227"
    seed: int = 42


@app.get("/health")
async def health():
    has_tiles = TILES_DIR.exists() and any(TILES_DIR.rglob("*.pbf"))
    has_pmtiles = PMTILES_PATH.exists()
    has_geojson = GEOJSON_PATH.exists()
    return {
        "status": "ok",
        "has_tiles": has_tiles,
        "has_pmtiles": has_pmtiles,
        "has_geojson": has_geojson,
        "cell_count": len(_cell_index),
        "has_classifications": bool(_latest_classifications),
        "has_verification": bool(_latest_verification),
    }


@app.post("/simulate")
async def simulate(req: SimulateRequest):
    """Run the full metamorphic engine and generate vector tiles."""
    t0 = time.time()
    logger.info(f"Simulate: center=({req.lat},{req.lng}), k_ring={req.k_ring}, res={req.resolution}, years={req.years}")

    try:
        np.random.seed(req.seed)
        geojson = _run_engine(req)
        _write_geojson(geojson)
        has_tippecanoe = _try_tippecanoe()
        elapsed = time.time() - t0
        logger.info(f"Simulation complete: {len(geojson['features'])} features, {elapsed:.1f}s")
        return {
            "status": "ok",
            "features": len(geojson["features"]),
            "elapsed_s": round(elapsed, 1),
            "has_tiles": has_tippecanoe,
            "geojson": str(GEOJSON_PATH),
            "classifications": len(_latest_classifications),
            "verification": _latest_verification.get("passed"),
        }
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tiles/{z}/{x}/{y}.pbf")
async def get_tile(z: int, x: int, y: int):
    tile_path = TILES_DIR / str(z) / str(x) / f"{y}.pbf"
    if not tile_path.exists():
        raise HTTPException(status_code=204)  # empty tile, not an error
    return FileResponse(
        tile_path,
        media_type="application/x-protobuf",
        headers={"Content-Encoding": "gzip", "Access-Control-Allow-Origin": "*"},
    )


@app.get("/scheme.pmtiles")
async def get_pmtiles():
    if not PMTILES_PATH.exists():
        raise HTTPException(status_code=404, detail="PMTiles not generated yet. POST /simulate first.")
    return FileResponse(PMTILES_PATH, media_type="application/octet-stream")


@app.get("/scheme.geojson")
async def get_geojson():
    if not GEOJSON_PATH.exists():
        raise HTTPException(status_code=404, detail="GeoJSON not generated yet. POST /simulate first.")
    return FileResponse(GEOJSON_PATH, media_type="application/geo+json")


@app.get("/cell/{h3_index}")
async def get_cell(h3_index: str):
    if h3_index in _cell_index:
        return _cell_index[h3_index]
    raise HTTPException(status_code=404, detail=f"Cell {h3_index} not found")


@app.get("/snapshots")
async def get_snapshots():
    return _simulation_summaries


@app.get("/classifications")
async def get_classifications():
    """Return block classifications."""
    if not _latest_classifications:
        raise HTTPException(status_code=404, detail="No classifications available. POST /simulate first.")
    # Serialize BlockClassification objects to dicts
    result = {}
    for h3_11, cls in _latest_classifications.items():
        if hasattr(cls, "management_class"):
            result[h3_11] = {
                "management_class": cls.management_class,
                "confidence": cls.confidence,
                "strategy_name": cls.strategy_name,
                "features": cls.features,
                "tree_count": cls.tree_count,
            }
        else:
            result[h3_11] = cls
    return result


@app.get("/verification")
async def get_verification():
    """Return latest verification results."""
    if not _latest_verification:
        raise HTTPException(status_code=404, detail="No verification results. POST /simulate first.")
    return _latest_verification


# ── Engine ──────────────────────────────────────────────────────────────────


def _run_engine(req: SimulateRequest) -> dict:
    """Run the metamorphic engine: load snapshot → simulate → export GeoJSON.

    Restructured flow with worker modules:
      1. Load snapshot + species DB
      2. Fill terrain (Worker 5 — terrain.py)
      3. Classify blocks (Worker 1 — classify.py)
      4. Compute scores (existing)
      5. Physarum corridors (existing)
      6. Generate planting scheme WITH classifications (Worker 2 — planting.py)
      7. Phase moves (Worker 3 — forecast.py)
      8. Build cell_to_block mapping for per-class GoL (Worker 4 — gol.py)
      9. Initialize GoL with per-class configs
     10. Step with phased interventions
     11. Verify (Worker 6 — verify.py)
     12. Export GeoJSON with management_class in properties
    """
    global _cell_index, _simulation_summaries, _latest_classifications, _latest_verification

    # ── 1. Load snapshot ─────────────────────────────────────────────────
    snapshot_path = Path(os.environ.get("SURVEYOR_OUTPUT", "../output"))
    snapshot_file = snapshot_path / f"{req.scan_id}_trees_res13.json"

    if not snapshot_file.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_file}")

    with open(snapshot_file) as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records)} trees from {snapshot_file.name}")

    # Load species database
    species_path = Path(os.environ.get("SPECIES_DB", "../species_db/species.json"))
    with open(species_path) as f:
        species_db = {s["id"]: s for s in json.load(f)}

    # Import simulation engines
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from surveyor.guild import compatibility_score
    from surveyor.sim.physarum import find_planting_corridors
    from surveyor.sim.planting import generate_planting_scheme
    from surveyor.scores.engine import compute_scores, WEIGHT_PRESETS

    # ── 2. Fill terrain (Worker 5) ───────────────────────────────────────
    from surveyor.sim.terrain import fill_terrain, compute_block_features
    records = fill_terrain(records)
    logger.info(f"Terrain filled for {len(records)} records")

    # ── 3. Classify blocks (Worker 1) ────────────────────────────────────
    from surveyor.sim.classify import classify_all_blocks
    classifications = classify_all_blocks(records, species_db)
    _latest_classifications = classifications
    logger.info(f"Classified {len(classifications)} blocks into management classes")

    # ── 4. Compute scores (existing) ─────────────────────────────────────
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        pd.DataFrame(records),
        geometry=[Point(r["lon"], r["lat"]) for r in records],
        crs="EPSG:4326",
    )
    scores = compute_scores(gdf, weights=WEIGHT_PRESETS["restoration"])

    # ── 5. Physarum corridors (existing) ─────────────────────────────────
    corridors = find_planting_corridors(records)

    # ── 6. Generate planting scheme WITH classifications (Worker 2) ──────
    scheme = generate_planting_scheme(
        records, scores, corridors, species_db, compatibility_score,
        classifications=classifications,  # NEW: class-aware species selection
    )

    # ── 7. Phase moves (Worker 3) ────────────────────────────────────────
    from surveyor.sim.forecast import scheme_to_phased_moves
    moves = scheme_to_phased_moves(scheme, species_db, classifications)
    moves_by_year = {}
    for m in moves:
        moves_by_year.setdefault(m["year"], []).append(m)
    logger.info(f"Phased {len(moves)} moves across {len(moves_by_year)} distinct years")

    # ── 8. Build cell_to_block mapping for GoL (Worker 4) ────────────────
    cell_to_block = {}
    for rec in records:
        h13 = rec.get("h3_13") or rec.get("h3")
        if not h13:
            continue
        try:
            parent = h3.cell_to_parent(h13, 11)
        except Exception:
            continue
        cls = classifications.get(parent)
        if cls:
            cell_to_block[h13] = cls.management_class

    logger.info(f"Mapped {len(cell_to_block)} cells to management classes")

    # ── 9. Initialize GoL with per-class configs ─────────────────────────
    from surveyor.sim.gol import ForestGoL, GoLConfig, GOL_CLASS_CONFIGS
    gol = ForestGoL(
        species_db, compatibility_score,
        config=GoLConfig(),
        block_configs=GOL_CLASS_CONFIGS,
        cell_to_block=cell_to_block,
    )
    gol.seed_from_snapshot(records)

    # ── 10. Step with phased interventions ───────────────────────────────
    checkpoints = [0, 1, 3, 5, 7, 10, 15, 20]
    if req.years > 20:
        checkpoints.extend([y for y in [30, 50, 75, 100] if y <= req.years])

    _simulation_summaries = []
    checkpoint_dir = DATA_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Collect checkpoint states for verification
    checkpoint_states = {}

    # Save year 0 state (initial snapshot)
    _save_checkpoint(gol.grid, species_db, 0, checkpoint_dir)
    checkpoint_states[0] = {
        h3_id: {"s": c.species, "h": round(c.height_m, 2), "hp": round(c.health, 3)}
        for h3_id, c in gol.grid.items()
    }

    for yr in range(1, req.years + 1):
        # Apply this year's phased moves
        if yr in moves_by_year:
            interventions = [
                {
                    "h3": m["h3"],
                    "action": m.get("action", "plant"),
                    "species": m["species"],
                }
                for m in moves_by_year[yr]
                if m.get("species") or m.get("action") == "thin"
            ]
            if interventions:
                gol.apply_intervention(interventions)

        summary = gol.step()

        if yr in checkpoints or yr == req.years:
            alive = sum(1 for c in gol.grid.values() if c.is_alive)
            species_set = set(c.species for c in gol.grid.values() if c.is_alive and c.species)
            _simulation_summaries.append({
                "year": yr,
                "alive": alive,
                "species_richness": len(species_set),
                "mean_height": round(np.mean([c.height_m for c in gol.grid.values() if c.is_alive]) if alive else 0, 2),
                "mean_health": round(np.mean([c.health for c in gol.grid.values() if c.is_alive]) if alive else 0, 3),
            })
            _save_checkpoint(gol.grid, species_db, yr, checkpoint_dir)
            # Capture checkpoint state for verification
            checkpoint_states[yr] = {
                h3_id: {"s": c.species, "h": round(c.height_m, 2), "hp": round(c.health, 3)}
                for h3_id, c in gol.grid.items()
            }

    # ── 11. Verify (Worker 6) ────────────────────────────────────────────
    from surveyor.sim.verify import verify_simulation
    verification = verify_simulation(checkpoint_states, classifications, species_db)
    _latest_verification = verification.to_dict()
    logger.info(f"Verification: {_latest_verification['summary']}")

    # ── 12. Export GeoJSON with management_class in properties ───────────
    features = []
    _cell_index = {}

    for h3_id, cell in gol.grid.items():
        try:
            boundary = h3.cell_to_boundary(h3_id)
            # h3 returns (lat, lng) tuples — GeoJSON needs [lng, lat]
            coords = [[lng, lat] for lat, lng in boundary]
            coords.append(coords[0])  # close the ring
        except Exception:
            continue

        sp_data = species_db.get(cell.species, {}) if cell.species else {}

        # Look up management class for this cell
        mgmt_class = cell_to_block.get(h3_id, "")
        mgmt_cls_obj = None
        if mgmt_class:
            try:
                parent_11 = h3.cell_to_parent(h3_id, 11)
                mgmt_cls_obj = classifications.get(parent_11)
            except Exception:
                pass

        props = {
            "h3": h3_id,
            "species": cell.species,
            "species_common": sp_data.get("common", cell.species),
            "stratum": sp_data.get("stratum", "unknown"),
            "succession": sp_data.get("succession", "unknown"),
            "height_m": round(cell.height_m, 2),
            "health": round(cell.health, 3),
            "canopy_cover": round(cell.canopy_cover, 3),
            "age_years": round(cell.age_years, 1),
            "alive": int(cell.is_alive),
            "growth_rate": sp_data.get("growth_rate", 0),
            "shade_tolerance": sp_data.get("shade_tolerance", 0),
            "drought_tolerance": sp_data.get("drought_tolerance", 0),
            "nitrogen_role": sp_data.get("nitrogen_role", "neutral"),
            # NEW: management class from block classification
            "management_class": mgmt_class,
            "strategy_name": mgmt_cls_obj.strategy_name if mgmt_cls_obj else "",
        }

        # Add score if available
        sc = scores.get(h3_id)
        if sc:
            props["score_overall"] = round(sc.get("overall", 0), 3)

        # Add corridor strength
        props["corridor"] = round(corridors.get(h3_id, 0), 3)

        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": props,
        }
        features.append(feature)
        _cell_index[h3_id] = props

    logger.info(f"Generated {len(features)} GeoJSON features")

    return {
        "type": "FeatureCollection",
        "features": features,
    }



# ── Checkpoint I/O ──

def _save_checkpoint(grid, species_db, year: int, checkpoint_dir: Path):
    """Save compact per-cell state at a checkpoint year."""
    states = {}
    for h3_id, cell in grid.items():
        if not cell.is_alive:
            states[h3_id] = {"s": None, "h": 0, "hp": 0}
            continue
        sp_data = species_db.get(cell.species, {}) if cell.species else {}
        states[h3_id] = {
            "s": cell.species,
            "h": round(cell.height_m, 2),
            "hp": round(cell.health, 3),
            "cc": round(cell.canopy_cover, 3),
            "st": sp_data.get("stratum", ""),
            "nr": sp_data.get("nitrogen_role", ""),
        }
    out = checkpoint_dir / f"year_{year:03d}.json"
    with open(out, "w") as f:
        json.dump(states, f, separators=(",", ":"))
    logger.info(f"Checkpoint saved: year {year}, {len(states)} cells -> {out}")


_checkpoint_cache: dict[int, dict] = {}


def _load_checkpoint(year: int) -> dict | None:
    """Load a checkpoint, with in-memory cache."""
    if year in _checkpoint_cache:
        return _checkpoint_cache[year]
    checkpoint_dir = DATA_DIR / "checkpoints"
    path = checkpoint_dir / f"year_{year:03d}.json"
    if not path.exists():
        # Find nearest available checkpoint
        available = sorted(int(f.stem.split("_")[1]) for f in checkpoint_dir.glob("year_*.json"))
        if not available:
            return None
        nearest = min(available, key=lambda y: abs(y - year))
        path = checkpoint_dir / f"year_{nearest:03d}.json"
    with open(path) as f:
        data = json.load(f)
    _checkpoint_cache[year] = data
    return data


@app.get("/state/{year}")
async def get_state(year: int):
    """Get per-cell planting state at a checkpoint year.

    Returns {h3_id: {s: species, h: height, hp: health, cc: canopy_cover, st: stratum, nr: nitrogen_role}}
    The frontend uses this to color hexagons based on timeline position.
    """
    data = _load_checkpoint(year)
    if not data:
        raise HTTPException(404, f"No checkpoint near year {year}")
    return JSONResponse(data, headers={"Cache-Control": "public, max-age=3600"})


@app.get("/checkpoints")
async def list_checkpoints():
    """List available checkpoint years."""
    checkpoint_dir = DATA_DIR / "checkpoints"
    if not checkpoint_dir.exists():
        return {"years": []}
    years = sorted(int(f.stem.split("_")[1]) for f in checkpoint_dir.glob("year_*.json"))
    return {"years": years}



def _write_geojson(geojson: dict):
    """Write GeoJSON to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(GEOJSON_PATH, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))
    size_mb = GEOJSON_PATH.stat().st_size / 1024 / 1024
    logger.info(f"GeoJSON written: {GEOJSON_PATH} ({size_mb:.1f} MB)")


def _try_tippecanoe() -> bool:
    """Run tippecanoe if available. Returns True if tiles were generated."""
    import shutil

    if not shutil.which("tippecanoe"):
        logger.warning("tippecanoe not found — skipping tile generation. GeoJSON endpoint still works.")
        return False

    TILES_DIR.mkdir(parents=True, exist_ok=True)

    base_args = [
        "-z", "18", "-Z", "8",
        "--no-feature-limit",
        "--no-tile-size-limit",
        "-l", "planting",
        "--force",
        str(GEOJSON_PATH),
    ]

    # 1. PMTiles archive
    cmd_pm = ["tippecanoe", "-o", str(PMTILES_PATH)] + base_args
    logger.info(f"Building PMTiles: {' '.join(cmd_pm)}")
    r1 = subprocess.run(cmd_pm, capture_output=True, text=True, timeout=300)
    if r1.returncode != 0:
        logger.error(f"tippecanoe PMTiles failed: {r1.stderr}")
        return False

    # 2. Tile directory for XYZ serving
    cmd_dir = ["tippecanoe", "-e", str(TILES_DIR), "--no-tile-compression"] + base_args
    logger.info(f"Building tile dir: {' '.join(cmd_dir)}")
    r2 = subprocess.run(cmd_dir, capture_output=True, text=True, timeout=300)
    if r2.returncode != 0:
        logger.error(f"tippecanoe tile dir failed: {r2.stderr}")

    tile_count = sum(1 for _ in TILES_DIR.rglob("*.pbf"))
    pmtiles_mb = PMTILES_PATH.stat().st_size / 1024 / 1024 if PMTILES_PATH.exists() else 0
    logger.info(f"Tiles generated: {tile_count} tiles, PMTiles {pmtiles_mb:.1f} MB")
    return True



# ── Perpetual mode state ──
_perpetual_gol = None           # ForestGoL instance (persists across seasons)
_perpetual_species_db = None    # species DB reference
_perpetual_year = 0             # current simulation year
_perpetual_season = 0           # current season within year (0-3)
_perpetual_running = False
_perpetual_lock = threading.Lock()
_ws_clients: list[WebSocket] = []


async def _broadcast(msg: dict):
    """Send a message to all connected WebSocket clients."""
    dead = []
    data = json.dumps(msg, separators=(",", ":"))
    for ws in _ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


def _perpetual_step() -> dict | None:
    """Run one GoL season step. Returns diff of changed cells."""
    global _perpetual_year, _perpetual_season

    with _perpetual_lock:
        if _perpetual_gol is None:
            return None

        # Snapshot pre-state
        pre = {}
        for h3_id, cell in _perpetual_gol.grid.items():
            pre[h3_id] = (cell.species, round(cell.height_m, 2), round(cell.health, 3))

        # Step
        _perpetual_gol.step()
        _perpetual_season += 1
        if _perpetual_season >= 4:
            _perpetual_season = 0
            _perpetual_year += 1

        # Compute diff — only cells that changed
        diff = {}
        for h3_id, cell in _perpetual_gol.grid.items():
            post = (cell.species, round(cell.height_m, 2), round(cell.health, 3))
            if post != pre.get(h3_id):
                sp_data = _perpetual_species_db.get(cell.species, {}) if cell.species else {}
                diff[h3_id] = {
                    "s": cell.species,
                    "h": post[1],
                    "hp": post[2],
                    "cc": round(cell.canopy_cover, 3),
                    "st": sp_data.get("stratum", ""),
                    "nr": sp_data.get("nitrogen_role", ""),
                }

        return {
            "type": "season",
            "year": _perpetual_year,
            "season": _perpetual_season,
            "changed": len(diff),
            "total": len(_perpetual_gol.grid),
            "diff": diff,
        }


@app.websocket("/ws")
async def ws_perpetual(websocket: WebSocket):
    """WebSocket endpoint for perpetual mode.

    Connect and receive real-time GoL diffs every ~2 seconds.
    Each message contains only the cells that changed that season.
    The forest evolves continuously — no endpoint, no convergence.
    """
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info(f"WebSocket client connected ({len(_ws_clients)} total)")

    try:
        # Send current state summary on connect
        await websocket.send_text(json.dumps({
            "type": "init",
            "year": _perpetual_year,
            "season": _perpetual_season,
            "running": _perpetual_running,
            "cells": len(_perpetual_gol.grid) if _perpetual_gol else 0,
        }))

        # Keep alive — client can send commands
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected ({len(_ws_clients)} remaining)")


@app.post("/perpetual/start")
async def start_perpetual():
    """Start perpetual mode — the forest evolves continuously.

    Seeds from the latest simulation state. Each season runs every 2s,
    diffs streamed to WebSocket clients.
    """
    global _perpetual_gol, _perpetual_species_db, _perpetual_year, _perpetual_season, _perpetual_running

    if _perpetual_running:
        return {"status": "already_running", "year": _perpetual_year}

    # Seed from in-memory cache or latest checkpoint file
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from surveyor.sim.gol import ForestGoL, GoLConfig
    from surveyor.guild import compatibility_score

    species_path = Path(os.environ.get("SPECIES_DB", "../species_db/species.json"))
    with open(species_path) as f:
        _perpetual_species_db = {s["id"]: s for s in json.load(f)}

    # Try to seed from checkpoint files if no live cache
    seed_data = list(_cell_index.values()) if _cell_index else None
    start_year = 20

    if not seed_data:
        # Load the latest checkpoint
        cp_dir = DATA_DIR / "checkpoints"
        if cp_dir.exists():
            cp_files = sorted(cp_dir.glob("year_*.json"), reverse=True)
            if cp_files:
                latest = cp_files[0]
                start_year = int(latest.stem.split("_")[1])
                with open(latest) as f:
                    cp_data = json.load(f)
                # Convert checkpoint format to seed records
                seed_data = []
                for h3_id, cell in cp_data.items():
                    if cell.get("s"):
                        seed_data.append({
                            "h3": h3_id,
                            "species": cell["s"],
                            "height_m": cell.get("h", 0),
                            "health": cell.get("hp", 0.5),
                            "lat": 0, "lon": 0,  # GoL uses h3 index, not coords
                        })
                logger.info(f"Seeding perpetual from checkpoint year {start_year}: {len(seed_data)} cells")

    if not seed_data:
        raise HTTPException(400, "No simulation data — run /simulate first or ensure checkpoints exist")

    config = GoLConfig(dt_years=0.25)  # quarter-year steps (seasons)
    _perpetual_gol = ForestGoL(_perpetual_species_db, compatibility_score, config)
    _perpetual_gol.seed_from_snapshot(seed_data)
    _perpetual_year = start_year
    _perpetual_season = 0
    _perpetual_running = True

    # Start background loop
    async def _run_loop():
        global _perpetual_running
        logger.info("Perpetual mode started")
        while _perpetual_running:
            result = _perpetual_step()
            if result and _ws_clients:
                await _broadcast(result)
            # Auto-save checkpoint every 10 sim years
            if _perpetual_year > 0 and _perpetual_season == 0 and _perpetual_year % 10 == 0:
                cp_dir = DATA_DIR / "checkpoints"
                cp_dir.mkdir(parents=True, exist_ok=True)
                cp_path = cp_dir / f"year_{_perpetual_year:03d}.json"
                if not cp_path.exists() and _perpetual_gol:
                    _save_checkpoint(_perpetual_gol.grid, _perpetual_species_db, _perpetual_year, cp_dir)
                    logger.info(f"Perpetual checkpoint saved: year {_perpetual_year}")
            await asyncio.sleep(2)  # 2s per season = ~8s per year
        logger.info("Perpetual mode stopped")

    asyncio.create_task(_run_loop())

    return {
        "status": "started",
        "year": _perpetual_year,
        "cells": len(_perpetual_gol.grid),
        "interval_s": 2,
    }


@app.post("/perpetual/stop")
async def stop_perpetual():
    """Stop perpetual mode."""
    global _perpetual_running
    _perpetual_running = False
    return {"status": "stopped", "year": _perpetual_year}


@app.get("/perpetual/status")
async def perpetual_status():
    """Current perpetual mode status."""
    return {
        "running": _perpetual_running,
        "year": _perpetual_year,
        "season": _perpetual_season,
        "clients": len(_ws_clients),
        "cells": len(_perpetual_gol.grid) if _perpetual_gol else 0,
    }



@app.get("/report")
async def planting_report():
    """Generate an HTML planting schedule report from checkpoint data.

    Shows every hex, every species, organized by phase/year.
    Open in browser and print to PDF.
    """
    import h3 as h3lib
    from collections import Counter

    checkpoint_dir = DATA_DIR / "checkpoints"
    if not checkpoint_dir.exists():
        raise HTTPException(404, "No checkpoints — run /simulate first")

    # Load key checkpoints for the planting phases
    phases = [0, 1, 3, 5, 7, 10, 15, 20]
    phase_data = {}
    for yr in phases:
        cp = checkpoint_dir / f"year_{yr:03d}.json"
        if cp.exists():
            with open(cp) as f:
                phase_data[yr] = json.load(f)

    if not phase_data:
        raise HTTPException(404, "No checkpoint files found")

    # Compute phase diffs
    html_sections = []
    prev_state = {}
    total_planted = 0
    species_totals = Counter()

    for yr in sorted(phase_data.keys()):
        state = phase_data[yr]
        new_plantings = {}
        deaths = {}
        species_changes = {}

        for h3_id, cell in state.items():
            prev = prev_state.get(h3_id, {})
            prev_sp = prev.get("s")
            curr_sp = cell.get("s")

            if curr_sp and not prev_sp:
                new_plantings[h3_id] = cell
            elif prev_sp and not curr_sp:
                deaths[h3_id] = prev
            elif curr_sp and prev_sp and curr_sp != prev_sp:
                species_changes[h3_id] = {"from": prev_sp, "to": curr_sp, "cell": cell}

        # Species counts at this year
        alive = {h: c for h, c in state.items() if c.get("s")}
        species_counts = Counter(c["s"] for c in alive.values())
        strata_counts = Counter(c.get("st", "unknown") for c in alive.values())

        # Build HTML for this phase
        section = f"""
        <div class="phase" style="page-break-before: always;">
            <h2>Year {yr}</h2>
            <div class="stats">
                <div class="stat"><span class="n">{len(alive):,}</span><span class="label">alive</span></div>
                <div class="stat"><span class="n">{len(new_plantings):,}</span><span class="label">new plantings</span></div>
                <div class="stat"><span class="n">{len(deaths):,}</span><span class="label">deaths</span></div>
                <div class="stat"><span class="n">{len(species_changes):,}</span><span class="label">species changes</span></div>
                <div class="stat"><span class="n">{len(species_counts)}</span><span class="label">species</span></div>
            </div>

            <h3>Species Distribution</h3>
            <table>
                <tr><th>Species</th><th>Count</th><th>%</th><th>Stratum</th></tr>
        """
        for sp, count in species_counts.most_common():
            pct = count / max(len(alive), 1) * 100
            # Find stratum for this species
            st = ""
            for c in alive.values():
                if c.get("s") == sp:
                    st = c.get("st", "")
                    break
            section += f'<tr><td>{sp}</td><td>{count:,}</td><td>{pct:.1f}%</td><td>{st}</td></tr>'
        section += "</table>"

        # Stratum breakdown
        section += "<h3>Stratum Layers</h3><table><tr><th>Stratum</th><th>Count</th><th>%</th></tr>"
        for st, count in strata_counts.most_common():
            pct = count / max(len(alive), 1) * 100
            section += f'<tr><td>{st}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>'
        section += "</table>"

        # New plantings detail (first 50)
        if new_plantings:
            section += f"<h3>New Plantings ({len(new_plantings):,} hexes)</h3>"
            section += "<table><tr><th>H3 Index</th><th>Species</th><th>Stratum</th><th>Height</th><th>Health</th></tr>"
            for h3_id, cell in list(new_plantings.items())[:50]:
                section += f'<tr><td class="mono">{h3_id[:16]}...</td><td>{cell.get("s","")}</td><td>{cell.get("st","")}</td><td>{cell.get("h",0)}m</td><td>{cell.get("hp",0):.2f}</td></tr>'
            if len(new_plantings) > 50:
                section += f'<tr><td colspan="5">... and {len(new_plantings)-50:,} more</td></tr>'
            section += "</table>"

        # Deaths detail (first 20)
        if deaths:
            section += f"<h3>Deaths ({len(deaths):,} hexes)</h3>"
            section += "<table><tr><th>H3 Index</th><th>Species Lost</th></tr>"
            for h3_id, cell in list(deaths.items())[:20]:
                section += f'<tr><td class="mono">{h3_id[:16]}...</td><td>{cell.get("s","")}</td></tr>'
            if len(deaths) > 20:
                section += f'<tr><td colspan="2">... and {len(deaths)-20:,} more</td></tr>'
            section += "</table>"

        section += "</div>"
        html_sections.append(section)
        prev_state = state
        total_planted += len(new_plantings)
        species_totals.update(species_counts)

    # Assemble full HTML
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Hooke Park — Metamorphic Planting Scheme</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #1a1a1a; }}
  h1 {{ font-size: 28px; border-bottom: 3px solid #1a1a1a; padding-bottom: 8px; }}
  h2 {{ font-size: 22px; color: #2d5016; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  h3 {{ font-size: 16px; color: #555; margin-top: 20px; }}
  .summary {{ background: #f5f5f0; padding: 20px; border-radius: 8px; margin: 20px 0; }}
  .stats {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 15px 0; }}
  .stat {{ text-align: center; padding: 10px 15px; background: white; border-radius: 6px; border: 1px solid #e0e0d8; }}
  .stat .n {{ display: block; font-size: 24px; font-weight: 700; color: #2d5016; }}
  .stat .label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  th {{ background: #f0f0e8; padding: 6px 10px; text-align: left; font-weight: 600; border-bottom: 2px solid #ddd; }}
  td {{ padding: 5px 10px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #fafaf5; }}
  .mono {{ font-family: 'SF Mono', monospace; font-size: 11px; color: #666; }}
  .phase {{ margin-bottom: 40px; }}
  @media print {{ .phase {{ page-break-before: always; }} body {{ font-size: 11px; }} }}
</style>
</head><body>
<h1>Hooke Park — Metamorphic Forest Planting Scheme</h1>
<div class="summary">
  <div class="stats">
    <div class="stat"><span class="n">{len(phase_data)}</span><span class="label">phases</span></div>
    <div class="stat"><span class="n">{total_planted:,}</span><span class="label">total plantings</span></div>
    <div class="stat"><span class="n">{len(species_totals)}</span><span class="label">species used</span></div>
    <div class="stat"><span class="n">{len(prev_state):,}</span><span class="label">total hexagons</span></div>
  </div>
  <p style="font-size:13px; color:#666;">
    Generated from {len(phase_data)} simulation checkpoints (years {", ".join(str(y) for y in sorted(phase_data.keys()))}).
    Each hexagon is an H3 res-13 cell (~1.2m). The metamorphic engine (GoL + physarum corridors)
    assigns species based on terrain, guild compatibility, and succession dynamics.
  </p>
</div>
{"".join(html_sections)}
</body></html>"""

    from fastapi.responses import HTMLResponse
    return HTMLResponse(html)



@app.get("/gif")
async def planting_gif():
    """Render an animated GIF of the planting scheme evolving over time.

    Each frame = one checkpoint year. Hexagons colored by species.
    Returns a GIF image directly.
    """
    import io
    import h3 as h3lib
    import numpy as np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        import imageio.v3 as iio
    except ImportError:
        raise HTTPException(500, "Install matplotlib + imageio: pip install matplotlib imageio")

    checkpoint_dir = DATA_DIR / "checkpoints"
    if not checkpoint_dir.exists():
        raise HTTPException(404, "No checkpoints")

    # Load planting phases
    phases = [0, 1, 3, 5, 7, 10, 15, 20]
    phase_data = {}
    for yr in phases:
        cp = checkpoint_dir / f"year_{yr:03d}.json"
        if cp.exists():
            with open(cp) as f:
                phase_data[yr] = json.load(f)

    if len(phase_data) < 2:
        raise HTTPException(404, "Need at least 2 checkpoints")

    # Build species → color map (deterministic hash, same as frontend)
    all_species = set()
    for state in phase_data.values():
        for cell in state.values():
            if cell.get("s"):
                all_species.add(cell["s"])

    species_colors = {}
    for sp in sorted(all_species):
        h = 0
        for c in sp:
            h = h * 31 + ord(c)
        h = h & 0xFFFFFFFF
        hue = (h % 360) / 360.0
        sat = 0.55 + (h % 30) / 100.0
        lit = 0.35 + (h % 20) / 100.0
        # HSL to RGB
        import colorsys
        r, g, b = colorsys.hls_to_rgb(hue, lit, sat)
        species_colors[sp] = (r, g, b)

    # Pre-compute hex boundaries (cache across frames)
    logger.info(f"Rendering GIF: {len(phase_data)} frames, {len(all_species)} species")
    hex_boundaries = {}
    sample_state = list(phase_data.values())[0]
    for h3_id in sample_state:
        try:
            boundary = h3lib.cell_to_boundary(h3_id)
            coords = [(lng, lat) for lat, lng in boundary]
            hex_boundaries[h3_id] = coords
        except Exception:
            pass

    # Compute bounds
    all_lngs = [c[0] for coords in hex_boundaries.values() for c in coords]
    all_lats = [c[1] for coords in hex_boundaries.values() for c in coords]
    min_lng, max_lng = min(all_lngs), max(all_lngs)
    min_lat, max_lat = min(all_lats), max(all_lats)
    pad = 0.001
    extent = [min_lng - pad, max_lng + pad, min_lat - pad, max_lat + pad]

    # Render frames
    frames = []
    for yr in sorted(phase_data.keys()):
        state = phase_data[yr]
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=100)
        fig.patch.set_facecolor("#0a0e14")
        ax.set_facecolor("#0a0e14")

        patches = []
        colors = []
        for h3_id, coords in hex_boundaries.items():
            cell = state.get(h3_id, {})
            sp = cell.get("s")
            if sp and cell.get("hp", 0) > 0:
                color = species_colors.get(sp, (0.4, 0.4, 0.4))
                health = cell.get("hp", 0.5)
                alpha_color = (*color, min(1.0, health * 0.8 + 0.15))
            else:
                alpha_color = (0.15, 0.15, 0.18, 0.3)
            patches.append(MplPolygon(coords, closed=True))
            colors.append(alpha_color)

        pc = PatchCollection(patches, facecolors=colors, edgecolors=(1, 1, 1, 0.06), linewidths=0.3)
        ax.add_collection(pc)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.axis("off")

        # Title
        alive = sum(1 for c in state.values() if c.get("s") and c.get("hp", 0) > 0)
        species_count = len(set(c.get("s") for c in state.values() if c.get("s")))
        ax.text(0.02, 0.97, f"Year {yr}", transform=ax.transAxes,
                fontsize=24, fontweight="bold", color="white", va="top", fontfamily="monospace")
        ax.text(0.02, 0.91, f"{alive:,} alive  |  {species_count} species",
                transform=ax.transAxes, fontsize=12, color="#888", va="top", fontfamily="monospace")

        # Species legend (top 8)
        from collections import Counter
        sp_counts = Counter(c.get("s") for c in state.values() if c.get("s"))
        for i, (sp, count) in enumerate(sp_counts.most_common(8)):
            color = species_colors.get(sp, (0.5, 0.5, 0.5))
            ax.add_patch(plt.Rectangle((0.75, 0.95 - i * 0.04), 0.02, 0.025,
                         transform=ax.transAxes, facecolor=color, edgecolor="none"))
            ax.text(0.78, 0.962 - i * 0.04, f"{sp.replace('_', ' ')} ({count:,})",
                    transform=ax.transAxes, fontsize=8, color="#aaa", va="top", fontfamily="monospace")

        # Render to numpy array
        fig.tight_layout(pad=0.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        frame = iio.imread(buf)
        frames.append(frame)
        logger.info(f"  Frame year {yr}: {alive:,} alive, {species_count} species")

    # Stitch into GIF
    gif_buf = io.BytesIO()
    iio.imwrite(gif_buf, frames, extension=".gif", duration=1500, loop=0)
    gif_buf.seek(0)
    gif_bytes = gif_buf.getvalue()
    logger.info(f"GIF complete: {len(frames)} frames, {len(gif_bytes) / 1024:.0f} KB")

    from fastapi.responses import Response
    return Response(content=gif_bytes, media_type="image/gif",
                    headers={"Cache-Control": "public, max-age=3600"})


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
