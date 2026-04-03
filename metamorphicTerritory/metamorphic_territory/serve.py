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

app = FastAPI(title="metamorphicTerritory", version="0.1.0")
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


# ── Engine ──────────────────────────────────────────────────────────────────


def _run_engine(req: SimulateRequest) -> dict:
    """Run the metamorphic engine: load snapshot → simulate → export GeoJSON."""
    global _cell_index, _simulation_summaries

    # Load the latest snapshot from the surveyor output
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
    from surveyor.sim.gol import ForestGoL, GoLConfig
    from surveyor.sim.physarum import find_planting_corridors
    from surveyor.sim.planting import generate_planting_scheme
    from surveyor.scores.engine import compute_scores, WEIGHT_PRESETS

    # Compute scores
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        pd.DataFrame(records),
        geometry=[Point(r["lon"], r["lat"]) for r in records],
        crs="EPSG:4326",
    )
    scores = compute_scores(gdf, weights=WEIGHT_PRESETS["restoration"])

    # Physarum corridors
    corridors = find_planting_corridors(records)

    # Generate planting scheme
    scheme = generate_planting_scheme(records, scores, corridors, species_db, compatibility_score)

    # Run GoL simulation
    config = GoLConfig(dt_years=1.0)
    gol = ForestGoL(species_db, compatibility_score, config)
    gol.seed_from_snapshot(records)

    # Apply planting moves
    for action in scheme.actions:
        gol.apply_intervention([{
            "h3": action.h3_id,
            "action": "plant",
            "species": action.species,
        }])

    # Run simulation, capture snapshots
    checkpoints = [0, 1, 3, 5, 7, 10, 15, 20]
    if req.years > 20:
        checkpoints.extend([y for y in [30, 50, 75, 100] if y <= req.years])

    _simulation_summaries = []
    checkpoint_dir = DATA_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Save year 0 state (initial planting)
    _save_checkpoint(gol.grid, species_db, 0, checkpoint_dir)
    for yr in range(1, req.years + 1):
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

    # Export to GeoJSON: each H3 cell → polygon with properties
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
        "--drop-densest-as-needed",
        "--extend-zooms-if-still-dropping",
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


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
