"""Multi-year forecast runner — produces Forest Game Records.

Runs the full pipeline:
  1. Score current state → classify each hex (conserve/transition/plant/opportunity)
  2. Generate planting scheme → convert to moves
  3. Run GoL baseline + intervention → capture checkpoints
  4. Record everything as FGR (Forest Game Record)

The FGR is the temporal export: moves + state checkpoints.
The frontend replays it like a Go game — each move placed on the hex grid over time.
"""

import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .fgr import (
    ForestGameRecord, Move, HexState,
    export_checkpoint, write_fgr, classify_hex_action,
    DEFAULT_CHECKPOINTS,
)

logger = logging.getLogger(__name__)


def generate_100yr_plan(
    snapshot_records: list[dict],
    species_db: dict,
    guild_scorer,
    scores: dict[str, dict],
    corridors: dict[str, float],
    output_dir: str = "output",
    scheme_name: str = "100yr_transition",
    checkpoints: Optional[list[int]] = None,
) -> Path:
    """Full pipeline: classify → plan → simulate → export FGR.

    This is the main entry point. Produces a complete Forest Game Record
    that the frontend can replay as a temporal visualization.

    Args:
        snapshot_records: current tree snapshot (87,935 records)
        species_db: species parameters
        guild_scorer: compatibility function
        scores: {h3: {vitality, resilience, ...}} from score engine
        corridors: {h3: corridor_strength} from physarum
        output_dir: where to write FGR files
        scheme_name: human-readable name
        checkpoints: years to snapshot (default: 0,1,3,5,7,10,15,20,30,50,75,100)

    Returns:
        Path to the FGR directory
    """
    from .gol import ForestGoL, GoLConfig
    from .planting import generate_planting_scheme

    if checkpoints is None:
        checkpoints = list(DEFAULT_CHECKPOINTS)

    scheme_id = f"{scheme_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    logger.info(f"Generating 100-year plan: {scheme_id}")

    # ── Phase 1: Classify every hex ──────────────────────────────────────

    by_h3 = {}
    for rec in snapshot_records:
        h = rec.get("h3_13") or rec.get("h3")
        if h:
            by_h3[h] = rec

    classifications = {}
    for h, rec in by_h3.items():
        sc = scores.get(h)
        classifications[h] = classify_hex_action(rec, sc)

    class_counts = {}
    for c in classifications.values():
        class_counts[c] = class_counts.get(c, 0) + 1
    logger.info(f"Classification: {class_counts}")

    # ── Phase 2: Generate moves (planting scheme → timed decisions) ──────

    scheme = generate_planting_scheme(
        snapshot_records, scores, corridors, species_db, guild_scorer,
    )

    # Convert planting actions to timed moves across phases
    moves = _scheme_to_moves(scheme, classifications, species_db, scores)
    logger.info(f"Generated {len(moves)} moves across {max(m.year for m in moves):.0f} years")

    # ── Phase 3: Run GoL with intervention ───────────────────────────────

    config = GoLConfig(dt_years=1.0)
    max_year = max(checkpoints)

    # Intervention run
    gol = ForestGoL(species_db, guild_scorer, config)
    gol.seed_from_snapshot(snapshot_records)

    checkpoint_states = {}

    # Year 0 = current state
    checkpoint_states[0] = export_checkpoint(
        gol.grid, 0, snapshot_records=snapshot_records, scores=scores,
    )

    # Apply moves at their scheduled year, capture checkpoints
    moves_by_year = {}
    for m in moves:
        yr = int(m.year)
        moves_by_year.setdefault(yr, []).append(m)

    for year in range(1, max_year + 1):
        # Apply this year's moves
        if year in moves_by_year:
            interventions = []
            for m in moves_by_year[year]:
                interventions.append({
                    "h3": m.h3,
                    "action": m.action,
                    "species": m.species or m.species_removed,
                })
            gol.apply_intervention(interventions)

        gol.step()

        if year in checkpoints:
            checkpoint_states[year] = export_checkpoint(gol.grid, year)
            alive = sum(1 for c in gol.grid.values() if c.is_alive)
            species_set = set(c.species for c in gol.grid.values() if c.is_alive and c.species)
            logger.info(f"Checkpoint year {year}: {alive} alive, {len(species_set)} species")

    # ── Phase 4: Build FGR ───────────────────────────────────────────────

    # Species legend
    species_legend = {}
    for sp_id, sp in species_db.items():
        species_legend[sp_id] = {
            "common": sp.get("common", sp_id),
            "stratum": sp.get("stratum", "canopy"),
            "color": sp.get("color", "#888888"),
        }

    # Checkpoint summaries
    summaries = {}
    for year, states in checkpoint_states.items():
        alive = [s for s in states if s.get("species")]
        species_set = set(s.get("species") for s in alive)
        summaries[year] = {
            "alive": len(alive),
            "empty": len(states) - len(alive),
            "species_richness": len(species_set),
            "mean_height_m": round(np.mean([s.get("height_m", 0) for s in alive]) if alive else 0, 1),
            "mean_health": round(np.mean([s.get("health", 0) for s in alive]) if alive else 0, 2),
        }

    fgr = ForestGameRecord(
        scheme_id=scheme_id,
        name=scheme_name,
        description=f"100-year transition plan for Hooke Park. {len(moves)} interventions across {len(class_counts)} action classes.",
        created=datetime.now().isoformat(),
        total_years=max_year,
        total_trees=len(by_h3),
        checkpoints=checkpoints,
        moves=moves,
        phases=_default_phases(),
        summaries=summaries,
        species_legend=species_legend,
    )

    return write_fgr(fgr, checkpoint_states, output_dir)


def _scheme_to_moves(scheme, classifications: dict, species_db: dict, scores: dict) -> list[Move]:
    """Convert a planting scheme + classifications into timed moves.

    Timing follows the 100-year phased approach:
      Year 1-2:  Thinning (transition zones — remove competing conifers)
      Year 3-5:  Pioneer planting (birch, alder, willow in corridors)
      Year 5-10: Secondary wave (oak, cherry, hazel, holly)
      Year 10-20: Climax establishment (beech, lime, yew)
      Year 20+:  Natural regeneration + monitoring
    """
    moves = []

    # Thinning moves (year 1-2) — transition zones
    thin_cells = [h for h, c in classifications.items() if c == "transition"]
    for i, h in enumerate(thin_cells):
        year = 1 + (i % 2)  # spread across 2 years
        rec = {}  # would need snapshot lookup for species_removed
        moves.append(Move(
            year=year,
            h3=h,
            action="thin",
            species_removed=None,  # will be filled from snapshot
            rationale="Transition zone: thin competing conifer to release broadleaf",
            priority=7,
            score_before=scores.get(h, {}).get("overall"),
        ))

    # Planting moves — phased by succession stage
    for action in scheme.actions:
        sp = species_db.get(action.species, {})
        succession = sp.get("succession", "secondary")

        if succession == "pioneer":
            year = 3 + hash(action.h3_id) % 3  # years 3-5
        elif succession == "secondary":
            year = 7 + hash(action.h3_id) % 4  # years 7-10
        elif succession == "climax":
            year = 12 + hash(action.h3_id) % 8  # years 12-19
        else:
            year = 5 + hash(action.h3_id) % 5

        moves.append(Move(
            year=year,
            h3=action.h3_id,
            action="plant",
            species=action.species,
            method=action.method,
            priority=action.priority,
            rationale=action.rationale,
            cluster_id=action.cluster_id,
            citations=action.citations,
            score_before=scores.get(action.h3_id, {}).get("overall"),
        ))

    # Natural regeneration events (year 20+) — opportunity zones let nature fill
    opportunity_cells = [h for h, c in classifications.items() if c == "natural_regen"]
    for h in opportunity_cells[:200]:  # cap to keep moves manageable
        moves.append(Move(
            year=20 + hash(h) % 10,  # years 20-29
            h3=h,
            action="monitor",
            rationale="Natural regeneration zone: monitor seed-rain colonization",
            priority=3,
        ))

    moves.sort(key=lambda m: (m.year, -m.priority))
    return moves


def _default_phases() -> list[dict]:
    """100-year plan phases for the timeline annotation."""
    return [
        {
            "name": "Assessment",
            "year_start": 0, "year_end": 1,
            "description": "Survey, score, classify every tree. Identify conservation zones, transition zones, opportunities.",
            "color": "#4FC3F7",
        },
        {
            "name": "Thinning",
            "year_start": 1, "year_end": 3,
            "description": "Selective removal of competing conifers in transition zones. Create light gaps for broadleaf establishment.",
            "color": "#FF8A65",
        },
        {
            "name": "Pioneer Planting",
            "year_start": 3, "year_end": 7,
            "description": "Birch, alder, willow along physarum corridors. N-fixers, fast canopy closure, windbreak establishment.",
            "color": "#81C784",
        },
        {
            "name": "Secondary Wave",
            "year_start": 7, "year_end": 15,
            "description": "Oak, cherry, hazel, holly in Miyawaki clusters. Guild-compatible underplanting beneath pioneer canopy.",
            "color": "#4DB6AC",
        },
        {
            "name": "Climax Establishment",
            "year_start": 15, "year_end": 30,
            "description": "Beech, lime, yew establishing in shade. Pioneers declining naturally. Multi-strata canopy forming.",
            "color": "#7986CB",
        },
        {
            "name": "Natural Succession",
            "year_start": 30, "year_end": 50,
            "description": "Self-organizing forest. Seed dispersal filling gaps. Mycorrhizal networks mature. Minimal intervention.",
            "color": "#A5D6A7",
        },
        {
            "name": "Mature Forest",
            "year_start": 50, "year_end": 100,
            "description": "Resilient mixed woodland. High carbon stocks. Complex structure. Self-sustaining ecosystem.",
            "color": "#2E7D32",
        },
    ]


# ── Standalone runner ────────────────────────────────────────────────────────

def run_forecast(
    snapshot_records: list[dict],
    species_db: dict,
    guild_scorer,
    interventions: Optional[list[dict]] = None,
    years: int = 30,
    checkpoints: Optional[list[int]] = None,
) -> dict:
    """Simple forecast runner (backward-compatible with original API).

    For FGR export, use generate_100yr_plan() instead.
    """
    from .gol import ForestGoL, GoLConfig

    if checkpoints is None:
        checkpoints = [3, 7, 10, 15, 30]
    checkpoints = [y for y in checkpoints if y <= years]

    config = GoLConfig()

    # Baseline
    baseline_gol = ForestGoL(species_db, guild_scorer, config)
    baseline_gol.seed_from_snapshot(snapshot_records)

    baseline_snapshots = {}
    for year in range(1, years + 1):
        summary = baseline_gol.step()
        if year in checkpoints:
            baseline_snapshots[year] = _gol_summary(baseline_gol, summary)

    result = {"baseline": baseline_snapshots}

    # Intervention
    if interventions:
        iv_gol = ForestGoL(species_db, guild_scorer, config)
        iv_gol.seed_from_snapshot(snapshot_records)
        iv_gol.apply_intervention(interventions)

        iv_snapshots = {}
        for year in range(1, years + 1):
            summary = iv_gol.step()
            if year in checkpoints:
                iv_snapshots[year] = _gol_summary(iv_gol, summary)

        result["intervention"] = iv_snapshots

    return result


def _gol_summary(gol, summary: dict) -> dict:
    alive = [c for c in gol.grid.values() if c.is_alive]
    species_counts = {}
    for cell in alive:
        species_counts[cell.species] = species_counts.get(cell.species, 0) + 1

    n = len(alive) or 1
    return {
        **summary,
        "species_richness": len(species_counts),
        "species_counts": species_counts,
        "mean_height_m": round(sum(c.height_m for c in alive) / n, 2),
        "mean_health": round(sum(c.health for c in alive) / n, 3),
        "canopy_cover_pct": round(sum(c.canopy_cover for c in alive) / max(1, len(gol.grid)) * 100, 1),
    }
