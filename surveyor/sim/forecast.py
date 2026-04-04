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
from surveyor.sim.planting import PHASE_SCHEDULE, CLASS_SPECIES_PALETTES

logger = logging.getLogger(__name__)


# ── Succession-stage offsets (years after phase_start) ───────────────────────
_SUCCESSION_OFFSET = {
    "pioneer":   0,
    "secondary": 3,
    "climax":    7,
    "shrub":     1,   # shrubs go in with pioneers
    "ground":    5,   # ground flora after canopy begins to close
}

# Classes that require thinning before planting
_THIN_FIRST_CLASSES = {"A", "D"}
_THIN_LEAD_YEARS = 2  # thinning precedes planting by this many years


def scheme_to_phased_moves(
    scheme,  # PlantingScheme
    species_db: dict,
    classifications: dict = None,  # {h3_11: BlockClassification}
) -> list:
    """Convert planting actions to temporally-phased moves.

    Timing is driven by two axes:
      1. Management class → PHASE_SCHEDULE base window (from planting.py)
      2. Species succession stage → offset within that window

    For class A/D (conifer monoculture / transition zones), thinning moves
    are inserted *before* planting moves by ``_THIN_LEAD_YEARS``.

    Returns list of dicts: [{h3, year, action, species, management_class}, ...]
    Sorted by year.
    """
    moves = []

    for action in scheme.actions:
        mgmt_class = action.management_class or ""
        h3_id = action.h3_id

        # ── Determine base year from PHASE_SCHEDULE ──────────────────
        if mgmt_class and mgmt_class in PHASE_SCHEDULE:
            phase_start, phase_end = PHASE_SCHEDULE[mgmt_class]
        else:
            # Fallback: treat as generic mid-priority
            phase_start, phase_end = (3, 15)

        # ── Look up succession stage ─────────────────────────────────
        sp_data = species_db.get(action.species, {})
        succession = sp_data.get("succession", "secondary")

        # Check CLASS_SPECIES_PALETTES for an authoritative stage lookup
        # (the palette keys *are* the succession categories)
        if mgmt_class and mgmt_class in CLASS_SPECIES_PALETTES:
            palette = CLASS_SPECIES_PALETTES[mgmt_class]
            for stage, spp_list in palette.items():
                if action.species in spp_list:
                    succession = stage
                    break

        offset = _SUCCESSION_OFFSET.get(succession, 3)
        plant_year = phase_start + offset

        # Clamp within the phase window (with a little slack for climax)
        plant_year = max(phase_start, min(plant_year, phase_end + 2))

        # Spread plants within the same year bucket using a stable hash
        # so that not all 15K moves land on a single year
        spread = abs(hash(h3_id)) % 3  # 0, 1, or 2 year jitter
        plant_year += spread
        plant_year = max(phase_start, min(plant_year, phase_end + 4))

        # ── Thinning move for class A / D ────────────────────────────
        if mgmt_class in _THIN_FIRST_CLASSES:
            thin_year = max(1, plant_year - _THIN_LEAD_YEARS)
            moves.append({
                "h3": h3_id,
                "year": thin_year,
                "action": "thin",
                "species": None,
                "species_removed": None,  # filled later from snapshot
                "management_class": mgmt_class,
                "method": None,
                "priority": action.priority,
                "rationale": (
                    f"Class {mgmt_class}: thin competing conifer "
                    f"to prepare for {sp_data.get('common', action.species)} "
                    f"planting in year {plant_year}"
                ),
                "cluster_id": action.cluster_id,
                "citations": action.citations,
            })

        # ── Planting move ────────────────────────────────────────────
        moves.append({
            "h3": h3_id,
            "year": plant_year,
            "action": "plant",
            "species": action.species,
            "species_removed": None,
            "management_class": mgmt_class,
            "method": action.method,
            "priority": action.priority,
            "rationale": action.rationale,
            "cluster_id": action.cluster_id,
            "citations": action.citations,
        })

    # Sort: year ascending, then priority descending
    moves.sort(key=lambda m: (m["year"], -(m["priority"] or 5)))
    return moves


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
    """Convert a planting scheme + classifications into timed Move objects.

    Delegates to ``scheme_to_phased_moves`` for PHASE_SCHEDULE-aware timing,
    then wraps the raw dicts as Move dataclass instances.  Also appends
    monitoring moves for natural-regeneration opportunity zones.
    """
    # ── Phased planting + thinning moves (via PHASE_SCHEDULE) ────────────
    raw_moves = scheme_to_phased_moves(scheme, species_db)

    moves = []
    for rm in raw_moves:
        moves.append(Move(
            year=rm["year"],
            h3=rm["h3"],
            action=rm["action"],
            species=rm.get("species"),
            species_removed=rm.get("species_removed"),
            method=rm.get("method"),
            priority=rm.get("priority", 5),
            rationale=rm.get("rationale", ""),
            cluster_id=rm.get("cluster_id"),
            citations=rm.get("citations", []),
            score_before=scores.get(rm["h3"], {}).get("overall"),
        ))

    # ── Natural regeneration monitoring (opportunity zones) ──────────────
    opportunity_cells = [h for h, c in classifications.items() if c == "natural_regen"]
    for h in opportunity_cells[:200]:  # cap to keep moves manageable
        moves.append(Move(
            year=20 + abs(hash(h)) % 10,  # years 20-29
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
