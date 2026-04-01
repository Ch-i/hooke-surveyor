"""Forest Game Record (FGR) — the temporal export format.

Inspired by SGF (Smart Game Format) for Go. Records agent decisions as
"moves" placed on the H3 grid, plus board-state checkpoints at key years.

The frontend replays this like watching a game of Go:
  - Timeline scrubs through years
  - Each move appears on the hex grid with rationale
  - Board state interpolates between checkpoints
  - Thought stream narrates the agent's reasoning

Format:
  {scheme_id}/
    fgr.json            → manifest + moves (the "game record")
    state_000.json      → board state at year 0 (87,935 hex states)
    state_003.json      → board state at year 3
    state_007.json      → ...
    state_010.json
    state_015.json
    state_030.json
    state_050.json
    state_100.json

Each hex state: {h3, species, height_m, health, canopy_cover, age_years, action_class}
  action_class: conserve | transition | plant | natural_regen | monitor | thin | opportunity

Each move: {year, h3, action, species, method, priority, rationale, score_before, score_after}
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default checkpoint years for a 100-year plan
DEFAULT_CHECKPOINTS = [0, 1, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]


@dataclass
class Move:
    """A single agent decision — a stone placed on the forest board."""

    year: float
    h3: str
    action: str  # plant | thin | underplant | coppice | protect | release | monitor
    species: Optional[str] = None  # species planted (or removed for thin)
    species_removed: Optional[str] = None  # what was thinned
    method: Optional[str] = None  # miyawaki_cluster | syntropic_row | individual | natural_regen
    priority: int = 5
    rationale: str = ""
    cluster_id: Optional[str] = None  # groups related moves
    score_before: Optional[float] = None  # overall score before intervention
    score_after: Optional[float] = None  # projected score after
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None and v != [] and v != ""}


@dataclass
class HexState:
    """State of a single hex at a checkpoint year."""

    h3: str
    species: Optional[str] = None
    height_m: float = 0.0
    health: float = 0.0
    canopy_cover: float = 0.0
    age_years: float = 0.0
    action_class: str = "monitor"  # conserve | transition | plant | natural_regen | monitor | thin | opportunity

    def to_dict(self) -> dict:
        d = asdict(self)
        # Compact: drop zero/null fields for 87K records
        return {k: v for k, v in d.items() if v is not None and v != 0 and v != 0.0 and v != ""}


@dataclass
class ForestGameRecord:
    """Complete game record: moves + checkpoints + metadata."""

    scheme_id: str
    name: str = ""
    description: str = ""
    created: str = ""
    bioregion: str = "dorset_maritime"
    total_years: int = 100
    total_trees: int = 87935
    checkpoints: list[int] = field(default_factory=lambda: list(DEFAULT_CHECKPOINTS))

    # Moves (the decisions)
    moves: list[Move] = field(default_factory=list)

    # Phase annotations
    phases: list[dict] = field(default_factory=list)

    # Summary statistics per checkpoint
    summaries: dict = field(default_factory=dict)  # {year: {alive, species_richness, ...}}

    # Species legend
    species_legend: dict = field(default_factory=dict)  # {species_id: {common, color, stratum}}


# ── Classification ───────────────────────────────────────────────────────────

def classify_hex_action(rec: dict, scores: Optional[dict] = None) -> str:
    """Classify what should happen to a hex based on current state + scores.

    Returns action_class: conserve | transition | plant | natural_regen | monitor | thin | opportunity
    """
    status = rec.get("status")
    species = rec.get("species_detected")
    height = rec.get("height_m", 0)
    ndvi = rec.get("ndvi", 0)
    guild = rec.get("guild_score", 0)
    risk = rec.get("risk_overall", 0)

    overall_score = scores.get("overall", 0.5) if scores else 0.5

    # Empty hex
    if not species or status == "lost" or height < 1:
        return "opportunity"

    # High-value tree — protect
    if overall_score > 0.7 and status == "persistent":
        return "conserve"

    # Conifer monoculture with low guild score — transition gradually
    species_group = rec.get("species_group", "")
    if "conifer" in species_group.lower() or species in ("scots_pine", "douglas_fir", "monterey_cypress"):
        if guild is not None and guild < 0.1:
            return "transition"

    # Declining tree — needs intervention
    if risk > 0.6:
        return "thin" if height > 15 else "plant"

    # Moderate score, could benefit from neighbours
    if overall_score < 0.4:
        return "natural_regen" if height > 5 else "plant"

    return "monitor"


# ── Export ────────────────────────────────────────────────────────────────────

def export_checkpoint(
    gol_grid: dict,
    year: int,
    snapshot_records: Optional[list[dict]] = None,
    scores: Optional[dict[str, dict]] = None,
) -> list[dict]:
    """Export a single checkpoint as a list of hex states.

    At year 0, uses snapshot_records directly. At future years, uses GoL grid.
    """
    if year == 0 and snapshot_records:
        states = []
        for rec in snapshot_records:
            h = rec.get("h3_13") or rec.get("h3")
            if not h:
                continue

            sc = scores.get(h) if scores else None
            action_class = classify_hex_action(rec, sc)

            states.append(HexState(
                h3=h,
                species=rec.get("species_detected"),
                height_m=round(rec.get("height_m", 0), 1),
                health=round(min(1.0, rec.get("ndvi", 0.5) / 0.8), 2),
                canopy_cover=round(min(1.0, rec.get("crown_area_m2", 0) / 1.44), 2),
                age_years=0,
                action_class=action_class,
            ).to_dict())
        return states

    # Future year — from GoL grid
    states = []
    for h, cell in gol_grid.items():
        states.append(HexState(
            h3=h,
            species=cell.species if cell.is_alive else None,
            height_m=round(cell.height_m, 1),
            health=round(cell.health, 2),
            canopy_cover=round(cell.canopy_cover, 2),
            age_years=round(cell.age_years, 1),
            action_class="conserve" if cell.health > 0.7 else ("monitor" if cell.is_alive else "opportunity"),
        ).to_dict())
    return states


def write_fgr(
    fgr: ForestGameRecord,
    checkpoint_states: dict[int, list[dict]],
    output_dir: str | Path,
):
    """Write a complete FGR to disk.

    Creates:
      {output_dir}/{scheme_id}/fgr.json
      {output_dir}/{scheme_id}/state_{year:03d}.json
    """
    output_dir = Path(output_dir) / fgr.scheme_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest + moves
    manifest = {
        "scheme_id": fgr.scheme_id,
        "name": fgr.name,
        "description": fgr.description,
        "created": fgr.created,
        "bioregion": fgr.bioregion,
        "total_years": fgr.total_years,
        "total_trees": fgr.total_trees,
        "checkpoints": fgr.checkpoints,
        "phases": fgr.phases,
        "summaries": fgr.summaries,
        "species_legend": fgr.species_legend,
        "move_count": len(fgr.moves),
        "moves": [m.to_dict() for m in fgr.moves],
    }

    fgr_path = output_dir / "fgr.json"
    with open(fgr_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"FGR manifest: {fgr_path} ({len(fgr.moves)} moves)")

    # Write checkpoint states
    for year, states in checkpoint_states.items():
        state_path = output_dir / f"state_{year:03d}.json"
        with open(state_path, "w") as f:
            json.dump(states, f, separators=(",", ":"))
        size_mb = state_path.stat().st_size / 1024 / 1024
        logger.info(f"State year {year}: {state_path} ({len(states)} hexes, {size_mb:.1f} MB)")

    logger.info(f"FGR complete: {output_dir} ({len(checkpoint_states)} checkpoints)")
    return output_dir
