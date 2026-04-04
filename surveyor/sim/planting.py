"""Planting scheme generator — combines scores, corridors, research, and GoL forecasting.

The planting engine:
1. Identifies intervention targets (low-score cells)
2. Overlays physarum corridors (where to connect)
3. Selects species from guild matrix + research knowledge
4. Groups into Miyawaki clusters or syntropic rows
5. Forecasts outcomes via GoL simulation
6. Outputs a prioritized planting scheme with citations
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h3
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PlantingAction:
    """A single planting action within a scheme."""

    h3_id: str
    species: str
    method: str  # miyawaki_cluster | syntropic_row | individual | underplant | natural_regen
    priority: int  # 1-10
    rationale: str
    cluster_id: Optional[str] = None
    window: str = ""  # e.g. "October-December 2026"
    companions: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)  # DOIs


@dataclass
class PlantingScheme:
    """A complete planting scheme for a block or area."""

    name: str
    area_h3_ids: list[str]
    actions: list[PlantingAction] = field(default_factory=list)
    total_trees: int = 0
    species_mix: dict = field(default_factory=dict)
    forecast_years: int = 0
    forecast_carbon_gain_tco2: float = 0.0
    forecast_biodiversity_gain: float = 0.0


def generate_planting_scheme(
    snapshot_records: list[dict],
    scores: dict[str, dict],
    corridors: dict[str, float],
    species_db: dict,
    guild_scorer,
    target_area: Optional[list[str]] = None,
    knowledge_dir: Optional[str] = None,
) -> PlantingScheme:
    """Generate a planting scheme for a target area.

    Args:
        snapshot_records: tree snapshot data
        scores: {h3: {vitality, resilience, symbiosis, ...}} from score engine
        corridors: {h3: corridor_strength} from physarum
        species_db: species database
        guild_scorer: compatibility function
        target_area: restrict to these H3 IDs (or None for full site)
        knowledge_dir: path to knowledge base for technique citations
    """
    # Index records by h3
    by_h3 = {rec.get("h3_13") or rec.get("h3"): rec for rec in snapshot_records}

    # Filter to target area
    if target_area:
        candidates = {h: by_h3[h] for h in target_area if h in by_h3}
    else:
        candidates = by_h3

    # Identify intervention cells: low overall score OR empty cells in corridors
    intervention_cells = []
    for h, rec in candidates.items():
        cell_score = scores.get(h, {}).get("overall", 0.5)
        corridor = corridors.get(h, 0)
        is_empty = rec.get("status") == "lost" or rec.get("species_detected") is None

        # Priority: low score + high corridor value = most important
        need = (1 - cell_score) * 0.6 + corridor * 0.4
        if need > 0.3 or is_empty:
            intervention_cells.append((h, rec, need, corridor))

    intervention_cells.sort(key=lambda x: x[2], reverse=True)
    # Cap interventions — focus on highest-need cells for a realistic plan
    max_interventions = min(len(intervention_cells), 15000)
    intervention_cells = intervention_cells[:max_interventions]
    logger.info(f"Identified {len(intervention_cells)} intervention cells (capped at {max_interventions})")

    # Select species for each cell
    actions = []
    for h, rec, need, corridor in intervention_cells:
        species = _select_species(h, rec, candidates, species_db, guild_scorer, corridor)
        method = _select_method(rec, corridor, species)

        actions.append(PlantingAction(
            h3_id=h,
            species=species,
            method=method,
            priority=min(10, max(1, int(need * 10))),
            rationale=_generate_rationale(rec, species, method, species_db),
            window=_planting_window(species, species_db),
        ))

    # Group into clusters
    actions = _cluster_actions(actions)

    # Load technique citations if available
    if knowledge_dir:
        _attach_citations(actions, knowledge_dir)

    # Build scheme
    species_mix = {}
    for a in actions:
        species_mix[a.species] = species_mix.get(a.species, 0) + 1

    scheme = PlantingScheme(
        name=f"scheme_{len(actions)}_trees",
        area_h3_ids=[a.h3_id for a in actions],
        actions=actions,
        total_trees=len(actions),
        species_mix=species_mix,
    )

    logger.info(f"Planting scheme: {len(actions)} trees, {len(species_mix)} species")
    return scheme


def _select_species(
    h3_id: str, rec: dict, all_records: dict,
    species_db: dict, guild_scorer, corridor_strength: float,
) -> str:
    """Select species based on site conditions, succession stage, and guild compatibility.

    Science: syntropic agroforestry (Gotsch) — pioneers establish first along
    corridors, N-fixers build soil, climax follows under canopy shelter.
    Simard (2012): ECM networks need pioneer birch/alder to establish before oak.
    """
    neighbours = rec.get("neighbours", [])
    nb_species = set()
    nb_heights = []
    for nb_h in neighbours:
        nb_rec = all_records.get(nb_h, {})
        sp = nb_rec.get("species_detected")
        if sp:
            nb_species.add(sp)
            nb_heights.append(nb_rec.get("height_m", 0))

    # Context: is this an open site or under existing canopy?
    own_height = rec.get("height_m", 0)
    avg_nb_height = sum(nb_heights) / max(len(nb_heights), 1)
    under_canopy = avg_nb_height > 8 or own_height > 10
    is_corridor = corridor_strength > 0.3
    is_empty = rec.get("status") == "lost" or rec.get("species_detected") is None

    # Determine which succession stage is appropriate for this cell
    # Gotsch principle: open ground → pioneers + N-fixers
    #                   under young canopy → secondary
    #                   under mature canopy → climax
    if is_empty and not under_canopy:
        target_succession = "pioneer"
    elif under_canopy and avg_nb_height > 15:
        target_succession = "climax"
    elif under_canopy:
        target_succession = "secondary"
    else:
        target_succession = "pioneer"

    # Corridors always get pioneers — they're the connection tissue
    if is_corridor and is_empty:
        target_succession = "pioneer"

    scores = {}
    for sp_id, sp in species_db.items():
        if sp.get("stratum") == "ground":
            continue

        score = 0.0

        # Succession match (strongest signal)
        if sp.get("succession") == target_succession:
            score += 0.35
        elif target_succession == "pioneer" and sp.get("succession") != "pioneer":
            score -= 0.3  # penalize non-pioneers in open ground

        # N-fixers get strong bonus in early succession (Simard 2012)
        if sp.get("nitrogen_role") == "fixer":
            if not under_canopy:
                score += 0.25  # critical for soil preparation
            else:
                score += 0.05

        # Guild compatibility
        for nb_sp in nb_species:
            score += guild_scorer(sp_id, nb_sp) * 0.2

        # Site suitability
        twi = rec.get("twi") or 8
        if twi < 6:
            score += sp.get("drought_tolerance", 0.5) * 0.15
        if under_canopy:
            score += sp.get("shade_tolerance", 0.5) * 0.2

        # Corridor connectivity: fast-growing species for rapid canopy closure
        if is_corridor:
            score += sp.get("growth_rate", 0.05) * 2.0

        scores[sp_id] = score

    if not scores:
        return "common_alder"  # N-fixer default

    return max(scores, key=scores.get)


def _select_method(rec: dict, corridor: float, species: str) -> str:
    """Choose planting method based on context."""
    if rec.get("status") == "lost" or rec.get("species_detected") is None:
        if corridor > 0.6:
            return "syntropic_row"
        return "miyawaki_cluster"
    if rec.get("height_m", 0) > 8:
        return "underplant"
    return "individual"


def _generate_rationale(rec: dict, species: str, method: str, species_db: dict) -> str:
    """Brief rationale for the planting action."""
    sp = species_db.get(species, {})
    parts = [f"{sp.get('common', species)} ({method.replace('_', ' ')})"]

    if rec.get("risk_drought", 0) > 0.5:
        parts.append(f"drought risk {rec['risk_drought']:.2f}")
    if rec.get("guild_score") is not None and rec["guild_score"] < 0:
        parts.append("poor guild compatibility")
    if rec.get("status") == "lost":
        parts.append("replacing lost tree")
    if sp.get("nitrogen_role") == "fixer":
        parts.append("N-fixation")

    return "; ".join(parts)


def _planting_window(species: str, species_db: dict) -> str:
    """Suggest planting window based on species type."""
    sp = species_db.get(species, {})
    if sp.get("succession") == "pioneer":
        return "March-April (spring, pioneer establishment)"
    return "October-December (dormancy, bare-root planting)"


def _cluster_actions(actions: list[PlantingAction]) -> list[PlantingAction]:
    """Group nearby actions into named clusters."""
    if not actions:
        return actions

    # Simple clustering by H3 res-11 parent
    clusters = {}
    for action in actions:
        parent = h3.cell_to_parent(action.h3_id, 11)
        clusters.setdefault(parent, []).append(action)

    cluster_id = 0
    for parent, cluster_actions in clusters.items():
        cluster_id += 1
        cid = f"cluster_{cluster_id:03d}"
        for action in cluster_actions:
            action.cluster_id = cid

    return actions


def _attach_citations(actions: list[PlantingAction], knowledge_dir: str):
    """Attach relevant technique citations from the knowledge base."""
    techniques_dir = Path(knowledge_dir) / "techniques"
    if not techniques_dir.exists():
        return

    # Load all techniques
    all_techniques = []
    for f in techniques_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
            if isinstance(data, list):
                all_techniques.extend(data)
            else:
                all_techniques.append(data)

    # Match techniques to actions by method/category
    method_to_category = {
        "miyawaki_cluster": "planting",
        "syntropic_row": "planting",
        "underplant": "planting",
        "individual": "planting",
        "natural_regen": "management",
    }

    for action in actions:
        category = method_to_category.get(action.method, "planting")
        relevant = [
            t for t in all_techniques
            if t.get("category") == category and t.get("source_doi")
        ]
        action.citations = [t["source_doi"] for t in relevant[:3]]
