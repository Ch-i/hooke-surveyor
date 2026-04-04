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

from surveyor.sim.classify import BlockClassification

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
    management_class: str = ""
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


# Per-class planting phase windows (from STRATEGY.md)
PHASE_SCHEDULE = {
    "C": (0, 1),   # Open ground: immediate
    "E": (0, 1),   # Riparian: immediate
    "G": (0, 1),   # Hedgerows: immediate
    "A": (3, 15),  # Monoculture: thin yr 1-3, plant yr 3+
    "D": (3, 15),  # Transition: thin yr 1-3, plant yr 3+
    "F": (3, 7),   # Ridgeline: secondary priority
    "B": (7, 20),  # Semi-natural: enrichment only, late
}

CLASS_SPECIES_PALETTES = {
    "A": {  # CCF transition: thin conifer, underplant broadleaf
        "pioneer": ["birch", "common_alder", "grey_alder"],
        "secondary": ["pedunculate_oak", "sweet_chestnut", "wild_cherry", "field_maple"],
        "climax": ["holly", "yew", "beech"],
        "shrub": ["hazel"],
    },
    "B": {  # Conserve + enrich understory
        "pioneer": [],  # no pioneers in established broadleaf
        "secondary": ["field_maple", "wild_cherry", "crab_apple"],
        "climax": ["holly", "yew", "wild_service", "small_leaved_lime"],
        "shrub": ["hazel", "spindle", "guelder_rose"],
        "ground": ["bluebell", "wood_anemone", "wild_garlic", "primrose"],
    },
    "C": {  # Syntropic Miyawaki: dense mixed pioneer
        "pioneer": ["birch", "common_alder", "grey_alder", "scots_pine", "goat_willow"],
        "secondary": ["pedunculate_oak", "sweet_chestnut", "wild_cherry", "douglas_fir"],
        "shrub": ["hawthorn", "blackthorn", "hazel", "elder"],
    },
    "D": {  # Alley cropping between retained conifers
        "pioneer": ["birch", "common_alder"],
        "secondary": ["pedunculate_oak", "sweet_chestnut", "field_maple"],
        "shrub": ["hazel", "hawthorn"],
    },
    "E": {  # Riparian N-fixer buffers
        "pioneer": ["common_alder", "grey_alder", "goat_willow", "white_willow"],
        "secondary": ["swamp_cypress"],
        "shrub": ["river_willow", "guelder_rose"],
    },
    "F": {  # Ridgeline drought resilience
        "pioneer": ["scots_pine", "birch", "monterey_cypress"],
        "secondary": ["sessile_oak", "sweet_chestnut", "cork_oak"],
        "shrub": ["hawthorn", "blackthorn"],
    },
    "G": {  # Dense hedgerows
        "pioneer": ["hawthorn", "blackthorn"],
        "secondary": ["field_maple", "wild_cherry"],
        "shrub": ["hazel", "dog_rose", "spindle", "guelder_rose", "dogwood", "elder"],
    },
}


def generate_planting_scheme(
    snapshot_records: list[dict],
    scores: dict[str, dict],
    corridors: dict[str, float],
    species_db: dict,
    guild_scorer,
    target_area: Optional[list[str]] = None,
    knowledge_dir: Optional[str] = None,
    classifications: dict[str, BlockClassification] = None,
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
        classifications: {h3_res11: BlockClassification} management class map
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
        # Look up management class for this hex
        mgmt_class = None
        if classifications:
            parent_11 = h3.cell_to_parent(h, 11)
            block_cls = classifications.get(parent_11)
            if block_cls:
                mgmt_class = block_cls.management_class

        species = _select_species(h, rec, candidates, species_db, guild_scorer, corridor, management_class=mgmt_class)
        method = _select_method(rec, corridor, species)

        actions.append(PlantingAction(
            h3_id=h,
            species=species,
            method=method,
            priority=min(10, max(1, int(need * 10))),
            rationale=_generate_rationale(rec, species, method, species_db),
            window=_planting_window(species, species_db),
            management_class=mgmt_class or "",
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
    management_class: str = None,
) -> str:
    """Select species using terrain, climate, succession, and timber objectives.

    When management_class is provided, candidates are filtered to only
    species present in that class's CLASS_SPECIES_PALETTES entry.

    Site data used (from LiDAR + COGs):
      slope_deg   — steep slopes get stabilization species
      twi         — wet areas get riparian, dry get drought-tolerant
      solar_par_kwh — north-facing shade-tolerant, south-facing timber/fruit
      height_m    — existing canopy determines succession stage
      ndvi        — vegetation health indicates soil quality

    Science grounding:
      Gotsch syntropic: stratified succession with N-fixer nurse crops
      Simard 2012: ECM networks need pioneer birch/alder before oak
      Dorset climate 2050: +2-3C, -15% summer rain → drought resilience
      Hedgerow technique: dense thorny shrubs on steep slopes
      ECM technique: oak/beech with birch/alder nurse rings
      Coppice technique: sweet chestnut/hazel for short-rotation timber
    """
    # ── Gather site context ──
    slope = rec.get("slope_deg") or 7  # median for Hooke Park
    twi = rec.get("twi") or 3  # topographic wetness index
    solar = rec.get("solar_par_kwh") or 900  # PAR kWh/m2/yr
    own_height = rec.get("height_m", 0)
    ndvi = rec.get("ndvi") or 0.6
    is_empty = rec.get("status") == "lost" or rec.get("species_detected") is None
    is_corridor = corridor_strength > 0.3

    # Neighbour context
    nb_species = set()
    nb_heights = []
    for nb_h in rec.get("neighbours", []):
        nb_rec = all_records.get(nb_h, {})
        sp = nb_rec.get("species_detected")
        if sp:
            nb_species.add(sp)
            nb_heights.append(nb_rec.get("height_m", 0))

    avg_nb_height = sum(nb_heights) / max(len(nb_heights), 1)
    under_canopy = avg_nb_height > 8 or own_height > 10
    under_mature = avg_nb_height > 15

    # ── Classify site type from terrain ──
    is_steep = slope > 15
    is_wet = twi > 4.5
    is_dry = twi < 1.5
    is_shaded = solar < 700  # north-facing or valley floor
    is_sunny = solar > 950

    # ── Determine succession target ──
    if is_empty and not under_canopy:
        target_succ = "pioneer"
    elif under_mature:
        target_succ = "climax"
    elif under_canopy:
        target_succ = "secondary"
    else:
        target_succ = "pioneer"

    if is_corridor and is_empty:
        target_succ = "pioneer"

    # ── Score each candidate species ──
    scores = {}
    for sp_id, sp in species_db.items():
        # Filter to class palette if management class is specified
        if management_class and management_class in CLASS_SPECIES_PALETTES:
            palette = CLASS_SPECIES_PALETTES[management_class]
            allowed = set()
            for spp_list in palette.values():
                allowed.update(spp_list)
            if sp_id not in allowed:
                continue

        stratum = sp.get("stratum", "canopy")
        if stratum == "ground" and not under_mature:
            continue  # ground layer only under established canopy

        score = 0.0

        # 1. SUCCESSION MATCH (0.30)
        sp_succ = sp.get("succession", "secondary")
        if sp_succ == target_succ:
            score += 0.30
        elif target_succ == "pioneer" and sp_succ != "pioneer":
            score -= 0.25

        # 2. TERRAIN SUITABILITY (0.25)
        shade_tol = sp.get("shade_tolerance", 0.5)
        drought_tol = sp.get("drought_tolerance", 0.5)

        # Steep slopes: hedgerow/stabilization species
        if is_steep:
            if stratum in ("shrub",) and drought_tol > 0.4:
                score += 0.15  # hawthorn, blackthorn, hazel for slope binding
            if sp.get("growth_rate", 0) > 0.1:
                score += 0.05  # fast root establishment
            if stratum == "emergent":
                score -= 0.10  # tall trees on steep slopes = windthrow risk

        # Wet areas: riparian species
        if is_wet:
            if drought_tol < 0.3:
                score += 0.15  # alder, willow, goat willow thrive in wet
            if sp.get("nitrogen_role") == "fixer":
                score += 0.10  # alder in riparian = N-fixation + bank stability
            if drought_tol > 0.7:
                score -= 0.15  # scots pine, cork oak don't belong in wet

        # Dry areas: drought-resilient species
        if is_dry:
            score += drought_tol * 0.20  # strong drought tolerance signal
            if drought_tol < 0.3:
                score -= 0.15  # willow, alder, beech struggle on dry sites

        # North-facing / shaded: shade-tolerant climax
        if is_shaded:
            score += shade_tol * 0.15  # beech, yew, holly, western hemlock
            if shade_tol < 0.2:
                score -= 0.10  # scots pine, birch need sun

        # South-facing / sunny: timber trees, fruit
        if is_sunny:
            if stratum in ("canopy", "emergent"):
                score += 0.08  # good timber growing conditions
            if shade_tol > 0.7:
                score -= 0.05  # shade lovers waste the sun

        # 3. TIMBER OBJECTIVE (0.15)
        # Hooke Park mandate: high quality timber
        if sp_id in ("pedunculate_oak", "sessile_oak"):
            score += 0.15  # 80-120yr premium hardwood
            if is_sunny and not is_steep:
                score += 0.05  # best sites for oak
        elif sp_id == "douglas_fir":
            score += 0.12  # 50yr construction timber, tallest tree
            if is_sunny:
                score += 0.05
        elif sp_id == "sweet_chestnut":
            score += 0.10  # 25yr coppice cycle, versatile
            if is_dry:
                score += 0.05  # drought tolerant
        elif sp_id in ("scots_pine", "birch"):
            score += 0.05  # construction/craft timber

        # 4. N-FIXER NURSE CROPS (0.15)
        # Science: Simard ECM — alder/birch nurse rings around oak
        if sp.get("nitrogen_role") == "fixer":
            if not under_canopy:
                score += 0.20  # critical for soil prep
            # Extra bonus if neighbours are heavy feeders (oak, ash)
            for nb_sp in nb_species:
                nb_data = species_db.get(nb_sp, {})
                if nb_data.get("nitrogen_role") == "heavy_feeder":
                    score += 0.10  # alder next to oak = perfect

        # 5. GUILD COMPATIBILITY (0.10)
        for nb_sp in nb_species:
            score += guild_scorer(sp_id, nb_sp) * 0.15

        # 6. CLIMATE RESILIENCE 2050 (0.05)
        # Dorset 2050: +2-3C, -15% summer rain
        if drought_tol > 0.5:
            score += 0.04  # future-proofing
        if sp_id in ("cork_oak", "monterey_cypress", "sweet_chestnut"):
            score += 0.03  # Mediterranean species becoming viable

        # 7. BIODIVERSITY BONUS
        # Rare strata get priority to fill all layers
        if stratum in ("understory", "shrub") and not under_canopy:
            score += 0.03  # structural diversity
        if stratum == "ground" and under_mature:
            score += 0.05  # bluebell, wild garlic under climax canopy

        # 8. CORRIDOR CONNECTIVITY
        if is_corridor:
            score += sp.get("growth_rate", 0.05) * 1.5  # fast growers for corridors

        scores[sp_id] = score

    if not scores:
        return "common_alder"

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
