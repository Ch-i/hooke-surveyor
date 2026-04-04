"""Block classification engine — assigns management class A-G to each H3 res-11 block.

Uses aggregated tree structure and terrain features to classify blocks into
silvicultural management classes from STRATEGY.md.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import h3
import numpy as np

logger = logging.getLogger(__name__)

CONIFER_SPECIES = {
    "scots_pine", "douglas_fir", "monterey_cypress",
    "western_hemlock", "coast_redwood", "swamp_cypress",
}

STRATEGY_NAMES = {
    "A": "Monoculture Conifer — CCF Transition",
    "B": "Semi-Natural Broadleaf — Conserve & Enrich",
    "C": "Open Ground — Syntropic Establishment",
    "D": "Transition Zone — Alley Cropping CCF",
    "E": "Riparian Corridor — N-Fixer Buffers",
    "F": "Ridgeline / Exposed — Drought Resilience",
    "G": "Estate Boundary — Dense Hedgerows",
}


@dataclass
class BlockClassification:
    """Classification result for a single H3 res-11 block."""
    h3_11: str
    management_class: str   # A, B, C, D, E, F, G
    confidence: float
    strategy_name: str
    features: dict = field(default_factory=dict)
    tree_count: int = 0


def _aggregate_block_features(trees: list[dict]) -> dict:
    """Compute aggregate features for a set of trees within a block.

    Returns a dict with keys:
        mean_height, height_cv, conifer_fraction, empty_fraction,
        mean_twi, mean_slope, tree_count, alive_count
    """
    total = len(trees)
    if total == 0:
        return {
            "mean_height": 0.0,
            "height_cv": 0.0,
            "conifer_fraction": 0.0,
            "empty_fraction": 1.0,
            "mean_twi": None,
            "mean_slope": None,
            "tree_count": 0,
            "alive_count": 0,
        }

    # Alive = not lost and has a species
    alive = [
        t for t in trees
        if t.get("status") != "lost" and t.get("species_detected") is not None
    ]
    alive_count = len(alive)

    # Empty fraction: lost OR no species
    empty_count = total - alive_count
    empty_fraction = empty_count / total

    # Heights from alive trees with height > 0
    heights = np.array([
        t["height_m"] for t in alive
        if t.get("height_m") is not None and t["height_m"] > 0
    ], dtype=np.float64)

    if len(heights) > 0:
        mean_height = float(np.mean(heights))
        std_height = float(np.std(heights))
        height_cv = std_height / mean_height if mean_height > 0 else 0.0
    else:
        mean_height = 0.0
        height_cv = 0.0

    # Conifer fraction among alive trees
    if alive_count > 0:
        conifer_count = sum(
            1 for t in alive
            if t.get("species_detected", "").lower() in CONIFER_SPECIES
        )
        conifer_fraction = conifer_count / alive_count
    else:
        conifer_fraction = 0.0

    # TWI — mean of available values
    twi_vals = [
        t["twi"] for t in trees
        if t.get("twi") is not None
    ]
    mean_twi = float(np.mean(twi_vals)) if twi_vals else None

    # Slope — mean of available values
    slope_vals = [
        t["slope_deg"] for t in trees
        if t.get("slope_deg") is not None
    ]
    mean_slope = float(np.mean(slope_vals)) if slope_vals else None

    return {
        "mean_height": round(mean_height, 2),
        "height_cv": round(height_cv, 3),
        "conifer_fraction": round(conifer_fraction, 3),
        "empty_fraction": round(empty_fraction, 3),
        "mean_twi": round(mean_twi, 2) if mean_twi is not None else None,
        "mean_slope": round(mean_slope, 2) if mean_slope is not None else None,
        "tree_count": total,
        "alive_count": alive_count,
    }


def _detect_boundary_blocks(all_h3_11: set[str]) -> set[str]:
    """Find blocks at the convex hull boundary.

    A block is a boundary block if any of its ring-1 neighbours
    is NOT in the set of all blocks.
    """
    boundary = set()
    for cell in all_h3_11:
        try:
            ring = h3.grid_ring(cell, 1)
        except Exception:
            # Pentagons or other edge cases
            ring = h3.grid_disk(cell, 1) - {cell}
        for neighbour in ring:
            if neighbour not in all_h3_11:
                boundary.add(cell)
                break
    return boundary


def classify_block(
    h3_11: str,
    trees: list[dict],
    is_boundary: bool = False,
    species_db: dict = None,
) -> BlockClassification:
    """Classify a single res-11 block based on aggregated tree features.

    Classification priority: E > F > C > A > D > B > G
    Terrain signals (E, F) override structure signals when available.
    """
    feats = _aggregate_block_features(trees)
    total = feats["tree_count"]

    mean_height = feats["mean_height"]
    height_cv = feats["height_cv"]
    conifer_frac = feats["conifer_fraction"]
    empty_frac = feats["empty_fraction"]
    mean_twi = feats["mean_twi"]
    mean_slope = feats["mean_slope"]

    cls = None
    confidence = 0.6

    # --- Priority 1: E — Riparian / Wet ---
    if mean_twi is not None and mean_twi > 4.5:
        cls = "E"
        confidence = 0.9 if mean_twi > 5.5 else 0.6

    # --- Priority 2: F — Ridgeline / Exposed ---
    if cls is None and mean_slope is not None and mean_slope > 20:
        cls = "F"
        confidence = 0.9 if mean_slope > 28 else 0.6

    # --- Priority 3: C — Open Ground / Cleared ---
    if cls is None and (empty_frac > 0.6 or mean_height < 2.0):
        cls = "C"
        if empty_frac > 0.8 or mean_height < 1.0:
            confidence = 0.9
        else:
            confidence = 0.6

    # --- Priority 4: A — Monoculture Conifer ---
    if cls is None and conifer_frac > 0.6 and mean_height > 15:
        cls = "A"
        confidence = 0.9 if conifer_frac > 0.8 and mean_height > 20 else 0.6

    # --- Priority 5: D — Transition / Mixed ---
    if cls is None and 0.3 < conifer_frac < 0.7 and mean_height > 8:
        cls = "D"
        confidence = 0.9 if 0.4 < conifer_frac < 0.6 else 0.6

    # --- Priority 6: B — Semi-Natural Broadleaf ---
    if cls is None and conifer_frac < 0.4 and height_cv > 0.3 and mean_height > 5:
        cls = "B"
        confidence = 0.9 if conifer_frac < 0.2 and height_cv > 0.45 else 0.6

    # --- Priority 7: G — Boundary / Edge ---
    if cls is None and is_boundary:
        cls = "G"
        confidence = 0.9

    # --- Fallback: assign G if boundary, otherwise B (broadleaf default) ---
    if cls is None:
        if is_boundary:
            cls = "G"
            confidence = 0.6
        else:
            # Default: most blocks at Hooke Park are broadleaf-dominant
            cls = "B"
            confidence = 0.6

    return BlockClassification(
        h3_11=h3_11,
        management_class=cls,
        confidence=confidence,
        strategy_name=STRATEGY_NAMES[cls],
        features=feats,
        tree_count=total,
    )


def classify_all_blocks(
    snapshot_records: list[dict],
    species_db: dict = None,
) -> dict[str, BlockClassification]:
    """Aggregate res-13 trees to res-11 blocks and classify each.

    Parameters
    ----------
    snapshot_records : list[dict]
        Each record must have at least ``h3_13`` (str).
        Optional but used: height_m, status, species_detected, twi, slope_deg.
    species_db : dict, optional
        Reserved for future species lookup enrichment.

    Returns
    -------
    dict[str, BlockClassification]
        Keyed by H3 res-11 index.
    """
    # Group trees by res-11 parent
    blocks: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for rec in snapshot_records:
        h3_13 = rec.get("h3_13")
        if not h3_13:
            skipped += 1
            continue
        try:
            parent = h3.cell_to_parent(h3_13, 11)
        except Exception:
            skipped += 1
            continue
        blocks[parent].append(rec)

    if skipped:
        logger.warning("Skipped %d records with invalid/missing h3_13", skipped)

    logger.info(
        "Grouped %d trees into %d res-11 blocks",
        len(snapshot_records) - skipped,
        len(blocks),
    )

    # Detect boundary blocks
    all_h3_11 = set(blocks.keys())
    boundary_blocks = _detect_boundary_blocks(all_h3_11)
    logger.info("Boundary blocks: %d / %d", len(boundary_blocks), len(all_h3_11))

    # Classify each block
    results: dict[str, BlockClassification] = {}
    for h3_11, trees in blocks.items():
        is_boundary = h3_11 in boundary_blocks
        results[h3_11] = classify_block(
            h3_11, trees, is_boundary=is_boundary, species_db=species_db,
        )

    # Log summary
    counts = Counter(c.management_class for c in results.values())
    total_blocks = len(results)
    logger.info("=== Block Classification Summary ===")
    for cls in sorted(counts):
        n = counts[cls]
        pct = n / total_blocks * 100 if total_blocks else 0
        logger.info(
            "  Class %s (%s): %d blocks (%.1f%%)",
            cls, STRATEGY_NAMES[cls], n, pct,
        )
    logger.info("Total: %d blocks classified", total_blocks)

    return results
