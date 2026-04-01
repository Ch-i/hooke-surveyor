"""Multi-dimensional score engine for trees and blocks.

Six dimensions (matching the bonete engine in CyberneticHooke frontend):

  1. Vitality     — NDVI trajectory, growth rate, crown health
  2. Resilience   — stability under stress, drought tolerance, genetic diversity
  3. Symbiosis    — guild compatibility, mycorrhizal network potential, N-cycling
  4. Productivity — carbon sequestration rate, timber value, growth velocity
  5. Biodiversity — species richness in neighbourhood, habitat structural diversity
  6. Soil         — TWI (moisture), slope stability, organic matter potential

Each dimension: 0–1. Overall score = weighted combination.
Management objectives shift the weights.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import geopandas as gpd

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Management objective weights (must sum to 1.0)."""

    vitality: float = 0.20
    resilience: float = 0.20
    symbiosis: float = 0.15
    productivity: float = 0.15
    biodiversity: float = 0.15
    soil: float = 0.15

    def normalize(self):
        total = (self.vitality + self.resilience + self.symbiosis
                 + self.productivity + self.biodiversity + self.soil)
        if total > 0:
            self.vitality /= total
            self.resilience /= total
            self.symbiosis /= total
            self.productivity /= total
            self.biodiversity /= total
            self.soil /= total


# Preset weight profiles for different management objectives
WEIGHT_PRESETS = {
    "balanced": ScoringWeights(),
    "carbon": ScoringWeights(vitality=0.15, resilience=0.10, symbiosis=0.10,
                              productivity=0.35, biodiversity=0.10, soil=0.20),
    "biodiversity": ScoringWeights(vitality=0.10, resilience=0.15, symbiosis=0.25,
                                    productivity=0.05, biodiversity=0.30, soil=0.15),
    "timber": ScoringWeights(vitality=0.25, resilience=0.15, symbiosis=0.05,
                              productivity=0.35, biodiversity=0.05, soil=0.15),
    "resilience": ScoringWeights(vitality=0.15, resilience=0.35, symbiosis=0.20,
                                  productivity=0.05, biodiversity=0.15, soil=0.10),
    "restoration": ScoringWeights(vitality=0.10, resilience=0.15, symbiosis=0.20,
                                   productivity=0.10, biodiversity=0.25, soil=0.20),
}


def compute_scores(
    trees: gpd.GeoDataFrame,
    weights: Optional[ScoringWeights] = None,
) -> dict[str, dict]:
    """Compute 6D scores for every tree.

    Returns {h3_13: {vitality, resilience, symbiosis, productivity, biodiversity, soil, overall}}
    """
    if weights is None:
        weights = WEIGHT_PRESETS["balanced"]
    weights.normalize()

    scores = {}

    for _, row in trees.iterrows():
        h = row.get("h3_13") or row.get("h3")
        if not h:
            continue

        v = _vitality(row)
        r = _resilience(row)
        s = _symbiosis(row)
        p = _productivity(row)
        b = _biodiversity(row)
        sl = _soil(row)

        overall = (
            weights.vitality * v + weights.resilience * r + weights.symbiosis * s
            + weights.productivity * p + weights.biodiversity * b + weights.soil * sl
        )

        scores[h] = {
            "vitality": round(v, 3),
            "resilience": round(r, 3),
            "symbiosis": round(s, 3),
            "productivity": round(p, 3),
            "biodiversity": round(b, 3),
            "soil": round(sl, 3),
            "overall": round(overall, 3),
        }

    logger.info(
        f"Scores computed for {len(scores)} trees. "
        f"Overall: μ={np.mean([s['overall'] for s in scores.values()]):.3f}"
    )
    return scores


def score_block(
    scores: dict[str, dict],
    block_h3_ids: list[str],
) -> dict:
    """Aggregate individual tree scores to a block level."""
    block_scores = [scores[h] for h in block_h3_ids if h in scores]
    if not block_scores:
        return {}

    dims = ["vitality", "resilience", "symbiosis", "productivity", "biodiversity", "soil", "overall"]
    return {
        dim: round(np.mean([s[dim] for s in block_scores]), 3)
        for dim in dims
    }


# ── Dimension scorers ────────────────────────────────────────────────────────

def _vitality(row) -> float:
    """NDVI health + growth trajectory."""
    score = 0.0
    n = 0

    ndvi = row.get("ndvi")
    if ndvi is not None and not np.isnan(ndvi):
        score += np.clip(ndvi / 0.8, 0, 1)
        n += 1

    growth = row.get("growth_m_yr")
    if growth is not None and not np.isnan(growth):
        score += np.clip(growth / 0.5, 0, 1)  # 0.5 m/yr = excellent
        n += 1

    trend = row.get("ndvi_trend")
    if trend is not None and not np.isnan(trend):
        # Positive trend = improving
        score += np.clip(0.5 + trend * 5, 0, 1)
        n += 1

    return score / max(1, n)


def _resilience(row) -> float:
    """Stability under stress, drought tolerance."""
    score = 0.5  # baseline

    # Low risk = high resilience
    risk = row.get("risk_overall")
    if risk is not None and not np.isnan(risk):
        score = 1.0 - risk

    # Multi-scan persistence = proven resilience
    n_scans = row.get("n_scans", 1)
    if n_scans >= 3:
        score = score * 0.7 + 0.3  # boost for persistent trees

    # Status
    status = row.get("status")
    if status == "persistent":
        score = min(1.0, score + 0.1)
    elif status == "lost":
        score *= 0.2

    return np.clip(score, 0, 1)


def _symbiosis(row) -> float:
    """Guild compatibility with neighbours."""
    guild = row.get("guild_score")
    if guild is not None and not np.isnan(guild):
        # guild_score is [-1, 1], map to [0, 1]
        return np.clip((guild + 1) / 2, 0, 1)

    # If no guild score, check species diversity in neighbourhood
    nb_species = row.get("neighbour_species", [])
    if nb_species:
        unique = len(set(s for s in nb_species if s and s != "empty"))
        return np.clip(unique / 4, 0, 1)  # 4+ unique species = excellent

    return 0.3  # unknown


def _productivity(row) -> float:
    """Carbon sequestration + growth rate."""
    score = 0.0
    n = 0

    carbon = row.get("carbon_tco2")
    if carbon is not None and not np.isnan(carbon):
        score += np.clip(carbon / 5.0, 0, 1)  # 5 tCO2 = excellent
        n += 1

    growth = row.get("growth_m_yr")
    if growth is not None and not np.isnan(growth):
        score += np.clip(growth / 0.5, 0, 1)
        n += 1

    height = row.get("height_m")
    if height is not None and not np.isnan(height):
        score += np.clip(height / 30, 0, 1)  # 30m = mature productive tree
        n += 1

    return score / max(1, n)


def _biodiversity(row) -> float:
    """Species richness in neighbourhood + structural diversity."""
    score = 0.0
    n = 0

    # Neighbour species richness
    nb_species = row.get("neighbour_species", [])
    if nb_species:
        unique = len(set(s for s in nb_species if s and s != "empty"))
        occupied = sum(1 for s in nb_species if s and s != "empty")
        score += np.clip(unique / 4, 0, 1) * 0.5  # species richness
        score += np.clip(occupied / 6, 0, 1) * 0.5  # occupancy
        n += 1

    # Height = structural diversity indicator
    height = row.get("height_m")
    if height is not None and not np.isnan(height):
        # Mid-height trees contribute most to structural diversity
        score += 0.5 * (1 - abs(height - 15) / 15)
        n += 1

    return np.clip(score / max(1, n), 0, 1)


def _soil(row) -> float:
    """Soil health indicators: moisture, stability, organic matter potential."""
    score = 0.0
    n = 0

    twi = row.get("twi")
    if twi is not None and not np.isnan(twi):
        # Sweet spot: TWI 6-10 (not too dry, not waterlogged)
        score += 1.0 - abs(twi - 8) / 8
        n += 1

    slope = row.get("slope_deg")
    if slope is not None and not np.isnan(slope):
        score += np.clip(1.0 - slope / 30, 0, 1)  # flat = stable
        n += 1

    # Canopy cover → organic matter input
    crown = row.get("crown_area_m2")
    if crown is not None and not np.isnan(crown):
        score += np.clip(crown / 30, 0, 1)  # large crown = lots of leaf litter
        n += 1

    return np.clip(score / max(1, n), 0, 1)
