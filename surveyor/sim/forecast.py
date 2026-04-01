"""Multi-year forecast runner — GoL simulation with intervention scenarios.

Runs counterfactual scenarios:
  A. Do nothing (baseline trajectory)
  B. Apply planting scheme (intervention)
  C. Compare outcomes (carbon, biodiversity, resilience)
"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_forecast(
    snapshot_records: list[dict],
    species_db: dict,
    guild_scorer,
    interventions: Optional[list[dict]] = None,
    years: int = 30,
    checkpoints: Optional[list[int]] = None,
) -> dict:
    """Run parallel forecasts: baseline vs intervention.

    Args:
        snapshot_records: current tree snapshot
        species_db: species parameters
        guild_scorer: compatibility function
        interventions: list of {"h3": str, "action": "plant"|"thin", "species": str}
        years: forecast horizon
        checkpoints: years to snapshot (default: [3, 7, 10, 15, 30])

    Returns:
        {
            "baseline": {year: summary},
            "intervention": {year: summary},  # only if interventions provided
            "comparison": {...},
        }
    """
    from .gol import ForestGoL, GoLConfig

    if checkpoints is None:
        checkpoints = [3, 7, 10, 15, 30]
    checkpoints = [y for y in checkpoints if y <= years]

    config = GoLConfig()

    # Baseline run
    logger.info(f"Running baseline forecast ({years} years)")
    baseline_gol = ForestGoL(species_db, guild_scorer, config)
    baseline_gol.seed_from_snapshot(snapshot_records)

    baseline_snapshots = {}
    for year in range(1, years + 1):
        summary = baseline_gol.step()
        if year in checkpoints:
            baseline_snapshots[year] = _snapshot_state(baseline_gol, summary)

    result = {"baseline": baseline_snapshots}

    # Intervention run
    if interventions:
        logger.info(f"Running intervention forecast ({years} years, {len(interventions)} actions)")
        intervention_gol = ForestGoL(species_db, guild_scorer, config)
        intervention_gol.seed_from_snapshot(snapshot_records)
        intervention_gol.apply_intervention(interventions)

        intervention_snapshots = {}
        for year in range(1, years + 1):
            summary = intervention_gol.step()
            if year in checkpoints:
                intervention_snapshots[year] = _snapshot_state(intervention_gol, summary)

        result["intervention"] = intervention_snapshots
        result["comparison"] = _compare(baseline_snapshots, intervention_snapshots, checkpoints)

    return result


def _snapshot_state(gol, summary: dict) -> dict:
    """Capture GoL state at a checkpoint."""
    alive = [c for c in gol.grid.values() if c.is_alive]

    species_counts = {}
    total_height = 0
    total_health = 0
    for cell in alive:
        species_counts[cell.species] = species_counts.get(cell.species, 0) + 1
        total_height += cell.height_m
        total_health += cell.health

    n = len(alive) or 1
    return {
        **summary,
        "species_richness": len(species_counts),
        "species_counts": species_counts,
        "mean_height_m": round(total_height / n, 2),
        "mean_health": round(total_health / n, 3),
        "canopy_cover_pct": round(sum(c.canopy_cover for c in alive) / max(1, len(gol.grid)) * 100, 1),
    }


def _compare(baseline: dict, intervention: dict, checkpoints: list[int]) -> dict:
    """Compare baseline vs intervention at each checkpoint."""
    comparison = {}
    for year in checkpoints:
        b = baseline.get(year, {})
        i = intervention.get(year, {})
        comparison[year] = {
            "alive_delta": i.get("alive", 0) - b.get("alive", 0),
            "species_richness_delta": i.get("species_richness", 0) - b.get("species_richness", 0),
            "mean_height_delta_m": round(i.get("mean_height_m", 0) - b.get("mean_height_m", 0), 2),
            "mean_health_delta": round(i.get("mean_health", 0) - b.get("mean_health", 0), 3),
            "canopy_cover_delta_pct": round(
                i.get("canopy_cover_pct", 0) - b.get("canopy_cover_pct", 0), 1
            ),
        }
    return comparison
