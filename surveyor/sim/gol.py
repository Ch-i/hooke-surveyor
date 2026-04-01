"""Game of Life on H3 grid — ecological succession, competition, mutualism.

Each H3 res-13 hex has a state:
    species, age, health (0-1), height_m, canopy_cover

Transition rules model:
    - Succession: pioneers establish, shade out, climax replaces
    - Competition: taller same-stratum neighbours suppress
    - Mutualism: guild-compatible neighbours boost health
    - Reproduction: seed dispersal to empty neighbours (distance by species)
    - Mortality: age, drought, competition stress
    - Intervention: planted species appear, thinned species removed
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import h3
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CellState:
    """State of a single H3 hex at a point in time."""

    h3_id: str
    species: Optional[str] = None  # None = empty
    age_years: float = 0
    health: float = 1.0  # 0 = dead, 1 = thriving
    height_m: float = 0.0
    canopy_cover: float = 0.0  # 0-1, fraction of hex covered

    @property
    def is_empty(self) -> bool:
        return self.species is None

    @property
    def is_alive(self) -> bool:
        return self.species is not None and self.health > 0


@dataclass
class GoLConfig:
    """Simulation parameters."""

    dt_years: float = 1.0  # time step
    competition_radius: int = 1  # H3 k-ring radius for competition
    seed_dispersal_radius: int = 2  # max rings for seed dispersal
    mortality_threshold: float = 0.1  # health below this → death
    natural_regen_prob: float = 0.02  # chance of spontaneous seedling per empty hex/year
    drought_stress: float = 0.0  # 0 = normal, 1 = severe (global modifier)


class ForestGoL:
    """Game of Life cellular automaton on the H3 res-13 grid.

    Runs forward in yearly steps, modeling succession + competition + mutualism.
    """

    def __init__(self, species_db: dict, guild_scorer, config: Optional[GoLConfig] = None):
        """
        Args:
            species_db: {species_id: {...}} from species.json
            guild_scorer: callable(sp_a, sp_b) → float [-1, 1]
            config: simulation parameters
        """
        self.species_db = species_db
        self.guild_score = guild_scorer
        self.config = config or GoLConfig()
        self.grid: dict[str, CellState] = {}
        self.year: float = 0
        self.history: list[dict] = []  # snapshots for playback

    def seed_from_snapshot(self, snapshot_records: list[dict]):
        """Initialize grid from a tree snapshot."""
        for rec in snapshot_records:
            h = rec.get("h3_13") or rec.get("h3")
            if not h:
                continue
            self.grid[h] = CellState(
                h3_id=h,
                species=rec.get("species_detected"),
                age_years=0,  # unknown — estimate from height
                health=min(1.0, rec.get("ndvi", 0.6) / 0.8),
                height_m=rec.get("height_m", 0),
                canopy_cover=min(1.0, rec.get("crown_area_m2", 0) / 1.44),  # hex area ~1.44m²
            )
        logger.info(f"GoL seeded with {len(self.grid)} cells, "
                     f"{sum(1 for c in self.grid.values() if c.is_alive)} alive")

    def step(self) -> dict:
        """Advance one time step. Returns summary of changes."""
        births, deaths, growth_events = 0, 0, 0
        new_grid = {}

        for h, cell in self.grid.items():
            neighbours = self._get_neighbours(h)

            if cell.is_alive:
                new_cell = self._update_alive(cell, neighbours)
                if not new_cell.is_alive:
                    deaths += 1
                else:
                    growth_events += 1
                new_grid[h] = new_cell
            else:
                new_cell = self._try_recruit(cell, neighbours)
                if new_cell.is_alive:
                    births += 1
                new_grid[h] = new_cell

        self.grid = new_grid
        self.year += self.config.dt_years

        summary = {
            "year": self.year,
            "alive": sum(1 for c in self.grid.values() if c.is_alive),
            "empty": sum(1 for c in self.grid.values() if c.is_empty),
            "births": births,
            "deaths": deaths,
        }
        self.history.append(summary)
        return summary

    def run(self, years: int) -> list[dict]:
        """Run simulation for N years."""
        logger.info(f"Running GoL for {years} years from year {self.year}")
        steps = int(years / self.config.dt_years)
        for _ in range(steps):
            self.step()
        logger.info(f"GoL complete: year {self.year}, "
                     f"{sum(1 for c in self.grid.values() if c.is_alive)} alive")
        return self.history

    def apply_intervention(self, interventions: list[dict]):
        """Apply planting/thinning interventions to the grid.

        Each intervention: {"h3": str, "action": "plant"|"thin", "species": str}
        """
        for iv in interventions:
            h = iv["h3"]
            if h not in self.grid:
                continue
            if iv["action"] == "plant":
                self.grid[h] = CellState(
                    h3_id=h, species=iv["species"],
                    age_years=0, health=0.8, height_m=0.3, canopy_cover=0.05,
                )
            elif iv["action"] == "thin":
                self.grid[h] = CellState(h3_id=h)  # empty

    # ── Internal rules ───────────────────────────────────────────────────

    def _update_alive(self, cell: CellState, neighbours: list[CellState]) -> CellState:
        """Update a living cell for one timestep."""
        sp = self.species_db.get(cell.species, {})
        dt = self.config.dt_years

        # Growth
        growth_rate = sp.get("growth_rate", 0.05)
        max_h = sp.get("max_height_m", 20)
        height_gain = growth_rate * (1 - cell.height_m / max_h) * cell.health * dt
        new_height = min(max_h, cell.height_m + height_gain)

        # Competition pressure (same-stratum neighbours)
        competition = 0.0
        mutualism = 0.0
        for nb in neighbours:
            if not nb.is_alive:
                continue
            nb_sp = self.species_db.get(nb.species, {})
            if nb_sp.get("stratum") == sp.get("stratum"):
                # Taller neighbour suppresses
                if nb.height_m > cell.height_m:
                    competition += 0.05
            # Guild compatibility
            score = self.guild_score(cell.species, nb.species)
            if score > 0:
                mutualism += score * 0.02
            elif score < 0:
                competition += abs(score) * 0.03

        # Shade tolerance modifies competition impact
        shade_tol = sp.get("shade_tolerance", 0.5)
        competition *= (1 - shade_tol)

        # Drought stress
        drought_tol = sp.get("drought_tolerance", 0.5)
        drought_hit = self.config.drought_stress * (1 - drought_tol) * 0.1

        # Health update
        health_delta = mutualism - competition - drought_hit
        new_health = np.clip(cell.health + health_delta * dt, 0, 1)

        # Mortality check
        if new_health < self.config.mortality_threshold:
            return CellState(h3_id=cell.h3_id)  # dead → empty

        return CellState(
            h3_id=cell.h3_id,
            species=cell.species,
            age_years=cell.age_years + dt,
            health=round(new_health, 3),
            height_m=round(new_height, 2),
            canopy_cover=min(1.0, cell.canopy_cover + 0.02 * dt),
        )

    def _try_recruit(self, cell: CellState, neighbours: list[CellState]) -> CellState:
        """Try to recruit a new tree into an empty cell via seed dispersal."""
        if cell.is_alive:
            return cell

        # Collect seed sources from neighbours
        seed_sources = {}
        for nb in neighbours:
            if nb.is_alive and nb.health > 0.5 and nb.age_years > 5:
                sp = nb.species
                seed_sources[sp] = seed_sources.get(sp, 0) + 1

        if not seed_sources:
            # Random natural regeneration (wind/bird dispersal)
            if np.random.random() < self.config.natural_regen_prob:
                # Pioneer species most likely
                pioneers = [
                    sid for sid, s in self.species_db.items()
                    if s.get("succession") == "pioneer" and s.get("stratum") in ("canopy", "understory")
                ]
                if pioneers:
                    species = np.random.choice(pioneers)
                    return CellState(
                        h3_id=cell.h3_id, species=species,
                        age_years=0, health=0.6, height_m=0.1, canopy_cover=0.01,
                    )
            return cell

        # Most abundant seed source wins (weighted by count)
        total = sum(seed_sources.values())
        if np.random.random() < min(0.3, total * 0.05):  # recruitment probability
            species = max(seed_sources, key=seed_sources.get)
            return CellState(
                h3_id=cell.h3_id, species=species,
                age_years=0, health=0.7, height_m=0.1, canopy_cover=0.01,
            )

        return cell

    def _get_neighbours(self, h: str) -> list[CellState]:
        """Get cell states for k-ring neighbours."""
        ring = h3.k_ring(h, self.config.competition_radius) - {h}
        return [self.grid[nb] for nb in ring if nb in self.grid]
