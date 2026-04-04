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
            ndvi = rec.get("ndvi") or 0.6
            height = rec.get("height_m") or 0
            crown = rec.get("crown_area_m2") or 0
            species = rec.get("species_detected")

            # Assign placeholder species from height when detection is missing
            # (crown segmentation produces geometry only, no species classification)
            if not species and height >= 2:
                if height > 15:
                    species = "scots_pine"    # dominant conifer at Hooke
                elif height > 5:
                    species = "birch"         # common secondary broadleaf
                else:
                    species = "hazel"         # understory/shrub layer

            self.grid[h] = CellState(
                h3_id=h,
                species=species,
                age_years=max(0, height * 2),  # rough age estimate from height
                health=min(1.0, ndvi / 0.8),
                height_m=height,
                canopy_cover=min(1.0, crown / 1.44),
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
        """Update a living cell — realistic for a working forest.

        Key principle: established trees (h > 5m) are resilient. They survived
        decades of competition already. Only genuine ecological stress kills:
        - Deep shade on shade-intolerant pioneers (succession driver)
        - Pioneer senescence after natural lifespan
        - Severe drought on moisture-dependent species

        New plantings (h < 2m) are vulnerable to all stresses.
        Mature trees get a stability bonus proportional to their size.
        """
        sp = self.species_db.get(cell.species, {})
        dt = self.config.dt_years
        succession = sp.get("succession", "secondary")

        # Maturity: established trees resist casual mortality
        is_established = cell.height_m > 5.0 or cell.age_years > 10
        is_seedling = cell.height_m < 2.0
        # Stability factor: bigger trees are harder to kill (0.1 for seedlings, 1.0 for 30m trees)
        stability = min(1.0, cell.height_m / 30.0) if is_established else 0.1

        # Growth
        growth_rate = sp.get("growth_rate", 0.05)
        max_h = sp.get("max_height_m", 20)
        height_gain = growth_rate * (1 - cell.height_m / max_h) * cell.health * dt
        new_height = min(max_h, cell.height_m + height_gain)

        # Competition + mutualism
        competition = 0.0
        mutualism = 0.0
        canopy_above = 0
        n_fixer_nearby = False

        for nb in neighbours:
            if not nb.is_alive:
                continue
            nb_sp = self.species_db.get(nb.species, {})

            # Same-stratum competition — only affects seedlings and suppressed trees
            if nb_sp.get("stratum") == sp.get("stratum"):
                if nb.height_m > cell.height_m * 1.5 and is_seedling:
                    competition += 0.06  # seedlings under taller same-stratum = real stress
                elif nb.height_m > cell.height_m * 1.2 and not is_established:
                    competition += 0.02

            # Canopy shading — only stresses shade-intolerant species
            if nb.height_m > cell.height_m + 5 and nb.canopy_cover > 0.4:
                canopy_above += 1

            # Guild compatibility
            score = self.guild_score(cell.species, nb.species)
            if score > 0:
                mutualism += score * 0.03
            elif score < 0 and is_seedling:
                competition += abs(score) * 0.02  # only seedlings suffer from antagonism

            # N-fixer facilitation
            if nb_sp.get("nitrogen_role") == "fixer" and nb.health > 0.4:
                n_fixer_nearby = True

        # Shade stress — drives succession but only kills shade-intolerant pioneers
        shade_tol = sp.get("shade_tolerance", 0.5)
        if canopy_above > 0 and shade_tol < 0.3:
            # Light-demanding pioneers under dense canopy genuinely struggle
            shade_stress = canopy_above * 0.06 * (1 - shade_tol)
            if is_established:
                shade_stress *= 0.3  # established pioneers resist longer
            competition += shade_stress

        # Pioneer senescence — natural lifespan (birch ~60yr, alder ~80yr)
        if succession == "pioneer":
            lifespan = 60 if sp.get("growth_rate", 0) > 0.1 else 80
            if cell.age_years > lifespan * 0.7:
                senescence = (cell.age_years - lifespan * 0.7) / lifespan * 0.05
                competition += senescence

        # N-fixer benefit
        if n_fixer_nearby:
            if sp.get("nitrogen_role") == "heavy_feeder":
                mutualism += 0.06
            else:
                mutualism += 0.02  # all species benefit somewhat from N

        # Established tree resilience — reduce total competition impact
        if is_established:
            competition *= (1.0 - stability * 0.7)  # 30m tree: competition reduced 70%

        # Drought stress (global modifier)
        drought_tol = sp.get("drought_tolerance", 0.5)
        drought_hit = self.config.drought_stress * (1 - drought_tol) * 0.05

        # Health update — mutualism heals, competition + drought hurts
        health_delta = mutualism - competition - drought_hit
        new_health = np.clip(cell.health + health_delta * dt, 0, 1)

        # Mortality — established trees need sustained stress to die
        mort_threshold = self.config.mortality_threshold
        if is_established:
            mort_threshold *= 0.5  # established trees survive at lower health

        if new_health < mort_threshold:
            return CellState(h3_id=cell.h3_id)  # dead

        # Canopy cover
        max_cover = min(1.0, new_height / max_h * 1.2)
        new_cover = min(max_cover, cell.canopy_cover + 0.02 * cell.health * dt)

        return CellState(
            h3_id=cell.h3_id,
            species=cell.species,
            age_years=cell.age_years + dt,
            health=round(new_health, 3),
            height_m=round(new_height, 2),
            canopy_cover=round(new_cover, 3),
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
        ring = set(h3.grid_disk(h, self.config.competition_radius)) - {h}
        return [self.grid[nb] for nb in ring if nb in self.grid]
