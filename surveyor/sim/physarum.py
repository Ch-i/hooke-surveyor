"""Physarum polycephalum network optimization on H3 grid.

Slime mold finds minimum-cost networks connecting food sources.
Applied to forestry: finds optimal ecological corridors for planting.

Nodes = H3 res-13 hexes (trees or empty)
Edges = connections between adjacent hexes
Food  = high-value ecological nodes (seed sources, habitat nodes, water)
Cost  = difficulty of establishment (slope, dry soil, competition)

The physarum finds the network that connects all food sources with
minimum total cost — this IS the optimal planting corridor layout.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import h3
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhysarumConfig:
    """Physarum model parameters."""

    n_agents: int = 500  # number of virtual agents exploring the grid
    decay_rate: float = 0.05  # pheromone decay per step
    deposit_rate: float = 1.0  # pheromone deposited by each agent
    sensitivity: float = 2.0  # how strongly agents follow pheromone
    food_strength: float = 10.0  # pheromone boost at food sources
    cost_weight: float = 1.0  # how much terrain cost affects movement
    steps: int = 200  # simulation steps
    prune_threshold: float = 0.1  # edges below this pheromone are pruned


@dataclass
class PhysarumAgent:
    """A single exploring agent."""

    h3_id: str
    heading: int = 0  # index into neighbour ring (0-5)


class PhysarumNetwork:
    """Physarum-inspired network optimizer.

    Finds the minimum spanning network connecting ecological food sources
    through the H3 grid, respecting terrain costs.
    """

    def __init__(self, config: Optional[PhysarumConfig] = None):
        self.config = config or PhysarumConfig()
        self.pheromone: dict[str, float] = {}  # h3 → pheromone level
        self.edge_flow: dict[tuple[str, str], float] = {}  # (h3_a, h3_b) → flow
        self.food_sources: set[str] = set()
        self.cost_map: dict[str, float] = {}  # h3 → traversal cost (0-1)
        self.grid_cells: set[str] = set()
        self.agents: list[PhysarumAgent] = []

    def setup(
        self,
        grid_cells: list[str],
        food_sources: list[str],
        cost_map: Optional[dict[str, float]] = None,
    ):
        """Initialize the network.

        Args:
            grid_cells: all H3 hex IDs in the domain
            food_sources: H3 IDs of high-value nodes to connect
            cost_map: {h3: cost} where 0=easy, 1=hard to traverse
        """
        self.grid_cells = set(grid_cells)
        self.food_sources = set(food_sources)
        self.cost_map = cost_map or {}

        # Initialize pheromone
        for h in grid_cells:
            self.pheromone[h] = 0.1
        for h in food_sources:
            self.pheromone[h] = self.config.food_strength

        # Spawn agents at food sources
        self.agents = []
        per_source = max(1, self.config.n_agents // max(1, len(food_sources)))
        for source in food_sources:
            for _ in range(per_source):
                self.agents.append(PhysarumAgent(
                    h3_id=source,
                    heading=np.random.randint(0, 6),
                ))

        logger.info(
            f"Physarum setup: {len(grid_cells)} cells, "
            f"{len(food_sources)} food sources, {len(self.agents)} agents"
        )

    def run(self) -> dict[str, float]:
        """Run the physarum simulation.

        Returns {h3: pheromone_level} — high pheromone = part of optimal network.
        """
        for step in range(self.config.steps):
            self._step()

            if step % 50 == 0:
                active = sum(1 for h, p in self.pheromone.items() if p > self.config.prune_threshold)
                logger.info(f"Step {step}: {active} active cells")

        return dict(self.pheromone)

    def extract_network(self) -> list[str]:
        """Extract the emergent network as a list of H3 cells above threshold.

        These are the cells where planting should happen — the ecological corridors.
        """
        threshold = self.config.prune_threshold
        network = [h for h, p in self.pheromone.items() if p > threshold]
        network.sort(key=lambda h: self.pheromone[h], reverse=True)
        logger.info(f"Network extracted: {len(network)} cells (of {len(self.grid_cells)} total)")
        return network

    def get_corridor_strength(self) -> dict[str, float]:
        """Get normalized corridor strength (0-1) for each cell.

        Use this for visualization — opacity = corridor strength.
        """
        if not self.pheromone:
            return {}
        max_p = max(self.pheromone.values())
        if max_p == 0:
            return {h: 0 for h in self.pheromone}
        return {h: p / max_p for h, p in self.pheromone.items()}

    # ── Internal ─────────────────────────────────────────────────────────

    def _step(self):
        """One simulation step: move agents → deposit → decay → reinforce food."""
        # Move each agent
        for agent in self.agents:
            self._move_agent(agent)

        # Deposit pheromone
        for agent in self.agents:
            self.pheromone[agent.h3_id] = (
                self.pheromone.get(agent.h3_id, 0) + self.config.deposit_rate
            )

        # Decay
        for h in self.pheromone:
            self.pheromone[h] *= (1 - self.config.decay_rate)

        # Reinforce food sources
        for h in self.food_sources:
            self.pheromone[h] = max(self.pheromone.get(h, 0), self.config.food_strength)

    def _move_agent(self, agent: PhysarumAgent):
        """Move an agent towards higher pheromone, biased by terrain cost."""
        neighbours = sorted(h3.k_ring(agent.h3_id, 1) - {agent.h3_id})
        valid = [n for n in neighbours if n in self.grid_cells]

        if not valid:
            return

        # Score each neighbour: pheromone attraction - terrain cost
        scores = []
        for nb in valid:
            pheromone = self.pheromone.get(nb, 0)
            cost = self.cost_map.get(nb, 0.5) * self.config.cost_weight
            score = max(0.001, pheromone ** self.config.sensitivity - cost)
            scores.append(score)

        # Stochastic selection (weighted by score)
        total = sum(scores)
        probs = [s / total for s in scores]
        idx = np.random.choice(len(valid), p=probs)

        agent.h3_id = valid[idx]


def find_planting_corridors(
    snapshot_records: list[dict],
    food_criteria: Optional[dict] = None,
    config: Optional[PhysarumConfig] = None,
) -> dict[str, float]:
    """High-level API: find optimal planting corridors from a tree snapshot.

    Args:
        snapshot_records: list of tree snapshot dicts
        food_criteria: how to identify food sources
            {"min_height": 15, "min_guild_score": 0.3, "status": "persistent"}
        config: physarum parameters

    Returns:
        {h3: corridor_strength} normalized 0-1
    """
    if food_criteria is None:
        food_criteria = {"min_height": 12, "status": "persistent"}

    all_cells = []
    food_cells = []
    cost_map = {}

    for rec in snapshot_records:
        h = rec.get("h3_13") or rec.get("h3")
        if not h:
            continue
        all_cells.append(h)

        # Determine if this is a food source
        is_food = True
        if "min_height" in food_criteria:
            if rec.get("height_m", 0) < food_criteria["min_height"]:
                is_food = False
        if "status" in food_criteria:
            if rec.get("status") != food_criteria["status"]:
                is_food = False
        if "min_guild_score" in food_criteria:
            if rec.get("guild_score", 0) < food_criteria["min_guild_score"]:
                is_food = False

        if is_food:
            food_cells.append(h)

        # Terrain cost: steep + dry = hard to establish
        slope = rec.get("slope_deg", 0)
        twi = rec.get("twi", 8)
        cost = np.clip(slope / 40, 0, 0.5) + np.clip(1 - twi / 12, 0, 0.5)
        cost_map[h] = cost

    physarum = PhysarumNetwork(config=config)
    physarum.setup(all_cells, food_cells, cost_map)
    physarum.run()
    return physarum.get_corridor_strength()
